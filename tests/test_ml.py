import json
import math

from leadfinder import ml, synth
from leadfinder.db import connect
from leadfinder.seed import apply_model, seed

NOW = ml.TRAIN_NOW
GOOD = {"specialty": "Pediatrics", "providers": 4, "ehr": "athenahealth"}
POOR = {"specialty": "Cardiology", "providers": 40, "ehr": "Epic"}


def model():
    return ml._fit(8000, 11)


def test_features_are_one_per_group():
    x = dict(zip(ml.FEATURES, ml.featurize("Pediatrics", 4, "athenahealth", [("/blog/waitlists", NOW - 86400), ("/demo", NOW - 9 * 86400)], NOW)))
    on = {k for k, v in x.items() if v}
    assert on == {"specialty:Pediatrics", "size:2-10", "ehr:integrated", "visit:/blog", "visit:/demo", "recent"}
    x = dict(zip(ml.FEATURES, ml.featurize("Urology", 40, None, [], NOW)))
    assert {k for k, v in x.items() if v} == {"specialty:other", "size:26+", "ehr:unknown"}


def test_the_simulated_history_is_the_same_for_the_same_seed():
    a, b = ml.simulate(300, 5), ml.simulate(300, 5)
    assert a == b and ml.simulate(300, 6) != a
    X, y, points = a
    assert len(X) == len(y) == len(points) == 300 and 0 < sum(y) < 300 and all(len(row) == len(ml.FEATURES) for row in X)


def test_the_model_beats_the_points_table_on_leads_it_never_saw():
    m = model()
    got, table = m["metrics"]["model"], m["metrics"]["points"]
    assert m["n_train"] == 5600 and m["n_test"] == 2400 and m["labels"] == "simulated"
    assert got["auc"] > table["auc"] > 0.7 and got["ap"] > table["ap"]
    assert got["top"] >= table["top"] and got["lift"] > 2
    assert abs(m["metrics"]["challenger"]["auc"] - got["auc"]) < 0.05           # trees gain little here: reported, not served


def test_it_learned_the_direction_of_the_hidden_rule_without_being_told():
    weights = {r["label"]: r["coef"] for r in ml.learned(model())}
    assert weights["Visited /demo"] > weights["Visited /pricing"] > weights["Visited /features"] > 0
    assert weights["2 to 10 providers"] > 0 > weights["26 or more providers"]
    assert weights["EHR the product integrates with"] > 0 > weights["Epic"]
    assert abs(weights["Visited /blog"]) < 0.15                                    # the simulator ignores blog visits; the points table pays for them


def test_a_prediction_is_the_average_lead_plus_its_reasons():
    m = model()
    visits = [("/demo", NOW - 86400), ("/pricing", NOW - 2 * 86400)]
    out = ml.predict(m | {"bands": m["bands"]}, GOOD, visits, now=NOW)
    assert out["band"] == "hot" and out["chance"] > 0.7 and out["reasons"][0]["effect"] > 0
    assert {r["source"] for r in out["reasons"]} == {"NPI record", "site visits"}
    # log-odds = the average lead's + every group's effect (also the ones too small to show)
    x = ml.featurize(GOOD["specialty"], GOOD["providers"], GOOD["ehr"], visits, NOW)
    average = m["intercept"] + sum(c * mu for c, mu in zip(m["coef"], m["means"]))
    effects = sum(c * (v - mu) for c, v, mu in zip(m["coef"], x, m["means"]))
    assert math.isclose(1 / (1 + math.exp(-(average + effects))), out["chance"], abs_tol=1e-3)
    poor = ml.predict(m, POOR, [], now=NOW)
    assert poor["band"] == "cool" and poor["chance"] < 0.1
    assert any(r["label"] == "No visit to /demo" and r["effect"] < 0 for r in poor["reasons"])


def test_the_hidden_rule_is_not_the_points_table():
    # same record, the simulator's two interaction cases: size on an integrated EHR, and /integrations on one
    base = synth.conversion_chance({"specialty": "Pediatrics", "providers": 4, "ehr": None}, [], NOW)
    assert synth.conversion_chance(GOOD, [], NOW) > base
    assert synth.conversion_chance(GOOD, [("/blog/waitlists", NOW - 30 * 86400)], NOW) == synth.conversion_chance(GOOD, [], NOW)


def test_train_stores_the_model_and_every_lead_gets_a_chance(tmp_path):
    conn = connect(tmp_path / "ml.db")
    assert ml.load(conn) is None and apply_model(conn) == 0
    seed(conn, n=12)                                             # before any model: points only
    assert conn.execute("SELECT COUNT(*) n FROM leads WHERE chance IS NOT NULL").fetchone()["n"] == 0
    m = ml.train(conn, 8000, 11)
    assert ml.load(conn)["version"] == m["version"] == "lr-8000-11" and apply_model(conn) == 12 and apply_model(conn) == 0
    row = conn.execute("SELECT chance, chance_band, chance_reasons, model_version FROM leads ORDER BY chance DESC LIMIT 1").fetchone()
    assert 0 < row["chance"] < 1 and row["chance_band"] in ("hot", "warm", "cool") and json.loads(row["chance_reasons"])
    conn.execute("UPDATE settings SET value=? WHERE key='model'", (json.dumps(m | {"features": ["something else"]}),))
    assert ml.load(conn) is None                                 # a model for other features is not served
