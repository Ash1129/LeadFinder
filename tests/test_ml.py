import json
import math

from leadfinder import ml, synth
from leadfinder.db import connect
from leadfinder.seed import apply_model, seed

NOW = ml.TRAIN_NOW
GOOD = {"specialty": "Pediatrics", "providers": 4, "ehr": "athenahealth"}
POOR = {"specialty": "Cardiology", "providers": 40, "ehr": "Epic"}


def model():
    return ml._benchmarked(ml.DEMO_RUNS)                    # the version fitted on the whole demo history


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
    m = ml.ensure(conn)
    assert ml.load(conn)["version"] == m["version"] == "demo-v6" and apply_model(conn) == 12 and apply_model(conn) == 0
    row = conn.execute("SELECT chance, chance_band, chance_reasons, model_version FROM leads ORDER BY chance DESC LIMIT 1").fetchone()
    assert 0 < row["chance"] < 1 and row["chance_band"] in ("hot", "warm", "cool") and json.loads(row["chance_reasons"])
    conn.execute("UPDATE settings SET value=? WHERE key='model'", (json.dumps(m | {"features": ["something else"]}),))
    assert ml.load(conn) is None                                 # a model for other features is not served


# ---------------- the runs and the scorecard ----------------
def card(top, auc, ap, ece, version="x"):
    return {"version": version, "metrics": {"model": {"top": top, "auc": auc, "ap": ap, "ece": ece}}}


def test_the_scorecard_wants_more_wins_than_losses_and_nothing_past_its_allowance():
    live = card(0.780, 0.810, 0.650, 0.020, "demo-v1")
    first = ml.verdict(live, None)
    assert first["approve"] and first["against"] is None and first["card"] == {}
    won = ml.verdict(card(0.775, 0.812, 0.652, 0.020), live)                       # two better, one worse within the allowance
    assert won["approve"] and won["card"] == {"top": "worse", "auc": "better", "ap": "better", "ece": "same"}
    assert won["sentence"] == "Went live over demo-v1: better on AUC and average precision; worse on top tenth, within the allowance."
    tied = ml.verdict(card(0.775, 0.812, 0.650, 0.020), live)                      # one each: not reason enough
    assert not tied["approve"] and tied["sentence"].startswith("Sent back: not better than demo-v1 overall")
    same = ml.verdict(card(0.780, 0.810, 0.650, 0.020), live)
    assert not same["approve"] and "the same as demo-v1 on every measure" in same["sentence"]
    blocked = ml.verdict(card(0.740, 0.830, 0.700, 0.010), live)                   # better on three, but the top tenth fell too far
    assert not blocked["approve"] and blocked["card"]["top"] == "blocked"
    assert blocked["sentence"] == "Sent back: top tenth 74.0% against demo-v1's 78.0%, worse by more than the allowance."
    assert not ml.verdict(card(0.790, 0.812, 0.652, 0.031), live)["approve"]       # calibration error is the one where lower is better


def test_each_month_adds_outcomes_and_every_version_is_judged_on_the_same_leads():
    fitted = [ml._candidate(m)["n_train"] for m in range(1, ml.DEMO_RUNS + 3)]
    assert fitted == [600, 1600, 2600, 3600, 4600, 5600, 6600, 7600]
    assert {ml._candidate(m)["n_test"] for m in range(1, ml.DEMO_RUNS + 3)} == {2400}
    assert len({json.dumps(ml._candidate(m)["metrics"]["points"]) for m in range(1, 8)}) == 1    # the table never changes
    assert ml._candidate(3) == ml._candidate(3) and "ece" in ml._candidate(3)["metrics"]["model"]


def test_a_new_database_gets_six_runs_one_of_them_live(tmp_path):
    conn = connect(tmp_path / "runs.db")
    live = ml.ensure(conn)
    runs = ml.runs(conn)
    assert [r["version"] for r in runs] == [f"demo-v{m}" for m in range(1, 7)]
    assert [r["status"] for r in runs].count("live") == 1 and runs[0]["status"] == "retired"
    assert {r["status"] for r in runs} == {"live", "retired", "sent_back"}
    assert next(r for r in runs if r["status"] == "live")["version"] == live["version"] and "challenger" in live["metrics"]
    for r in runs:                                                     # every decision is the scorecard's, against the live one of its day
        assert r["verdict"].startswith("Went live" if r["status"] != "sent_back" else "Sent back")
        assert (r["against"] is None) == (r["month"] == 1)
    assert ml.ensure(conn)["version"] == live["version"] and len(ml.runs(conn)) == 6      # a second start changes nothing


def test_train_adds_a_month_and_the_rules_decide(tmp_path):
    conn = connect(tmp_path / "month.db")
    seed(conn, n=8)
    live = ml.ensure(conn)
    apply_model(conn)
    out = ml.train(conn)
    assert out["version"] == "demo-v7" and out["model"]["n_train"] == 6600 and out["status"] in ("live", "sent_back")
    assert ml.load(conn)["version"] == (out["version"] if out["status"] == "live" else live["version"])
    assert apply_model(conn) == (8 if out["status"] == "live" else 0)             # leads are rescored only when the live model changed
    cells = ml.side_by_side(conn)[-1]["cells"]
    assert set(cells) == {"top", "auc", "ap", "ece"} and cells["top"]["value"].endswith("%") and cells["top"]["mark"]


def test_a_database_from_before_the_runs_is_given_them(tmp_path):
    conn = connect(tmp_path / "old.db")
    conn.execute("INSERT INTO settings(key, value) VALUES('model', ?)", (json.dumps(ml._candidate(2) | {"version": "lr-8000-11"}),))
    assert ml.load(conn)["version"] == "lr-8000-11"
    assert ml.ensure(conn)["version"].startswith("demo-v") and len(ml.runs(conn)) == 6
