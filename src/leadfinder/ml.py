"""The trained scorer: a logistic regression that turns a practice's record and visits into a chance of converting,
with each feature's share of that chance shown to the rep.

Training needs outcomes, and there are no real ones yet, so the history is simulated (synth.py): thousands of
made-up practices, each with a won-or-lost label drawn from the simulator's hidden rule. The model is fitted on
70% of them and judged on the other 30%, beside the points table (scoring.py) and a gradient-boosted challenger.
Those numbers say how well each scorer recovers the simulator's pattern, relative to the others; they are not a
forecast of real conversion, and the pages say so wherever a percentage appears.

Logistic regression is the live model because every prediction decomposes exactly: the log-odds are the average
lead's plus one term per feature, so "why 34%" has a true answer, not an approximation. The challenger is there to
show what a model that can learn interactions gains, and is reported, not served.

scikit-learn fits and measures; serving is the stored coefficients and a sigmoid, so a page view never loads it and
nothing is unpickled. The live model is one JSON value in the settings table.

There is more than one version. The history arrives a simulated month at a time; each month a new version is fitted
on everything so far and marked against the live one on a scorecard of four measures (`MEASURES`, `verdict`). It
goes live only if it wins the card; either way it is kept in the `model_runs` table with the decision, and the
model's page shows every version side by side.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import random
import sqlite3
import time
from typing import Iterable, Optional

from . import synth
from .db import get_setting, set_setting
from .scoring import EHR_INTEGRATED, SPECIALTY_POINTS, score_practice

SETTING = "model"
TRAIN_NOW = 1_800_000_000.0             # the simulated history's "now": fixed, so the same seed trains the same model
PAGES = ("/demo", "/pricing", "/integrations", "/features", "/blog")
SIZES = ("solo", "2-10", "11-25", "26+")
EHR_KINDS = ("integrated", "epic", "other", "unknown")
FEATURES = ([f"specialty:{s}" for s in SPECIALTY_POINTS] + ["specialty:other"] + [f"size:{s}" for s in SIZES]
            + [f"ehr:{k}" for k in EHR_KINDS] + [f"visit:{p}" for p in PAGES] + ["recent"])
GROUPS = ("specialty", "size", "ehr") + tuple(f"visit:{p}" for p in PAGES) + ("recent",)
SHOW_AT = 0.05                          # log-odds: smaller shares than this are left off a lead's page
TOP_SHARE = 0.10                        # "the top tenth": the leads a team would actually call


def size_of(providers: int) -> str:
    return "solo" if providers <= 1 else ("2-10" if providers <= 10 else ("11-25" if providers <= 25 else "26+"))


def ehr_kind(ehr: Optional[str]) -> str:
    return "unknown" if not ehr else ("integrated" if ehr in EHR_INTEGRATED else ("epic" if ehr == "Epic" else "other"))


def featurize(specialty: str, providers: int, ehr: Optional[str], visits: Iterable[tuple[str, float]], now: float) -> list[float]:
    """The practice as the model sees it: one 0/1 per name in FEATURES."""
    visits = list(visits)
    pages = {("/blog" if page.startswith("/blog") else page) for page, _ in visits}
    on = {f"specialty:{specialty if specialty in SPECIALTY_POINTS else 'other'}", f"size:{size_of(providers)}",
          f"ehr:{ehr_kind(ehr)}"} | {f"visit:{p}" for p in pages}
    if visits and max(at for _, at in visits) >= now - 7 * 86400:
        on.add("recent")
    return [1.0 if name in on else 0.0 for name in FEATURES]


def simulate(n: int, seed: int) -> tuple[list[list[float]], list[int], list[int]]:
    """`n` simulated past leads: their features, whether each converted, and what the points table gave it."""
    rng = random.Random(seed)
    X, y, points = [], [], []
    for _ in range(n):
        p, seen = synth.practice(rng), synth.visits(rng, TRAIN_NOW)
        X.append(featurize(p["specialty"], p["providers"], p["ehr"], seen, TRAIN_NOW))
        y.append(1 if rng.random() < synth.conversion_chance(p, seen, TRAIN_NOW, rng) else 0)
        points.append(score_practice(p["specialty"], p["providers"], p["ehr"], seen, now=TRAIN_NOW).total)
    return X, y, points


def _measure(y: list[int], scores: list[float], chances: bool = False) -> dict:
    """How well a ranking finds the converters: area under the ROC curve, average precision, and the share of
    converters among the top tenth with its lift over the base rate. With `chances` (the scores are probabilities),
    also the calibration error: the leads in ten equal groups by chance, and the average gap between what each
    group was given and how often it converted."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    k = max(1, round(len(y) * TOP_SHARE))
    # ties (the points table has many) are broken by position, the same for every scorer
    order = sorted(range(len(y)), key=lambda i: (-scores[i], i))
    hit = sum(y[i] for i in order[:k]) / k
    base = sum(y) / len(y)
    out = {"auc": round(float(roc_auc_score(y, scores)), 3), "ap": round(float(average_precision_score(y, scores)), 3),
           "top": round(hit, 3), "lift": round(hit / base, 2) if base else None}
    if chances:
        size, gap = len(y) / 10, 0.0
        for g in range(10):
            group = order[round(g * size):round((g + 1) * size)]
            gap += abs(sum(scores[i] for i in group) - sum(y[i] for i in group))
        out["ece"] = round(gap / len(y), 4)
    return out


# ---------------------------------------------------------------- the runs: one model a simulated month
# The history arrives a month at a time, as outcomes would. The first model is fitted on the first outcomes; each
# month after adds more and a new version is fitted on everything so far. Every version is judged on the same
# held-out leads, which none of them trained on, so the columns on the model's page can be read against each other.
POOL_N, POOL_SEED = 8000, 11            # the simulated history the demo starts with; its last 30% is the held-out test
FIRST, MONTH, DEMO_RUNS = 600, 1000, 6  # outcomes behind the first model, outcomes a month adds, runs made on first start
STATUS_WORDS = {"live": "live", "retired": "was live", "sent_back": "sent back"}
# The scorecard a new version is put to, against the live one, on the held-out leads:
# (key in the metrics, name, +1 when higher is better, the allowance: how much worse still counts as noise, why it is there).
MEASURES = (("top", "Top tenth", 1, 0.025, "the leads a team calls first; the allowance is about one standard error on that many leads"),
            ("auc", "AUC", 1, 0.005, "the whole ranking, top to bottom"),
            ("ap", "Average precision", 1, 0.005, "how tightly the converters are packed at the top"),
            ("ece", "Calibration error", -1, 0.01, "whether a 30% chance keeps meaning about 30 in 100"))


def show(key: str, value: float) -> str:
    """A measure as the pages print it."""
    return f"{value * 100:.1f}%" if key == "top" else (f"{value * 100:.1f} points" if key == "ece" else f"{value:.3f}")


@functools.lru_cache(maxsize=1)
def _pool() -> tuple:
    X, y, points = simulate(POOL_N, POOL_SEED)
    return X, y, points, int(POOL_N * 0.7)                    # the rows are independent draws: the first 70% train


@functools.lru_cache(maxsize=32)
def _later_month(month: int) -> tuple:
    """A month after the demo's history runs out: fresh simulated outcomes, the same for the same month."""
    X, y, _ = simulate(MONTH, POOL_SEED * 1000 + month)
    return X, y


def _window(month: int) -> tuple[list[list[float]], list[int]]:
    """Everything known by the end of `month`: the outcomes a version made that month is fitted on."""
    X, y, _, cut = _pool()
    n = min(cut, FIRST + MONTH * (month - 1))
    Xa, ya = list(X[:n]), list(y[:n])
    for m in range(DEMO_RUNS + 1, month + 1):
        Xm, ym = _later_month(m)
        Xa += Xm
        ya += ym
    return Xa, ya


@functools.lru_cache(maxsize=32)
def _candidate(month: int) -> dict:
    """The version fitted at the end of `month`, measured on the held-out leads beside the points table."""
    from sklearn.linear_model import LogisticRegression
    X, y, points, cut = _pool()
    Xa, ya = _window(month)
    Xb, yb = X[cut:], y[cut:]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xa, ya)
    train_p = sorted(float(p) for p in lr.predict_proba(Xa)[:, 1])
    return {
        "version": f"demo-v{month}",
        "month": month,
        "kind": "logistic regression",
        "labels": "simulated",
        "features": FEATURES,
        "coef": [round(float(c), 4) for c in lr.coef_[0]],
        "intercept": round(float(lr.intercept_[0]), 4),
        "means": [round(sum(row[j] for row in Xa) / len(Xa), 4) for j in range(len(FEATURES))],
        "bands": {"hot": round(train_p[int(len(train_p) * 0.8)], 4), "warm": round(train_p[int(len(train_p) * 0.5)], 4)},
        "n_train": len(ya), "n_test": len(yb), "base_rate": round(sum(yb) / len(yb), 3), "top_share": TOP_SHARE,
        "metrics": {
            "model": _measure(yb, [float(p) for p in lr.predict_proba(Xb)[:, 1]], chances=True),
            "points": _measure(yb, [float(s) for s in points[cut:]]),
        },
    }


@functools.lru_cache(maxsize=8)
def _benchmarked(month: int) -> dict:
    """The version with the challenger beside it: gradient-boosted trees fitted on the same outcomes. Only the live
    version gets one (it is the benchmark on the model's page), so a start fits one set of trees, not six."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y, _, cut = _pool()
    Xa, ya = _window(month)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, random_state=0).fit(Xa, ya)
    model = copy.deepcopy(_candidate(month))
    model["metrics"]["challenger"] = _measure(y[cut:], [float(p) for p in gb.predict_proba(X[cut:])[:, 1]], chances=True)
    return model


def verdict(candidate: dict, live: Optional[dict]) -> dict:
    """The sign-off: {approve, sentence, against, card}. The card marks each measure better, same or worse than the
    live version's, or `blocked` when it is worse by more than the allowance. A candidate goes live when nothing
    is blocked and it is better on more measures than it is worse on: it has seen more outcomes than the live
    version, so a card it wins is reason enough, and a card it ties or loses is not."""
    if live is None:
        return {"approve": True, "against": None, "card": {}, "sentence": "Went live: the first model, with nothing to compare it against."}
    c, l, name = candidate["metrics"]["model"], live["metrics"]["model"], live["version"]
    card = {}
    for key, _, sign, allowance, _ in MEASURES:
        gain = round(sign * (c[key] - l[key]), 4)
        card[key] = "better" if gain > 0 else ("same" if gain == 0 else ("worse" if gain >= -allowance else "blocked"))
    names = {key: label.lower() if key != "auc" else label for key, label, *_ in MEASURES}
    listed = lambda mark: [names[k] for k in card if card[k] == mark]                                   # noqa: E731
    both = lambda k: f"{show(k, c[k])} against {name}'s {show(k, l[k])}"                               # noqa: E731
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]                 # noqa: E731
    blocked = [k for k in card if card[k] == "blocked"]
    if blocked:
        k = blocked[0]
        return {"approve": False, "against": name, "card": card,
                "sentence": f"Sent back: {names[k]} {both(k)}, worse by more than the allowance."}
    better, worse = listed("better"), listed("worse")
    tail = f"; worse on {join(worse)}, within the allowance" if worse else ""
    if len(better) > len(worse):
        return {"approve": True, "against": name, "card": card, "sentence": f"Went live over {name}: better on {join(better)}{tail}."}
    if not better and not worse:
        return {"approve": False, "against": name, "card": card, "sentence": f"Sent back: the same as {name} on every measure."}
    return {"approve": False, "against": name, "card": card,
            "sentence": f"Sent back: not better than {name} overall" + (f" (better on {join(better)}{tail})." if better else f" (worse on {join(worse)}).")}


def runs(conn: sqlite3.Connection) -> list[dict]:
    """Every version in the registry, oldest first: version, month, status, verdict, trained_at and the model."""
    rows = conn.execute("SELECT version, month, status, verdict, against, card, trained_at, model FROM model_runs ORDER BY month").fetchall()
    return [dict(r) | {"model": json.loads(r["model"]), "card": json.loads(r["card"]), "word": STATUS_WORDS[r["status"]]} for r in rows]


def side_by_side(conn: sqlite3.Connection) -> list[dict]:
    """The registry as the model's page shows it: each version's measures as printed, with the mark each got on its
    scorecard (against the version that was live when it was judged, which the page names)."""
    out = runs(conn)
    for r in out:
        m = r["model"]["metrics"]["model"]
        r |= {"n_train": r["model"]["n_train"], "lift": m["lift"],
              "cells": {key: {"value": show(key, m[key]), "mark": r["card"].get(key)} for key, *_ in MEASURES}}
    return out


def _add_run(conn: sqlite3.Connection, month: int, now: Optional[float] = None) -> dict:
    """Fit the version for `month`, put it to the sign-off, and record it. The one that goes live replaces the live
    model (the previous one is kept as `retired`); the rest are kept as `sent_back`, with the reason."""
    now = time.time() if now is None else now
    candidate = _candidate(month)
    decision = verdict(candidate, load(conn))
    if decision["approve"]:
        candidate = _benchmarked(month)
        conn.execute("UPDATE model_runs SET status='retired' WHERE status='live'")
        set_setting(conn, SETTING, copy.deepcopy(candidate) | {"trained_at": now})
    conn.execute("INSERT INTO model_runs(version, month, status, verdict, against, card, trained_at, model) VALUES(?,?,?,?,?,?,?,?)",
                 (candidate["version"], month, "live" if decision["approve"] else "sent_back", decision["sentence"],
                  decision["against"], json.dumps(decision["card"]), now, json.dumps(candidate)))
    return {"version": candidate["version"], "status": "live" if decision["approve"] else "sent_back",
            "verdict": decision["sentence"], "model": candidate}


def train(conn: sqlite3.Connection, now: Optional[float] = None) -> dict:
    """One more simulated month: fit a version on everything so far and let the rules decide whether it goes live.
    On an empty registry this is the first model."""
    last = conn.execute("SELECT MAX(month) m FROM model_runs").fetchone()["m"] or 0
    return _add_run(conn, last + 1, now)


def load(conn: sqlite3.Connection) -> Optional[dict]:
    """The live model, or None when there is none or it was fitted for other features than this code's."""
    model = get_setting(conn, SETTING)
    return model if model and model.get("features") == FEATURES else None


def ensure(conn: sqlite3.Connection) -> dict:
    """The live model. A database without a registry (a new one, or one from before there were runs) gets the
    demo's: the first model and a version for each simulated month after it, each put to the sign-off in turn."""
    live = load(conn)
    if live is None or not conn.execute("SELECT 1 FROM model_runs WHERE version=? AND status='live'", (live["version"],)).fetchone():
        conn.execute("DELETE FROM model_runs")
        conn.execute("DELETE FROM settings WHERE key=?", (SETTING,))
        for _ in range(DEMO_RUNS):
            train(conn)
    return load(conn)


# ---------------------------------------------------------------- serving: coefficients and a sigmoid
def _label(group: str, practice: dict, on: bool) -> tuple[str, str]:
    """(what the rep reads, where the fact came from) for one group of features on one practice."""
    if group == "specialty":
        return f"Specialty: {practice['specialty']}", "NPI record"
    if group == "size":
        n = practice["providers"]
        return f"{n} provider{'s' if n != 1 else ''}", "NPI record"
    if group == "ehr":
        ehr = practice.get("ehr")
        return (f"Runs {ehr}" + (", which the product integrates with" if ehr in EHR_INTEGRATED else "") if ehr else "EHR unknown"), "NPI record"
    if group == "recent":
        return ("Visited in the last 7 days" if on else "No visit in the last 7 days"), "site visits"
    page = group.split(":", 1)[1]
    return (f"Visited {page}" if on else f"No visit to {page}"), "site visits"


def predict(model: dict, practice: dict, visits: Iterable[tuple[str, float]], now: Optional[float] = None) -> dict:
    """The chance, its band, and why: one line per fact, each with how far it moved the log-odds from the average
    simulated lead's (so the lines and the average add up to the prediction exactly)."""
    x = featurize(practice["specialty"], practice["providers"], practice.get("ehr"), visits, time.time() if now is None else now)
    coef, means = model["coef"], model["means"]
    z = model["intercept"] + sum(c * v for c, v in zip(coef, x))
    chance = 1 / (1 + math.exp(-z))
    reasons = []
    for group in GROUPS:
        idx = [j for j, name in enumerate(model["features"]) if name == group or (group in ("specialty", "size", "ehr") and name.startswith(group + ":"))]
        effect = sum(coef[j] * (x[j] - means[j]) for j in idx)
        if abs(effect) >= SHOW_AT:
            label, source = _label(group, practice, any(x[j] for j in idx))
            reasons.append({"label": label, "effect": round(effect, 3), "source": source})
    reasons.sort(key=lambda r: -abs(r["effect"]))
    band = "hot" if chance >= model["bands"]["hot"] else ("warm" if chance >= model["bands"]["warm"] else "cool")
    return {"chance": round(chance, 4), "band": band, "reasons": reasons, "version": model["version"]}


def learned(model: dict) -> list[dict]:
    """The coefficients for the model's page, strongest push up first, in plain words."""
    names = {"recent": "Visited in the last 7 days", "ehr:integrated": "EHR the product integrates with", "ehr:epic": "Epic",
             "ehr:other": "Another EHR", "ehr:unknown": "EHR unknown", "size:solo": "Solo practice", "size:2-10": "2 to 10 providers",
             "size:11-25": "11 to 25 providers", "size:26+": "26 or more providers", "specialty:other": "Another specialty"}
    rows = []
    for name, c in zip(model["features"], model["coef"]):
        kind, _, value = name.partition(":")
        rows.append({"label": names.get(name) or (f"Visited {value}" if kind == "visit" else value), "coef": c,
                     "odds": round(math.exp(c), 2), "source": "site visits" if kind in ("visit", "recent") else "NPI record"})
    return sorted(rows, key=lambda r: -r["coef"])
