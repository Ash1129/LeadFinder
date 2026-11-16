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
nothing is unpickled. The model is one JSON value in the settings table.
"""
from __future__ import annotations

import copy
import functools
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


def _measure(y: list[int], scores: list[float]) -> dict:
    """How well a ranking finds the converters: area under the ROC curve, average precision, and the share of
    converters among the top tenth with its lift over the base rate."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    k = max(1, round(len(y) * TOP_SHARE))
    # ties (the points table has many) are broken by position, the same for every scorer
    top = sorted(range(len(y)), key=lambda i: (-scores[i], i))[:k]
    hit = sum(y[i] for i in top) / k
    base = sum(y) / len(y)
    return {"auc": round(float(roc_auc_score(y, scores)), 3), "ap": round(float(average_precision_score(y, scores)), 3),
            "top": round(hit, 3), "lift": round(hit / base, 2) if base else None}


@functools.lru_cache(maxsize=4)
def _fit(n: int, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    X, y, points = simulate(n, seed)
    cut = int(n * 0.7)                                         # the rows are independent draws: the first 70% train
    Xa, ya, Xb, yb = X[:cut], y[:cut], X[cut:], y[cut:]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xa, ya)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, random_state=0).fit(Xa, ya)
    train_p = sorted(float(p) for p in lr.predict_proba(Xa)[:, 1])
    return {
        "version": f"lr-{n}-{seed}",
        "kind": "logistic regression",
        "labels": "simulated",
        "features": FEATURES,
        "coef": [round(float(c), 4) for c in lr.coef_[0]],
        "intercept": round(float(lr.intercept_[0]), 4),
        "means": [round(sum(row[j] for row in Xa) / len(Xa), 4) for j in range(len(FEATURES))],
        "bands": {"hot": round(train_p[int(len(train_p) * 0.8)], 4), "warm": round(train_p[int(len(train_p) * 0.5)], 4)},
        "n_train": len(ya), "n_test": len(yb), "base_rate": round(sum(yb) / len(yb), 3), "top_share": TOP_SHARE,
        "metrics": {
            "model": _measure(yb, [float(p) for p in lr.predict_proba(Xb)[:, 1]]),
            "points": _measure(yb, [float(s) for s in points[cut:]]),
            "challenger": _measure(yb, [float(p) for p in gb.predict_proba(Xb)[:, 1]]),
        },
    }


def train(conn: sqlite3.Connection, n: int = 8000, seed: int = 11) -> dict:
    """Fit on a simulated history, measure on the held-out part, and make the result the live model."""
    model = copy.deepcopy(_fit(n, seed)) | {"trained_at": time.time()}
    set_setting(conn, SETTING, model)
    return model


def load(conn: sqlite3.Connection) -> Optional[dict]:
    model = get_setting(conn, SETTING)
    return model if model and model.get("features") == FEATURES else None      # a model for other features is not this code's


def ensure(conn: sqlite3.Connection) -> dict:
    return load(conn) or train(conn)


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
