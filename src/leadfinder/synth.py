"""The generators behind every synthetic thing: what a made-up practice looks like, what its visits look like, and,
for training only, whether it would have converted.

`seed.py` uses the first two to fill the demo database; `ml.py` uses all three to build the labelled history a real
deployment would have collected from sales outcomes. The conversion rule here is the simulator's own secret: it is
not the points table (scoring.py) and the model never sees it, only the outcomes it produces. It shares the points
table's general direction but not its weights, ignores blog visits, adds an interaction the table cannot express
(a practice of the right size on an integrated EHR), and has noise for everything a record does not show. So a
benchmark on it says which scorer recovers a hidden pattern better, not what real practices do.
"""
from __future__ import annotations

import math
import random
from typing import Optional

PLACES = ["Maple Grove", "Cedar Ridge", "Lakeside", "Riverbend", "Stonebridge", "Willow Creek", "Harbor View",
          "Fox Hollow", "Pine Hill", "Meadowbrook", "Elm Street", "Northgate", "Bayside", "Clearwater", "Oak Park"]
SPECIALTIES = ["Family Medicine", "Pediatrics", "Dermatology", "Internal Medicine", "Orthopedic Surgery",
               "Physical Therapy", "Cardiology", "Ophthalmology"]
SUFFIX = {"Family Medicine": "Family Medicine", "Pediatrics": "Pediatrics", "Dermatology": "Dermatology",
          "Internal Medicine": "Internal Medicine Associates", "Orthopedic Surgery": "Orthopedics",
          "Physical Therapy": "Physical Therapy", "Cardiology": "Heart Center", "Ophthalmology": "Eye Care"}
CITIES = [("Ithaca", "NY"), ("Rochester", "NY"), ("Syracuse", "NY"), ("Burlington", "VT"), ("Hartford", "CT"),
          ("Scranton", "PA"), ("Albany", "NY"), ("Worcester", "MA"), ("Portland", "ME"), ("Providence", "RI")]
EHRS = ["athenahealth", "eClinicalWorks", "NextGen", "Epic", "Practice Fusion", None]
PROVIDERS = [1, 2, 3, 4, 5, 6, 8, 12, 18, 40]
VISIT_COUNTS = [0, 1, 1, 2, 3, 4, 6]
PAGES = ["/demo", "/pricing", "/integrations", "/features", "/blog/no-show-rates", "/blog/waitlists", "/"]
PAGE_WEIGHTS = [2, 3, 2, 3, 3, 2, 6]
VISIT_DAYS = 21
INTEGRATED = {"athenahealth", "eClinicalWorks", "NextGen"}


def practice(rng: random.Random) -> dict:
    """The facts a registry record gives: specialty, providers, city, state, EHR (None when unknown)."""
    city, state = rng.choice(CITIES)
    return {"specialty": rng.choice(SPECIALTIES), "providers": rng.choice(PROVIDERS), "city": city, "state": state,
            "ehr": rng.choice(EHRS)}


def visits(rng: random.Random, now: float) -> list[tuple[str, float]]:
    """(page, unix time) pairs over the last three weeks; often none."""
    return [(rng.choices(PAGES, PAGE_WEIGHTS)[0], now - rng.uniform(0, VISIT_DAYS) * 86400)
            for _ in range(rng.choice(VISIT_COUNTS))]


# ---------------------------------------------------------------- the hidden rule (training labels only)
_SPECIALTY = {"Family Medicine": 0.9, "Pediatrics": 1.1, "Dermatology": 0.3, "Internal Medicine": 0.7,
              "Orthopedic Surgery": 0.2, "Physical Therapy": 0.6, "Cardiology": -0.2, "Ophthalmology": 0.0}
_PAGE = {"/demo": 1.3, "/pricing": 0.9, "/integrations": 0.5, "/features": 0.2}
_BASE, _NOISE = -3.4, 0.6


def conversion_chance(p: dict, seen: list[tuple[str, float]], now: float, rng: Optional[random.Random] = None) -> float:
    """The simulator's chance that this practice books a demo. With `rng`, unobserved noise is added."""
    z = _BASE + _SPECIALTY.get(p["specialty"], 0.0)
    n = p["providers"]
    z += -0.6 if n <= 1 else (0.8 if n <= 10 else (0.2 if n <= 25 else -0.9))
    ehr = p.get("ehr")
    z += 0.7 if ehr in INTEGRATED else (-0.8 if ehr == "Epic" else (-0.1 if ehr else 0.0))
    if 2 <= n <= 10 and ehr in INTEGRATED:
        z += 0.5                                              # the right size on an EHR the product works inside
    pages = {page for page, _ in seen}
    z += sum(w for page, w in _PAGE.items() if page in pages)
    if "/integrations" in pages and ehr in INTEGRATED:
        z += 0.6
    if seen and max(at for _, at in seen) >= now - 7 * 86400:
        z += 0.5
    if rng is not None:
        z += rng.gauss(0, _NOISE)
    return 1 / (1 + math.exp(-z))
