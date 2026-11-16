"""Synthetic data in the shape of the real thing: independent practices as the NPI registry describes them, and the
website visits a tracking tool would tie to each.

Everything here is made up and marked so (`source` says "synthetic"): the practice names, the people, and the NPIs,
which are random ten-digit numbers with a valid check digit (the registry's Luhn rule over the 80840 prefix), so
they look like NPIs to any validator but are not anyone's. Emails use the reserved `.example` domain. Swapping this
module for a call to the NPPES API is the data agent's job; the rest of the service only reads the tables.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time

from . import outreach
from .scoring import score_practice

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
FIRST = ["Alex", "Jordan", "Priya", "Marcus", "Elena", "Sam", "Tara", "Devon", "Lena", "Omar", "Grace", "Hugo"]
LAST = ["Nguyen", "Patel", "Okafor", "Silva", "Kowalski", "Bennett", "Haddad", "Larsen", "Moreau", "Ito"]
PAGES = ["/demo", "/pricing", "/integrations", "/features", "/blog/no-show-rates", "/blog/waitlists", "/"]
PAGE_WEIGHTS = [2, 3, 2, 3, 3, 2, 6]


def npi_check_digit(first9: str) -> int:
    """The NPI check digit: Luhn over '80840' + the first nine digits."""
    digits = [int(d) for d in "80840" + first9]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10


def valid_npi(npi: str) -> bool:
    return len(npi) == 10 and npi.isdigit() and npi[0] in "12" and npi_check_digit(npi[:9]) == int(npi[9])


def _npi(rng: random.Random) -> str:
    first9 = "1" + "".join(str(rng.randrange(10)) for _ in range(8))
    return first9 + str(npi_check_digit(first9))


def seed(conn: sqlite3.Connection, n: int = 40, seed: int = 7, now: float | None = None) -> dict:
    """Fill an empty database with `n` practices, their visits over the last three weeks, and a scored lead with a
    drafted email for each. Returns counts. Does nothing if practices exist already."""
    if conn.execute("SELECT COUNT(*) n FROM practices").fetchone()["n"]:
        return {"practices": 0, "visits": 0, "leads": 0}
    rng = random.Random(seed)
    now = time.time() if now is None else now
    names, visits_made = set(), 0
    for _ in range(n):
        specialty = rng.choice(SPECIALTIES)
        while True:
            name = f"{rng.choice(PLACES)} {SUFFIX[specialty]}"
            if name not in names:
                names.add(name)
                break
            specialty = rng.choice(SPECIALTIES)
        city, state = rng.choice(CITIES)
        providers = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, 18, 40])
        contact = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        slug = "".join(ch for ch in name.lower() if ch.isalnum())[:24]
        cur = conn.execute("INSERT INTO practices(npi, name, specialty, providers, city, state, ehr, contact_name, contact_email, source) "
                           "VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (_npi(rng), name, specialty, providers, city, state, rng.choice(EHRS), contact,
                            f"office@{slug}.example", "synthetic"))
        pid = cur.lastrowid
        for _ in range(rng.choice([0, 1, 1, 2, 3, 4, 6])):
            page = rng.choices(PAGES, PAGE_WEIGHTS)[0]
            conn.execute("INSERT INTO visits(practice_id, page, at) VALUES(?,?,?)", (pid, page, now - rng.uniform(0, 21) * 86400))
            visits_made += 1
    leads = score_all(conn, now=now)
    return {"practices": n, "visits": visits_made, "leads": leads}


def score_all(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Score every practice that has no lead yet and draft its email."""
    now = time.time() if now is None else now
    made = 0
    for p in conn.execute("SELECT * FROM practices WHERE id NOT IN (SELECT practice_id FROM leads)").fetchall():
        visits = [(r["page"], r["at"]) for r in conn.execute("SELECT page, at FROM visits WHERE practice_id=?", (p["id"],))]
        s = score_practice(p["specialty"], p["providers"], p["ehr"], visits, now=now)
        reasons = [r.as_dict() for r in s.reasons]
        subject, body = outreach.draft(dict(p), reasons)
        conn.execute("INSERT INTO leads(practice_id, score, band, reasons, subject, body, status, scored_at) VALUES(?,?,?,?,?,?,?,?)",
                     (p["id"], s.total, s.band, json.dumps(reasons), subject, body, "new", now))
        made += 1
    return made
