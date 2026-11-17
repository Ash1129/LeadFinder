"""Synthetic data in the shape of the real thing: independent practices as the NPI registry describes them, and the
website visits a tracking tool would tie to each.

Everything here is made up and marked so (`source` says "synthetic"): the practice names, the people, and the NPIs,
which are random ten-digit numbers with a valid check digit (the registry's Luhn rule over the 80840 prefix), so
they look like NPIs to any validator but are not anyone's. Emails use the reserved `.example` domain. Swapping this
module for a call to the NPPES API is the data agent's job; the rest of the service only reads the tables.
What a made-up practice and its visits look like is in synth.py, which the model's simulated history shares.
"""
from __future__ import annotations

import json
import random
import sqlite3
import time

from . import ml, synth
from .scoring import score_practice
from .synth import PLACES, SUFFIX
from .visit_analysis import PageView, VisitAnalysisInput, analyze_visits
from .writer import Offline

FIRST = ["Alex", "Jordan", "Priya", "Marcus", "Elena", "Sam", "Tara", "Devon", "Lena", "Omar", "Grace", "Hugo"]
LAST = ["Nguyen", "Patel", "Okafor", "Silva", "Kowalski", "Bennett", "Haddad", "Larsen", "Moreau", "Ito"]


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
    call brief for each (no email: that is written on request). Returns counts. Does nothing if practices exist already."""
    if conn.execute("SELECT COUNT(*) n FROM practices").fetchone()["n"]:
        return {"practices": 0, "visits": 0, "leads": 0}
    rng = random.Random(seed)
    now = time.time() if now is None else now
    names, visits_made = set(), 0
    for _ in range(n):
        p = synth.practice(rng)
        while True:
            name = f"{rng.choice(PLACES)} {SUFFIX[p['specialty']]}"
            if name not in names:
                names.add(name)
                break
            p = synth.practice(rng)
        contact = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        slug = "".join(ch for ch in name.lower() if ch.isalnum())[:24]
        cur = conn.execute("INSERT INTO practices(npi, name, specialty, providers, city, state, ehr, contact_name, contact_email, source) "
                           "VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (_npi(rng), name, p["specialty"], p["providers"], p["city"], p["state"], p["ehr"], contact,
                            f"office@{slug}.example", "synthetic"))
        for page, at in synth.visits(rng, now):
            conn.execute("INSERT INTO visits(practice_id, page, at) VALUES(?,?,?)", (cur.lastrowid, page, at))
            visits_made += 1
    leads = score_all(conn, now=now)
    apply_model(conn, now=now)
    return {"practices": n, "visits": visits_made, "leads": leads}


def seed_demo_review_cases(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Add labeled synthetic review examples once, without resetting a demo user's later decisions."""
    now = time.time() if now is None else now
    practices = conn.execute("SELECT id, name, city, state FROM practices ORDER BY id LIMIT 2").fetchall()
    if not practices:
        raise RuntimeError("demo review cases require at least one seeded practice")
    known = practices[0]
    second = practices[-1]
    cases = [
        {
            "request_id": "demo-review-unidentified-v1",
            "lead_id": "demo-visitor-unidentified",
            "practice_name_raw": None,
            "domain_or_org_hint": None,
            "location_hint": None,
            "company_identified": False,
            "identification_confidence": 0.0,
            "pages": (("/pricing", 90.0), ("/integrations", 45.0)),
            "review_status": "pending",
            "practice_id": None,
            "note": None,
        },
        {
            "request_id": "demo-review-low-confidence-v1",
            "lead_id": "demo-visitor-low-confidence",
            "practice_name_raw": second["name"],
            "domain_or_org_hint": None,
            "location_hint": f"{second['city']}, {second['state']}",
            "company_identified": True,
            "identification_confidence": 0.35,
            "pages": (("/request-a-demo", 75.0), ("/security", 50.0)),
            "review_status": "pending",
            "practice_id": None,
            "note": None,
        },
        {
            "request_id": "demo-review-verified-v1",
            "lead_id": "demo-visitor-verified",
            "practice_name_raw": known["name"],
            "domain_or_org_hint": None,
            "location_hint": f"{known['city']}, {known['state']}",
            "company_identified": False,
            "identification_confidence": 0.0,
            "pages": (("/pricing", 60.0),),
            "review_status": "identity_verified",
            "practice_id": known["id"],
            "note": "Synthetic demo only: reviewer matched the hint to a seeded practice.",
        },
        {
            "request_id": "demo-review-closed-v1",
            "lead_id": "demo-visitor-closed",
            "practice_name_raw": None,
            "domain_or_org_hint": None,
            "location_hint": None,
            "company_identified": False,
            "identification_confidence": 0.0,
            "pages": (("/demo", 40.0),),
            "review_status": "closed",
            "practice_id": None,
            "note": "Synthetic demo only: no reliable practice match.",
        },
    ]
    made = 0
    for index, case in enumerate(cases):
        created_at = now - (len(cases) - index) * 3600
        window_end = created_at
        data = VisitAnalysisInput(
            lead_id=case["lead_id"],
            practice_name_raw=case["practice_name_raw"],
            domain_or_org_hint=case["domain_or_org_hint"],
            location_hint=case["location_hint"],
            company_identified=case["company_identified"],
            identification_confidence=case["identification_confidence"],
            page_views=tuple(
                PageView(f"{case['lead_id']}-view-{i}", page, window_end - i * 120, dwell)
                for i, (page, dwell) in enumerate(case["pages"])
            ),
            window_end=window_end,
            patient_data_stripped=True,
        )
        result = analyze_visits(data)
        if result.routing_decision != "human_review":
            raise RuntimeError(f"demo case {case['request_id']} did not route to human review")
        resolved = case["review_status"] != "pending"
        cur = conn.execute(
            "INSERT INTO visit_analyses(request_id, lead_id, routing_decision, review_status, is_demo, result_json, "
            "created_by, created_at, practice_id, resolved_by, resolved_at, resolution_note) "
            "VALUES(?,?,'human_review',?,1,?,'morgan',?,?,?,?,?) ON CONFLICT(request_id) DO NOTHING",
            (case["request_id"], case["lead_id"], case["review_status"],
             json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False),
             created_at, case["practice_id"], "morgan" if resolved else None,
             created_at + 600 if resolved else None, case["note"]),
        )
        made += cur.rowcount
    return made


def score_all(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Score every practice that has no lead yet and write its call brief from the template. No email is written
    here: a lead starts without one, and gets its first when a person asks for it on the lead's page."""
    now = time.time() if now is None else now
    writer = Offline()
    made = 0
    for p in conn.execute("SELECT * FROM practices WHERE id NOT IN (SELECT practice_id FROM leads)").fetchall():
        visits = [(r["page"], r["at"]) for r in conn.execute("SELECT page, at FROM visits WHERE practice_id=?", (p["id"],))]
        s = score_practice(p["specialty"], p["providers"], p["ehr"], visits, now=now)
        reasons = [r.as_dict() for r in s.reasons]
        pages = [page for page, _ in visits]
        conn.execute("INSERT INTO leads(practice_id, score, band, reasons, subject, body, status, scored_at, written_by, brief, brief_by) "
                     "VALUES(?,?,?,?,'','',?,?,'',?,?)",
                     (p["id"], s.total, s.band, json.dumps(reasons), "new", now,
                      json.dumps(writer.brief(dict(p), reasons, pages)), writer.name))
        made += 1
    return made


def apply_model(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Give every lead the live model's chance, band and reasons, if it has none from this version of the model.
    Decided leads are scored too: the audit log keeps what was on screen when each was decided."""
    model = ml.load(conn)
    if model is None:
        return 0
    done = 0
    rows = conn.execute("SELECT l.id, p.id pid, p.specialty, p.providers, p.ehr FROM leads l JOIN practices p ON p.id = l.practice_id "
                        "WHERE l.model_version IS NOT ?", (model["version"],)).fetchall()
    for r in rows:
        visits = [(v["page"], v["at"]) for v in conn.execute("SELECT page, at FROM visits WHERE practice_id=?", (r["pid"],))]
        out = ml.predict(model, dict(r), visits, now=now)
        conn.execute("UPDATE leads SET chance=?, chance_band=?, chance_reasons=?, model_version=? WHERE id=?",
                     (out["chance"], out["band"], json.dumps(out["reasons"]), out["version"], r["id"]))
        done += 1
    return done
