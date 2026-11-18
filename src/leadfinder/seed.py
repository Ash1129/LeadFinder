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
VISIT_PAGES = {
    "demo_or_trial": "/demo",
    "pricing": "/pricing",
    "ehr_integration": "/integrations",
    "usability": "/features",
    "blog": "/blog",
    "security": "/security",
    "billing": "/billing",
    "specialty_case_study": "/case-studies",
    "careers": "/careers",
    "about": "/about",
    "homepage": "/",
    "unmapped": "/unmapped",
}


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


def seed(conn: sqlite3.Connection, n: int = 40, seed: int = 7, now: float | None = None,
         practices: str = "synthetic") -> dict:
    """Fill an empty database with `n` practices, their visits over the last three weeks, and a scored lead with a
    call brief for each (no email: that is written on request). Returns counts. Does nothing if practices exist already.

    `practices="npi"` takes the practices from the real NPI table (data/npi_practices.json, see npi.py) instead of
    making them up; their visits are still simulated, since there is no tracking connector."""
    if conn.execute("SELECT COUNT(*) n FROM practices").fetchone()["n"]:
        return {"practices": 0, "visits": 0, "leads": 0}
    if practices not in ("synthetic", "npi"):
        raise ValueError("practices must be 'synthetic' or 'npi'")
    rng = random.Random(seed)
    now = time.time() if now is None else now
    if practices == "npi":
        return _seed_npi(conn, rng, n, now)
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


def _seed_npi(conn: sqlite3.Connection, rng: random.Random, n: int, now: float) -> dict:
    """`n` practices drawn (seeded) from the NPI table, with simulated visits. EHR stays unknown and there is no
    contact email: the registry has neither, and neither is invented."""
    from . import npi              # npi imports this module for valid_npi; import it only when it is needed

    records = sorted(npi.load_table(), key=lambda r: r.npi)
    visits_made = 0
    for record in rng.sample(records, min(n, len(records))):
        cur = conn.execute("INSERT INTO practices(npi, name, specialty, providers, city, state, ehr, contact_name, "
                           "contact_email, source) VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (record.npi, record.name, record.specialty, record.providers, record.city, record.state,
                            None, record.contact_name, None, "npi"))
        for page, at in synth.visits(rng, now):
            conn.execute("INSERT INTO visits(practice_id, page, at) VALUES(?,?,?)", (cur.lastrowid, page, at))
            visits_made += 1
    made = conn.execute("SELECT COUNT(*) n FROM practices").fetchone()["n"]
    leads = score_all(conn, now=now)
    apply_model(conn, now=now)
    return {"practices": made, "visits": visits_made, "leads": leads}


def process_local_visit_handoff(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    visitor_ref: str,
    practice_id: int,
    analysis: dict,
    identity_method: str,
    created_by: str,
    is_demo: bool = False,
    now: float | None = None,
) -> dict:
    """Use a verified local practice record, attach gate views, and create or refresh its scored lead."""
    if identity_method not in ("automatic", "human_review"):
        raise ValueError("identity_method must be automatic or human_review")
    now = time.time() if now is None else now
    analysis_json = json.dumps(analysis, sort_keys=True, separators=(",", ":"), allow_nan=False)
    existing = conn.execute("SELECT * FROM lead_handoffs WHERE request_id=?", (request_id,)).fetchone()
    if existing is not None:
        if (existing["analysis_json"] != analysis_json or existing["practice_id"] != practice_id or
                existing["identity_method"] != identity_method):
            raise ValueError("request_id was already used for a different verified visit")
        lead = conn.execute(
            "SELECT score, chance, model_version FROM leads WHERE id=?", (existing["lead_record_id"],)
        ).fetchone()
        return dict(existing) | (dict(lead) if lead else {})

    practice_row = conn.execute("SELECT * FROM practices WHERE id=?", (practice_id,)).fetchone()
    if practice_row is None:
        raise LookupError("verified practice no longer exists in the local practice table")
    practice = dict(practice_row)
    for view in analysis["pages_viewed"]:
        page = VISIT_PAGES.get(view["category"], "/unmapped")
        conn.execute(
            "INSERT INTO visits(practice_id, page, at, source_request_id, source_view_id) VALUES(?,?,?,?,?) "
            "ON CONFLICT(source_request_id, source_view_id) DO NOTHING",
            (practice_id, page, view["viewed_at"], request_id, view["view_id"]),
        )

    visit_rows = conn.execute(
        "SELECT page, at FROM visits WHERE practice_id=? ORDER BY at", (practice_id,)
    ).fetchall()
    visits = [(row["page"], row["at"]) for row in visit_rows]
    score = score_practice(practice["specialty"], practice["providers"], practice["ehr"], visits, now=now)
    reasons = [reason.as_dict() for reason in score.reasons]
    model = ml.load(conn)
    if model is None:
        raise RuntimeError("the lead model is not available for scoring")
    prediction = ml.predict(model, practice, visits, now=now)

    lead = conn.execute("SELECT id, brief_by, status, subject, body FROM leads WHERE practice_id=?",
                        (practice_id,)).fetchone()
    if lead is None:
        pages = [page for page, _ in visits]
        brief = Offline().brief(practice, reasons, pages)
        cursor = conn.execute(
            "INSERT INTO leads(practice_id, score, band, reasons, subject, body, status, scored_at, written_by, brief, "
            "brief_by, chance, chance_band, chance_reasons, model_version) "
            "VALUES(?,?,?,?,?,'','new',?,'',?,?,?,?,?,?)",
            (practice_id, score.total, score.band, json.dumps(reasons), "", now, json.dumps(brief), "template",
             prediction["chance"], prediction["band"], json.dumps(prediction["reasons"]), prediction["version"]),
        )
        lead_id = cursor.lastrowid
    else:
        lead_id = lead["id"]
        conn.execute(
            "UPDATE leads SET score=?, band=?, reasons=?, scored_at=?, chance=?, chance_band=?, chance_reasons=?, "
            "model_version=? WHERE id=?",
            (score.total, score.band, json.dumps(reasons), now, prediction["chance"], prediction["band"],
             json.dumps(prediction["reasons"]), prediction["version"], lead_id),
        )
        if lead["status"] == "new" and not lead["subject"] and not lead["body"] and lead["brief_by"] == "template":
            pages = [page for page, _ in visits]
            brief = Offline().brief(practice, reasons, pages)
            conn.execute("UPDATE leads SET brief=? WHERE id=?", (json.dumps(brief), lead_id))

    record = {
        "source": practice["source"],
        "practice_id": practice["id"],
        "npi": practice["npi"],
        "name": practice["name"],
        "specialty": practice["specialty"],
        "providers": practice["providers"],
        "city": practice["city"],
        "state": practice["state"],
        "ehr": practice["ehr"],
        "ehr_source": practice["source"] if practice["ehr"] else "unknown",
        "external_research_performed": False,
    }
    record_json = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cursor = conn.execute(
        "INSERT INTO lead_handoffs(request_id, visitor_ref, identity_method, practice_id, lead_record_id, analysis_json, "
        "practice_record_json, created_by, created_at, is_demo) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (request_id, visitor_ref, identity_method, practice_id, lead_id, analysis_json, record_json,
         created_by, now, int(is_demo)),
    )
    return {
        "id": cursor.lastrowid,
        "request_id": request_id,
        "practice_id": practice_id,
        "lead_record_id": lead_id,
        "identity_method": identity_method,
        "analysis_json": analysis_json,
        "practice_record_json": record_json,
        "created_at": now,
        "is_demo": int(is_demo),
        "score": score.total,
        "chance": prediction["chance"],
        "model_version": prediction["version"],
    }


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
    for row in conn.execute(
        "SELECT id, request_id, lead_id, practice_id, result_json FROM visit_analyses "
        "WHERE is_demo=1 AND review_status='identity_verified' AND handoff_status='not_started'"
    ).fetchall():
        handoff = process_local_visit_handoff(
            conn,
            request_id=row["request_id"],
            visitor_ref=row["lead_id"],
            practice_id=row["practice_id"],
            analysis=json.loads(row["result_json"]),
            identity_method="human_review",
            created_by="morgan",
            is_demo=True,
            now=now,
        )
        conn.execute(
            "UPDATE visit_analyses SET handoff_status='complete', lead_record_id=?, handoff_note=NULL WHERE id=?",
            (handoff["lead_record_id"], row["id"]),
        )
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
