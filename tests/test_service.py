from fastapi.testclient import TestClient
import time

from leadfinder import auth
from leadfinder.service.security import CSP

from .conftest import form_value, login


def first_new_lead(app) -> int:
    return app.state.conn.execute("SELECT id FROM leads WHERE status='new' ORDER BY score DESC LIMIT 1").fetchone()["id"]


# ---------------- logging in and security ----------------
def test_pages_need_a_session(client):
    r = client.get("/leads?status=all", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Fleads%3Fstatus%3Dall"
    assert client.post("/lead/1/reject", data={"reason": "fit"}).status_code == 401


def test_login_page_shows_demo_logins_and_wrong_password_fails(client):
    page = client.get("/login").text
    assert "riley" in page and auth.DEMO_PASSWORD in page and "LeadFinder" in page
    r = client.post("/login", data={"username": "riley", "password": "wrong-password"})
    assert r.status_code == 401 and "do not match" in r.text


def test_security_headers_and_the_one_script(client):
    r = client.get("/login")
    assert r.headers["content-security-policy"] == CSP and "script-src 'self'" in CSP
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert "<script>" not in r.text
    js = client.get("/assets/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and "outline" in js.text


def test_unknown_host_and_cross_site_writes_are_refused(sdr):
    assert sdr.get("/", headers={"host": "evil.example"}).status_code == 400
    r = sdr.post("/lead/1/reject", data={"reason": "fit"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_theme_cookie_is_written_into_the_page(sdr):
    r = sdr.post("/theme", data={"theme": "dark", "next": "/leads"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/leads"
    assert 'data-theme="dark"' in sdr.get("/leads").text
    sdr.post("/theme", data={"theme": "nonsense", "next": "//evil.example"})
    assert "data-theme" not in sdr.get("/").text.split(">", 2)[1]


# ---------------- the pages ----------------
def test_overview_and_leads_render_in_the_house_style(sdr):
    home = sdr.get("/").text
    assert "waiting for review" in home and 'class="kicker"' in home and 'class="fnotes"' in home and 'class="info"' in home
    assert "Cormorant+Garamond" in home and '<script src="/assets/app.js" defer></script>' in home
    leads = sdr.get("/leads").text
    assert "data-sortable" in leads and 'aria-current="page"' in leads


def write_email(client, lead_id: int) -> str:
    """Press the button that writes the lead's first email; the page that comes back."""
    r = client.post(f"/lead/{lead_id}/write", data={"what": "email"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("wrote=email")
    return client.get(f"/lead/{lead_id}").text


def test_lead_page_shows_reasons_and_facts(app, sdr):
    lead_id = first_new_lead(app)
    page = sdr.get(f"/lead/{lead_id}").text
    assert "every point and its source" in page and "NPI record" in page and "synthetic" in page


def test_no_lead_has_an_email_until_someone_asks_for_one(app, sdr):
    conn = app.state.conn
    assert conn.execute("SELECT COUNT(*) n FROM leads WHERE subject != '' OR body != '' OR written_by != ''").fetchone()["n"] == 0
    lead_id = first_new_lead(app)
    page = sdr.get(f"/lead/{lead_id}").text
    assert "not written yet" in page and "Write the email</button>" in page
    assert 'name="subject"' not in page and "Approve the email" not in page and "guardrail would stop" not in page
    assert "0 emails written" in sdr.get("/").text
    page = write_email(sdr, lead_id)
    assert 'name="subject"' in page and "Approve the email" in page and "drafted, waiting for you" in page
    row = conn.execute("SELECT subject, body, written_by FROM leads WHERE id=?", (lead_id,)).fetchone()
    assert row["subject"] and row["body"] and row["written_by"] == "template"
    assert conn.execute("SELECT COUNT(*) n FROM leads WHERE body != ''").fetchone()["n"] == 1
    assert "1 email written" in sdr.get("/").text


def test_a_lead_can_be_rejected_without_an_email(app, sdr):
    lead_id = first_new_lead(app)
    seen = form_value(sdr.get(f"/lead/{lead_id}").text, "seen")
    r = sdr.post(f"/lead/{lead_id}/reject", data={"reason": "fit", "seen": seen}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("done=rejected")
    page = sdr.get(f"/lead/{lead_id}").text
    assert "none written: rejected" in page and "No email was written for this lead" in page


def test_an_approval_needs_an_email(app, sdr):
    lead_id = first_new_lead(app)
    seen = form_value(sdr.get(f"/lead/{lead_id}").text, "seen")
    r = sdr.post(f"/lead/{lead_id}/approve", data={"subject": "", "body": "", "seen": seen})
    assert r.status_code == 422 and "guardrail stopped" in r.text
    assert app.state.conn.execute("SELECT status FROM leads WHERE id=?", (lead_id,)).fetchone()["status"] == "new"


def test_how_it_works_page_and_its_place_in_the_sidebar(app, sdr, manager):
    home = sdr.get("/").text
    assert '<a class="nav" href="/how">How it works</a>' in home
    page = sdr.get("/how")
    assert page.status_code == 200 and '<a class="nav" href="/how" aria-current="page">How it works</a>' in page.text
    text = page.text
    assert "Already-identified demo leads" in text and "40 practices" in text and "0 emails written" in text and "Write the email" in text
    assert "intent gate before practice lookup" in text and "No local practice match or lead score happens before this gate." in text
    assert "sample visits were not run through the gate above" in text
    assert '<svg class="workflow-svg"' in text and 'role="img"' in text
    assert "human_review" in text and "Logistic regression" in text and "ResearchAgent" in text
    assert "standalone module but is not connected" in text
    assert 'class="kicker"' in text and 'class="fnotes"' in text and "What is real today" in text
    assert 'href="/audit"' not in text.split("<main")[1] and 'href="/audit"' in manager.get("/how").text.split("<main")[1]
    assert sdr.get("/how", follow_redirects=False).status_code == 200 and TestClient(app).get("/how", follow_redirects=False).status_code == 303


def test_sdr_cannot_open_the_audit_log_but_the_manager_can(sdr, manager):
    assert sdr.get("/audit", headers={"accept": "text/html"}).status_code == 403
    assert manager.get("/audit").status_code == 200


def test_missing_lead_is_a_404(sdr):
    assert sdr.get("/lead/99999").status_code == 404


# ---------------- visit-analysis integration and human review ----------------
def test_demo_starts_with_labeled_review_examples(app, sdr):
    rows = app.state.conn.execute(
        "SELECT review_status, COUNT(*) n FROM visit_analyses WHERE is_demo=1 GROUP BY review_status"
    ).fetchall()
    assert {row["review_status"]: row["n"] for row in rows} == {
        "pending": 2, "identity_verified": 1, "closed": 1
    }
    queue = sdr.get("/review-queue")
    assert "2 pending" in queue.text and "synthetic demo" in queue.text
    review_id = app.state.conn.execute(
        "SELECT id FROM visit_analyses WHERE is_demo=1 AND review_status='pending' ORDER BY id LIMIT 1"
    ).fetchone()["id"]
    detail = sdr.get(f"/review/{review_id}")
    assert "Synthetic demo case" in detail.text and "not collected from a real visitor" in detail.text
    verified = app.state.conn.execute(
        "SELECT id, handoff_status, lead_record_id FROM visit_analyses "
        "WHERE is_demo=1 AND review_status='identity_verified'"
    ).fetchone()
    assert verified["handoff_status"] == "complete" and verified["lead_record_id"] is not None
    resolved = sdr.get(f"/review/{verified['id']}")
    assert "Local synthetic-record handoff complete" in resolved.text
    assert "not external web research" in resolved.text


def review_payload(request_id="visit-request-1"):
    now = time.time()
    return {
        "request_id": request_id,
        "lead_id": "visitor-ref-1",
        "practice_name_raw": "North Clinic",
        "domain_or_org_hint": "north.example",
        "location_hint": "Boston, MA",
        "company_identified": False,
        "identification_confidence": 0.2,
        "page_views": [
            {"view_id": "view-1", "page_url": "https://north.example/pricing?email=private@example.test",
             "viewed_at": now - 60, "seconds_on_page": 30},
            {"view_id": "view-2", "page_url": "https://north.example/integrations", "viewed_at": now - 30,
             "seconds_on_page": 20},
        ],
        "window_end": now,
        "patient_data_stripped": True,
    }


def test_scrubbed_gate_review_is_persisted_and_resolved_in_the_app(app, sdr):
    payload = review_payload()
    response = sdr.post("/api/visit-analysis", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["routing_decision"] == "human_review"
    assert result["review_status"] == "pending" and result["persisted"] is True
    review_id = result["analysis_id"]

    stored = app.state.conn.execute("SELECT * FROM visit_analyses WHERE id=?", (review_id,)).fetchone()
    assert stored["review_status"] == "pending" and stored["created_by"] == "riley"
    assert "private@example.test" not in stored["result_json"]
    assert "pricing" in stored["result_json"] and "https://north.example" not in stored["result_json"]
    queue = sdr.get("/review-queue")
    assert queue.status_code == 200 and f"/review/{review_id}" in queue.text and "North Clinic" in queue.text
    page = sdr.get(f"/review/{review_id}")
    assert page.status_code == 200 and "Verify this practice's identity" in page.text
    practice_id = app.state.conn.execute("SELECT id FROM practices ORDER BY id LIMIT 1").fetchone()["id"]
    seen = form_value(page.text, "seen")
    verified = sdr.post(
        f"/review/{review_id}/verify",
        data={"practice_id": practice_id, "note": "Matched against approved source", "seen": seen},
        follow_redirects=False,
    )
    assert verified.status_code == 303 and verified.headers["location"].endswith("done=identity_verified")
    saved = app.state.conn.execute(
        "SELECT review_status, practice_id, resolved_by, resolution_note FROM visit_analyses WHERE id=?",
        (review_id,),
    ).fetchone()
    assert (saved["review_status"], saved["practice_id"], saved["resolved_by"], saved["resolution_note"]) == (
        "identity_verified", practice_id, "riley", "Matched against approved source"
    )
    resolved_page = sdr.get(f"/review/{review_id}")
    assert "Local synthetic-record handoff complete" in resolved_page.text
    assert "not external web research" in resolved_page.text
    handoff = app.state.conn.execute(
        "SELECT identity_method, lead_record_id, practice_record_json FROM lead_handoffs WHERE request_id=?",
        (payload["request_id"],),
    ).fetchone()
    assert handoff["identity_method"] == "human_review"
    assert '"external_research_performed":false' in handoff["practice_record_json"]
    assert f'href="/lead/{handoff["lead_record_id"]}"' in resolved_page.text
    assert app.state.conn.execute(
        "SELECT COUNT(*) n FROM visits WHERE source_request_id=?", (payload["request_id"],)
    ).fetchone()["n"] == 2

    repeated = sdr.post("/api/visit-analysis", json=payload).json()
    assert repeated["analysis_id"] == review_id and repeated["persisted"] is True
    assert repeated["review_status"] == "identity_verified"
    changed = review_payload()
    changed["practice_name_raw"] = "Different Clinic"
    assert sdr.post("/api/visit-analysis", json=changed).status_code == 409


def test_automatic_and_human_identity_share_the_local_lead_handoff(app, sdr):
    conn = app.state.conn
    practice = conn.execute("SELECT id, name, city, state FROM practices ORDER BY id LIMIT 1").fetchone()
    practice_id = practice["id"]
    manual = review_payload("human-handoff")
    manual_result = sdr.post("/api/visit-analysis", json=manual).json()
    page = sdr.get(f"/review/{manual_result['analysis_id']}")
    verified = sdr.post(
        f"/review/{manual_result['analysis_id']}/verify",
        data={"practice_id": practice_id, "seen": form_value(page.text, "seen")},
        follow_redirects=False,
    )
    assert verified.status_code == 303
    human_handoff = conn.execute(
        "SELECT lead_record_id FROM lead_handoffs WHERE request_id=?", (manual["request_id"],)
    ).fetchone()
    human_lead = conn.execute("SELECT score, chance FROM leads WHERE id=?", (human_handoff["lead_record_id"],)).fetchone()

    automatic = review_payload("automatic-handoff")
    automatic.update({
        "practice_name_raw": practice["name"],
        "location_hint": f"{practice['city']}, {practice['state']}",
        "company_identified": True,
        "identification_confidence": 0.95,
    })
    response = sdr.post("/api/visit-analysis", json=automatic)
    assert response.status_code == 200
    result = response.json()
    assert result["routing_decision"] == "research"
    assert result["handoff_status"] == "complete"
    assert result["handoff"]["identity_method"] == "automatic"
    assert result["handoff"]["practice_record"]["source"] == "synthetic"
    assert result["handoff"]["practice_record"]["external_research_performed"] is False
    assert result["handoff"]["lead_record_id"] == human_handoff["lead_record_id"]
    assert result["handoff"]["score"] == human_lead["score"]
    assert result["handoff"]["chance"] == human_lead["chance"]

    before = conn.execute(
        "SELECT COUNT(*) n FROM visits WHERE source_request_id=?", (automatic["request_id"],)
    ).fetchone()["n"]
    repeated = sdr.post("/api/visit-analysis", json=automatic).json()
    after = conn.execute(
        "SELECT COUNT(*) n FROM visits WHERE source_request_id=?", (automatic["request_id"],)
    ).fetchone()["n"]
    assert repeated["analysis_id"] == result["analysis_id"] and before == after == 2
    assert conn.execute(
        "SELECT COUNT(*) n FROM lead_handoffs WHERE request_id=?", (automatic["request_id"],)
    ).fetchone()["n"] == 1

    no_match = review_payload("automatic-no-match")
    no_match.update({
        "practice_name_raw": practice["name"],
        "location_hint": "Different City, ZZ",
        "company_identified": True,
        "identification_confidence": 0.95,
    })
    unmatched = sdr.post("/api/visit-analysis", json=no_match).json()
    assert unmatched["routing_decision"] == "research"
    assert unmatched["handoff_status"] == "local_practice_not_found"
    assert unmatched["persisted"] is False
    assert conn.execute(
        "SELECT COUNT(*) n FROM lead_handoffs WHERE request_id=?", (no_match["request_id"],)
    ).fetchone()["n"] == 0

    cursor = conn.execute(
        "INSERT INTO practices(npi, name, specialty, providers, city, state, ehr, contact_name, contact_email, source) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("12345678901", "New Local Clinic", "Family Medicine", 4, "Cambridge", "MA", None, "", "", "synthetic"),
    )
    new_practice_id = cursor.lastrowid
    new_lead = review_payload("automatic-create-lead")
    new_lead.update({
        "practice_name_raw": "New Local Clinic",
        "location_hint": "Cambridge, MA",
        "company_identified": True,
        "identification_confidence": 0.95,
    })
    created = sdr.post("/api/visit-analysis", json=new_lead).json()
    assert created["handoff"]["practice_id"] == new_practice_id
    assert conn.execute("SELECT practice_id FROM leads WHERE id=?", (created["handoff"]["lead_record_id"],)).fetchone()[
        "practice_id"
    ] == new_practice_id


def test_unverified_or_unsanitized_visits_cannot_enter_review_queue(app, sdr):
    payload = review_payload("unsanitized-request")
    payload["patient_data_stripped"] = False
    response = sdr.post("/api/visit-analysis", json=payload)
    assert response.status_code == 200
    assert response.json()["routing_decision"] == "stop" and response.json()["persisted"] is False
    assert app.state.conn.execute("SELECT COUNT(*) n FROM visit_analyses WHERE is_demo=0").fetchone()["n"] == 0

    low_intent = review_payload("low-intent-request")
    low_intent["page_views"] = [
        {"view_id": "view-blog", "page_url": "https://north.example/blog/pricing-tips",
         "viewed_at": low_intent["window_end"] - 30, "seconds_on_page": 20}
    ]
    response = sdr.post("/api/visit-analysis", json=low_intent)
    assert response.status_code == 200
    assert response.json()["routing_decision"] == "stop" and response.json()["persisted"] is False
    assert app.state.conn.execute("SELECT COUNT(*) n FROM visit_analyses WHERE is_demo=0").fetchone()["n"] == 0
    assert "2 pending" in sdr.get("/review-queue").text


def test_visit_analysis_api_requires_login_and_review_close_is_audited(app, sdr):
    payload = review_payload("close-request")
    assert TestClient(app).post("/api/visit-analysis", json=payload).status_code == 401
    review_id = sdr.post("/api/visit-analysis", json=payload).json()["analysis_id"]
    page = sdr.get(f"/review/{review_id}")
    seen = form_value(page.text, "seen")
    closed = sdr.post(
        f"/review/{review_id}/close",
        data={"reason": "no_match", "note": "No trustworthy directory match", "seen": seen},
        follow_redirects=False,
    )
    assert closed.status_code == 303 and closed.headers["location"].endswith("done=closed")
    saved = app.state.conn.execute(
        "SELECT review_status, resolved_by, resolution_note FROM visit_analyses WHERE id=?", (review_id,)
    ).fetchone()
    assert tuple(saved) == ("closed", "riley", "no reliable practice match: No trustworthy directory match")


# ---------------- deciding ----------------
def test_approve_logs_the_decision_and_the_edit(app, sdr, manager):
    lead_id = first_new_lead(app)
    page = write_email(sdr, lead_id)
    seen = form_value(page, "seen")
    subject = form_value(page, "subject") + " (edited)"
    body = app.state.conn.execute("SELECT body FROM leads WHERE id=?", (lead_id,)).fetchone()["body"]
    r = sdr.post(f"/lead/{lead_id}/approve", data={"subject": subject, "body": body, "seen": seen}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("done=approved")
    row = app.state.conn.execute("SELECT * FROM audit WHERE lead_id=?", (lead_id,)).fetchone()
    assert row["username"] == "riley" and row["action"] == "approved" and row["edited"] == 1 and row["subject"] == subject
    assert "Edited before approving" in manager.get("/audit").text


def test_guardrail_blocks_an_approval(app, sdr):
    lead_id = first_new_lead(app)
    seen = form_value(sdr.get(f"/lead/{lead_id}").text, "seen")
    r = sdr.post(f"/lead/{lead_id}/approve", data={"subject": "Hi", "body": "We guarantee it. " * 5, "seen": seen})
    assert r.status_code == 422 and "guardrail stopped" in r.text
    assert app.state.conn.execute("SELECT status FROM leads WHERE id=?", (lead_id,)).fetchone()["status"] == "new"


def test_a_stale_decision_is_refused(app, sdr):
    lead_id = first_new_lead(app)
    seen = form_value(sdr.get(f"/lead/{lead_id}").text, "seen")
    other = login(TestClient(app), "jo")
    assert other.post(f"/lead/{lead_id}/reject", data={"reason": "size", "seen": seen}, follow_redirects=False).status_code == 303
    r = sdr.post(f"/lead/{lead_id}/reject", data={"reason": "fit", "seen": seen}, follow_redirects=False)
    assert r.headers["location"].endswith("stale=1")
    rows = app.state.conn.execute("SELECT username, reason FROM audit WHERE lead_id=?", (lead_id,)).fetchall()
    assert [(r["username"], r["reason"]) for r in rows] == [("jo", "size")]
    assert "Someone decided this lead" in sdr.get(f"/lead/{lead_id}?stale=1").text


def test_reject_needs_a_known_reason(app, sdr):
    lead_id = first_new_lead(app)
    seen = form_value(sdr.get(f"/lead/{lead_id}").text, "seen")
    assert sdr.post(f"/lead/{lead_id}/reject", data={"reason": "because", "seen": seen}).status_code == 400


# ---------------- the trained model ----------------
def test_leads_are_ranked_by_the_models_chance(app, sdr):
    rows = app.state.conn.execute("SELECT l.id, l.chance, l.score FROM leads l WHERE status='new' ORDER BY chance DESC").fetchall()
    assert all(r["chance"] is not None for r in rows)
    page = sdr.get("/leads").text
    order = [int(x.split('"')[0]) for x in page.split('href="/lead/')[1:]]
    assert order == [r["id"] for r in rows] and "Chance</th>" in page and "Points</th>" in page
    home = sdr.get("/").text
    assert f'href="/lead/{rows[0]["id"]}"' in home.split("ov-top")[-1] and "trained model" in home


def test_lead_page_shows_the_chance_and_what_moved_it(app, sdr):
    lead = app.state.conn.execute("SELECT id, chance, score FROM leads ORDER BY chance DESC LIMIT 1").fetchone()
    page = sdr.get(f"/lead/{lead['id']}").text
    assert "Chance of converting" in page and f"{round(lead['chance'] * 100)}%" in page and "raises" in page
    assert f"The points table says {lead['score']}" in page and "not a forecast of real conversion" in page


def test_model_page_shows_the_benchmark_and_says_the_labels_are_simulated(sdr):
    page = sdr.get("/model").text
    assert "Benchmark" in page and "Trained model" in page and "Points table" in page and "Challenger" in page
    assert "demo-v6" in page and "simulated" in page and "Visited /demo" in page


def test_model_page_shows_every_version_and_the_scorecard(app, sdr):
    page = sdr.get("/model").text
    runs = app.state.conn.execute("SELECT version, status, verdict FROM model_runs ORDER BY month").fetchall()
    assert len(runs) == 6 and all(r["version"] in page and r["verdict"].replace("'", "&#39;") in page for r in runs)
    assert "Every version" in page and "the live one of 6 versions" in page and "was live" in page and "sent back" in page
    assert "How a version goes live" in page and "Calibration error" in page and "2.5 points" in page and "1 point<" in page
    live = next(r["version"] for r in runs if r["status"] == "live")
    assert f"model {live}" in sdr.get("/lead/1").text and f"{live}." in sdr.get("/how").text


def test_the_audit_log_keeps_the_chance_that_was_on_screen(app, sdr, manager):
    lead = app.state.conn.execute("SELECT id, chance, model_version FROM leads WHERE status='new' ORDER BY chance DESC LIMIT 1").fetchone()
    seen = form_value(sdr.get(f"/lead/{lead['id']}").text, "seen")
    sdr.post(f"/lead/{lead['id']}/reject", data={"reason": "fit", "seen": seen})
    row = app.state.conn.execute("SELECT chance, model_version FROM audit WHERE lead_id=?", (lead["id"],)).fetchone()
    assert (row["chance"], row["model_version"]) == (lead["chance"], lead["model_version"])
    assert f"{round(lead['chance'] * 100)}%" in manager.get("/audit").text
