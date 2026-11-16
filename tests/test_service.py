from fastapi.testclient import TestClient

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
    assert "The path of a lead" in text and "40 practices" in text and "0 emails written" in text and "Write the email" in text
    assert 'class="kicker"' in text and 'class="fnotes"' in text and "What is real today" in text
    assert 'href="/audit"' not in text.split("<main")[1] and 'href="/audit"' in manager.get("/how").text.split("<main")[1]
    assert sdr.get("/how", follow_redirects=False).status_code == 200 and TestClient(app).get("/how", follow_redirects=False).status_code == 303


def test_sdr_cannot_open_the_audit_log_but_the_manager_can(sdr, manager):
    assert sdr.get("/audit", headers={"accept": "text/html"}).status_code == 403
    assert manager.get("/audit").status_code == 200


def test_missing_lead_is_a_404(sdr):
    assert sdr.get("/lead/99999").status_code == 404


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
    assert "lr-8000-11" in page and "simulated" in page and "Visited /demo" in page


def test_the_audit_log_keeps_the_chance_that_was_on_screen(app, sdr, manager):
    lead = app.state.conn.execute("SELECT id, chance, model_version FROM leads WHERE status='new' ORDER BY chance DESC LIMIT 1").fetchone()
    seen = form_value(sdr.get(f"/lead/{lead['id']}").text, "seen")
    sdr.post(f"/lead/{lead['id']}/reject", data={"reason": "fit", "seen": seen})
    row = app.state.conn.execute("SELECT chance, model_version FROM audit WHERE lead_id=?", (lead["id"],)).fetchone()
    assert (row["chance"], row["model_version"]) == (lead["chance"], lead["model_version"])
    assert f"{round(lead['chance'] * 100)}%" in manager.get("/audit").text
