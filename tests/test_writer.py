import pytest
from fastapi.testclient import TestClient

from leadfinder import outreach
from leadfinder.writer import FALLBACK_BETA, Claude, Offline, WriterError, make_writer

from .conftest import FakeAnthropic, login

PRACTICE = {"name": "Maple Grove Pediatrics", "specialty": "Pediatrics", "providers": 4, "city": "Ithaca", "state": "NY",
            "ehr": "athenahealth", "contact_name": "Priya Patel", "contact_email": "office@maplegrove.example", "npi": "1234567893"}
REASONS = [{"label": "Visited /demo", "points": 20, "source": "site visits"},
           {"label": "Runs athenahealth, which the product integrates with", "points": 15, "source": "NPI record"}]
VISITS = ["/demo", "/pricing"]
EMAIL = {"subject": "Booking at Maple Grove Pediatrics",
         "body": "Hi Priya,\n\nSomeone at Maple Grove looked at our demo page this week. Clearwell Scheduling works inside "
                 "athenahealth, so your front desk keeps one calendar.\n\nWould 20 minutes next week be useful?\n\nThe Clearwell team"}
BRIEF = {"opener": "Hi Priya, I'm calling about scheduling at Maple Grove Pediatrics.",
         "why_now": "Someone there looked at the demo page this week.",
         "talking_points": ["It works inside athenahealth.", "Four providers is the size it fits.", "The waitlist refills cancelled slots."],
         "questions": ["How do families book today?", "What happens to a cancelled slot?", "Who decides on scheduling?"],
         "objections": [{"objection": "We book in athenahealth already.", "reply": "It works inside it and adds the waitlist."},
                        {"objection": "We are too busy to switch.", "reply": "Setup runs beside what you do now."}]}


# ---------------- the template writer ----------------
def test_offline_brief_has_the_shape_and_passes_its_checks():
    w = Offline()
    brief = w.brief(PRACTICE, REASONS, VISITS)
    assert outreach.check_brief(brief, PRACTICE) == []
    assert "Priya" in brief["opener"] and "/demo" in brief["why_now"]
    solo_epic = PRACTICE | {"providers": 1, "ehr": "Epic"}
    brief = w.brief(solo_epic, REASONS, [])
    assert outreach.check_brief(brief, solo_epic) == [] and any("no direct integration" in t for t in brief["talking_points"])


def test_make_writer_follows_the_environment(monkeypatch):
    for k in ("LEADFINDER_WRITER", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert not make_writer().is_model
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert make_writer().is_model
    monkeypatch.setenv("LEADFINDER_WRITER", "template")
    assert not make_writer().is_model
    with pytest.raises(ValueError):
        make_writer("gpt")


# ---------------- Claude, behind a fake client ----------------
def test_claude_email_request_and_the_footer_code_adds():
    fake = FakeAnthropic(EMAIL)
    subject, body = Claude(client=fake).email(PRACTICE, REASONS, VISITS)
    assert subject == EMAIL["subject"] and body.endswith(outreach.FOOTER) and outreach.check(subject, body) == []
    sent = fake.requests[0]
    assert sent["model"] == "claude-opus-5-5" and sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["output_config"]["format"]["type"] == "json_schema" and "thinking" not in sent and "temperature" not in sent
    said = sent["messages"][0]["content"]
    assert "Maple Grove Pediatrics" in said and "/pricing" in said
    assert "office@maplegrove.example" not in said and "1234567893" not in said      # only what the email needs


def test_claude_email_with_patient_words_or_an_unbacked_claim_is_not_kept():
    for bad in ("We saw your patient list.", "We guarantee fewer no-shows."):
        fake = FakeAnthropic(EMAIL | {"body": EMAIL["body"] + "\n" + bad})
        with pytest.raises(WriterError, match="did not pass the checks"):
            Claude(client=fake).email(PRACTICE, REASONS, VISITS)


def test_claude_text_that_is_not_on_the_record_is_not_kept():
    with pytest.raises(WriterError, match="EHR that is not on the practice's record"):
        Claude(client=FakeAnthropic(EMAIL | {"body": EMAIL["body"].replace("athenahealth", "Cerner")})).email(PRACTICE, REASONS, VISITS)
    with pytest.raises(WriterError, match="12 providers; the record says 4"):
        Claude(client=FakeAnthropic(BRIEF | {"why_now": "With 12 providers they are growing."})).brief(PRACTICE, REASONS, VISITS)
    assert outreach.ungrounded("Built for 2 to 10 providers; your 4 providers fit. No Epic integration yet.", PRACTICE) == []


def test_claude_brief_shape_is_checked():
    assert Claude(client=FakeAnthropic(BRIEF)).brief(PRACTICE, REASONS, VISITS) == BRIEF
    with pytest.raises(WriterError, match="3 questions"):
        Claude(client=FakeAnthropic(BRIEF | {"questions": ["Only one?"]})).brief(PRACTICE, REASONS, VISITS)


def test_claude_refusal_and_api_errors_become_writer_errors():
    import anthropic
    import httpx2
    with pytest.raises(WriterError, match="declined"):
        Claude(client=FakeAnthropic("refusal")).email(PRACTICE, REASONS, VISITS)
    with pytest.raises(WriterError, match="cut off"):
        Claude(client=FakeAnthropic("max_tokens")).email(PRACTICE, REASONS, VISITS)
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    with pytest.raises(WriterError, match="could not be reached"):
        Claude(client=FakeAnthropic(anthropic.APIConnectionError(request=request))).email(PRACTICE, REASONS, VISITS)
    limited = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)
    with pytest.raises(WriterError, match="rate limited"):
        Claude(client=FakeAnthropic(limited)).email(PRACTICE, REASONS, VISITS)


# ---------------- on the page ----------------
def first_lead(app):
    return app.state.conn.execute("SELECT l.id, p.name, p.providers, p.ehr, p.contact_name FROM leads l JOIN practices p "
                                  "ON p.id = l.practice_id WHERE l.status='new' ORDER BY l.score DESC LIMIT 1").fetchone()


def test_template_app_shows_the_seeded_brief_and_no_claude_button(app, sdr):
    page = sdr.get(f"/lead/{first_lead(app)['id']}").text
    assert "Call brief" in page and "written by the template" in page and "Three questions" in page
    assert "Have Claude write" not in page


def test_claude_writes_the_email_and_the_audit_log_says_so(claude_app, fake):
    sdr, lead = login(TestClient(claude_app)), first_lead(claude_app)
    page = sdr.get(f"/lead/{lead['id']}").text
    assert "Have Claude write it" in page and "Have Claude write the brief" in page
    body = f"Hi there,\n\nA note for {lead['name']} about online booking and the waitlist.\n\nWould 20 minutes next week be useful?"
    fake.replies.append({"subject": "A note on booking", "body": body})
    r = sdr.post(f"/lead/{lead['id']}/write", data={"what": "email"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("wrote=email")
    row = claude_app.state.conn.execute("SELECT subject, body, written_by FROM leads WHERE id=?", (lead["id"],)).fetchone()
    assert row["written_by"] == "claude" and row["subject"] == "A note on booking" and row["body"].endswith(outreach.FOOTER)
    page = sdr.get(f"/lead/{lead['id']}?wrote=email").text
    assert "Written by Claude." in page and "Have Claude write it again" in page
    seen = page.split('name="seen" value="')[1].split('"')[0]
    sdr.post(f"/lead/{lead['id']}/approve", data={"subject": row["subject"], "body": row["body"], "seen": seen})
    audit = claude_app.state.conn.execute("SELECT written_by, edited FROM audit WHERE lead_id=?", (lead["id"],)).fetchone()
    assert (audit["written_by"], audit["edited"]) == ("claude", 0)


def test_a_failed_write_leaves_the_draft_and_says_why(claude_app, fake):
    sdr, lead = login(TestClient(claude_app)), first_lead(claude_app)
    before = claude_app.state.conn.execute("SELECT subject, body, brief FROM leads WHERE id=?", (lead["id"],)).fetchone()
    fake.replies += ["refusal", {"subject": "Hi", "body": "We guarantee results for every practice, every single time."}]
    r = sdr.post(f"/lead/{lead['id']}/write", data={"what": "brief"})
    assert r.status_code == 502 and "Not written." in r.text and "declined" in r.text
    r = sdr.post(f"/lead/{lead['id']}/write", data={"what": "email"})
    assert r.status_code == 502 and "did not pass the checks" in r.text
    after = claude_app.state.conn.execute("SELECT subject, body, brief FROM leads WHERE id=?", (lead["id"],)).fetchone()
    assert tuple(after) == tuple(before)


def test_an_email_is_not_rewritten_after_the_decision(claude_app, fake):
    sdr, lead = login(TestClient(claude_app)), first_lead(claude_app)
    seen = sdr.get(f"/lead/{lead['id']}").text.split('name="seen" value="')[1].split('"')[0]
    sdr.post(f"/lead/{lead['id']}/reject", data={"reason": "fit", "seen": seen})
    r = sdr.post(f"/lead/{lead['id']}/write", data={"what": "email"}, follow_redirects=False)
    assert r.headers["location"].endswith("stale=1") and fake.requests == []
    assert sdr.post(f"/lead/{lead['id']}/write", data={"what": "poem"}).status_code == 400
