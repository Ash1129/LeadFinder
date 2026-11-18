"""The outreach routes: the bounded revision loop on write, the SDR's send-back, the dry-run send step and the
feedback summary on the audit page. Claude is the fake client from conftest; nothing reaches the network."""
import pytest
from fastapi.testclient import TestClient

from leadfinder import outreach, revision
from leadfinder.sender import DryRun
from leadfinder.service.app import create_app
from leadfinder.writer import Claude

from .conftest import FakeAnthropic, login


def good(name: str) -> dict:
    return {"subject": "A note on booking",
            "body": f"Hi there,\n\nA note for {name} about online booking and the waitlist.\n\nWould 20 minutes next week be useful?"}


def bad(name: str) -> dict:
    return good(name) | {"body": good(name)["body"] + "\nWe guarantee fewer no-shows."}


@pytest.fixture
def full_fake():
    return FakeAnthropic()


@pytest.fixture
def full_app(tmp_path, full_fake):
    """Claude writing, two revisions allowed, sending in dry-run mode."""
    return create_app(tmp_path / "full.db", demo=True, writer=Claude(client=full_fake), revisions=2, sender=DryRun())


def top_lead(app) -> dict:
    return dict(app.state.conn.execute("SELECT l.id, p.name FROM leads l JOIN practices p ON p.id = l.practice_id "
                                       "WHERE l.status='new' ORDER BY l.score DESC LIMIT 1").fetchone())


def seen(client, lead_id: int) -> str:
    return client.get(f"/lead/{lead_id}").text.split('name="seen" value="')[1].split('"')[0]


def write(client, fake, lead, *replies):
    fake.replies += list(replies)
    return client.post(f"/lead/{lead['id']}/write", data={"what": "email"}, follow_redirects=False)


def count(app, sql: str, *args) -> int:
    return app.state.conn.execute(sql, args).fetchone()[0]


# ---------------- nothing changes where the new steps are off ----------------
def test_template_app_has_no_send_back_and_no_send_button(app, sdr):
    lead_id = top_lead(app)["id"]
    sdr.post(f"/lead/{lead_id}/write", data={"what": "email"})
    page = sdr.get(f"/lead/{lead_id}").text
    assert "Approve the email" in page and "send-back" not in page and "Send back" not in page
    assert sdr.post(f"/lead/{lead_id}/send-back", data={"note": "shorter", "seen": seen(sdr, lead_id)}).status_code == 400
    assert sdr.post(f"/lead/{lead_id}/send").status_code == 404                          # sending is off by default
    assert count(app, "SELECT COUNT(*) FROM draft_attempts WHERE lead_id=? AND passed=1 AND kept=1", lead_id) == 1


def test_by_default_a_failed_draft_is_not_revised(claude_app, fake):
    sdr, lead = login(TestClient(claude_app)), top_lead(claude_app)
    r = write(sdr, fake, lead, bad(lead["name"]))
    assert r.status_code == 502 and "did not pass the checks" in r.text and len(fake.requests) == 1
    assert count(claude_app, "SELECT COUNT(*) FROM draft_attempts WHERE lead_id=? AND passed=0 AND kept=0", lead["id"]) == 1


# ---------------- the revision loop ----------------
def test_a_failed_draft_is_revised_and_every_try_is_logged(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    r = write(sdr, full_fake, lead, bad(lead["name"]), good(lead["name"]))
    assert r.status_code == 303 and r.headers["location"].endswith("wrote=email")
    assert "The checker stopped it" in full_fake.requests[1]["messages"][0]["content"]
    row = full_app.state.conn.execute("SELECT body FROM leads WHERE id=?", (lead["id"],)).fetchone()
    assert "guarantee" not in row["body"] and row["body"].endswith(outreach.FOOTER)
    tries = full_app.state.conn.execute("SELECT attempt, passed, kept FROM draft_attempts WHERE lead_id=? ORDER BY attempt",
                                        (lead["id"],)).fetchall()
    assert [tuple(t) for t in tries] == [(1, 0, 0), (2, 1, 1)]


def test_after_two_revisions_the_draft_is_left_as_it_was(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    r = write(sdr, full_fake, lead, *[bad(lead["name"])] * 3)
    assert r.status_code == 502 and "after 3 tries" in r.text and len(full_fake.requests) == 3
    assert count(full_app, "SELECT COUNT(*) FROM leads WHERE id=? AND body=''", lead["id"]) == 1


# ---------------- send-back ----------------
def test_send_back_rewrites_the_draft_and_keeps_the_lead_undecided(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, good(lead["name"]))
    page = sdr.get(f"/lead/{lead['id']}").text
    assert "Send back</button>" in page and "2 of 2 left" in page
    full_fake.replies.append(good(lead["name"]) | {"subject": "Booking, briefly"})
    r = sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "Shorter, lead with the waitlist", "seen": seen(sdr, lead["id"])},
                 follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("sentback=1")
    said = full_fake.requests[-1]["messages"][0]["content"]
    assert "lead with the waitlist" in said and "A note on booking" in said
    assert count(full_app, "SELECT COUNT(*) FROM leads WHERE id=? AND status='new' AND subject='Booking, briefly'", lead["id"]) == 1
    assert count(full_app, "SELECT COUNT(*) FROM audit") == 0                         # not a decision: the audit log is untouched
    page = sdr.get(f"/lead/{lead['id']}?sentback=1").text
    assert "Sent back and rewritten." in page and "Sent back · 1" in page and "1 of 2 left" in page
    row = full_app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead["id"],)).fetchone()
    sdr.post(f"/lead/{lead['id']}/approve", data={"subject": row["subject"], "body": row["body"], "seen": seen(sdr, lead["id"])})
    assert count(full_app, "SELECT COUNT(*) FROM leads WHERE id=? AND status='approved'", lead["id"]) == 1


def test_send_back_is_bounded_and_needs_a_note(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, good(lead["name"]))
    r = sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "   ", "seen": seen(sdr, lead["id"])})
    assert r.status_code == 422 and "Say what to change" in r.text
    for _ in range(revision.MAX_SEND_BACKS):
        full_fake.replies.append(good(lead["name"]))
        sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "again", "seen": seen(sdr, lead["id"])})
    calls = len(full_fake.requests)
    r = sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "once more", "seen": seen(sdr, lead["id"])})
    assert r.status_code == 409 and "the most allowed" in r.text and len(full_fake.requests) == calls
    assert "0 of 2 left" in sdr.get(f"/lead/{lead['id']}").text


def test_a_failed_send_back_leaves_the_draft_and_is_logged(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, good(lead["name"]))
    before = tuple(full_app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead["id"],)).fetchone())
    full_fake.replies.append("refusal")
    r = sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "shorter", "seen": seen(sdr, lead["id"])})
    assert r.status_code == 502 and "The draft is unchanged." in r.text
    assert tuple(full_app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead["id"],)).fetchone()) == before
    assert count(full_app, "SELECT COUNT(*) FROM send_backs WHERE lead_id=? AND outcome='failed'", lead["id"]) == 1


def test_a_decided_lead_cannot_be_sent_back(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, good(lead["name"]))
    token = seen(sdr, lead["id"])
    sdr.post(f"/lead/{lead['id']}/reject", data={"reason": "fit", "seen": token})
    calls = len(full_fake.requests)
    r = sdr.post(f"/lead/{lead['id']}/send-back", data={"note": "shorter", "seen": token}, follow_redirects=False)
    assert r.headers["location"].endswith("stale=1") and len(full_fake.requests) == calls


# ---------------- the dry-run send step ----------------
def test_an_approved_email_can_be_queued_once_and_is_never_sent(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, good(lead["name"]))
    assert sdr.post(f"/lead/{lead['id']}/send", follow_redirects=False).headers["location"].endswith("stale=1")   # not approved yet
    row = full_app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead["id"],)).fetchone()
    sdr.post(f"/lead/{lead['id']}/approve", data={"subject": row["subject"], "body": row["body"], "seen": seen(sdr, lead["id"])})
    assert "Queue it (dry run)" in sdr.get(f"/lead/{lead['id']}").text
    r = sdr.post(f"/lead/{lead['id']}/send", follow_redirects=False)
    assert r.headers["location"].endswith("queued=1")
    sdr.post(f"/lead/{lead['id']}/send")
    out = full_app.state.conn.execute("SELECT * FROM outbox WHERE lead_id=?", (lead["id"],)).fetchall()
    assert len(out) == 1 and out[0]["status"] == "dry_run" and out[0]["sender"] == "dry-run" and out[0]["queued_by"] == "riley"
    page = sdr.get(f"/lead/{lead['id']}?queued=1").text
    assert "Recorded, not sent." in page and "In the outbox (dry-run)" in page and "Queue it (dry run)" not in page


# ---------------- feedback ----------------
def test_the_manager_sees_the_feedback_summary_and_sdrs_do_not(full_app, full_fake):
    sdr, lead = login(TestClient(full_app)), top_lead(full_app)
    write(sdr, full_fake, lead, bad(lead["name"]), good(lead["name"]))
    row = full_app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead["id"],)).fetchone()
    sdr.post(f"/lead/{lead['id']}/approve", data={"subject": row["subject"], "body": row["body"] + "\nThanks.",
                                                  "seen": seen(sdr, lead["id"])})
    assert sdr.get("/api/outreach-feedback").status_code == 403
    manager = login(TestClient(full_app), "morgan")
    data = manager.get("/api/outreach-feedback").json()
    assert data["decisions"]["approved_edited"] == 1 and data["draft_checks"]["passed_after_revision"] == 1
    assert data["draft_checks"]["problems"] == {"it makes a claim the seller cannot back": 1}
    page = manager.get("/audit").text
    assert "Outreach feedback" in page and "1 approved, 0 rejected" in page and "Approved after an edit" in page
