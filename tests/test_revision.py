"""The draft-check-revise loop, the send-back rewrite, the dry-run sender and the feedback summary, without the web app.

These need no FastAPI: Claude is a fake client (the same shape as conftest.FakeAnthropic) and the store is SQLite in
memory. The routes that use them are tested in test_outreach_flow.py.
"""
import json
import time
from types import SimpleNamespace

import pytest

from leadfinder import feedback, outreach, revision
from leadfinder.db import connect
from leadfinder.sender import DryRun, Message, SendError, make_sender
from leadfinder.writer import Claude, Offline, WriterError

PRACTICE = {"name": "Maple Grove Pediatrics", "specialty": "Pediatrics", "providers": 4, "city": "Ithaca", "state": "NY",
            "ehr": "athenahealth", "contact_name": "Priya Patel", "contact_email": "office@maplegrove.example", "npi": "1234567893"}
REASONS = [{"label": "Visited /demo", "points": 20, "source": "site visits"},
           {"label": "Runs athenahealth, which the product integrates with", "points": 15, "source": "NPI record"}]
VISITS = ["/demo", "/pricing"]
GOOD = {"subject": "Booking at Maple Grove Pediatrics",
        "body": "Hi Priya,\n\nSomeone at Maple Grove looked at our demo page this week. Clearwell Scheduling works inside "
                "athenahealth, so your front desk keeps one calendar.\n\nWould 20 minutes next week be useful?\n\nThe Clearwell team"}
CLAIM = GOOD | {"body": GOOD["body"] + "\nWe guarantee fewer no-shows."}
WRONG_EHR = GOOD | {"body": GOOD["body"].replace("athenahealth", "Cerner")}


class Fake:
    """Answers `beta.messages.create` with queued replies and keeps every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return SimpleNamespace(stop_reason=reply, content=[])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(reply))])

    def said(self, i: int) -> str:
        return self.requests[i]["messages"][0]["content"]


# ---------------- the loop ----------------
def test_a_draft_that_passes_is_kept_after_one_try():
    fake = Fake(GOOD)
    out = revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=2)
    assert out.subject == GOOD["subject"] and out.body.endswith(outreach.FOOTER)
    assert [a.passed for a in out.attempts] == [True] and out.revisions == 0 and len(fake.requests) == 1


def test_a_failed_draft_is_revised_with_what_the_checks_found():
    fake = Fake(CLAIM, GOOD)
    out = revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=2)
    assert [a.passed for a in out.attempts] == [False, True] and out.revisions == 1
    assert "guarantee" in out.attempts[0].problems[0] and "guarantee" not in out.body
    second = fake.said(1)
    assert "The checker stopped it" in second and "guarantee" in second and "We guarantee fewer no-shows." in second
    assert outreach.FOOTER not in second                     # the model is never shown the footer to copy or change


def test_the_loop_stops_after_two_revisions_and_nothing_is_kept():
    fake = Fake(CLAIM, WRONG_EHR, CLAIM, GOOD)
    with pytest.raises(WriterError, match="did not pass the checks after 3 tries") as e:
        revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=2)
    assert len(fake.requests) == 3 and [a.passed for a in e.value.attempts] == [False, False, False]
    assert "Cerner" in e.value.attempts[1].problems[0]


def test_with_no_revisions_a_failed_draft_is_refused_as_before():
    fake = Fake(CLAIM)
    with pytest.raises(WriterError) as e:
        revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=0)
    assert str(e.value).startswith("Claude's draft did not pass the checks: ") and len(fake.requests) == 1
    assert len(e.value.attempts) == 1


def test_revisions_are_capped_at_two():
    fake = Fake(CLAIM, CLAIM, CLAIM, GOOD)
    with pytest.raises(WriterError):
        revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=9)
    assert len(fake.requests) == 3


def test_a_writer_error_mid_loop_carries_the_attempts_so_far():
    fake = Fake(CLAIM, "refusal")
    with pytest.raises(WriterError, match="declined") as e:
        revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=2)
    assert len(e.value.attempts) == 1 and not e.value.attempts[0].passed


def test_claude_email_still_behaves_as_before():
    subject, body = Claude(client=Fake(GOOD)).email(PRACTICE, REASONS, VISITS)
    assert subject == GOOD["subject"] and body.endswith(outreach.FOOTER)
    with pytest.raises(WriterError, match="did not pass the checks"):
        Claude(client=Fake(CLAIM)).email(PRACTICE, REASONS, VISITS)


# ---------------- send-back ----------------
def test_a_send_back_rewrites_from_the_note_and_the_previous_draft():
    fake = Fake(GOOD)
    previous = ("Old subject", "Hi Priya,\n\nAn older draft about booking.\n\n--\n" + outreach.FOOTER)
    out = revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, note="Shorter, and lead with the waitlist.",
                               previous=previous)
    said = fake.said(0)
    assert "sent it back with this note" in said and "lead with the waitlist" in said and "An older draft" in said
    assert "rules above, which always win" in said and outreach.FOOTER not in said
    assert out.attempts[0].passed


def test_a_send_back_note_cannot_break_the_rules():
    fake = Fake(CLAIM, CLAIM, CLAIM)
    with pytest.raises(WriterError):
        revision.write_email(Claude(client=fake), PRACTICE, REASONS, VISITS, revisions=2, note="Promise them a guarantee.",
                             previous=(GOOD["subject"], GOOD["body"]))


def test_the_template_cannot_be_sent_back_but_writes_as_before():
    out = revision.write_email(Offline(), PRACTICE, REASONS, VISITS, revisions=2)
    assert (out.subject, out.body) == outreach.draft(PRACTICE, REASONS) and out.attempts[0].passed
    with pytest.raises(WriterError, match="template cannot be sent back"):
        revision.write_email(Offline(), PRACTICE, REASONS, VISITS, note="shorter", previous=(out.subject, out.body))
    assert not revision.can_revise(Offline()) and revision.can_revise(Claude(client=Fake()))


def test_notes_are_trimmed_and_required():
    assert revision.clean_note("  shorter \n please ") == "shorter please"
    assert len(revision.clean_note("x" * 900)) == revision.NOTE_LIMIT
    with pytest.raises(ValueError):
        revision.clean_note("   ")


def test_revisions_setting_reads_the_environment(monkeypatch):
    monkeypatch.delenv("LEADFINDER_REVISIONS", raising=False)
    assert revision.revisions_from_env() == 0
    monkeypatch.setenv("LEADFINDER_REVISIONS", "2")
    assert revision.revisions_from_env() == 2
    for bad in ("3", "-1", "two"):
        monkeypatch.setenv("LEADFINDER_REVISIONS", bad)
        with pytest.raises(ValueError):
            revision.revisions_from_env()


# ---------------- the send step ----------------
def test_sending_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("LEADFINDER_SENDER", raising=False)
    assert make_sender() is None and make_sender("off") is None
    monkeypatch.setenv("LEADFINDER_SENDER", "dry-run")
    assert isinstance(make_sender(), DryRun)
    with pytest.raises(ValueError):
        make_sender("smtp")


def test_the_dry_run_records_and_rechecks():
    subject, body = outreach.draft(PRACTICE, REASONS)
    receipt = DryRun().send(Message("Priya Patel", "office@maplegrove.example", subject, body))
    assert receipt.status == "dry_run" and "not sent" in receipt.detail
    with pytest.raises(SendError, match="no longer passes"):
        DryRun().send(Message("Priya Patel", "office@maplegrove.example", subject, body.replace("unsubscribe", "")))
    with pytest.raises(SendError, match="no email address"):
        DryRun().send(Message("Priya Patel", "", subject, body))


# ---------------- the feedback summary ----------------
def test_feedback_summary_counts_what_is_logged():
    conn = connect(":memory:")
    assert feedback.summary(conn)["decisions"]["approval_rate"] is None          # nothing yet, nothing divided by zero
    conn.execute("INSERT INTO practices(id, npi, name, specialty, providers, city, state, source) "
                 "VALUES(1, '1', 'A', 'Pediatrics', 4, 'Ithaca', 'NY', 'test'), (2, '2', 'B', 'Pediatrics', 4, 'Ithaca', 'NY', 'test'), "
                 "(3, '3', 'C', 'Pediatrics', 4, 'Ithaca', 'NY', 'test')")
    now = time.time()
    for lead, status in ((1, "approved"), (2, "approved"), (3, "rejected")):
        conn.execute("INSERT INTO leads(id, practice_id, score, band, reasons, subject, body, status, scored_at) "
                     "VALUES(?,?,50,'warm','[]','s','b',?,?)", (lead, lead, status, now))
    conn.execute("INSERT INTO audit(lead_id, username, action, edited, at, written_by) VALUES(1,'riley','approved',0,?,'claude'),"
                 "(2,'riley','approved',1,?,'claude')", (now, now))
    conn.execute("INSERT INTO audit(lead_id, username, action, reason, at, written_by) VALUES(3,'jo','rejected','fit',?,'template')", (now,))
    conn.execute("INSERT INTO send_backs(lead_id, username, note, subject, body, outcome, at) VALUES(2,'riley','shorter','s','b','rewritten',?)", (now,))
    rows = [("r1", 1, 0, '["it makes a claim the seller cannot back (“guarantee”)"]'), ("r1", 2, 1, "[]"),
            ("r2", 1, 1, "[]"),
            ("r3", 1, 0, '["it names an EHR that is not on the practice\'s record (Cerner)"]'),
            ("r3", 2, 0, '["it makes a claim the seller cannot back (“#1”)"]')]
    for rnd, n, passed, problems in rows:
        conn.execute("INSERT INTO draft_attempts(lead_id, round, purpose, attempt, writer, subject, body, problems, passed, username, at) "
                     "VALUES(1,?,'write',?,'claude','s','b',?,?,'riley',?)", (rnd, n, problems, passed, now))
    conn.execute("INSERT INTO outbox(lead_id, to_email, subject, body, sender, status, queued_by, queued_at) "
                 "VALUES(1,'a@b.example','s','b','dry-run','dry_run','riley',?)", (now,))
    s = feedback.summary(conn)
    assert s["decisions"] | {"by_writer": None, "rejected_by_reason": None} == {
        "decided": 3, "approved": 2, "approved_as_drafted": 1, "approved_edited": 1, "rejected": 1, "rejected_by_reason": None,
        "by_writer": None, "approval_rate": 0.667, "edit_rate_of_approvals": 0.5}
    assert s["decisions"]["rejected_by_reason"] == {"fit": 1}
    assert s["decisions"]["by_writer"]["claude"] == {"approved_as_drafted": 1, "approved_edited": 1, "rejected": 0}
    assert s["send_backs"] == {"total": 1, "rewritten": 1, "failed": 0, "leads": 1, "leads_approved_after": 1}
    ck = s["draft_checks"]
    assert (ck["rounds"], ck["passed_first_try"], ck["passed_after_revision"], ck["never_passed"]) == (3, 1, 1, 1)
    assert ck["problems"] == {"it makes a claim the seller cannot back": 2, "it names an EHR that is not on the practice's record": 1}
    assert s["outbox"] == {"dry_run": 1}
