import smtplib

import pytest
from fastapi.testclient import TestClient

from leadfinder import outreach
from leadfinder.service.app import create_app
from leadfinder.service.email_html import BRAND, LOGO_CID, LOGO_URL, branded
from leadfinder.service.mailer import Mailer, SignInRefused
from leadfinder.writer import Offline

from .conftest import form_value, login

BOX = "tester@mail.example"


class FakeMailer(Mailer):
    """A configured mailer that keeps what it would have sent instead of opening a connection."""

    def __init__(self, fail: Exception | None = None):
        super().__init__(user=BOX, password="app-password", sender=BOX, test_to=BOX)
        self.fail, self.sent = fail, []

    def send(self, intended, subject, text, html, from_name="", images=None):
        if self.fail:
            raise self.fail
        m = self.message(intended, subject, text, html, from_name, images)
        self.sent.append(m)
        return {"to": m["To"], "from": m["From"], "subject": m["Subject"]}


@pytest.fixture
def mailer():
    return FakeMailer()


@pytest.fixture
def mail_app(tmp_path, mailer):
    return create_app(tmp_path / "mail.db", demo=True, writer=Offline(), mailer=mailer)


def approved_lead(app, client) -> int:
    """Write and approve the top lead's email; the lead's id."""
    lead_id = app.state.conn.execute("SELECT id FROM leads WHERE status='new' ORDER BY score DESC LIMIT 1").fetchone()["id"]
    client.post(f"/lead/{lead_id}/write", data={"what": "email"})
    page = client.get(f"/lead/{lead_id}").text
    row = app.state.conn.execute("SELECT subject, body FROM leads WHERE id=?", (lead_id,)).fetchone()
    r = client.post(f"/lead/{lead_id}/approve", data={"subject": row["subject"], "body": row["body"], "seen": form_value(page, "seen")},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("done=approved")
    return lead_id


# ---------------------------------------------------------------- the mailer
def test_nothing_is_configured_without_a_test_mailbox():
    assert not Mailer.from_env({}).configured
    assert not Mailer.from_env({"LEADFINDER_SMTP_USER": BOX, "LEADFINDER_SMTP_PASS": "x"}).configured
    m = Mailer.from_env({"LEADFINDER_SMTP_USER": BOX, "LEADFINDER_SMTP_PASS": "x", "LEADFINDER_SMTP_TEST_TO": "me@mail.example"})
    assert m.configured and (m.host, m.port, m.sender, m.test_to) == ("smtp.gmail.com", 587, BOX, "me@mail.example")
    with pytest.raises(RuntimeError, match="not set up"):
        Mailer().send("a@practice.example", "s", "t", "<p>t</p>")


def test_every_message_goes_to_the_test_mailbox_and_says_who_it_was_for():
    m = FakeMailer().message("kim@practice.example", "Scheduling for Oak Park", "plain words", "<p>branded</p>", "Riley Chen")
    assert m["To"] == BOX and m["Subject"] == "[for kim@practice.example] Scheduling for Oak Park"
    assert m["From"] == f"Riley Chen <{BOX}>" and "kim@practice.example" not in m["To"]
    parts = {p.get_content_type(): p.get_content() for p in m.iter_parts()}
    assert parts["text/plain"].strip() == "plain words" and parts["text/html"].strip() == "<p>branded</p>"


def test_the_logo_travels_inside_the_message():
    png = b"\x89PNG\r\n\x1a\n-not-really"
    m = FakeMailer().message("kim@practice.example", "s", "plain", '<img src="cid:logo">', images={"logo": png})
    plain, related = m.iter_parts()
    assert plain.get_content_type() == "text/plain" and related.get_content_type() == "multipart/related"
    html, image = related.iter_parts()
    assert html.get_content_type() == "text/html" and image.get_content_type() == "image/png"
    assert image["Content-ID"] == "<logo>" and image.get_content_disposition() == "inline" and image.get_content() == png


def test_a_refused_sign_in_says_what_to_change():
    class Dropped:
        def login(self, user, password):
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
    with pytest.raises(SignInRefused, match="app password"):
        FakeMailer()._login(Dropped())


# ---------------------------------------------------------------- the branded email
def test_branded_email_has_the_logo_the_words_and_the_footer():
    _, body = outreach.draft({"name": "Oak Park Pediatrics", "specialty": "Pediatrics", "providers": 4, "city": "Ithaca", "ehr": None,
                              "contact_name": "Kim Lee"}, [])
    html = branded(body + "\n\nP.S. <b>not markup</b> & https://clearwell.example/demo.", "Riley Chen")
    assert f">{BRAND}</span>" in html and f'<img src="cid:{LOGO_CID}"' in html and f'alt="{BRAND} logo"' in html
    assert "Online booking" in html and "font-size:24px" in html.split("Hi Kim,")[0][-200:]       # the product line; the greeting set taller
    assert f'<img src="{LOGO_URL}"' in branded(body, logo=LOGO_URL)                                 # as a page of the service shows it
    assert "Hi Kim," in html and "200 Example Street" in html and "unsubscribe" in html and "Sent by Riley Chen" in html
    assert "&lt;b&gt;not markup&lt;/b&gt; &amp;" in html and '<a href="https://clearwell.example/demo"' in html
    assert "\n--\n" not in html and "<script" not in html and not outreach.check("Scheduling", body)


# ---------------------------------------------------------------- sending from the lead's page
def test_without_a_mailer_there_is_no_send_button_and_nothing_sends(app, sdr):
    lead_id = approved_lead(app, sdr)
    page = sdr.get(f"/lead/{lead_id}").text
    assert f"/lead/{lead_id}/send" not in page and "as it will be sent" not in page
    assert sdr.post(f"/lead/{lead_id}/send").status_code == 409
    assert "nothing is sent" in sdr.get("/").text and "no way to send mail" in sdr.get("/how").text


def test_an_approved_email_is_sent_once_branded_and_logged(mail_app, mailer):
    sdr, manager = login(TestClient(mail_app), "riley"), login(TestClient(mail_app), "morgan")
    lead_id = approved_lead(mail_app, sdr)
    page = sdr.get(f"/lead/{lead_id}").text
    assert f'action="/lead/{lead_id}/send"' in page and "as it will be sent" in page and BOX in page and f'<img src="{LOGO_URL}"' in page
    logo = TestClient(mail_app).get(LOGO_URL)                                       # the picture needs no session, like the script
    assert logo.status_code == 200 and logo.headers["content-type"] == "image/png" and logo.content[:4] == b"\x89PNG"
    r = sdr.post(f"/lead/{lead_id}/send", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("done=sent")
    lead = mail_app.state.conn.execute("SELECT l.subject, l.body, p.contact_email FROM leads l JOIN practices p ON p.id=l.practice_id "
                                       "WHERE l.id=?", (lead_id,)).fetchone()
    (m,) = mailer.sent
    assert m["To"] == BOX and m["Subject"] == f"[for {lead['contact_email']}] {lead['subject']}" and m["From"] == f"Riley Chen <{BOX}>"
    plain, related = m.iter_parts()
    html, image = related.iter_parts()
    assert plain.get_content().strip() == lead["body"].strip() and BRAND in html.get_content() and "200 Example Street" in html.get_content()
    assert f'src="cid:{LOGO_CID}"' in html.get_content() and image["Content-ID"] == f"<{LOGO_CID}>" and image.get_content()[:4] == b"\x89PNG"
    row = mail_app.state.conn.execute("SELECT username, note, body FROM audit WHERE lead_id=? AND action='sent'", (lead_id,)).fetchone()
    assert (row["username"], row["note"], row["body"]) == ("riley", f"to {BOX}", lead["body"])
    page = sdr.get(f"/lead/{lead_id}?done=sent").text
    assert "<b>Sent.</b>" in page and "approved and sent" in page and "as it was sent" in page and f'action="/lead/{lead_id}/send"' not in page
    assert sdr.post(f"/lead/{lead_id}/send", follow_redirects=False).headers["location"] == f"/lead/{lead_id}" and len(mailer.sent) == 1
    log = manager.get("/audit").text
    assert "1 approved, 0 rejected, 1 sent" in log and f"Branded email sent to {BOX}" in log
    assert "only to a test mailbox" in sdr.get("/").text and "no email goes to a practice" in sdr.get("/how").text


def test_only_an_approved_email_can_be_sent(mail_app, mailer):
    sdr = login(TestClient(mail_app), "riley")
    lead_id = mail_app.state.conn.execute("SELECT id FROM leads WHERE status='new' ORDER BY score DESC LIMIT 1").fetchone()["id"]
    assert sdr.post(f"/lead/{lead_id}/send").status_code == 409                      # nothing written, nothing approved
    sdr.post(f"/lead/{lead_id}/write", data={"what": "email"})
    assert sdr.post(f"/lead/{lead_id}/send").status_code == 409 and not mailer.sent  # written, not approved
    assert TestClient(mail_app).post(f"/lead/{lead_id}/send").status_code == 401     # and never without a session


def test_a_failed_send_says_why_and_logs_nothing(tmp_path):
    mailer = FakeMailer(fail=SignInRefused("smtp.gmail.com refused the sign-in for tester@mail.example: 535 bad credentials"))
    app = create_app(tmp_path / "fail.db", demo=True, writer=Offline(), mailer=mailer)
    sdr = login(TestClient(app), "riley")
    lead_id = approved_lead(app, sdr)
    r = sdr.post(f"/lead/{lead_id}/send")
    assert r.status_code == 502 and "<b>Not sent.</b>" in r.text and "refused the sign-in" in r.text and "Nothing went out" in r.text
    assert app.state.conn.execute("SELECT COUNT(*) n FROM audit WHERE action='sent'").fetchone()["n"] == 0
    assert f'action="/lead/{lead_id}/send"' in sdr.get(f"/lead/{lead_id}").text       # still there to try again
