"""Sending an approved email from the service, over SMTP, the way AuxoCase's mailer.py does.

"Send it" on an approved lead's page sends the email from here: the plain text and the branded HTML (email_html.py)
as one multipart message, from the configured mailbox with the SDR's name on it. The logo goes inside the
message, beside the HTML, so it shows without the mail client fetching anything.

Every email goes to a test mailbox, never to the practice. The practices are synthetic or have no address on record,
and outreach to real practices is not agreed yet, so the address the email was for is put in the subject instead:
"[for <the practice's contact>] <subject>". Without a test mailbox set, nothing sends.

Settings, read from the environment:
  LEADFINDER_SMTP_USER, LEADFINDER_SMTP_PASS   the mailbox and its password (for Gmail, an app password)
  LEADFINDER_SMTP_TEST_TO                      where every email goes; unset, nothing sends
  LEADFINDER_SMTP_HOST, LEADFINDER_SMTP_PORT   default smtp.gmail.com, 587 (STARTTLS; 465 is SSL)
  LEADFINDER_SMTP_FROM                         the address on the From line, default the user; the SDR's name is put in front
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr
from typing import Mapping, Optional

TIMEOUT_S = 20


@dataclass
class Mailer:
    user: str = ""
    password: str = ""
    host: str = "smtp.gmail.com"
    port: int = 587
    sender: str = ""
    test_to: str = ""

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Mailer":
        env = os.environ if env is None else env
        user = env.get("LEADFINDER_SMTP_USER", "").strip()
        return cls(user=user, password=env.get("LEADFINDER_SMTP_PASS", ""),
                   host=env.get("LEADFINDER_SMTP_HOST", "").strip() or "smtp.gmail.com", port=int(env.get("LEADFINDER_SMTP_PORT") or 587),
                   sender=env.get("LEADFINDER_SMTP_FROM", "").strip() or user, test_to=env.get("LEADFINDER_SMTP_TEST_TO", "").strip())

    @property
    def configured(self) -> bool:
        """A mailbox to send from and a test mailbox to send to; without both, the page has no Send button."""
        return bool(self.user and self.password and self.test_to)

    def message(self, intended: str, subject: str, text: str, html: str, from_name: str = "",
                images: Optional[Mapping[str, bytes]] = None) -> EmailMessage:
        """The email as it goes: to the test mailbox, the subject saying who it was for (`intended`: the practice's
        contact address, or its name when there is none), text and HTML alternatives, the sender's name on From.
        `images`: PNGs by content id, carried with the HTML, which shows one as <img src="cid:the-id">."""
        m = EmailMessage()
        m["To"], m["Subject"] = self.test_to, f"[for {intended}] {subject}"
        m["From"] = formataddr((from_name, self.sender)) if from_name else self.sender
        m.set_content(text)
        m.add_alternative(html, subtype="html")
        for cid, png in (images or {}).items():
            m.get_payload()[-1].add_related(png, "image", "png", cid=f"<{cid}>", filename=f"{cid}.png", disposition="inline")
        return m

    def send(self, intended: str, subject: str, text: str, html: str, from_name: str = "",
             images: Optional[Mapping[str, bytes]] = None) -> dict:
        """Send it; returns {to, from, subject}: where it went and how it was headed. Raises SignInRefused when the
        mailbox will not sign the service in, and what smtplib raises for anything else."""
        if not self.configured:
            raise RuntimeError("sending is not set up: LEADFINDER_SMTP_USER, LEADFINDER_SMTP_PASS and LEADFINDER_SMTP_TEST_TO are not all set")
        m = self.message(intended, subject, text, html, from_name, images)
        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=TIMEOUT_S) as s:
                self._login(s)
                s.send_message(m)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=TIMEOUT_S) as s:
                s.starttls()
                self._login(s)
                s.send_message(m)
        return {"to": m["To"], "from": m["From"], "subject": m["Subject"]}

    def _login(self, s) -> None:
        """Sign in, and say why not in words the SDR can act on. Gmail answers an account password with a 534 (an app
        password is required) and drops the line, which smtplib reports as a disconnection."""
        try:
            s.login(self.user, self.password)
        except smtplib.SMTPAuthenticationError as e:
            raise SignInRefused(f"{self.host} refused the sign-in for {self.user}: {e.smtp_error.decode(errors='replace').splitlines()[0]}") from e
        except smtplib.SMTPServerDisconnected as e:
            hint = ("; for Gmail, LEADFINDER_SMTP_PASS must be an app password (Google account › Security › 2-Step Verification › "
                    "App passwords), not the account password") if "gmail" in self.host else ""
            raise SignInRefused(f"{self.host} closed the connection at sign-in for {self.user}{hint}") from e


class SignInRefused(Exception):
    """The mailbox would not sign the service in; the message says what to change."""
