"""The send step, kept off until the team agrees who owns sending and which mail provider it uses.

LEADFINDER_SENDER picks it:
- unset or "off" (the default): there is no send step. An approval only marks the email ready to send, exactly as
  before, and the lead page shows no send button.
- "dry-run": an approved email can be queued. The dry-run sender checks the words once more and records the message
  in the outbox table with status "dry_run"; it never opens a network connection and never emails anyone.

A real provider (SMTP, SendGrid, ...) is one more class with the same `send` method, added once the team decides.
Whatever it is, the app only ever hands it an email an SDR approved, which passes the guardrail again here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

from . import outreach


class SendError(RuntimeError):
    """The message was not handed over; the text is fit to show on the page."""


@dataclass(frozen=True)
class Message:
    to_name: Optional[str]
    to_email: str
    subject: str
    body: str


@dataclass(frozen=True)
class Receipt:
    status: str            # "dry_run" here; a real provider would say e.g. "accepted"
    detail: str


class Sender(Protocol):
    name: str

    def send(self, message: Message) -> Receipt: ...


def preflight(message: Message) -> None:
    """The last gate before any sender: an address to send to, and words that still pass the guardrail."""
    if "@" not in (message.to_email or ""):
        raise SendError("the lead has no email address to send to")
    problems = outreach.check(message.subject, message.body)
    if problems:
        raise SendError("the approved email no longer passes the checks: " + "; ".join(problems))


@dataclass
class DryRun:
    """Records instead of sending. Nothing leaves the machine."""
    name: str = "dry-run"

    def send(self, message: Message) -> Receipt:
        preflight(message)
        return Receipt("dry_run", "recorded in the outbox; not sent (dry run)")


def make_sender(kind: Optional[str] = None) -> Optional[Sender]:
    """None when sending is off (the default); otherwise the sender LEADFINDER_SENDER names."""
    kind = (kind or os.environ.get("LEADFINDER_SENDER") or "off").strip().lower()
    if kind == "off":
        return None
    if kind == "dry-run":
        return DryRun()
    raise ValueError("LEADFINDER_SENDER must be off or dry-run (no real mail provider is agreed yet)")
