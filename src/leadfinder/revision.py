"""Draft, check, revise: the bounded loop between the writer and the guardrail, and the SDR's send-back.

The brainstorm's outreach step: draft the email, check it in code, and when the checks stop it, ask the writer for
at most two revisions, each told exactly what the checks found. An SDR may also send a draft back with a note; the
writer rewrites it from that note, and the result passes the same checks and the same bounded loop.

Every attempt is returned (`Drafted.attempts`, or `WriterError.attempts` when nothing passed) so the app can log it:
how often a first draft fails, and on what, is evidence for the course rubric's draft-check measure.

Only a writer that can revise (one with `draft_email`, i.e. Claude) is looped or sent back. The template writer
cannot take a note, so it is called once exactly as before, and its draft is shown with the guardrail's verdict.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from . import outreach
from .writer import Writer, WriterError

MAX_REVISIONS = 2          # the brainstorm's bound: at most two revisions after the first draft
MAX_SEND_BACKS = 2         # how many times one lead's draft may be sent back by an SDR
NOTE_LIMIT = 500


@dataclass
class Attempt:
    n: int                 # 1 is the first draft, 2 and 3 are revisions
    subject: str
    body: str
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class Drafted:
    subject: str
    body: str
    attempts: list[Attempt]

    @property
    def revisions(self) -> int:
        return len(self.attempts) - 1


def can_revise(writer: Writer) -> bool:
    return callable(getattr(writer, "draft_email", None))


def revisions_from_env() -> int:
    """LEADFINDER_REVISIONS: how many revisions the app asks for when a draft fails the checks, 0 to 2. Unset is 0,
    which is how the app behaved before this loop existed (a failed draft is refused with the reasons)."""
    raw = os.environ.get("LEADFINDER_REVISIONS", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError("LEADFINDER_REVISIONS must be a whole number from 0 to 2") from e
    if not 0 <= n <= MAX_REVISIONS:
        raise ValueError("LEADFINDER_REVISIONS must be a whole number from 0 to 2")
    return n


def problems_with(subject: str, body: str, practice: dict) -> list[str]:
    """Everything that stops a model's email: the guardrail, plus nothing about the practice that is not on record."""
    return outreach.check(subject, body) + outreach.ungrounded(f"{subject}\n{body}", practice)


def clean_note(note: str) -> str:
    note = " ".join((note or "").split())
    if not note:
        raise ValueError("say what to change before sending the draft back")
    return note[:NOTE_LIMIT]


def write_email(writer: Writer, practice: dict, reasons: list[dict], visits: list[str], *, revisions: int = 0,
                note: Optional[str] = None, previous: Optional[tuple[str, str]] = None) -> Drafted:
    """The first email (or, with `note` and `previous`, a sent-back draft's rewrite), checked, with up to `revisions`
    more tries when the checks stop it. Raises WriterError, carrying `.attempts`, when no attempt passes or the
    writer fails; nothing should be kept then."""
    revisions = max(0, min(int(revisions), MAX_REVISIONS))
    attempts: list[Attempt] = []
    if not can_revise(writer):
        if note is not None:
            raise WriterError("the template cannot be sent back with a note; edit the draft yourself instead")
        subject, body = writer.email(practice, reasons, visits)              # unchanged: the template is never refused here
        attempts.append(Attempt(1, subject, body, outreach.check(subject, body)))
        return Drafted(subject, body, attempts)
    try:
        subject, body = writer.draft_email(practice, reasons, visits, previous=previous, note=note)
        attempts.append(Attempt(1, subject, body, problems_with(subject, body, practice)))
        while not attempts[-1].passed and len(attempts) <= revisions:
            last = attempts[-1]
            subject, body = writer.draft_email(practice, reasons, visits, previous=(last.subject, last.body),
                                               problems=last.problems, note=note)
            attempts.append(Attempt(len(attempts) + 1, subject, body, problems_with(subject, body, practice)))
    except WriterError as e:
        e.attempts = attempts
        raise
    if not attempts[-1].passed:
        tried = f" after {len(attempts)} tries" if len(attempts) > 1 else ""
        error = WriterError(f"Claude's draft did not pass the checks{tried}: " + "; ".join(attempts[-1].problems))
        error.attempts = attempts
        raise error
    return Drafted(attempts[-1].subject, attempts[-1].body, attempts)
