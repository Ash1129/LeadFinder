"""Who writes the first email and the call brief: Claude, or the templates when no model is reachable.

- `Claude`: the Anthropic Messages API (the official SDK), with a JSON schema for each answer so the reply is
  always the fields the page needs. It reads ANTHROPIC_API_KEY (or another credential the SDK finds).
- `Offline`: no model at all; the email is `outreach.draft`'s template and the brief is built from the same
  reasons by fixed rules. Every test runs on it, and so does the app when no credential is set.

The model only ever sees public business facts about the practice (the NPI-style record, the pages visited, the
score's reasons) and what the product does. What it writes is checked by code before it is kept (`outreach.check`,
`outreach.check_brief`): no patient information, no claims the seller cannot back, nothing about the practice that
is not on its record. The footer with the postal address and the opt-out is added by code, not asked of the model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from . import outreach

MODEL = os.environ.get("LEADFINDER_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = f"""You write for the sales development reps at {outreach.SELLER} Health, which sells {outreach.PRODUCT} to \
independent medical practices in the United States.

What {outreach.PRODUCT} does, and all you may say it does: online appointment booking, automated appointment \
reminders, and a waitlist that offers cancelled slots to waiting people. It integrates with these EHRs today: \
{", ".join(sorted(outreach.INTEGRATED_EHRS))}. It has no Epic integration yet; with any other EHR it runs beside \
it. It is built for practices of 2 to 10 providers.

You are given one practice: its public business record (specialty, number of providers, city, EHR if known, the \
office contact), the pages someone there visited on our site, and the reasons our scoring gave it its score. Use \
only those facts about the practice. If a fact is not given, do not guess it and do not imply it.

Hard rules, because a person at a medical practice reads or hears this:
- Never mention patients' information, any individual patient, diagnoses or treatment. This is business outreach \
to a business address.
- No prices, discounts, statistics, customer names, guarantees, or compliance claims (for example about HIPAA).
- Plain, specific, courteous. No hype, no exclamation marks, no emoji. Write the way a thoughtful colleague would."""

EMAIL_TASK = """Write the first email to this practice's office contact.

- Subject: under 60 characters, specific to the practice, not clickbait.
- Body: 90 to 140 words, plain text, short paragraphs. Open with the contact's first name. Say why you are \
writing to this practice in particular, using one or two of the facts given. End with one small, concrete ask \
(a 20-minute look next week). Sign off as "The {seller} team".
- Do not add a footer, an address or an unsubscribe line; those are added after you."""

BRIEF_TASK = """Write a call brief the rep reads in the minute before phoning this practice's office contact.

- opener: the first sentence the rep says after hello, naming why they are calling this practice.
- why_now: one sentence on what makes this practice worth a call today, from the facts given.
- talking_points: exactly three, each one sentence, each tied to a fact about the practice or the product.
- questions: exactly three open questions to learn what the record does not show (how they book today, where \
no-shows hurt, who decides).
- objections: exactly two this practice is likely to raise given its record, each with a one-sentence honest reply."""

REVISE_TASK = """Rewrite this email to the same practice. Everything in the first task still applies.

{why}

The draft to rewrite:
Subject: {subject}

{body}"""

EMAIL_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
    "additionalProperties": False,
}
BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string"},
        "why_now": {"type": "string"},
        "talking_points": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
        "objections": {"type": "array", "items": {
            "type": "object",
            "properties": {"objection": {"type": "string"}, "reply": {"type": "string"}},
            "required": ["objection", "reply"],
            "additionalProperties": False,
        }},
    },
    "required": ["opener", "why_now", "talking_points", "questions", "objections"],
    "additionalProperties": False,
}


class WriterError(RuntimeError):
    """The writer could not produce something usable; the message is fit to show on the page."""


class Writer(Protocol):
    name: str            # "claude" or "template": what the page and the audit log say wrote it
    is_model: bool

    def email(self, practice: dict, reasons: list[dict], visits: list[str]) -> tuple[str, str]: ...
    def brief(self, practice: dict, reasons: list[dict], visits: list[str]) -> dict: ...


def facts(practice: dict, reasons: list[dict], visits: list[str]) -> str:
    """The one thing the model is told about the practice, as JSON with sorted keys (the same lead reads the same)."""
    return json.dumps({
        "practice": {k: practice.get(k) for k in ("name", "specialty", "providers", "city", "state", "ehr", "contact_name")},
        "pages_visited": visits,
        "score_reasons": [{"reason": r["label"], "points": r["points"], "source": r["source"]} for r in reasons],
    }, sort_keys=True, indent=1)


# ---------------------------------------------------------------- no model
@dataclass
class Offline:
    name: str = "template"
    is_model: bool = False

    def email(self, practice: dict, reasons: list[dict], visits: list[str]) -> tuple[str, str]:
        return outreach.draft(practice, reasons)

    def brief(self, practice: dict, reasons: list[dict], visits: list[str]) -> dict:
        first = (practice.get("contact_name") or "").split(" ")[0] or "the office manager"
        ehr, n = practice.get("ehr"), practice["providers"]
        integrated = ehr in outreach.INTEGRATED_EHRS
        pages = sorted({("/blog" if v.startswith("/blog") else v) for v in visits if v != "/"})
        top = next((r["label"] for r in reasons if r["points"] > 0), "It is an independent practice")
        why = (f"Someone there looked at {' and '.join(pages[:2])} on our site." if pages
               else f"{top}.")
        points = [
            f"{practice['specialty']} practices lose booked time to no-shows; reminders and the waitlist are built for that.",
            (f"It works inside {ehr}, so the front desk keeps one calendar." if integrated else
             f"It runs beside {ehr or 'the EHR they have'}; say plainly that there is no direct integration yet."),
            (f"With {n} providers they are the size the product was built for." if 2 <= n <= 10 else
             f"With {n} provider{'s' if n != 1 else ''} they are outside the 2-to-10 size it was built for; ask before assuming a fit."),
        ]
        objections = [
            {"objection": "We already book through our EHR." if ehr else "We are happy with how we book now.",
             "reply": "That is common; the question is whether cancelled slots get refilled, which is what the waitlist does."},
            {"objection": f"Does it work with {ehr}?" if ehr else "Will it work with our system?",
             "reply": ("Yes, it reads and writes the schedule there." if integrated else
                       "Not directly today; it runs beside it, and I can show you what that hand-off looks like.")},
        ]
        return {
            "opener": f"Hi {first}, I'm calling from {outreach.SELLER} about scheduling at {practice['name']}.",
            "why_now": why,
            "talking_points": points,
            "questions": ["How do people book with you today, and how much of it is by phone?",
                          "When someone cancels the day before, what happens to that slot?",
                          "Who would decide on a change to how you schedule?"],
            "objections": objections,
        }


# ---------------------------------------------------------------- Claude
@dataclass
class Claude:
    model: str = MODEL
    name: str = "claude"
    is_model: bool = True
    client: Any = field(default=None, repr=False)        # an anthropic.Anthropic; made on first use, or injected by a test

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(timeout=90.0, max_retries=1)
        return self.client

    def _ask(self, task: str, practice: dict, reasons: list[dict], visits: list[str], schema: dict) -> dict:
        import anthropic
        try:
            # Structured output: the reply's first text block is JSON in `schema`. The effort is set because this
            # model's default is medium and a short email does not need more. If the model declines for policy
            # reasons the API retries on its default fallback model inside the same call.
            response = self._client().beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=SYSTEM,
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": f"{task}\n\nThe practice:\n{facts(practice, reasons, visits)}"}],
            )
        except anthropic.AuthenticationError as e:
            raise WriterError("Claude refused the credential; check ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            raise WriterError("Claude is rate limited right now; try again in a minute") from e
        except anthropic.APIStatusError as e:
            raise WriterError(f"Claude answered with an error ({e.status_code})") from e
        except anthropic.APIConnectionError as e:
            raise WriterError("Claude could not be reached") from e
        if response.stop_reason == "refusal":
            raise WriterError("Claude declined to write this")
        if response.stop_reason == "max_tokens":
            raise WriterError("Claude's answer was cut off")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except ValueError as e:
            raise WriterError("Claude's answer was not the JSON asked for") from e
        if not isinstance(data, dict):
            raise WriterError("Claude's answer was not the JSON asked for")
        return data

    def draft_email(self, practice: dict, reasons: list[dict], visits: list[str], *, previous: Optional[tuple[str, str]] = None,
                    problems: Optional[list[str]] = None, note: Optional[str] = None) -> tuple[str, str]:
        """One draft, unchecked: the caller checks it (`email` below, or the revision loop in revision.py).
        With `previous`, Claude rewrites that draft, told what the checks found (`problems`) and/or what the SDR
        asked for when sending it back (`note`). The footer is added by code either way."""
        task = EMAIL_TASK.format(seller=outreach.SELLER)
        if previous is not None:
            why = []
            if problems:
                why.append("The checker stopped it for these reasons; fix every one:\n- " + "\n- ".join(problems))
            if note:
                why.append("The sales rep sent it back with this note. Follow it unless it conflicts with the rules "
                           f"above, which always win:\n{note}")
            old_body = previous[1].split("\n\n--\n")[0]                 # the model never sees or writes the footer
            task += "\n\n" + REVISE_TASK.format(why="\n\n".join(why) or "Improve it.", subject=previous[0], body=old_body)
        data = self._ask(task, practice, reasons, visits, EMAIL_SCHEMA)
        subject = str(data.get("subject", "")).strip()
        body = str(data.get("body", "")).replace("\r\n", "\n").strip()
        return subject, f"{body}\n\n--\n{outreach.FOOTER}"              # the footer is code's, never the model's

    def email(self, practice: dict, reasons: list[dict], visits: list[str]) -> tuple[str, str]:
        subject, body = self.draft_email(practice, reasons, visits)
        problems = outreach.check(subject, body) + outreach.ungrounded(f"{subject}\n{body}", practice)
        if problems:
            raise WriterError("Claude's draft did not pass the checks: " + "; ".join(problems))
        return subject, body

    def brief(self, practice: dict, reasons: list[dict], visits: list[str]) -> dict:
        data = self._ask(BRIEF_TASK, practice, reasons, visits, BRIEF_SCHEMA)
        problems = outreach.check_brief(data, practice)
        if problems:
            raise WriterError("Claude's brief did not pass the checks: " + "; ".join(problems))
        return data


def make_writer(kind: Optional[str] = None) -> Writer:
    """`kind`: "claude" or "template". None reads LEADFINDER_WRITER, and failing that picks Claude when a
    credential is set (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN) and the templates otherwise."""
    kind = kind or os.environ.get("LEADFINDER_WRITER") or (
        "claude" if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") else "template")
    if kind not in ("claude", "template"):
        raise ValueError("LEADFINDER_WRITER must be claude or template")
    return Claude() if kind == "claude" else Offline()
