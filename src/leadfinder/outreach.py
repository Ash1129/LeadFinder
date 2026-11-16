"""The drafted first email and the guardrail every draft passes before an SDR can approve it.

The draft is a template filled from the practice's public business facts and the reasons it scored well; nothing
about patients ever enters it. The guardrail checks the words an SDR is about to approve (the template's or their
edit): no patient information, no claims the seller cannot back, and the CAN-SPAM footer (who sends it, a postal
address and a way to opt out) still there.
"""
from __future__ import annotations

import re

SELLER = "Clearwell"
PRODUCT = "Clearwell Scheduling"
FOOTER = ("Clearwell Health, Inc. · 200 Example Street, Suite 4, Ithaca, NY 14850\n"
          "Not the right person, or would rather not hear from us? Reply \"unsubscribe\" and we will stop.")

PATIENT_TERMS = ("patient name", "your patient", "patients'", "diagnosis", "diagnoses", "diagnosed", "medical record", "mrn", "date of birth",
                 "dob", "ssn", "social security", "insurance id", "member id", "treatment of", "phi")
CLAIM_TERMS = ("guarantee", "hipaa certified", "hipaa-certified", "fda approved", "fda-approved", "100%", "risk-free",
               "#1", "best in the country", "cure")


def draft(practice: dict, reasons: list[dict]) -> tuple[str, str]:
    """(subject, body) for a practice row (name, specialty, providers, city, ehr, contact_name)."""
    first = (practice.get("contact_name") or "").split(" ")[0] or "there"
    ehr = practice.get("ehr")
    hook = {
        True: f"It works inside {ehr}, so your front desk keeps one calendar instead of two.",
        False: "It runs beside the EHR you have today, and we can show you how the hand-off works.",
    }[bool(ehr) and any("integrates" in r["label"] for r in reasons)]
    visited_pricing = any(r["label"].startswith("Visited /pricing") for r in reasons)
    visited_demo = any(r["label"].startswith("Visited /demo") for r in reasons)
    ask = ("You looked at our demo page, so I'll keep it short: would 20 minutes next week to walk through it with "
           "your schedule in mind be useful?") if visited_demo else (
          "You had a look at our pricing, so I'm glad to go through what a practice your size would pay, line by line."
          if visited_pricing else "Would a 20-minute look next week be useful?")
    n = practice["providers"]
    size = f"A team of {n} providers is the size it was built for. " if 2 <= n <= 10 else ""
    subject = f"Scheduling for {practice['name']}"
    body = (f"Hi {first},\n\n"
            f"I work with independent {practice['specialty'].lower()} practices in and around {practice['city']} on "
            f"{PRODUCT}: online booking, reminders and a waitlist that fills cancelled slots. {size}"
            f"{hook}\n\n{ask}\n\nBest,\nThe {SELLER} team\n\n--\n{FOOTER}")
    return subject, body


def check(subject: str, body: str) -> list[str]:
    """What stops a draft from being approved; [] when it may go."""
    problems = []
    text = f"{subject}\n{body}".lower()
    if not subject.strip():
        problems.append("the subject is empty")
    if len(body.strip()) < 40:
        problems.append("the body is too short to be an email")
    def says(t: str) -> bool:       # as a word: "phi" is not in "Philadelphia", "cure" not in "secure"
        return re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", text) is not None

    for t in PATIENT_TERMS:
        if says(t):
            problems.append(f"it mentions patient information (“{t}”); outreach uses public business facts only")
            break
    for t in CLAIM_TERMS:
        if says(t):
            problems.append(f"it makes a claim the seller cannot back (“{t}”)")
            break
    if "unsubscribe" not in text or "street" not in text:
        problems.append("the footer with the postal address and the way to opt out is missing (CAN-SPAM)")
    return problems
