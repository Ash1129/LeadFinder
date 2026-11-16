"""The drafted first email and the guardrail every draft passes before an SDR can approve it.

The draft is a template filled from the practice's public business facts and the reasons it scored well; nothing
about patients ever enters it. The guardrail checks the words an SDR is about to approve (the template's or their
edit): no patient information, no claims the seller cannot back, and the CAN-SPAM footer (who sends it, a postal
address and a way to opt out) still there.
"""
from __future__ import annotations

import re

from .scoring import EHR_INTEGRATED

INTEGRATED_EHRS = EHR_INTEGRATED
KNOWN_EHRS = ("athenahealth", "eClinicalWorks", "NextGen", "Epic", "Practice Fusion", "Cerner", "Oracle Health", "Allscripts",
              "Veradigm", "Kareo", "Tebra", "DrChrono", "AdvancedMD", "Meditech", "ModMed", "Greenway")
SELLER = "Clearwell"
PRODUCT = "Clearwell Scheduling"
FOOTER = ("Clearwell Health, Inc. · 200 Example Street, Suite 4, Ithaca, NY 14850\n"
          "Not the right person, or would rather not hear from us? Reply \"unsubscribe\" and we will stop.")

PATIENT_TERMS = ("patient name", "your patient", "diagnosis", "diagnoses", "diagnosed", "medical record", "mrn", "date of birth",
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


def _says(text: str, term: str) -> bool:
    """As a word: "phi" is not in "Philadelphia", "cure" not in "secure"."""
    return re.search(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])", text.lower()) is not None


def check_words(text: str) -> list[str]:
    """Patient information and claims the seller cannot back, in any text a practice will read or hear."""
    problems = []
    for t in PATIENT_TERMS:
        if _says(text, t):
            problems.append(f"it mentions patient information (“{t}”); outreach uses public business facts only")
            break
    for t in CLAIM_TERMS:
        if _says(text, t):
            problems.append(f"it makes a claim the seller cannot back (“{t}”)")
            break
    return problems


def check(subject: str, body: str) -> list[str]:
    """What stops a draft from being approved; [] when it may go."""
    problems = []
    text = f"{subject}\n{body}"
    if not subject.strip():
        problems.append("the subject is empty")
    if len(body.strip()) < 40:
        problems.append("the body is too short to be an email")
    problems += check_words(text)
    if "unsubscribe" not in text.lower() or "street" not in text.lower():
        problems.append("the footer with the postal address and the way to opt out is missing (CAN-SPAM)")
    return problems


def ungrounded(text: str, practice: dict) -> list[str]:
    """Things a model's text says about the practice that its record does not: an EHR it is not on record as
    running (the ones the product integrates with may be named as product facts), or another number of providers."""
    problems = []
    own = (practice.get("ehr") or "").lower()
    for ehr in KNOWN_EHRS:
        if ehr.lower() != own and ehr not in INTEGRATED_EHRS and _says(text, ehr) and not (ehr == "Epic" and "no epic" in text.lower()):
            problems.append(f"it names an EHR that is not on the practice's record ({ehr})")
            break
    sized = re.sub(r"2\s*(?:to|-|–|-to-)\s*10", "", text.lower())            # "2 to 10 providers" is the product's range
    for n in re.findall(r"(\d+)[ -]provider", sized):
        if int(n) != practice.get("providers"):
            problems.append(f"it says {n} providers; the record says {practice.get('providers')}")
            break
    return problems


def check_brief(brief: dict, practice: dict) -> list[str]:
    """What stops a call brief from being kept: the wrong shape, the same words an email may not use, or a fact
    about the practice that is not on its record."""
    problems = []
    if not all(isinstance(brief.get(k), str) and brief[k].strip() for k in ("opener", "why_now")):
        problems.append("the opener or the reason to call is missing")
    for key, n in (("talking_points", 3), ("questions", 3), ("objections", 2)):
        items = brief.get(key)
        if not isinstance(items, list) or len(items) != n:
            problems.append(f"it should have {n} {key.replace('_', ' ')}")
    objections = brief.get("objections") if isinstance(brief.get("objections"), list) else []
    if not all(isinstance(o, dict) and str(o.get("objection", "")).strip() and str(o.get("reply", "")).strip() for o in objections):
        problems.append("an objection is missing its reply")
    if problems:
        return problems
    text = "\n".join([brief["opener"], brief["why_now"], *map(str, brief["talking_points"]), *map(str, brief["questions"]),
                      *(f"{o['objection']} {o['reply']}" for o in objections)])
    return check_words(text) + ungrounded(text, practice)
