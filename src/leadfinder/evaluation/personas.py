"""Persona-based visit simulator with a separate ground-truth file.

Each simulated visitor gets
    - a `VisitAnalysisInput` (exactly what `visit_analysis.analyze_visits` receives), and
    - a `GroundTruth` row (persona, true practice, whether the gate *should* pass them, whether they convert).
The two are returned separately and must stay separate: ground truth never enters lead state or any model input.

Every URL in the catalog carries the category a person would give it (`intended`), so the evaluation can measure
taxonomy coverage: how often `visit_analysis` puts a page in the right category and tier.

Conversion reuses the practice part of the app's hidden rule (`synth.conversion_chance` with no visits) and adds the
persona's buying intent and unobserved noise. All weights are assumptions, versioned in `GT_VERSION`.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .. import synth
from ..visit_analysis import PageView, VisitAnalysisInput

GT_VERSION = "gt-v1"
DAY = 86400.0

PERSONA_MIX = {"browser": 0.45, "researcher": 0.20, "ready_to_buy": 0.10, "administrator": 0.15, "owner": 0.10}
SHOULD_PASS = {"researcher", "ready_to_buy", "administrator", "owner"}
INTENT = {"browser": -1.5, "researcher": 0.8, "ready_to_buy": 1.6, "administrator": 0.3, "owner": 1.0}
HIDDEN_NOISE_SD = 0.6

# (url, intended category). Includes realistic URLs the taxonomy may not expect.
URLS: dict[str, list[tuple[str, str]]] = {
    "demo": [("/demo", "demo_or_trial"), ("/request-demo", "demo_or_trial"), ("/free-trial", "demo_or_trial"),
             ("/request-a-demo", "demo_or_trial"), ("/schedule-demo", "demo_or_trial")],
    "pricing": [("/pricing", "pricing"), ("/plans", "pricing"), ("/pricing?plan=pro", "pricing")],
    "integration": [("/integrations", "ehr_integration"), ("/integrations/athenahealth", "ehr_integration"),
                    ("/integrates-with-eclinicalworks", "ehr_integration"), ("/integrations/nextgen", "ehr_integration")],
    "security": [("/security", "security"), ("/hipaa", "security"), ("/data-export", "security")],
    "billing": [("/billing", "billing"), ("/revenue-cycle", "billing")],
    "case_study": [("/case-studies/pediatrics", "specialty_case_study"),
                   ("/case-studies/family-medicine", "specialty_case_study"),
                   ("/customers/dermatology", "specialty_case_study")],
    "features": [("/features", "usability"), ("/features/online-booking", "usability"), ("/scheduling", "usability")],
    "blog": [("/blog/no-show-rates", "blog"), ("/blog/waitlists", "blog"), ("/blog/patient-reminders-101", "blog"),
             ("/blog/ehr-integration-tips", "blog"), ("/blog/pricing-your-services", "blog"),
             ("/blog/hipaa-checklist", "blog"), ("/webinars/no-show-rates", "blog")],
    "careers": [("/careers", "careers"), ("/careers/billing-specialist", "careers")],
    "about": [("/about", "about"), ("/about/team", "about")],
    "home": [("/", "homepage")],
}

# Which page groups each persona favors (weights), and how much it browses.
BEHAVIOR = {
    "browser": {"groups": {"blog": 6, "home": 3, "about": 2, "careers": 2, "features": 1},
                "sessions": (1, 1), "pages": (1, 2), "p_identified": 0.40},
    "researcher": {"groups": {"integration": 3, "security": 3, "features": 3, "case_study": 2, "blog": 2, "pricing": 1, "home": 1},
                   "sessions": (2, 4), "pages": (2, 5), "p_identified": 0.70},
    "ready_to_buy": {"groups": {"pricing": 4, "demo": 4, "integration": 2, "features": 1, "home": 1},
                     "sessions": (1, 3), "pages": (2, 4), "p_identified": 0.80},
    "administrator": {"groups": {"billing": 4, "integration": 3, "features": 3, "pricing": 2, "home": 1},
                      "sessions": (1, 2), "pages": (1, 3), "p_identified": 0.65},
    "owner": {"groups": {"pricing": 3, "case_study": 3, "security": 2, "demo": 2, "home": 1},
              "sessions": (1, 3), "pages": (1, 4), "p_identified": 0.70},
}


@dataclass(frozen=True)
class Practice:
    practice_id: str
    name: str
    domain: str
    city: str
    state: str
    specialty: str
    providers: int
    ehr: str | None


@dataclass(frozen=True)
class GroundTruth:
    lead_id: str
    persona: str
    true_practice_id: str
    should_pass_gate: bool
    p_convert: float
    converted: bool
    gt_version: str = GT_VERSION


def make_practices(n: int, rng: random.Random) -> list[Practice]:
    """Synthetic directory. Names repeat across cities on purpose (e.g. two 'Lakeside Pediatrics'), like real data."""
    out, domains = [], set()
    for i in range(n):
        p = synth.practice(rng)
        name = f"{rng.choice(synth.PLACES)} {synth.SUFFIX[p['specialty']]}"
        slug = "".join(ch for ch in name.lower() if ch.isalnum())
        domain, k = f"{slug}.example", 2
        while domain in domains:
            domain, k = f"{slug}{k}.example", k + 1
        domains.add(domain)
        out.append(Practice(f"PR{i:05d}", name, domain, p["city"], p["state"], p["specialty"], p["providers"], p["ehr"]))
    return out


def conversion_probability(practice: Practice, persona: str, rng: random.Random) -> float:
    base = synth.conversion_chance(
        {"specialty": practice.specialty, "providers": practice.providers, "ehr": practice.ehr}, [], 0.0
    )
    z = math.log(base / (1 - base)) + INTENT[persona] + rng.gauss(0, HIDDEN_NOISE_SD)
    return 1 / (1 + math.exp(-z))


def _views(persona: str, lead_id: str, now: float, rng: random.Random) -> tuple[tuple[PageView, ...], list[str]]:
    b = BEHAVIOR[persona]
    groups, weights = zip(*b["groups"].items())
    views, intended = [], []
    n_sessions = rng.randint(*b["sessions"])
    starts = sorted(now - rng.uniform(0, 25) * DAY for _ in range(n_sessions))
    for s, start in enumerate(starts):
        t = start
        for _ in range(rng.randint(*b["pages"])):
            url, category = rng.choice(URLS[rng.choices(groups, weights)[0]])
            dwell = rng.uniform(10, 240)
            views.append(PageView(f"{lead_id}-v{len(views)}", url, t, round(dwell, 1)))
            intended.append(category)
            t += dwell + rng.uniform(5, 120)  # well inside the 30-minute session timeout
    return tuple(views), intended


def _name_hint(practice: Practice, rng: random.Random) -> tuple[str, str]:
    """What a tracking tool reports as the company name, and which kind of noise was applied."""
    r = rng.random()
    name = practice.name
    if r < 0.45:
        return name, "exact"
    if r < 0.65:
        abbrev = (name.replace("Pediatrics", "Peds").replace("Internal Medicine Associates", "IM Assoc.")
                  .replace("Family Medicine", "Family Med").replace("Physical Therapy", "PT")
                  .replace("Orthopedics", "Ortho").replace("Dermatology", "Derm"))
        return (abbrev, "abbreviation") if abbrev != name else (name.upper(), "case")
    if r < 0.85:
        i = rng.randrange(1, len(name) - 1)
        return name[:i] + name[i + 1:], "typo"
    return name.split(" ")[0] + " " + name.split(" ")[-1] + " LLC", "legal_suffix"


def simulate(n: int = 2000, seed: int = 42, n_practices: int = 400, now: float = 1_795_000_000.0):
    """Return (practices, inputs, truth, extras). `extras[lead_id]` keeps evaluation-only metadata
    (intended URL categories and the hint-noise type) and, like `truth`, must never reach the pipeline."""
    rng = random.Random(seed)
    practices = make_practices(n_practices, rng)
    personas, mix = zip(*PERSONA_MIX.items())
    inputs, truth, extras = [], [], {}
    for i in range(n):
        lead_id = f"SIM{i:05d}"
        persona = rng.choices(personas, mix)[0]
        practice = rng.choice(practices)
        views, intended = _views(persona, lead_id, now, rng)
        identified = rng.random() < BEHAVIOR[persona]["p_identified"]
        if identified:
            name, noise = _name_hint(practice, rng)
            domain = rng.choice([practice.domain, f"www.{practice.domain}", None])
            location = rng.choice([f"{practice.city}, {practice.state}", practice.city, None])
            confidence = round(min(1.0, max(0.05, rng.gauss(0.8, 0.15))), 3)
        else:
            name, noise, domain, location, confidence = None, "unidentified", None, None, 0.0
        inputs.append(VisitAnalysisInput(lead_id, name, domain, location, identified, confidence, views, now, True))
        p = conversion_probability(practice, persona, rng)
        truth.append(GroundTruth(lead_id, persona, practice.practice_id, persona in SHOULD_PASS, round(p, 4),
                                 rng.random() < p))
        extras[lead_id] = {"intended_categories": intended, "hint_noise": noise}
    return practices, inputs, truth, extras


def truth_rows(truth: list[GroundTruth]) -> list[dict]:
    return [asdict(t) for t in truth]
