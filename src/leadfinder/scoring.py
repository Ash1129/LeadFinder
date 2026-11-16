"""The explainable score: a points table over the facts that define a practice and what it did on the site.

Every point comes with a reason a rep can check, and each reason names its source (the NPI-style record, or the
visit log), so a score is never a number without its why. The weights are a starting table, meant to be moved by
what SDRs reject and approve (the feedback loop in the proposal); they live here, in one place, for that reason.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

SPECIALTY_POINTS = {
    "Family Medicine": 20,
    "Pediatrics": 18,
    "Dermatology": 16,
    "Internal Medicine": 15,
    "Orthopedic Surgery": 12,
    "Physical Therapy": 10,
}
OTHER_SPECIALTY = 4
EHR_INTEGRATED = {"athenahealth", "eClinicalWorks", "NextGen"}     # the seller's product reads and writes these today
PAGE_POINTS = {"/demo": 20, "/pricing": 15, "/integrations": 8, "/features": 4, "/blog": 2}
PAGE_CAP = 40
RECENT_DAYS = 7
RECENT_POINTS = 10
BANDS = ((70, "hot"), (45, "warm"), (0, "cool"))


@dataclass(frozen=True)
class Reason:
    label: str
    points: int
    source: str          # "NPI record" or "site visits"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Score:
    total: int
    band: str
    reasons: list[Reason]


def band_of(total: int) -> str:
    return next(name for floor, name in BANDS if total >= floor)


def _page_key(page: str) -> str:
    return "/blog" if page.startswith("/blog") else page


def score_practice(specialty: str, providers: int, ehr: Optional[str], visits: Iterable[tuple[str, float]],
                   now: Optional[float] = None) -> Score:
    """`visits`: (page, unix time) pairs. Returns the 0-100 total, its band and the reasons, largest effect first."""
    now = time.time() if now is None else now
    reasons: list[Reason] = []

    sp = SPECIALTY_POINTS.get(specialty)
    reasons.append(Reason(f"{specialty}: {'a target specialty' if sp else 'outside the target specialties'}",
                          sp if sp is not None else OTHER_SPECIALTY, "NPI record"))

    if providers <= 1:
        reasons.append(Reason("Solo practice: little room for a scheduling tool", 4, "NPI record"))
    elif providers <= 10:
        reasons.append(Reason(f"{providers} providers: the size the product is built for", 20, "NPI record"))
    elif providers <= 25:
        reasons.append(Reason(f"{providers} providers: large for an independent practice", 10, "NPI record"))
    else:
        reasons.append(Reason(f"{providers} providers: likely buys through a health system's committees", 0, "NPI record"))

    if not ehr:
        reasons.append(Reason("EHR unknown: integration not yet checked", 5, "NPI record"))
    elif ehr in EHR_INTEGRATED:
        reasons.append(Reason(f"Runs {ehr}, which the product integrates with", 15, "NPI record"))
    else:
        reasons.append(Reason(f"Runs {ehr}: no integration yet", 0, "NPI record"))

    visits = list(visits)
    seen: dict[str, int] = {}
    for page, _ in visits:
        k = _page_key(page)
        if k in PAGE_POINTS:
            seen[k] = seen.get(k, 0) + 1
    page_total = 0
    for k, n in sorted(seen.items(), key=lambda kv: -PAGE_POINTS[kv[0]]):
        pts = min(PAGE_POINTS[k], PAGE_CAP - page_total)        # each kind of page counts once, the sum is capped
        if pts <= 0:
            break
        page_total += pts
        reasons.append(Reason(f"Visited {k}{f' ({n} times)' if n > 1 else ''}", pts, "site visits"))
    if not visits:
        reasons.append(Reason("No site visits on record", 0, "site visits"))
    elif max(at for _, at in visits) >= now - RECENT_DAYS * 86400:
        reasons.append(Reason(f"Visited in the last {RECENT_DAYS} days", RECENT_POINTS, "site visits"))

    reasons.sort(key=lambda r: -r.points)
    total = max(0, min(100, sum(r.points for r in reasons)))
    return Score(total=total, band=band_of(total), reasons=reasons)
