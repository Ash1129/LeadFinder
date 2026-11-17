"""Evaluate practice matching in research.py against the true practice behind each simulated visitor.

Uses the module's own deterministic safeguards (`rank_candidates`, MATCH_THRESHOLD, MATCH_MARGIN,
AMBIGUOUS_THRESHOLD). The LLM in the research loop can only *confirm* a match that passes these safeguards, so this
is the best case for the agent: if the deterministic layer matches the wrong practice, the agent will too.

The directory lookup is a stand-in for the practice-directory adapter: it returns every practice whose name is reasonably close
to the hint, or whose domain matches, capped at research.MAX_DIRECTORY_RESULTS.
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from .. import research as rs
from .personas import GroundTruth, Practice


def _lookup(req: rs.ResearchRequest, practices: list[Practice]) -> list[rs.PracticeCandidate]:
    want_name = rs._normalized(req.practice_name)
    want_domain = rs._domain(req.domain_hint)
    scored = []
    for p in practices:
        sim = difflib.SequenceMatcher(None, want_name, rs._normalized(p.name)).ratio()
        if sim >= 0.5 or (want_domain and rs._domain(p.domain) == want_domain):
            scored.append((sim, p))
    scored.sort(key=lambda x: -x[0])
    return [
        rs.PracticeCandidate(p.practice_id, p.name, p.domain, f"{p.city}, {p.state}", p.specialty, p.providers)
        for _, p in scored[: rs.MAX_DIRECTORY_RESULTS]
    ]


def decide(ranked) -> tuple[str, str | None]:
    """Same rule as ResearchAgent._result for a model that picks the top candidate."""
    if not ranked:
        return "not_found", None
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if top.confidence >= rs.MATCH_THRESHOLD and (second is None or top.confidence - second.confidence >= rs.MATCH_MARGIN):
        return "matched", top.practice_id
    if top.confidence >= rs.AMBIGUOUS_THRESHOLD:
        return "ambiguous", None
    return "not_found", None


def match_metrics(inputs, truth: list[GroundTruth], practices: list[Practice], extras: dict, only_ids=None) -> dict:
    truth_by_id = {t.lead_id: t for t in truth}
    outcomes = Counter()
    by_noise = defaultdict(Counter)
    for i in inputs:
        if not i.company_identified or (only_ids is not None and i.lead_id not in only_ids):
            continue
        t = truth_by_id[i.lead_id]
        req = rs.ResearchRequest(i.practice_name_raw, i.domain_or_org_hint, i.location_hint)
        ranked = rs.rank_candidates(req, _lookup(req, practices))[:3]
        status, chosen = decide(ranked)
        if status == "matched":
            outcome = "correct_match" if chosen == t.true_practice_id else "WRONG_match"
        elif status == "ambiguous":
            in_top3 = any(c.practice_id == t.true_practice_id for c in ranked)
            outcome = "ambiguous_truth_in_top3" if in_top3 else "ambiguous_truth_missing"
        else:
            outcome = "not_found"
        outcomes[outcome] += 1
        by_noise[extras[i.lead_id]["hint_noise"]][outcome] += 1
        has_domain = "with_domain" if i.domain_or_org_hint else "no_domain"
        by_noise[has_domain][outcome] += 1
    n = sum(outcomes.values())
    return {"n": n, "outcomes": dict(outcomes), "by_noise": {k: dict(v) for k, v in sorted(by_noise.items())}}
