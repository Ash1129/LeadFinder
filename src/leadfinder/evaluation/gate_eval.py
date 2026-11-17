"""Evaluate the visit-analysis gate and page taxonomy against persona ground truth.

Runs the real `visit_analysis.analyze_visits`. A proposed taxonomy change can be evaluated without editing
visit_analysis.py by passing `classifier=` (see `proposed_classify`), so the before/after comparison is reproducible
and the change can be adopted separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

from .. import visit_analysis as va
from .personas import GroundTruth

LOW_SECTIONS = {"blog": "blog", "news": "blog", "articles": "blog", "webinars": "blog", "resources": "blog",
                "careers": "careers", "jobs": "careers"}
DEMO_ALIASES = ("request-a-demo", "schedule-demo", "book-a-demo", "book-demo", "get-a-demo")

_original_classify = va._classify


def proposed_classify(page_url: str):
    """Taxonomy v2 proposal: (1) a page inside a content section (blog, careers, ...) keeps that section's low tier
    even if its slug contains a keyword like 'pricing' or 'ehr'; (2) common demo URL variants count as demo;
    (3) /customers/<specialty> counts as a specialty case study."""
    path = va._safe_path(page_url)
    first = path.strip("/").split("/")[0]
    if first in LOW_SECTIONS:
        category = LOW_SECTIONS[first]
        return va._PageClassification(category, category, va.LOW_TIER)
    if first in DEMO_ALIASES:
        return va._PageClassification("demo/trial", "demo_or_trial", va.HIGH_TIER, is_demo=True)
    if first == "customers":
        return _original_classify(path.replace("/customers/", "/case-studies/", 1))
    return _original_classify(page_url)


@contextmanager
def using_classifier(classifier):
    if classifier is None:
        yield
        return
    with mock.patch.object(va, "_classify", classifier):
        yield


@dataclass
class GateRun:
    label: str
    results: dict  # lead_id -> VisitAnalysisResult


def run_gate(inputs, config: va.ThresholdConfig = va.ThresholdConfig(), classifier=None, label: str = "") -> GateRun:
    with using_classifier(classifier):
        results = {i.lead_id: va.analyze_visits(i, config) for i in inputs}
    return GateRun(label or config.version, results)


def _identified(r) -> bool:
    return r.company_identified and r.threshold_fail_reason not in ("unidentified", "low_identification")


def gate_metrics(run: GateRun, truth: list[GroundTruth]) -> dict:
    by_persona = defaultdict(lambda: {"n": 0, "identified": 0, "passed": 0, "passed_identified": 0})
    tp = fn = fp = tn = 0
    lost = Counter()
    for t in truth:
        r = run.results[t.lead_id]
        row = by_persona[t.persona]
        ident = _identified(r)
        row["n"] += 1
        row["identified"] += ident
        row["passed"] += r.passes_threshold
        row["passed_identified"] += r.passes_threshold and ident
        if ident:
            if t.should_pass_gate:
                tp += r.passes_threshold
                fn += not r.passes_threshold
            else:
                fp += r.passes_threshold
                tn += not r.passes_threshold
        if t.converted:
            lost[r.threshold_fail_reason or "passed"] += 1
    personas = {
        p: {
            "n": v["n"],
            "identified_rate": v["identified"] / v["n"],
            "pass_rate_all": v["passed"] / v["n"],
            "pass_rate_identified": v["passed_identified"] / max(v["identified"], 1),
        }
        for p, v in sorted(by_persona.items())
    }
    return {
        "label": run.label,
        "by_persona": personas,
        "recall_real_leads": tp / max(tp + fn, 1),
        "false_pass_rate_browsers": fp / max(fp + tn, 1),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "converters_by_outcome": dict(lost.most_common()),
        "passed": sum(r.passes_threshold for r in run.results.values()),
    }


def taxonomy_coverage(run: GateRun, extras: dict, classifier=None) -> dict:
    """Share of simulated page views whose category matches the intended one, plus the misclassified URLs."""
    total = correct = 0
    wrong = Counter()
    with using_classifier(classifier):
        for lead_id, result in run.results.items():
            intended = extras[lead_id]["intended_categories"]
            for page, want in zip(result.pages_viewed, intended):
                total += 1
                if page.category == want:
                    correct += 1
                else:
                    wrong[(want, page.category)] += 1
    # Map wrong pairs back to example URLs for the report.
    return {"page_views": total, "accuracy": correct / max(total, 1), "errors": wrong.most_common()}


def misclassified_urls(classifier=None) -> list[tuple[str, str, str, str]]:
    """Every catalog URL whose category differs from the intended one: (url, intended, got, tier)."""
    from .personas import URLS
    out = []
    fn = classifier or _original_classify
    for group in URLS.values():
        for url, want in group:
            got = fn(url)
            if got.category != want:
                out.append((url, want, got.category, got.tier))
    return out
