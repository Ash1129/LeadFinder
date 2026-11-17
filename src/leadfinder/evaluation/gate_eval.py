"""Compare taxonomy v2 in the real gate with the preserved v1 baseline."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

from .. import visit_analysis as va
from .personas import GroundTruth

def baseline_v1_classify(page_url: str) -> va._PageClassification:
    """Preserve the pre-v2 classifier for reproducible before/after evaluation."""
    path = va._safe_path(page_url)
    if va._path_has(path, "demo", "request-demo", "demo-request", "trial", "free-trial"):
        return va._PageClassification("demo/trial", "demo_or_trial", va.HIGH_TIER, is_demo=True)
    if va._path_has(path, "pricing", "plans", "plans-and-pricing"):
        return va._PageClassification("pricing", "pricing", va.HIGH_TIER, ("pricing",))
    if va._path_has(path, "integration", "integrations", "ehr", "integrates-with"):
        ehr = next((name for name in va.EHR_NAMES if name.casefold().replace(" ", "-") in path), None)
        return va._PageClassification("EHR integration", "ehr_integration", va.HIGH_TIER, ("integration",),
                                      ("integrations",), ehr_hint=ehr)
    if va._path_has(path, "security", "hipaa", "privacy", "data-exit", "data-portability", "data-export"):
        return va._PageClassification("security", "security", va.MEDIUM_TIER, ("security",))
    if va._path_has(path, "billing", "revenue-cycle", "revenue-cycle-management"):
        return va._PageClassification("billing", "billing", va.MEDIUM_TIER, product_lines=("billing",))

    parts = {part for part in path.split("/") if part}
    specialty = next((value for slug, value in va.SPECIALTY_SLUGS.items() if slug in parts), None)
    if va._path_has(path, "case-study", "case-studies", "customer-stories") and specialty:
        return va._PageClassification("specialty case study", "specialty_case_study", va.MEDIUM_TIER,
                                      specialty_hint=specialty)
    if va._path_has(path, "features", "workflow", "usability", "booking", "scheduling", "appointments"):
        product = "scheduling" if va._path_has(path, "booking", "scheduling", "appointments") else "workflow"
        return va._PageClassification("workflow/features", "usability", va.MEDIUM_TIER, ("usability",), (product,))
    if va._path_has(path, "blog", "news", "articles"):
        return va._PageClassification("blog", "blog", va.LOW_TIER)
    if va._path_has(path, "careers", "jobs", "employment"):
        return va._PageClassification("careers", "careers", va.LOW_TIER)
    if va._path_has(path, "about", "company"):
        return va._PageClassification("about", "about", va.LOW_TIER)
    if path == "/":
        return va._PageClassification("/", "homepage", va.LOW_TIER)
    return va._PageClassification("unmapped", "unmapped", va.LOW_TIER)


@contextmanager
def using_classifier(classifier=None, taxonomy_version: str | None = None):
    with mock.patch.object(va, "_classify", classifier or va._classify):
        if taxonomy_version is None:
            yield
        else:
            with mock.patch.object(va, "TAXONOMY_VERSION", taxonomy_version):
                yield


@dataclass
class GateRun:
    label: str
    results: dict  # lead_id -> VisitAnalysisResult


def run_gate(
    inputs,
    config: va.ThresholdConfig = va.ThresholdConfig(),
    classifier=None,
    label: str = "",
    taxonomy_version: str | None = None,
) -> GateRun:
    with using_classifier(classifier, taxonomy_version):
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
        "recall_should_pass": tp / max(tp + fn, 1),
        "false_pass_rate_browsers": fp / max(fp + tn, 1),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "converters_by_outcome": dict(lost.most_common()),
        "passed": sum(r.passes_threshold for r in run.results.values()),
    }


def taxonomy_coverage(run: GateRun, extras: dict) -> dict:
    """Share of simulated page views whose category matches the intended one, plus the misclassified URLs."""
    total = correct = 0
    wrong = Counter()
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
    fn = classifier or va._classify
    for group in URLS.values():
        for url, want in group:
            got = fn(url)
            if got.category != want:
                out.append((url, want, got.category, got.tier))
    return out
