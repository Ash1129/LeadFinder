"""Tests for the evaluation package: simulator properties, label separation, metrics and the taxonomy proposal."""
from dataclasses import fields

from leadfinder import visit_analysis as va
from leadfinder.evaluation import gate_eval, match_eval
from leadfinder.evaluation.personas import GroundTruth, simulate


def test_same_seed_gives_same_simulation():
    a = simulate(150, seed=3)
    b = simulate(150, seed=3)
    assert a[1] == b[1] and a[2] == b[2]


def test_ground_truth_never_appears_in_pipeline_inputs():
    _, inputs, truth, _ = simulate(100, seed=1)
    input_fields = {f.name for f in fields(va.VisitAnalysisInput)}
    assert not input_fields & {"persona", "converted", "p_convert", "should_pass_gate", "true_practice_id"}
    assert all(isinstance(t, GroundTruth) for t in truth)
    assert all(i.patient_data_stripped for i in inputs)


def test_simulated_inputs_run_through_the_real_gate():
    _, inputs, truth, _ = simulate(300, seed=2)
    metrics = gate_eval.gate_metrics(gate_eval.run_gate(inputs), truth)
    assert 0.0 <= metrics["recall_should_pass"] <= 1.0
    assert sum(metrics["confusion"].values()) <= len(inputs)


def test_baseline_and_current_runs_record_their_taxonomy_versions():
    _, inputs, _, _ = simulate(20, seed=4)
    baseline = gate_eval.run_gate(
        inputs,
        classifier=gate_eval.baseline_v1_classify,
        taxonomy_version="v1",
    )
    current = gate_eval.run_gate(inputs)

    assert {result.taxonomy_version for result in baseline.results.values()} == {"v1"}
    assert {result.taxonomy_version for result in current.results.values()} == {"v2"}


def test_current_taxonomy_keeps_content_pages_low_tier_and_catches_demo_variants():
    expected = {
        "/blog/pricing-your-services": ("blog", va.LOW_TIER),
        "/blog/ehr-integration-tips": ("blog", va.LOW_TIER),
        "/blog/hipaa-checklist": ("blog", va.LOW_TIER),
        "/webinars/no-show-rates": ("blog", va.LOW_TIER),
        "/careers/billing-specialist": ("careers", va.LOW_TIER),
        "/request-a-demo": ("demo_or_trial", va.HIGH_TIER),
        "/schedule-demo": ("demo_or_trial", va.HIGH_TIER),
        "/book-a-demo": ("demo_or_trial", va.HIGH_TIER),
        "/book-demo": ("demo_or_trial", va.HIGH_TIER),
        "/get-a-demo": ("demo_or_trial", va.HIGH_TIER),
        "/customers/pediatrics": ("specialty_case_study", va.MEDIUM_TIER),
    }
    for url, (category, tier) in expected.items():
        result = va._classify(url)
        assert (result.category, result.tier) == (category, tier)

    assert gate_eval.baseline_v1_classify("/blog/pricing-your-services").tier == va.HIGH_TIER
    assert gate_eval.baseline_v1_classify("/schedule-demo").category == "unmapped"
    # Unchanged where v1 was already right.
    for url in ("/pricing", "/integrations/athenahealth", "/security", "/blog/no-show-rates", "/"):
        assert gate_eval.baseline_v1_classify(url).category == va._classify(url).category


def test_classifier_patch_is_scoped():
    with gate_eval.using_classifier(gate_eval.baseline_v1_classify):
        assert va._classify("/request-a-demo").category == "unmapped"
    assert va._classify("/request-a-demo").category == "demo_or_trial"


def test_match_decision_follows_research_thresholds():
    from leadfinder.research import CandidateMatch

    def c(pid, conf):
        return CandidateMatch(pid, "n", None, None, None, None, "npi", conf, "r")

    assert match_eval.decide([c("A", 0.95), c("B", 0.80)]) == ("matched", "A")
    assert match_eval.decide([c("A", 0.95), c("B", 0.90)])[0] == "ambiguous"
    assert match_eval.decide([c("A", 0.30)])[0] == "not_found"
    assert match_eval.decide([]) == ("not_found", None)
