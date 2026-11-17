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
    assert 0.0 <= metrics["recall_real_leads"] <= 1.0
    assert sum(metrics["confusion"].values()) <= len(inputs)


def test_proposed_taxonomy_keeps_content_pages_low_tier_and_catches_demo_variants():
    assert gate_eval.proposed_classify("/blog/pricing-your-services").tier == va.LOW_TIER
    assert gate_eval.proposed_classify("/careers/billing-specialist").category == "careers"
    assert gate_eval.proposed_classify("/request-a-demo").category == "demo_or_trial"
    assert gate_eval.proposed_classify("/customers/pediatrics").category == "specialty_case_study"
    # Unchanged where v1 was already right.
    for url in ("/pricing", "/integrations/athenahealth", "/security", "/blog/no-show-rates", "/"):
        assert gate_eval.proposed_classify(url).category == va._classify(url).category


def test_classifier_patch_is_scoped():
    with gate_eval.using_classifier(gate_eval.proposed_classify):
        assert va._classify("/request-a-demo").category == "demo_or_trial"
    assert va._classify("/request-a-demo").category == "unmapped"


def test_match_decision_follows_research_thresholds():
    from leadfinder.research import CandidateMatch

    def c(pid, conf):
        return CandidateMatch(pid, "n", None, None, None, None, "npi", conf, "r")

    assert match_eval.decide([c("A", 0.95), c("B", 0.80)]) == ("matched", "A")
    assert match_eval.decide([c("A", 0.95), c("B", 0.90)])[0] == "ambiguous"
    assert match_eval.decide([c("A", 0.30)])[0] == "not_found"
    assert match_eval.decide([]) == ("not_found", None)
