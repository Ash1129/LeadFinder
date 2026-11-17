import json

import pytest

from leadfinder.visit_analysis import (
    PageView,
    ThresholdConfig,
    VisitAnalysisInput,
    analyze_visits,
)

NOW = 1_800_000_000.0


def view(view_id, page, at=NOW, seconds=60):
    return PageView(view_id, page, at, seconds)


def request(views=(), **overrides):
    values = {
        "lead_id": "lead-1",
        "practice_name_raw": "Maple Grove Pediatrics",
        "domain_or_org_hint": "maplegrove.example",
        "location_hint": "Ithaca, NY",
        "company_identified": True,
        "identification_confidence": 0.9,
        "page_views": tuple(views),
        "window_end": NOW,
        "patient_data_stripped": True,
    }
    return VisitAnalysisInput(**(values | overrides))


def test_high_tier_page_passes_and_queries_are_not_in_feature_output():
    result = analyze_visits(request([
        view("p1", "https://site.example/pricing?patient=not-for-model"),
        view("p2", "/integrations/athenahealth"),
    ]))

    assert result.passes_threshold
    assert result.threshold_reasons == ("high_tier_page", "criteria_breadth_2")
    assert result.max_intent_tier == result.intent_level == "high"
    assert result.criteria_breadth == 2
    assert result.ehr_of_interest == "athenahealth"
    assert "not-for-model" not in json.dumps(result.as_dict())
    assert result.pages_viewed[0].page == "pricing"


def test_relative_page_paths_are_classified():
    result = analyze_visits(request([view("p1", "pricing?campaign=spring")]))

    assert result.page_category_counts["pricing"] == 1
    assert result.max_intent_tier == "high"


def test_distinct_criteria_pass_but_repeated_page_views_do_not_inflate_breadth():
    pages = [view(f"p{i}", "/pricing") for i in range(3)]
    result = analyze_visits(request(pages))

    assert result.criteria_breadth == 1
    assert result.threshold_reasons == ("high_tier_page", "repeat_interest")
    assert result.demo_without_depth is False


def test_medium_tier_repeat_sessions_passes_and_sessions_respect_timeout():
    result = analyze_visits(request([
        view("p1", "/security/hipaa", NOW - 3601),
        view("p2", "/billing", NOW),
    ]))

    assert result.session_count == 2
    assert result.threshold_reasons == ("repeat_interest",)
    assert result.max_intent_tier == "medium"


def test_exact_session_timeout_gap_stays_in_the_same_session():
    result = analyze_visits(request([
        view("p1", "/billing", NOW - 1800),
        view("p2", "/billing", NOW),
    ]))

    assert result.session_count == 1
    assert result.pages_per_session == 2.0


def test_three_medium_pages_in_one_session_pass_the_depth_rule():
    result = analyze_visits(request([
        view("p1", "/billing", NOW - 120),
        view("p2", "/billing", NOW - 60),
        view("p3", "/billing", NOW),
    ]))

    assert result.session_count == 1
    assert result.pages_per_session == 3.0
    assert result.threshold_reasons == ("repeat_interest",)


def test_low_tier_only_visits_fail_with_an_explainable_reason():
    result = analyze_visits(request([
        view("p1", "/blog/no-show-rates"),
        view("p2", "/careers"),
    ]))

    assert not result.passes_threshold
    assert result.threshold_reasons == ()
    assert result.threshold_fail_reason == "only_low_tier_pages"
    assert result.page_category_counts["blog"] == 1
    assert result.page_category_counts["careers"] == 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"company_identified": False}, "unidentified"),
        ({"identification_confidence": 0.49}, "low_identification"),
        ({"patient_data_stripped": False}, "patient_data_not_scrubbed"),
    ],
)
def test_hard_gate_requirements_block_even_high_intent_visits(overrides, reason):
    result = analyze_visits(request([view("p1", "/demo")], **overrides))

    assert not result.passes_threshold
    assert result.threshold_fail_reason == reason


def test_unscrubbed_visit_data_is_not_used_for_features():
    result = analyze_visits(request(
        [view("patient-name", "/pricing?patient=private")],
        patient_data_stripped=False,
    ))

    assert result.visit_ids == ()
    assert result.pages_viewed == ()
    assert result.page_category_counts["pricing"] == 0
    assert "private" not in json.dumps(result.as_dict())


def test_session_and_timing_metrics_use_only_sorted_views_in_the_lookback():
    result = analyze_visits(request([
        view("newer", "/pricing", NOW - 86400, 30),
        view("old", "/demo", NOW - 40 * 86400),
        view("older", "/blog", NOW - 2 * 86400, 90),
        view("future", "/demo", NOW + 100),
    ]))

    assert result.visit_ids == ("older", "newer")
    assert result.session_count == 2
    assert result.days_span == 1 and result.recency_days == 1
    assert result.total_time_on_site == 120
    assert result.avg_time_per_page == 60


def test_lookback_window_includes_both_boundary_timestamps():
    result = analyze_visits(request([
        view("start", "/demo", NOW - 30 * 86400),
        view("end", "/demo", NOW),
    ]))

    assert result.visit_ids == ("start", "end")
    assert result.days_span == 30


def test_specialty_case_study_billing_and_role_hints_are_deterministic():
    result = analyze_visits(request([
        view("p1", "/case-studies/family-medicine"),
        view("p2", "/billing"),
        view("p3", "/integrations/athenahealth"),
    ]))

    assert result.specialty_interest_hint == "Family Medicine"
    assert result.product_line_interest == ("billing", "integrations")
    assert result.visitor_role_guess == "administrator"
    assert result.field_sources["visitor_role_guess"] == "inferred"
    assert result.field_sources["ehr_of_interest"] == "visit_log"


def test_empty_window_has_no_session_and_cannot_pass():
    result = analyze_visits(request([]))

    assert result.session_count == 0
    assert result.recency_days is None
    assert result.threshold_fail_reason == "single_shallow_session"
    assert "no_visits_in_lookback_window" in result.warnings


def test_schema_and_threshold_versions_are_exposed_as_json():
    result = analyze_visits(
        request([view("p1", "/demo")]),
        ThresholdConfig(version="v2"),
    )
    serialized = result.as_dict()

    assert serialized["schema_version"] == "visit-v2"
    assert serialized["taxonomy_version"] == "v2"
    assert serialized["threshold_config_version"] == "v2"
    assert serialized["pages_viewed"][0]["session_id"] == "lead-1:s1"
    assert serialized["threshold_reasons"] == ["high_tier_page"]


def test_threshold_and_visit_inputs_reject_invalid_values():
    with pytest.raises(ValueError, match="identification_confidence"):
        request(identification_confidence=1.1)
    with pytest.raises(ValueError, match="seconds_on_page"):
        view("p1", "/pricing", seconds=-1)
    with pytest.raises(ValueError, match="repeat_min_pages_per_session"):
        ThresholdConfig(repeat_min_pages_per_session=float("nan"))
