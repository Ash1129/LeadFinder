import json

import httpx
import pytest

from leadfinder.research import (
    MAX_TOOL_CALLS,
    AgentToolCall,
    HttpxPracticeWebsite,
    ModelTurn,
    PracticeCandidate,
    PracticeLookupError,
    ResearchAgent,
    ResearchRequest,
    WebsiteEvidence,
    WebsiteFetchError,
    extract_ehr_mentions,
)


class FakeModel:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.messages = []

    def next_turn(self, messages):
        self.messages.append(messages)
        return self.turns.pop(0)


class FakeDirectory:
    def __init__(self, candidates=(), error=None):
        self.candidates = list(candidates)
        self.error = error
        self.requests = []

    def lookup(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.candidates


class FakeWebsite:
    def __init__(self, evidence=None, error=None):
        self.evidence = evidence or WebsiteEvidence("https://maplegrove.example/", ())
        self.error = error
        self.domains = []

    def inspect(self, domain):
        self.domains.append(domain)
        if self.error:
            raise self.error
        return self.evidence


def call(call_id, name, **arguments):
    return AgentToolCall(call_id, name, arguments)


def turn(*calls):
    return ModelTurn(calls, tuple({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                                  for c in calls), "tool_use")


def test_research_matches_by_domain_and_returns_sourced_website_ehr():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics", "https://www.maplegrove.example/about",
                                  "Ithaca, NY", "Pediatrics", 4)
    directory = FakeDirectory([candidate])
    website = FakeWebsite(WebsiteEvidence("https://maplegrove.example/", ("athenahealth",)))
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_domain_location")),
        turn(call("2", "fetch_practice_website", practice_id="npi-1")),
        turn(call("3", "complete_research", candidate_id="npi-1", rationale="Exact domain and matching location.")),
    )

    result = ResearchAgent(directory, website, model).research(
        ResearchRequest("Maple Grove Pediatrics", "maplegrove.example", "Ithaca, NY")
    )

    assert result.status == "matched" and result.practice == candidate
    assert result.match_confidence == 1.0 and result.tool_calls == 3
    assert result.ehr.value == "athenahealth" and result.ehr.source == "website:https://maplegrove.example/"
    assert result.specialty.value == "Pediatrics" and result.specialty.source == "npi"
    assert website.domains == ["maplegrove.example"]
    assert result.as_dict()["practice_id"] == "npi-1"


def test_ambiguous_names_are_not_auto_matched_even_if_the_model_picks_one():
    candidates = [
        PracticeCandidate("one", "Maple Grove Pediatrics", "one.example", "Ithaca, NY", "Pediatrics", 4),
        PracticeCandidate("two", "Maple Grove Pediatrics", "two.example", "Ithaca, NY", "Pediatrics", 3),
    ]
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_only")),
        turn(call("2", "fetch_practice_website", practice_id="one")),
        turn(call("3", "complete_research", candidate_id="one", rationale="The first candidate looks likely.")),
    )

    result = ResearchAgent(FakeDirectory(candidates), FakeWebsite(), model).research(
        ResearchRequest("Maple Grove Pediatrics", location_hint="Ithaca, NY")
    )

    assert result.status == "ambiguous"
    assert result.practice is None
    assert [candidate.practice_id for candidate in result.top_candidates] == ["one", "two"]
    assert any("threshold and margin" in warning for warning in result.warnings)


def test_unknown_ehr_stays_unknown_when_the_website_has_multiple_vendor_mentions():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics", "maplegrove.example")
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_only")),
        turn(call("2", "fetch_practice_website", practice_id="npi-1")),
        turn(call("3", "complete_research", candidate_id="npi-1", rationale="Directory name matches.")),
    )
    website = FakeWebsite(WebsiteEvidence("https://maplegrove.example/", ("athenahealth", "NextGen")))

    result = ResearchAgent(FakeDirectory([candidate]), website, model).research(
        ResearchRequest("Maple Grove Pediatrics")
    )

    assert result.status == "matched"
    assert result.ehr.value is None and result.ehr.source == "unknown"
    assert any("multiple EHRs" in warning for warning in result.warnings)


def test_model_cannot_fetch_a_website_for_a_candidate_not_returned_by_lookup():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics")
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_only")),
        turn(call("2", "fetch_practice_website", practice_id="not-returned"),
             call("3", "complete_research", candidate_id="npi-1", rationale="Exact name.")),
    )
    website = FakeWebsite()

    result = ResearchAgent(FakeDirectory([candidate]), website, model).research(
        ResearchRequest("Maple Grove Pediatrics")
    )

    assert result.status == "matched"
    assert website.domains == []
    assert any("unreturned practice" in warning for warning in result.warnings)


def test_model_must_try_the_directory_before_completing():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics")
    model = FakeModel(
        turn(call("1", "complete_research", candidate_id="", rationale="No match.")),
        turn(call("2", "practice_lookup", strategy="name_only"),
             call("3", "complete_research", candidate_id="npi-1", rationale="Exact name.")),
    )
    directory = FakeDirectory([candidate])

    result = ResearchAgent(directory, FakeWebsite(), model).research(ResearchRequest("Maple Grove Pediatrics"))

    assert result.status == "matched"
    assert len(directory.requests) == 1
    assert any("before practice lookup" in warning for warning in result.warnings)


def test_directory_failure_is_reported_as_failed_not_no_match():
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_domain_location")),
        turn(call("2", "complete_research", candidate_id="", rationale="Lookup failed.")),
    )
    result = ResearchAgent(FakeDirectory(error=PracticeLookupError("data source unavailable")), FakeWebsite(), model).research(
        ResearchRequest("Maple Grove Pediatrics")
    )

    assert result.status == "failed"
    assert any("directory lookup failed" in warning for warning in result.warnings)


def test_research_stops_at_the_absolute_tool_call_limit():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics")
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_only")),
        turn(call("2", "fetch_practice_website", practice_id="npi-1")),
    )

    result = ResearchAgent(FakeDirectory([candidate]), FakeWebsite(), model, max_tool_calls=2).research(
        ResearchRequest("Maple Grove Pediatrics")
    )

    assert MAX_TOOL_CALLS == 10
    assert result.status == "partial" and result.tool_calls == 2
    assert any("tool-call limit" in warning for warning in result.warnings)


def test_directory_and_website_failures_remain_distinct():
    candidate = PracticeCandidate("npi-1", "Maple Grove Pediatrics", "maplegrove.example")
    model = FakeModel(
        turn(call("1", "practice_lookup", strategy="name_only")),
        turn(call("2", "fetch_practice_website", practice_id="npi-1")),
        turn(call("3", "complete_research", candidate_id="npi-1", rationale="Exact name.")),
    )

    result = ResearchAgent(
        FakeDirectory([candidate]), FakeWebsite(error=WebsiteFetchError("page unavailable")), model
    ).research(ResearchRequest("Maple Grove Pediatrics"))

    assert result.status == "matched"
    assert result.ehr.value is None and result.ehr.source == "unknown"
    assert any("website lookup failed" in warning for warning in result.warnings)


def test_website_extraction_returns_only_cued_ehr_names_not_raw_html():
    html = (
        "<html><body>Our EHR is athenahealth."
        "<p>We like Epic as a TV show.</p>"
        "<script>We use NextGen for our EHR.</script>"
        "<p>Patient record details are not part of the output.</p></body></html>"
    )

    assert extract_ehr_mentions(html) == ("athenahealth",)


def test_http_website_fetch_is_bounded_and_only_returns_structured_evidence():
    html = b"<html><body>Our EHR is athenahealth. Patient Alice record 123.</body></html>"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=html)
    ))
    website = HttpxPracticeWebsite(client=client, resolver=lambda host: ("93.184.216.34",))

    evidence = website.inspect("maplegrove.example")

    assert evidence == WebsiteEvidence("https://maplegrove.example/", ("athenahealth",))
    assert "Alice" not in json.dumps(evidence.__dict__)
    client.close()


def test_http_website_fetch_rejects_non_public_dns_before_request():
    website = HttpxPracticeWebsite(resolver=lambda host: ("127.0.0.1",))

    with pytest.raises(WebsiteFetchError, match="public IP"):
        website.inspect("maplegrove.example")


@pytest.mark.parametrize("maximum", [0, MAX_TOOL_CALLS + 1])
def test_research_tool_call_limit_must_stay_within_contract(maximum):
    with pytest.raises(ValueError, match="max_tool_calls"):
        ResearchAgent(FakeDirectory(), FakeWebsite(), FakeModel(), max_tool_calls=maximum)
