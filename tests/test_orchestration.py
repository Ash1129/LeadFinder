"""Orchestration tests. These use fake research and writer objects, so they need no API key, no network and
no LangGraph install. Run with: pytest tests/test_orchestration.py"""
import json

import pytest

from leadfinder import orchestration as orch
from leadfinder.outreach import FOOTER
from leadfinder.research import CandidateMatch, ResearchResult, SourcedFact
from leadfinder.visit_analysis import PageView, VisitAnalysisInput
from leadfinder.writer import Offline, WriterError

NOW = 1_800_000_000.0


def visit_input(views=None, **overrides):
    views = views if views is not None else (
        PageView("p1", "https://site.example/pricing", NOW, 60),
        PageView("p2", "/integrations/athenahealth", NOW, 60),
    )
    values = dict(lead_id="lead-1", practice_name_raw="Maple Grove Pediatrics",
                  domain_or_org_hint="maplegrove.example", location_hint="Ithaca, NY",
                  company_identified=True, identification_confidence=0.9, page_views=tuple(views),
                  window_end=NOW, patient_data_stripped=True)
    return VisitAnalysisInput(**(values | overrides))


def fact(value, source="npi"):
    return SourcedFact(value, source)


def research_result(status="matched", **overrides):
    values = dict(status=status, practice=None, match_confidence=0.9, top_candidates=(),
                  specialty=fact("Pediatrics"), provider_count=fact(5), location=fact("Ithaca, NY"),
                  ehr=fact("athenahealth", "website"), rationale="test", warnings=(), tool_calls=3)
    return ResearchResult(**(values | overrides))


class FakeResearch:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def research(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeWriter:
    """Returns the given drafts in order; an Exception in the list is raised."""
    is_model = True

    def __init__(self, *drafts, name="fake"):
        self.drafts, self.name, self.calls = list(drafts), name, 0

    def email(self, practice, reasons, visits):
        item = self.drafts[min(self.calls, len(self.drafts) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


GOOD = ("Scheduling for Maple Grove Pediatrics",
        "Hi there, we help independent pediatric practices keep one calendar. Worth a short call this week?\n\n--\n" + FOOTER)
BAD = ("Following up on your patient records", "We saw patient chart access on your site. Let us talk about it today.")


def deps(research=None, writer=None, **kw):
    return orch.Deps(research_agent=research or FakeResearch(research_result()), writer=writer or Offline(), **kw)


def run(state_input=None, **kw):
    return orch.run_lead({"lead_id": "lead-1", "visit_input": state_input or visit_input()}, deps(**kw))


# ---- happy path and approval pause

def test_good_lead_pauses_for_approval_and_never_sends():
    sent = []
    state = run(writer=FakeWriter(GOOD), sender=sent.append)
    assert state["outcome"] == "awaiting_approval"
    assert state["scoring"]["total"] > 0
    assert sent == []
    assert [t["step"] for t in state["trace"]][-1] == "approval"


def test_approve_then_send_calls_sender_once():
    sent = []
    d = deps(writer=FakeWriter(GOOD), sender=sent.append)
    paused = orch.run_lead({"lead_id": "lead-1", "visit_input": visit_input()}, d)
    paused["approval"] = {"decision": "approve"}
    done = orch.run_lead(paused, d, start="approval")
    assert done["outcome"] == "sent"
    assert len(sent) == 1


def test_reject_ends_without_sending():
    d = deps(writer=FakeWriter(GOOD))
    paused = orch.run_lead({"lead_id": "lead-1", "visit_input": visit_input()}, d)
    paused["approval"] = {"decision": "reject"}
    assert orch.run_lead(paused, d, start="approval")["outcome"] == "rejected"


def test_sdr_edit_is_rechecked_before_it_can_go_out():
    d = deps(writer=FakeWriter(GOOD))
    paused = orch.run_lead({"lead_id": "lead-1", "visit_input": visit_input()}, d)
    paused["approval"] = {"decision": "edit", "subject": BAD[0], "body": BAD[1]}
    after = orch.run_lead(paused, d, start="approval")
    assert after["outcome"] != "sent"


# ---- guardrail: fake patient strings

def test_unstripped_patient_data_blocks_everything():
    research = FakeResearch(research_result())
    state = run(visit_input(patient_data_stripped=False), research=research)
    assert state["outcome"] == "blocked"
    assert research.calls == 0
    assert "visit" not in state


def test_fake_patient_strings_never_reach_downstream_state():
    views = (PageView("p1", "https://site.example/pricing?patient_name=Jane+Doe&dob=1980-01-01", NOW, 60),
             PageView("p2", "https://site.example/patient-portal/jane-doe-diagnosis", NOW, 30),
             PageView("p3", "/integrations/athenahealth", NOW, 60))
    state = run(visit_input(views), writer=FakeWriter(GOOD))
    blob = json.dumps({k: v for k, v in state.items() if k != "visit_input"}, default=str)
    for secret in ("Jane", "1980-01-01", "patient_name", "diagnosis"):
        assert secret not in blob
    assert any("patient" in w for w in state["warnings"])
    assert all("?" not in v.page_url for v in state["visit_input"].page_views)


# ---- failure routing

def test_low_intent_lead_stops_at_gate_with_reason_and_no_research():
    research = FakeResearch(research_result())
    state = run(visit_input((PageView("p1", "/blog/hello", NOW, 5),)), research=research)
    assert state["outcome"] == "stopped"
    assert state["reason"]
    assert research.calls == 0


def test_ambiguous_research_goes_partial_and_skips_scoring():
    state = run(research=FakeResearch(research_result("ambiguous")))
    assert state["outcome"] == "partial"
    assert "scoring" not in state


@pytest.mark.parametrize("status", ["not_found", "failed"])
def test_no_match_goes_to_review_queue(status):
    state = run(research=FakeResearch(research_result(status)))
    assert state["outcome"] == "review_queue"
    assert "outreach" not in state


def test_research_tool_crash_goes_to_review_queue_not_a_crash():
    state = run(research=FakeResearch(error=RuntimeError("directory down")))
    assert state["outcome"] == "review_queue"
    assert any("RuntimeError" in w for w in state["warnings"])


def test_unknown_ehr_is_not_guessed():
    state = run(research=FakeResearch(research_result(ehr=fact(None, "unknown"))), writer=FakeWriter(GOOD))
    assert state["outcome"] == "awaiting_approval"
    assert state["research"]["facts"]["ehr"]["value"] is None


# ---- draft / check loop

def test_bad_draft_is_revised_once_then_passes():
    writer = FakeWriter(BAD, GOOD)
    state = run(writer=writer)
    assert state["outcome"] == "awaiting_approval"
    assert writer.calls == 2
    assert state["outreach"]["revisions"] == 1


def test_draft_that_keeps_failing_stops_after_two_revisions():
    writer = FakeWriter(BAD)
    state = run(writer=writer)
    assert state["outcome"] == "review_queue"
    assert writer.calls == orch.MAX_REVISIONS + 1


def test_writer_failure_retries_then_falls_back_to_template():
    state = run(writer=FakeWriter(WriterError("down")), fallback_writer=Offline())
    assert state["outcome"] == "awaiting_approval"
    assert state["outreach"]["writer"] == "template"


def test_writer_failure_without_fallback_goes_to_review_queue():
    state = run(writer=FakeWriter(WriterError("down")))
    assert state["outcome"] == "review_queue"


# ---- routers are pure and cover every outcome

def test_every_run_ends_in_a_known_outcome():
    cases = [run(), run(research=FakeResearch(research_result("ambiguous"))),
             run(visit_input(patient_data_stripped=False))]
    assert all(c["outcome"] in orch.OUTCOMES for c in cases)
