"""The pipeline with the real NPI practice table behind the research agent (committed snapshot; no network).
The only fakes are the research model (scripted tool calls instead of Claude) and the website fetcher."""
import pytest

from leadfinder import orchestration as orch
from leadfinder.npi import TablePracticeDirectory
from leadfinder.research import (AgentToolCall, ModelTurn, PracticeLookupRequest, ResearchRequest, WebsiteEvidence,
                                  rank_candidates)
from leadfinder.visit_analysis import PageView, VisitAnalysisInput
from leadfinder.writer import Offline

NOW = 1_800_000_000.0


class ScriptedModel:
    def __init__(self, *turns):
        self.turns = list(turns)

    def next_turn(self, messages):
        if not self.turns:
            return ModelTurn((), (), "end_turn")
        return self.turns.pop(0)


class NoWebsite:
    def inspect(self, domain):
        return WebsiteEvidence(f"https://{domain}/", ())


def turn(call_id, name, **arguments):
    call = AgentToolCall(call_id, name, arguments)
    return ModelTurn((call,), ({"type": "tool_use", "id": call_id, "name": name, "input": arguments},), "tool_use")


def clear_match_record():
    """A real practice the research safeguards accept on name and city alone: it ranks well above every other
    candidate the table returns (some real practices have near-identical siblings and are rightly 'ambiguous')."""
    directory = TablePracticeDirectory.from_file()
    for record in directory.records:
        if not record.specialty:
            continue
        found = directory.lookup(PracticeLookupRequest(record.name, None, record.location, "name_domain_location"))
        ranked = rank_candidates(ResearchRequest(record.name, None, record.location), found)
        if ranked and ranked[0].practice_id == record.practice_id and ranked[0].confidence >= 0.9 and (
                len(ranked) == 1 or ranked[0].confidence - ranked[1].confidence >= 0.15):
            return record
    raise AssertionError("no practice in the table is a clear match")


def visit(name, location):
    return VisitAnalysisInput(
        lead_id="lead-npi", practice_name_raw=name, domain_or_org_hint=None, location_hint=location,
        company_identified=True, identification_confidence=0.9,
        page_views=(PageView("p1", "/pricing", NOW, 60), PageView("p2", "/integrations/athenahealth", NOW, 60)),
        window_end=NOW, patient_data_stripped=True)


def deps(model):
    agent = orch.build_research_agent(website=NoWebsite(), model=model)
    return orch.Deps(research_agent=agent, writer=Offline())


def test_default_directory_is_the_npi_table():
    agent = orch.build_research_agent(website=NoWebsite(), model=ScriptedModel())
    assert isinstance(agent.directory, TablePracticeDirectory)


def test_real_practice_goes_all_the_way_to_approval():
    record = clear_match_record()
    model = ScriptedModel(
        turn("1", "practice_lookup", strategy="name_domain_location"),
        turn("2", "complete_research", candidate_id=record.practice_id, rationale="Exact name in the same city."))
    state = orch.run_lead({"lead_id": "lead-npi", "visit_input": visit(record.name, record.location)},
                          deps(model))
    assert state["research"]["status"] == "matched"
    assert state["research"]["practice_id"] == record.practice_id
    assert state["research"]["facts"]["ehr"]["value"] is None       # the registry has no EHR; it is not guessed
    assert state["outcome"] == "awaiting_approval"
    assert state["outreach"]["problems"] == []


def test_practice_not_in_the_table_goes_to_the_review_queue():
    model = ScriptedModel(turn("1", "practice_lookup", strategy="name_only"))
    state = orch.run_lead({"lead_id": "lead-npi",
                           "visit_input": visit("Zzyzx Qwerty Holographic Clinic", "Ithaca, NY")}, deps(model))
    assert state["outcome"] == "review_queue"
    assert "scoring" not in state
