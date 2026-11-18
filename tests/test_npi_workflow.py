"""The NPI practice table connected to the orchestration: real registry practices go through research, scoring,
drafting and the approval pause. No API key and no network: research runs on the rules model over the committed
table, the writer is the template writer, and the website fetcher is never reached (registry rows have no domain)."""
import pytest

from leadfinder import npi
from leadfinder import orchestration as orch
from leadfinder.research import (
    MATCH_THRESHOLD,
    AnthropicResearchModel,
    PracticeCandidate,
    ResearchAgent,
    ResearchRequest,
    RulesResearchModel,
    WebsiteEvidence,
    make_research_model,
)
from leadfinder.visit_analysis import PageView, VisitAnalysisInput
from leadfinder.writer import Offline

NOW = 1_800_000_000.0


class NoWebsite:
    def inspect(self, domain):
        raise AssertionError("registry candidates have no domain, so no website is fetched")


@pytest.fixture(scope="module")
def table():
    return npi.load_table()


@pytest.fixture(scope="module")
def target(table):
    return next(r for r in table if r.city == "Ithaca")


def agent(directory=None, website=None):
    return ResearchAgent(directory or npi.TablePracticeDirectory.from_file(), website or NoWebsite(),
                         RulesResearchModel())


def visit(name, location, views=None, **overrides):
    views = views or (PageView("v1", "/pricing", NOW - 3600, 80), PageView("v2", "/demo", NOW - 1800, 60))
    values = dict(lead_id="lead-npi", practice_name_raw=name, domain_or_org_hint=None, location_hint=location,
                  company_identified=True, identification_confidence=0.9, page_views=tuple(views),
                  window_end=NOW, patient_data_stripped=True)
    return VisitAnalysisInput(**(values | overrides))


def run(visit_input, research_agent=None):
    deps = orch.Deps(research_agent=research_agent or agent(), writer=Offline(), fallback_writer=Offline())
    return orch.run_lead({"lead_id": "lead-npi", "visit_input": visit_input}, deps)


# ---------------------------------------------------------------- the rules model

def test_rules_model_matches_a_real_practice_from_a_casual_hint(target):
    hint = target.name.lower().replace(",", "")
    result = agent().research(ResearchRequest(hint, None, target.location)).as_dict()
    assert result["status"] == "matched" and result["practice_id"] == target.practice_id
    assert result["tool_calls"] == 2                       # one lookup, one completion
    assert result["facts"]["ehr"] == {"value": None, "source": "unknown"}


def test_rules_model_falls_back_to_name_only_when_the_location_finds_nothing(target):
    result = agent().research(ResearchRequest(target.name, None, "Nowhere, ZZ")).as_dict()
    assert result["tool_calls"] == 3                       # location lookup, name-only lookup, completion
    assert target.practice_id in [c["practice_id"] for c in result["top_candidates"]]


def test_rules_model_selects_nothing_for_an_unknown_practice():
    result = agent().research(ResearchRequest("Zebra Quantum Wellness Collective", None, "Ithaca, NY")).as_dict()
    assert result["status"] in ("not_found", "ambiguous") and result["practice_id"] is None


def test_rules_model_inspects_a_website_before_completing():
    class OneCandidate:
        def lookup(self, request):
            return [PracticeCandidate("p1", "Maple Grove Pediatrics", "maplegrove.example", "Ithaca, NY",
                                      "Pediatrics", 4)]

    class Website:
        domains = []

        def inspect(self, domain):
            self.domains.append(domain)
            return WebsiteEvidence(f"https://{domain}/", ("athenahealth",))

    website = Website()
    result = agent(OneCandidate(), website).research(
        ResearchRequest("Maple Grove Pediatrics", None, "Ithaca, NY")).as_dict()
    assert website.domains == ["maplegrove.example"]
    assert result["status"] == "matched"
    assert result["facts"]["ehr"]["value"] == "athenahealth"


def test_rules_model_leaves_ties_to_the_agents_safeguards():
    twins = [npi.PracticeRecord(npi=n, name="Lakeside Family Medicine", specialty="Family Medicine",
                                taxonomy_code="207Q00000X", taxonomy_desc="Family Medicine", street=s, city="Ithaca",
                                state="NY", zip5="14850", phone=None, providers=2,
                                providers_source="npi:clinicians-at-address", contact_name=None, contact_title=None,
                                enumerated=None, last_updated=None)
             for n, s in (("1000000004", "1 Main St"), ("1000000012", "9 Oak Ave"))]
    result = agent(npi.TablePracticeDirectory(twins)).research(
        ResearchRequest("Lakeside Family Medicine", None, "Ithaca, NY")).as_dict()
    assert result["status"] == "ambiguous" and result["practice_id"] is None
    assert result["match_confidence"] >= MATCH_THRESHOLD   # confident, but tied: the margin rule refuses


# ---------------------------------------------------------------- the whole pipeline on real practices

def test_a_real_practice_goes_through_the_pipeline_to_the_approval_pause(target):
    state = run(visit(target.name.lower().replace(",", ""), target.location))
    assert state["outcome"] == "awaiting_approval"
    assert state["research"]["practice_id"] == target.practice_id
    assert [t["step"] for t in state["trace"]] == ["ingest", "visit_features", "research", "score", "draft",
                                                    "check", "approval"]
    assert state["outreach"]["problems"] == []
    assert target.name in state["outreach"]["subject"] + state["outreach"]["body"]   # registry name, not the hint


def test_an_unlisted_practice_never_reaches_scoring_or_drafting():
    state = run(visit("Zebra Quantum Wellness Collective", "Ithaca, NY"))
    assert state["outcome"] in ("review_queue", "partial")
    assert "scoring" not in state and "outreach" not in state


def test_default_deps_use_the_real_table_and_rules_without_a_credential(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LEADFINDER_RESEARCH", "LEADFINDER_WRITER"):
        monkeypatch.delenv(name, raising=False)
    deps = orch.default_deps()
    assert isinstance(deps.research_agent.directory, npi.TablePracticeDirectory)
    assert isinstance(deps.research_agent.model, RulesResearchModel)
    assert isinstance(deps.writer, Offline) and isinstance(deps.fallback_writer, Offline)
    assert deps.sender is None


def test_research_agent_picks_claude_when_a_credential_is_set(monkeypatch):
    monkeypatch.delenv("LEADFINDER_RESEARCH", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    assert isinstance(orch.build_research_agent().model, AnthropicResearchModel)   # built only; no request is made
    with pytest.raises(ValueError):
        make_research_model("oracle")


def test_demo_runs_offline():
    state = orch.demo()
    assert state["outcome"] == "awaiting_approval" and state["research"]["practice_id"].startswith("npi:")


def test_langgraph_graph_pauses_then_finishes_on_real_practices(target, caplog):
    pytest.importorskip("langgraph")
    graph = orch.build_graph(orch.Deps(research_agent=agent(), writer=Offline(), fallback_writer=Offline()))
    config = {"configurable": {"thread_id": "lead-npi"}}
    graph.invoke({"lead_id": "lead-npi", "visit_input": visit(target.name, target.location), "warnings": [],
                  "trace": []}, config)
    paused = graph.get_state(config)
    assert paused.next == ("approval",)
    assert paused.values["research"]["practice_id"] == target.practice_id
    graph.update_state(config, {"approval": {"decision": "approve"}})
    assert graph.invoke(None, config)["outcome"] == "sent"
    # Restoring the paused state uses only registered types, so LangGraph logs no deprecation warning.
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]
