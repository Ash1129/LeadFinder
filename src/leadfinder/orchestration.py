"""LeadFinder orchestration: how one lead moves through the pipeline, step by step.

The order is fixed (no LLM decides what runs next). Each step is a plain function that reads the shared lead
state and returns only the fields it changes. Small router functions decide the next step. LangGraph (see
`build_graph`) wires these same functions together and adds checkpointing and the approval pause; the
functions themselves have no LangGraph dependency, so they can be tested directly.

Pipeline:
  ingest -> visit_features -> [gate] -> research -> [match?] -> score -> draft <-> check (max 2 revisions)
         -> approval (pause) -> send

Every lead ends in exactly one outcome:
  awaiting_approval  draft is ready and waiting for the SDR (the graph pauses here, nothing is sent)
  sent               the SDR approved and the send step ran
  rejected           the SDR rejected the lead
  partial            research found several possible practices; top candidates saved for a person to pick
  review_queue       something needs a person (identity unclear, research tool down, draft kept failing)
  stopped            the lead did not pass the gate; reason is logged
  blocked            the input was unsafe (patient data not removed); nothing downstream ran
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import Annotated, Any, Callable, Optional, TypedDict
from urllib.parse import urlsplit, urlunsplit

from . import outreach
from .research import ResearchAgent, ResearchRequest
from .scoring import score_practice
from .visit_analysis import PageView, VisitAnalysisInput, analyze_visits
from .writer import Offline, Writer, WriterError, make_writer

MAX_REVISIONS = 2

# The visit gate labels pages by category ("pricing"); the scorer expects URL-style keys ("/pricing").
# This mirrors VISIT_PAGES in seed.py (the app's existing handoff). Once this branch is merged, import that
# instead of keeping a second copy.
SCORER_PAGES = {"demo_or_trial": "/demo", "pricing": "/pricing", "ehr_integration": "/integrations",
                "usability": "/features", "blog": "/blog"}

OUTCOMES = ("awaiting_approval", "sent", "rejected", "partial", "review_queue", "stopped", "blocked")


class LeadState(TypedDict, total=False):
    """The shared lead object. `warnings` and `trace` append; every other field is overwritten by its step."""

    lead_id: str
    visit_input: VisitAnalysisInput
    visit: dict          # output of visit_features
    research: dict       # output of research
    scoring: dict
    outreach: dict       # subject, body, problems, revisions, writer
    approval: Optional[dict]   # {"decision": "approve"|"edit"|"send_back"|"reject", "subject"?, "body"?}
    outcome: str
    reason: str
    warnings: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


@dataclass
class Deps:
    """Things the steps need from outside, passed in so tests can use fakes."""

    research_agent: ResearchAgent
    writer: Writer
    fallback_writer: Optional[Writer] = None      # used if the main writer fails twice (template writer)
    sender: Optional[Callable[[dict], None]] = None  # real email send; None means "approved, not sent"


def build_research_agent(directory=None, website=None, model=None) -> ResearchAgent:
    """The research agent the pipeline uses. By default it looks practices up in the committed NPI table
    (`npi.TablePracticeDirectory`), checks homepages over HTTPS, and lets research.make_research_model() choose who
    drives the bounded tool loop: Claude when a credential is set, fixed rules otherwise. Any part can be replaced,
    which is how the tests run offline."""
    from .npi import TablePracticeDirectory
    from .research import HttpxPracticeWebsite, make_research_model

    return ResearchAgent(directory or TablePracticeDirectory.from_file(),
                         website or HttpxPracticeWebsite(),
                         model or make_research_model())


def default_deps(research_agent: Optional[ResearchAgent] = None, writer: Optional[Writer] = None,
                 sender: Optional[Callable[[dict], None]] = None) -> Deps:
    """Deps with the real parts: build_research_agent() over the NPI table, the app's writer choice (make_writer)
    with the template writer as fallback, and no sender, so an approval marks a lead ready but sends nothing."""
    return Deps(research_agent=research_agent or build_research_agent(), writer=writer or make_writer(),
                fallback_writer=Offline(), sender=sender)


def _trace(step: str, **info: Any) -> list:
    return [{"step": step, **info}]


# ---------------------------------------------------------------- steps

def ingest(state: LeadState, deps: Deps) -> dict:
    """Compliance guardrail. Fail closed if patient data was not stripped; drop page views whose URL looks
    like it carries patient information; remove query strings and fragments from everything kept."""
    data = state["visit_input"]
    if not data.patient_data_stripped:
        return {"outcome": "blocked", "reason": "patient data was not stripped upstream",
                "trace": _trace("ingest", result="blocked")}
    kept, dropped = [], 0
    for view in data.page_views:
        parts = urlsplit(view.page_url)
        raw = f"{parts.path} {parts.query}"
        if outreach.check_words(raw):
            dropped += 1          # count only; the text itself is never logged
            continue
        clean = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        kept.append(PageView(view.view_id, clean, view.viewed_at, view.seconds_on_page))
    update: dict = {"visit_input": replace(data, page_views=tuple(kept)),
                    "trace": _trace("ingest", result="ok", dropped_views=dropped)}
    if dropped:
        update["warnings"] = [f"ingest dropped {dropped} page view(s) that looked like patient data"]
    return update


def visit_features(state: LeadState, deps: Deps) -> dict:
    result = analyze_visits(state["visit_input"])
    update: dict = {"visit": result.as_dict(),
                    "trace": _trace("visit_features", decision=result.routing_decision)}
    if result.routing_decision == "stop":
        update.update(outcome="stopped", reason=result.threshold_fail_reason or "did not pass the lead gate")
    elif result.routing_decision == "human_review":
        update.update(outcome="review_queue", reason="identity needs a person before research")
    return update


def research(state: LeadState, deps: Deps) -> dict:
    visit = state["visit"]
    name = visit.get("practice_name_raw")
    if not name:
        return {"outcome": "review_queue", "reason": "no practice name to research",
                "trace": _trace("research", status="no_name")}
    try:
        result = deps.research_agent.research(
            ResearchRequest(practice_name=name, domain_hint=visit.get("domain_or_org_hint"),
                            location_hint=visit.get("location_hint")))
    except Exception as exc:  # a research tool failing must never crash the whole run
        return {"research": {"status": "failed"}, "warnings": [f"research raised {type(exc).__name__}"],
                "trace": _trace("research", status="failed")}
    data = result.as_dict()
    return {"research": data, "warnings": list(result.warnings),
            "trace": _trace("research", status=data["status"], tool_calls=data["tool_calls"])}


def score(state: LeadState, deps: Deps) -> dict:
    facts = state["research"]["facts"]
    specialty = facts["specialty"]["value"] or "unknown"
    providers = facts["provider_count"]["value"] or 0
    ehr = facts["ehr"]["value"]        # unknown EHR stays None; it is never guessed
    visits = [(SCORER_PAGES.get(p["category"], "/unmapped"), p["viewed_at"]) for p in state["visit"]["pages_viewed"]]
    result = score_practice(specialty, int(providers), ehr, visits, now=state["visit"]["window_end"])
    data = {"total": result.total, "band": result.band, "reasons": [r.as_dict() for r in result.reasons]}
    return {"scoring": data, "trace": _trace("score", total=result.total, band=result.band)}


def _practice_row(state: LeadState) -> dict:
    """The practice as the writer and checker expect it. The name is the directory's record for the matched practice
    when research returned one, not the visitor's raw hint (which can be lower case, abbreviated or misspelled). A
    fact research could not find gets a neutral stand-in so the draft reads naturally; the EHR stays None (never
    guessed) and is simply not mentioned."""
    research = state["research"]
    facts = research["facts"]
    matched = next((c for c in research.get("top_candidates") or [] if c["practice_id"] == research.get("practice_id")),
                   None)
    name = matched["name"] if matched else state["visit"]["practice_name_raw"]
    return {"name": name, "specialty": facts["specialty"]["value"] or "medical",
            "providers": facts["provider_count"]["value"] or 0, "city": facts["location"]["value"] or "your area",
            "ehr": facts["ehr"]["value"], "contact_name": None}


def draft(state: LeadState, deps: Deps) -> dict:
    """Write the email. One retry if the writer fails, then fall back to the template writer."""
    prior = state.get("outreach") or {}
    revisions = prior.get("revisions", 0) + (1 if prior.get("problems") else 0)
    reasons = state["scoring"]["reasons"]
    pages = [p["page"] for p in state["visit"]["pages_viewed"]]
    row = _practice_row(state)
    writers = [deps.writer, deps.writer] + ([deps.fallback_writer] if deps.fallback_writer else [])
    last_error = ""
    for writer in writers:
        try:
            subject, body = writer.email(row, reasons, pages)
            return {"outreach": {"subject": subject, "body": body, "problems": [], "revisions": revisions,
                                 "writer": writer.name},
                    "trace": _trace("draft", writer=writer.name, revisions=revisions)}
        except WriterError as exc:
            last_error = str(exc)
    return {"outcome": "review_queue", "reason": f"writer failed: {last_error}",
            "trace": _trace("draft", result="writer_failed")}


def check(state: LeadState, deps: Deps) -> dict:
    """Deterministic draft check: patient references, empty text, claims the practice record does not support."""
    out = dict(state["outreach"])
    text = f"{out['subject']}\n{out['body']}"
    problems = outreach.check(out["subject"], out["body"]) + outreach.ungrounded(text, _practice_row(state))
    out["problems"] = problems
    return {"outreach": out, "trace": _trace("check", problems=len(problems), revisions=out["revisions"])}


def approval(state: LeadState, deps: Deps) -> dict:
    """Pause point. With no decision yet the lead stays awaiting_approval; nothing is ever auto-sent."""
    decision = state.get("approval")
    if not decision:
        return {"outcome": "awaiting_approval", "trace": _trace("approval", result="waiting")}
    if decision["decision"] == "edit":          # SDR edited the text: re-run the same checks on their version
        out = dict(state["outreach"], subject=decision["subject"], body=decision["body"], writer="sdr")
        return {"outreach": out, "trace": _trace("approval", result="edited")}
    return {"trace": _trace("approval", result=decision["decision"])}


def send(state: LeadState, deps: Deps) -> dict:
    if deps.sender:
        deps.sender({"lead_id": state["lead_id"], **{k: state["outreach"][k] for k in ("subject", "body")}})
    return {"outcome": "sent", "trace": _trace("send", sent=bool(deps.sender))}


# ---------------------------------------------------------------- routers (pure: state in, next step name out)

END = "__end__"


def after_ingest(state: LeadState) -> str:
    return END if state.get("outcome") == "blocked" else "visit_features"


def after_features(state: LeadState) -> str:
    return END if state.get("outcome") in ("stopped", "review_queue") else "research"


def after_research(state: LeadState) -> str:
    """matched -> score. ambiguous/partial -> save top candidates and stop (partial). Anything else -> review queue."""
    if state.get("outcome"):
        return END
    status = state["research"]["status"]
    if status == "matched":
        return "score"
    return "finish_partial" if status in ("ambiguous", "partial") else "finish_review"


def after_draft(state: LeadState) -> str:
    return END if state.get("outcome") == "review_queue" else "check"


def after_check(state: LeadState) -> str:
    out = state["outreach"]
    if not out["problems"]:
        return "approval"
    return "draft" if out["revisions"] < MAX_REVISIONS else "finish_blocked"


def after_approval(state: LeadState) -> str:
    decision = (state.get("approval") or {}).get("decision")
    return {"approve": "send", "edit": "check", "send_back": "draft", "reject": "finish_rejected"}.get(
        decision, END)   # no decision: pause and end the run here; the checkpoint keeps the state


# terminal steps: set the outcome and a plain-language reason
def finish_partial(state: LeadState, deps: Deps) -> dict:
    return {"outcome": "partial", "reason": "several possible practices; a person should pick the right one",
            "trace": _trace("finish", outcome="partial")}


def finish_review(state: LeadState, deps: Deps) -> dict:
    return {"outcome": "review_queue", "reason": f"research status: {state['research']['status']}",
            "trace": _trace("finish", outcome="review_queue")}


def finish_blocked(state: LeadState, deps: Deps) -> dict:
    return {"outcome": "review_queue", "reason": "draft failed the checks after the maximum revisions",
            "trace": _trace("finish", outcome="review_queue")}


def finish_rejected(state: LeadState, deps: Deps) -> dict:
    return {"outcome": "rejected", "reason": "SDR rejected the lead", "trace": _trace("finish", outcome="rejected")}


STEPS = {"ingest": ingest, "visit_features": visit_features, "research": research, "score": score,
         "draft": draft, "check": check, "approval": approval, "send": send,
         "finish_partial": finish_partial, "finish_review": finish_review,
         "finish_blocked": finish_blocked, "finish_rejected": finish_rejected}

ROUTES = {"ingest": after_ingest, "visit_features": after_features, "research": after_research,
          "draft": after_draft, "check": after_check, "approval": after_approval}
FIXED_NEXT = {"score": "draft", "send": END, "finish_partial": END, "finish_review": END,
              "finish_blocked": END, "finish_rejected": END}


def run_lead(state: LeadState, deps: Deps, start: str = "ingest") -> LeadState:
    """Run the same steps and routers without LangGraph (tests, scripts). Stops at END or an approval pause."""
    state = dict(state)
    state.setdefault("warnings", [])
    state.setdefault("trace", [])
    step = start
    while step != END:
        update = STEPS[step](state, deps)
        for key, value in update.items():
            if key in ("warnings", "trace"):
                state[key] = state[key] + value
            else:
                state[key] = value
        step = ROUTES[step](state) if step in ROUTES else FIXED_NEXT[step]
        if step == "approval" and not state.get("approval"):
            update = approval(state, deps)          # reach the pause, record it, stop
            state["outcome"] = update["outcome"]
            state["trace"] = state["trace"] + update["trace"]
            break
    return state


def build_graph(deps: Deps, checkpointer=None):
    """Wire the steps into a LangGraph StateGraph. LangGraph is imported here so the rest of this file works
    without it. The graph pauses before `approval`; resume it after writing the SDR decision into the state."""
    from langgraph.graph import END as LG_END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    def lg(name: str):
        return lambda state: STEPS[name](state, deps)

    def target(name: str) -> str:
        return LG_END if name == END else name

    graph = StateGraph(LeadState)
    for name in STEPS:
        graph.add_node(name, lg(name))
    graph.set_entry_point("ingest")
    for name, router in ROUTES.items():
        options = {n: target(n) for n in _targets_of(name)}
        graph.add_conditional_edges(name, router, options)
    for name, nxt in FIXED_NEXT.items():
        graph.add_edge(name, target(nxt))
    return graph.compile(checkpointer=checkpointer or MemorySaver(serde=checkpoint_serializer()),
                         interrupt_before=["approval"])


# The state's own types that a checkpoint stores. Restoring a type it was not told about makes LangGraph log a
# deprecation warning ("will be blocked in a future version"); registering them is what that warning asks for.
CHECKPOINT_TYPES = [("leadfinder.visit_analysis", "PageView"), ("leadfinder.visit_analysis", "VisitAnalysisInput")]


def checkpoint_serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def _targets_of(name: str) -> list[str]:
    return {"ingest": ["visit_features", END], "visit_features": ["research", END],
            "research": ["score", "finish_partial", "finish_review", END],
            "draft": ["check", END], "check": ["approval", "draft", "finish_blocked"],
            "approval": ["send", "check", "draft", "finish_rejected", END]}[name]


def demo(practice_hint: Optional[str] = None) -> LeadState:
    """Run one simulated visit for a real practice in the NPI table through the whole pipeline, with the template
    writer, and stop at the approval pause. The visit is made up; the practice and its facts are not."""
    import time

    from . import npi

    records = npi.load_table()
    target = next((r for r in records if practice_hint and practice_hint.casefold() in r.name.casefold()), None) \
        or next(r for r in records if r.city == "Ithaca")
    now = time.time()
    visit = VisitAnalysisInput(
        lead_id="demo-1", practice_name_raw=target.name.lower().replace(",", ""), domain_or_org_hint=None,
        location_hint=target.location, company_identified=True, identification_confidence=0.9,
        page_views=(PageView("v1", "/pricing", now - 3600, 75), PageView("v2", "/demo", now - 1800, 60)),
        window_end=now, patient_data_stripped=True)
    return run_lead({"lead_id": "demo-1", "visit_input": visit}, default_deps(writer=Offline()))


if __name__ == "__main__":
    import json
    import sys

    result = demo(" ".join(sys.argv[1:]) or None)
    research_out = result.get("research", {})
    print(json.dumps({
        "outcome": result.get("outcome"),
        "practice_id": research_out.get("practice_id"),
        "research_status": research_out.get("status"),
        "facts": research_out.get("facts"),
        "score": result.get("scoring", {}).get("total"),
        "draft_subject": result.get("outreach", {}).get("subject"),
        "steps": [t["step"] for t in result.get("trace", [])],
        "warnings": result.get("warnings"),
    }, indent=1))
