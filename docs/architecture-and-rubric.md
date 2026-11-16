# LeadFinder: brainstorm handoff and rubric evidence

Compared with the checked-in implementation on 2026-11-16. This is a status and handoff document: the brainstorm's
target design must not be described as functionality that already runs.

## What the prototype does today

- **Problem and users:** helps health-tech sales development representatives prioritize independent medical
  practices and review business outreach; a sales manager can inspect decision history.
- **Inputs and research:** the app uses seeded, synthetic practice records and flat page/timestamp visits. There is
  no real NPPES import, visitor/session identity, healthcare page taxonomy, or lead-threshold gate. A standalone
  bounded research module now defines the agent loop, match safeguards, source-tagged result and website EHR
  extraction, but no real directory adapter is connected and the app does not call it.
- **Scoring:** a hand-set points table and an explainable logistic-regression scorer rank all seeded practices.
  Their benchmark, including a gradient-boosted challenger, uses synthetic outcomes and held-out synthetic leads;
  it does not establish real-world conversion performance. There is no LLM scorer benchmark.
- **Generative AI and oversight:** when configured, Claude writes an email or call brief only after an SDR asks.
  Code checks generated text; an SDR may edit and approve or reject, and the decision is audited. Nothing is sent.
  Email and brief writing are single calls; the separate research module uses a bounded Claude tool-use loop but
  is not connected to the app.
- **Responsible-AI boundary:** the app's outreach checks and model input selection address some content risks.
  There is no raw-visit ingestion pipeline or ingestion-time patient-data scrubber. Do not connect real tracking
  data or send raw URLs/forms to a model until that guardrail is implemented and tested. Call the standalone
  research module only with allowlisted company-level name/domain/location hints after the gate and scrub.
- **Evaluation:** automated tests cover scoring, synthetic-model comparisons, outreach checks, and web-service
  behavior. The measured model results are against the simulator's labels, not real sales outcomes. SDR feedback
  is audited but does not update the model.
- **Reproducibility:** the README documents setup, dependencies, configuration, and tests; synthetic generators
  use seeds. The optional Claude integration is disclosed and tests do not call the API.

## Target architecture already specified in the brainstorm

The brainstorm's decisions and their current implementation status:

1. Use a fixed LangGraph workflow with a shared typed lead state, deterministic routing, state/checkpoint
   persistence, and per-node traceability; do not use an LLM supervisor.
2. Apply an ingestion allowlist and patient-data scrub before any lead or model sees visit data. Compute structured
   visit features in code, then stop unidentified/low-confidence or low-intent visits at a recall-oriented,
   versioned gate before costly research.
3. After the gate, run a one-call intent summary in parallel with a bounded ReAct prospect-research loop
   (maximum 10 tool calls). The standalone loop is now implemented in `src/leadfinder/research.py`; connecting
   Daniel's practice table and wiring it into the graph remain. Unknown EHR is explicit, never guessed. Ambiguous
   matches retain candidates and confidence rather than inventing a match.
4. Cross-check EHR and specialty hints, run the ML and LLM scorers in parallel, and preserve both scores and
   explainable reasons.
5. Draft and check outreach with at most two revisions. Pause for an SDR decision; only an approved or edited
   message can reach a send step. A send-back is bounded, a rejection is logged, and no response never auto-sends.
6. Keep simulation labels separate from lead state. Use SDR decision events and ground truth for evaluation; do not
   claim the system learns from feedback until a versioned, tested scoring update exists.

### Intent gate: owner and rule

The intent gate is a cheap, deterministic visit-analysis step that stops unqualified traffic before the more costly
practice-research agent runs. It is not the research agent and does not decide the final lead ranking.

**Sasha owns the visit-analysis features and gate** in the brainstorm. The proposed rule requires an identified
practice and identification confidence of at least 0.5; then it passes if any of these holds:

- a high-tier page was viewed (for example pricing, demo/trial, or EHR integration);
- at least two distinct evaluation criteria were viewed; or
- a medium-or-higher tier page was viewed and the visit had at least two sessions or at least three pages per
  session.

The gate records pass/fail and reasons. **Hanna owns evaluation of the gate**: use the separate persona ground truth
to measure recall and noise, then recommend threshold changes. The numeric cutoffs and healthcare taxonomy are
starting proposals, not validated results. The gate is not implemented in this app yet.

## Open decisions — resolve before integration

The proposed direction is recorded below where one exists. Owners should confirm the contract in a shared schema
or team decision before dependent work is integrated.

| Owner from brainstorm | Decision or deliverable still needed | Proposed direction / unblock condition |
| --- | --- | --- |
| **Daniel — data agent** | Deliverable format (file or API), field names/types, stable practice ID, normalization owner, and ambiguous-match cutoff. | Use the selected **option A**: a bounded lookup tool over the practice table, not a live LLM data agent. `research.py` expects lookup requests with name/domain/location and a strategy, and candidates with stable `practice_id`, name, domain, location, specialty, provider count, source, and optional EHR plus its source. Its current match defaults (0.85 minimum and 0.10 margin) are heuristics, not probabilities; Daniel should confirm or replace them. The fetcher currently checks only the HTTPS homepage. Confirm the table contract, match policy, and whether homepage-only EHR research is sufficient. |
| **Hanna — evaluation** (suggested owner; not confirmed) | Who builds the visit simulator; persona mix, EHR list, page taxonomy coverage, and the separate ground-truth schema. | The brainstorm suggests evaluation owns generator and labels. Confirm that owner, then test seeded repeatability, unmatched/noisy names, all personas, and no label leakage. |
| **Ashwin — lead scoring** | Exact scorer input/output contract, handling of unknown EHR and partial research, score-disagreement rule, and whether/how an LLM scorer can be evaluated. | Preserve both scores and reasons for comparison. Agree on fields with visit/research owners before implementing the graph join. |
| **Gabriel — outreach and approval** | Checker/revision behavior, approval/edit/send-back/reject payloads, feedback format, and send-step ownership. | Keep sending disabled until explicitly agreed and implemented. The current app supports edit-and-approve or reject only; it has no send-back, automatic revision loop, or mail provider. |
| **Mahika — orchestration** | Integration plan between the existing FastAPI/SQLite prototype and the proposed LangGraph state, checkpointing, scheduling, and retries. | LangGraph is the brainstorm's selected design, but it is not a dependency or runtime in this checkout. The research module is callable independently and emits `ResearchResult.as_dict()`; agree on the graph/state mapping and failure routing before wiring it in. |
| **All owners** | Confirm schemas and the proposed safety/evaluation acceptance criteria. | Test fake patient-like strings at ingestion and prove none reach persisted lead state or model prompts; separately measure threshold recall/noise, practice-match accuracy, draft-check failures, latency, and failure/partial/review-queue behavior. Record results and at least one test-driven iteration. |

The standalone research loop is implemented; its concrete directory adapter, app/graph integration, and the open
contracts above are not. Until those are closed, keep the current prototype runnable and clearly label the remaining
graph/simulation work as planned. Do not silently substitute synthetic assumptions for real data contracts or make
an unknown EHR into a guessed fact.

## Course rubric: evidence and remaining work

The rubric totals **150 points**: 140 primarily team-based and 10 individually assessed. This table distinguishes
evidence in the current checkout from work needed for the brainstorm's target project.

| Category | Current evidence | Gap to address / evidence to prepare |
| --- | --- | --- |
| **1. Problem definition and AI justification — 15** | README states a health-tech lead-prioritization problem and intended SDR workflow. | State the target-user constraints and desired outcome precisely; justify learned ranking and constrained text generation against a simpler rules-only workflow, and scope what can be completed in the project timeline. |
| **2. AI solution design and reasoning — 20** | Explainable logistic regression is compared with a points table and tree challenger; current UI documents the five-step prototype; the bounded research loop is specified in code. | Publish the final graph/data flows. Explain why visit analysis is deterministic, how the bounded tool-using research module uses evidence and match safeguards, and why human approval is retained. |
| **3. AI implementation and integration — 35** | Working FastAPI/SQLite prototype, model training/serving, optional Anthropic writing, tested content checks and human decisions; standalone research module with fake-adapter tests. | The module has no real directory adapter or app/graph integration. The visit gate, LLM scoring comparison, revision loop, and send workflow are also not implemented. Demonstrate a functioning core path without presenting planned steps as complete. |
| **4. Testing, evaluation, iterative improvement — 25** | Tests cover the research loop with fake adapters and held-out synthetic scorer comparisons. | Add persona-ground-truth cases, threshold and match measures, failure/latency evidence, safety injection tests, and a documented change made because of observed results. Synthetic-label metrics are not real-world effectiveness evidence. |
| **5. Responsible AI, ethics, alignment — 15** | Outreach text is checked; generation uses selected business facts; human review is required; sending is disabled; limitations of simulated conversion are disclosed. | Implement and test pre-model ingestion minimization/scrubbing, provenance/confidence and unknown handling, match ambiguity review, bounded retries, and explicit failure reporting. Discuss privacy, bias, false identification, hallucinations, and misuse. |
| **6. Documentation and reproducibility — 10** | README has setup/configuration/test instructions; seeded synthetic generation and model versions support repeatability. This handoff records design status and unresolved interfaces. | Document final architecture, data/model/prompt versions, external services, graph configuration, evaluation procedure/results, and any access/setup requirements after decisions are closed. |
| **7. Presentation and live demo — 20** | A runnable UI and tutorial video are present. | Prepare a demo of the functioning path, show the human gate and safety behavior, report measured results and failure cases, and prepare team members to explain design trade-offs. |
| **8. Individual contribution and reflection — 10** | The brainstorm names owners for visit analysis, research, data, scoring, outreach, orchestration, and evaluation. | Keep an evidence trail of each person's actual contributions; each member should reflect on technical challenges, skills learned, and AI limitations. |
