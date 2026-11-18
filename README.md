# LeadFinder

An AI lead finder for health-tech sales to independent medical practices. Website visits and an NPI-style practice
record go into a trained model that gives each practice a chance of converting, with what moved that chance shown
line by line, and into a hand-set points table kept beside it for comparison. Each lead has a call brief, and no
email until an SDR asks for one with a button; it is then written from public business facts only, and the SDR
approves, edits or rejects it, with every decision logged. The How it works page walks through the five steps. Nothing is sent from the app; an approval marks the email ready to send.

Leads are ranked by the trained model, a logistic regression (`src/leadfinder/ml.py`). There are no real sales
outcomes yet, so it learns from a simulated history (`src/leadfinder/synth.py`): 8,000 made-up past leads, each won
or lost by a rule the simulator keeps to itself. It is fitted on 70% of them and judged on the other 30%, beside the
points table and a gradient-boosted challenger; the Scoring model page shows that benchmark and what the model
learned. The numbers compare the scorers with each other. They are not a forecast of real conversion.

The email and the call brief start as templates; with an
Anthropic API key set, an SDR can have Claude write either one from the lead's page. Claude sees only the practice's
public record, its visits and the score's reasons, and code checks what it writes before it is kept: no patient
information, no claims the seller cannot back, no EHR or provider count that is not on the record.

It is a FastAPI service with server-rendered Jinja2 pages and SQLite storage. The pages use the same broadsheet style
as AuxoCase (the palette and type of ashwinprajeeth.com, light and dark themes), and their one script is
`/assets/app.js` under a `script-src 'self'` policy.

## Team handoff: brainstorm decisions and rubric

**The brainstorm describes a target pipeline, not the current end-to-end application.** This checkout is a working
lead-scoring and human-reviewed outreach prototype. The visit gate is integrated into the authenticated app at
`POST /api/visit-analysis`; scrubbed, behavior-qualified review cases persist in SQLite and SDRs can resolve or close them in the
Visit review UI. The gate still depends on the caller's `patient_data_stripped` assertion, and the UI only matches
against practices already in the app. A human-verified case and an automatically identified case with a unique
exact local name match (narrowed by location when supplied) share a handoff: derived visit categories are attached
to the local practice and
its existing lead is created or rescored. This is synthetic local-record lookup and scoring, not external research.
The bounded prospect-research module in `src/leadfinder/research.py` has no real practice-directory adapter on
`master`. A LangGraph orchestration prototype in `src/leadfinder/orchestration.py` calls the ResearchAgent and
points scorer, but it is not connected to FastAPI/SQLite. Do not present this as a live
visitor-tracking-to-research workflow.

The target post-gate flow is: the ResearchAgent sends name/domain/location lookup requests to a bounded
`PracticeDirectory` adapter, receives candidate records, applies code-based match safeguards, and may inspect a
returned candidate's HTTPS homepage for recognized EHR evidence. It returns a sourced research result for a later
cross-check and scoring step. The adapter is a data lookup tool, not a second autonomous LLM agent. This flow is
not wired into the running app on `master`: the NPI-table/adapter work remains proposed in
[PR #3](https://github.com/Ash1129/LeadFinder/pull/3), while the orchestration prototype from
[PR #2](https://github.com/Ash1129/LeadFinder/pull/2) is now on `master` but is not connected to FastAPI/SQLite.
That prototype uses a points scorer; parallel LLM scoring and full app input/output and failure contracts remain
open. The NPI table provides stable `npi:<digits>` practice IDs but has no domain or EHR data; a website check
therefore needs a separately supplied, validated domain.

### Owner handoff

- **Daniel (@Dami047) — data:** done in [docs/npi-data.md](./docs/npi-data.md). `src/leadfinder/npi.py` builds a
  practice table of 742 real independent practices from the NPI registry (`data/npi_practices.json`) and
  `npi.TablePracticeDirectory` is the `PracticeDirectory` adapter over it. IDs are `npi:<NPI>`; duplicate NPIs of one
  business are merged; provider counts are estimates; the registry has no domains or EHRs, so neither is filled.
  Matching on real names (`docs/npi-match-report.md`) supports keeping the 0.85 / 0.10 cutoffs. The table is not yet
  wired into the app's visit handoff or LangGraph (Mahika); `LEADFINDER_PRACTICES=npi` seeds the demo from it.
- **Sasha (@Sasha-Shkolnikov) — visit analysis:** deterministic features and gate are in
  `src/leadfinder/visit_analysis.py`; taxonomy v2 handles content sections before slug keywords, common demo
  paths, and customer/specialty pages. The authenticated app endpoint runs the gate, stores scrubbed review cases, and
  routes behavior-qualified unresolved/low-confidence visits to the Visit review queue. Validate against
  representative real-site URLs; keep thresholds and inferred hints provisional. The endpoint does not scrub
  input; the caller must not set `patient_data_stripped=true` until a real ingestion scrubber exists.
- **Hanna (@hannamlt) — evaluation:** the persona simulator and separate ground truth are now in
  `src/leadfinder/evaluation/`. Validate the persona/should-pass labels and the assumed identification rates;
  report gate recall, false passes, taxonomy coverage, and match accuracy to Sasha/Daniel. Keep labels out of
  `ResearchResult` and lead state.
- **Ashwin (@Ash1129) — scoring:** agree on the shared lead-state fields with Sasha, Daniel, and Mahika; decide
  whether/how to consume inferred visit hints and research facts, and how unknown EHR, partial research, and
  ambiguous candidates affect scores and reasons.
- **Gabriel (@gabrielmaurad) — outreach:** use sourced research facts; confirm draft checks/revisions and approval
  feedback. Sending stays disabled until its owner and provider are agreed.
- **Mahika (@ma2255-cu) — orchestration:** integrate the LangGraph prototype with FastAPI/SQLite and persisted
  review outcomes; confirm checkpointing, traces, retries, and routing for `human_review`, `partial`, and `failed`
  outcomes after owners confirm the input/output contracts. Do not send unresolved identities to research or
  duplicate the research loop.

**Dependency order:** confirm privacy and shared schemas -> review/merge and integrate the proposed directory
adapter, and revise the standalone gate from evaluation evidence -> connect the graph to the app and persisted
review outcomes -> connect scoring/outreach -> evaluate and document
results. These modules are deliberately not frozen contracts; the named owners should update their assumptions
and tests as real data and evaluation decisions arrive. Full questions and rubric evidence are in
[docs/architecture-and-rubric.md](./docs/architecture-and-rubric.md).

### Provisional choices to revisit

- **Visits and privacy:** `visit_analysis.py` assumes opaque view IDs, one lead/visitor's events, and already
  allowlisted, scrubbed company-level inputs. `patient_data_stripped` is only a caller assertion, not a scrubber.
  Agree on identity, fields, retention, and upstream scrubbing before connecting real visits. The simulator's
  identification rates are assumptions, not measured tracking coverage; do not raise them as a substitute for
  improving the real identity source.
- **Unresolved identity routing:** the result routes scrubbed, unidentified or low-confidence visits to
  `human_review` only when their behavior meets the intent rule; otherwise they stop. The authenticated
  `/api/visit-analysis` endpoint persists only scrubbed `human_review` cases, and the Visit review page records an SDR's
  verification against an existing practice or closes the case with a reason. Verification and a unique exact
  automatic local name match (narrowed by location when supplied) share the same synthetic-record visit
  attachment and lead-scoring handoff. This does not
  invoke the separate research agent or any external source. `passes_threshold` remains false for review cases;
  consumers should route on `routing_decision`. The API does not scrub input.
- **Gate and inferred hints:** taxonomy v2 is implemented based on simulated URLs, but still needs representative
  real-site validation. The 30-day lookback, 30-minute session timeout, 0.5 identification cutoff, other gate
  rules, and owner/administrator heuristics remain provisional. Hanna's evaluation should drive Sasha's revisions;
  Ashwin should decide whether inferred role/EHR/specialty/product hints belong in scoring.
- **Research and shared state:** `research.py` uses provisional match heuristics (0.85 match, 0.45 ambiguity,
  0.10 margin), a 10-call cap, and homepage-only EHR evidence. Daniel should confirm or replace matching/data
  assumptions; Ashwin and Mahika should agree on the shared state and treatment of partial/ambiguous results.
  See the architecture handoff for the detailed owner-by-owner update points.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn asgi:app --port 8010
```

Open http://127.0.0.1:8010. The first start takes a few seconds while the model is trained; after that it is read
from the database. On first start an empty `data/leadfinder.db` is also filled with 40 synthetic practices, their
visits, their scored leads and call briefs (no emails: those are written on request), and the demo logins listed on the login page (`riley` and `jo` are SDRs,
`morgan` is the sales manager; the password is `leadfinder`). Set `LEADFINDER_DEMO=0` to start empty instead.
Demo mode also seeds two pending identity-review examples, one verified example, and one closed example. Each is
marked **synthetic demo** in the review UI and is not a real visitor or tracking observation.

### Sending visit analyses to the gate

A logged-in staff client with `act_on_leads` permission can submit allowlisted, already-scrubbed JSON to
`POST /api/visit-analysis`; the endpoint uses the app's staff session, not a separate service token. There is no
tracking connector in this checkout. Supply a unique `request_id` for retry-safe submission. Behavior-qualified
unresolved cases are persisted in **Visit review**; an SDR may verify an existing local practice or close the case.
Qualified identified cases hand off only when the name/location exactly and uniquely match a local practice. Both
paths use the same local visit-attachment and lead-scoring routine. No external directory lookup or web research
occurs, and no new practice is created. The app does not perform input scrubbing; do not send real visit data until
an approved upstream scrubber and identity source are in place.

To start on real practices from the NPI registry instead of made-up ones, set `LEADFINDER_PRACTICES=npi` on a fresh
database: 40 practices are drawn from `data/npi_practices.json` (their visits are still simulated, their EHR is
unknown and they have no email address, since the registry has neither). `python -m leadfinder.npi build` rebuilds
that table from the public registry; see [docs/npi-data.md](./docs/npi-data.md).

To have Claude write the emails and call briefs, put `ANTHROPIC_API_KEY=...` in a `.env` file beside `asgi.py`
(it is gitignored; `.env.example` lists every setting) and restart. The tests never call the API.

The first start fits six versions of the model, one for each simulated month of history, and a scorecard decides
which is live (the Scoring model page shows all six side by side with the decision on each). To add a month, fit the
next version and put it to the scorecard:

```bash
.venv/bin/python run.py train
```

```bash
.venv/bin/python -m pytest -q
```

## Layout

| Path | What it is |
| --- | --- |
| `src/leadfinder/ml.py` | the trained scorer: features, training, the benchmark, and each prediction's reasons |
| `src/leadfinder/synth.py` | what a made-up practice and its visits look like, and the simulator's hidden conversion rule |
| `src/leadfinder/scoring.py` | the hand-set points table |
| `src/leadfinder/outreach.py` | the template email and the checks (no patient data, no claims we can't back, the CAN-SPAM footer, nothing off the record) |
| `src/leadfinder/writer.py` | who writes the email and the call brief: Claude through the Anthropic SDK, or the templates |
| `src/leadfinder/seed.py` | synthetic practices with valid-format NPIs, and their visits |
| `src/leadfinder/npi.py` | the data agent: NPI registry client, practice-table builder, entity merge, and `TablePracticeDirectory` |
| `data/npi_practices.json` | the practice table: real independent practices from the NPI registry, with provenance |
| `src/leadfinder/research.py` | bounded Claude tool-use research, candidate safeguards, website EHR cues, and the adapter contract |
| `src/leadfinder/orchestration.py` | fixed LangGraph workflow prototype that runs the gate, research, points scoring, checked drafting, and approval routing |
| `src/leadfinder/visit_analysis.py` | deterministic page/session features and the versioned pre-research intent gate |
| `src/leadfinder/service/app.py` | authenticated visit-analysis API and human-review queue integration |
| `src/leadfinder/service/templates/review_queue.html` and `visit_review.html` | pending review queue and reviewer decision pages |
| `tests/test_visit_analysis.py` | focused feature, gate, privacy-boundary, and serialization tests |
| `src/leadfinder/evaluation/npi_match.py` | practice matching on the real NPI table and the cutoff sweep (writes `docs/npi-match-report.md`) |
| `src/leadfinder/evaluation/` | persona visit simulator with separate ground truth; evaluation of the gate, page taxonomy and practice matching (`python -m leadfinder.evaluation` writes `docs/evaluation-report.md`) |
| `src/leadfinder/auth.py` | accounts, roles (SDR, sales manager), scrypt passwords, sessions |
| `src/leadfinder/service/` | the app, the security middleware and headers, the templates, `static/app.js` |

## Not built yet (from the proposal)

- Integration of the LangGraph prototype with FastAPI/SQLite and persisted human-review outcomes. The current
  graph runs the gate, ResearchAgent, points scorer, checked drafting, and approval routing; the app still does
  not invoke it. Parallel ML/LLM scoring and the post-gate intent summary remain planned.
- Enrichment from the real NPPES registry instead of `seed.py`.
- The concrete NPI table/`PracticeDirectory` adapter is proposed in PR #3 but is not on `master`; the research
  module cannot use it in the running app until the PR is reviewed, merged, and wired in.
- Persona-based visit simulation and evaluation now exist (`src/leadfinder/evaluation/`, results in
  `docs/evaluation-report.md`). Taxonomy v2 is applied in the standalone gate; its simulation results are not
  evidence of real-traffic performance. The proposed research match-margin change is not applied and needs
  Daniel's review with real directory data. Not yet covered: the scoring benchmark against this ground truth and
  the SDR feedback loop.
- Real outcomes: the model's labels are simulated, and SDR approvals and rejections do not retrain it yet.
- The LLM-judgment row of the benchmark (Claude ranking the same held-out leads).
- Integration of the research module into an application workflow; its website fetcher is limited to a candidate's
  HTTPS homepage and exact, locally extracted EHR cues. Current identity handoffs use only the local practice table
  and scoring logic, not the research module or external sources.
- An actual ingestion-time patient-data scrubber and approved company-identity source. The gate API only checks
  the caller's `patient_data_stripped` flag; it is not a privacy filter. Also open: per-node graph traces/retries,
  draft revision/send-back, and feedback-driven scoring. Existing outreach checks are not a substitute for the
  proposed ingestion guardrail.
