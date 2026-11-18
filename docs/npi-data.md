# Practice data: the NPI table and the directory adapter

Owner: Daniel (@Dami047), data agent. This answers the data questions the README and
[architecture-and-rubric.md](./architecture-and-rubric.md) left open for the data owner, and records what was built.

## What was built

| Piece | Where | What it does |
| --- | --- | --- |
| Registry client and table builder | `src/leadfinder/npi.py` (`python -m leadfinder.npi build`) | Queries the public NPPES API (v2.1, no key) for organizations in each territory city and specialty, keeps independent practices, estimates their size, merges duplicate NPIs, writes the table |
| Practice table | `data/npi_practices.json` | 742 real practices (800 NPIs) in the 10 territory cities, retrieved 2026-11-18, committed so the app, tests and evaluation run offline on one snapshot |
| Directory adapter | `npi.TablePracticeDirectory` | Implements `research.PracticeDirectory` over the table: the integration point `research.py` was waiting for |
| Name normalization | `research.normalize_name` | Drops legal forms (PC, PLLC, LLC, Inc, MD...), joins split initials ("P.C."), and ignores "and"/"&" before names are compared |
| Matching evaluation | `src/leadfinder/evaluation/npi_match.py` → [npi-match-report.md](./npi-match-report.md) | Measures matching on real registry names and sweeps the cutoffs |
| Real practices in the app | `LEADFINDER_PRACTICES=npi` | Seeds the demo from the table instead of made-up practices (visits are still simulated) |
| Connection to the workflow | `orchestration.build_research_agent()`, `default_deps()` | The orchestration graph's research step searches the table. Claude makes the research decisions when a credential is set; otherwise `research.RulesResearchModel` follows fixed rules (look up, check the website if there is a domain, select the top candidate only above the threshold) under the same safeguards |
| Tests | `tests/test_npi.py` | Parser, filters, provider estimate, entity merge, API paging and errors (mocked, no network), directory, an end-to-end research run on real table rows, the committed table's validity, seeding, the lead page |

## Decisions on the open questions

**Deliverable format: a file, not an API.** Option A from the brainstorm: a bounded lookup over a table built ahead of
time. A live registry call per lead would make results change between runs and depend on a free public service being
up during the demo. `TablePracticeDirectory.from_file()` loads `data/npi_practices.json`; rebuilding is one command.

**Stable practice ID: `npi:<10-digit NPI>`.** An NPI is issued once and never reused. When several NPIs are one
business (below), the lowest is the ID and all of them are kept in `npis`. The ID changes only if that NPI is
deactivated and the table is rebuilt.

**Fields and provenance.** Each candidate carries `practice_id`, `name`, `location` ("City, ST"), `specialty`,
`provider_count` and `source = "npi"`. The table also keeps the NPPES taxonomy code and description, street, ZIP,
practice phone, the authorized official's name and title (the registry's listed contact), enumeration and last-update
dates, and `providers_source`.

| Field | Where it comes from | Can it be wrong? |
| --- | --- | --- |
| name, NPI, street, city, state | NPPES record, LOCATION address (not the mailing address) | As current as the practice keeps its registration |
| specialty | primary taxonomy code mapped to the scorer's eight specialties | Practices whose primary code is outside the eight are left out |
| provider_count | **estimate**: individual clinicians registered at the same address with the same specialty, floored at 1 | Yes. A lower bound for most; inflated for groups that sit inside a hospital building |
| domain | none: the registry has no websites | Always empty, so the `domain_only` strategy finds nothing |
| EHR | none: the registry has no EHR | Stays unknown unless website research finds a single named EHR |

**Normalization owner: the data side**, in `research.normalize_name`, used by both the directory and `rank_candidates`.

**Match cutoffs: keep 0.85 match, 0.10 margin.** The sweep on real names (below) puts the current setting at the knee:
91.5% of hints for listed practices match the right record, none match the wrong one, and 1.3% of hints for unlisted
practices land on a differently named real practice. 0.90 would cut correct matches by 9 points to save 0.6 points of
false matches; 0.80 adds 4 points of correct matches but nearly triples the different-practice false matches (3.6%).
A false match means outreach to the wrong practice, which costs more than a miss (a miss goes to the review queue), so
the stricter side of the knee is right. The simulated evaluation's suggestion to lower the margin to 0.07 is not
supported on real data: at 0.05 the margin buys 0.3 points of correct matches for 0.5 points more false matches,
largely because the duplicate names that motivated it are now merged before matching.

**Website domains and homepage-only EHR research.** The registry cannot supply domains, so website research only
works when the visit identity source supplies a domain. Homepage-only EHR detection is therefore a later question; with
NPI data alone, EHR stays unknown, which the scorer already treats as its own category. Nothing is guessed.

## How the data changed after testing

The matching evaluation drove two changes. Both are measured on the same 742 practices, split 70/30 into directory and
held out, with 8 hint forms per practice (4,152 queries that should match, 1,784 that should not).

1. **Entity resolution.** The first evaluation showed false matches that were not different practices at all: the
   registry lists one business under several NPIs (Maine Medical Partners has four in Portland, ME). 58 of 800 NPIs
   (7%) were duplicates of another practice in the same city. Left apart, two identical names tie, so the research
   agent can never match either. They are now merged into one practice with every NPI kept.
2. **Name normalization.** Registry names carry legal forms that a company-name hint from visit tracking usually drops.

| Matching at 0.85 / 0.10 | correct | wrong | false match (unlisted practice) | of which a differently named practice |
| --- | --- | --- | --- | --- |
| before: punctuation-only names | 63.1% | 0.0% | 1.3% | 1.1% |
| after: `normalize_name` | 91.5% | 0.0% | 2.4% | 1.3% |

The trade-off is stated, not hidden: normalization also makes "X Physical Therapy, PC" in one city match "X Physical
Therapy LLC" in another when the hint has no location, so same-name false matches rose from 4 to 18 of 1,784.

Where the false matches come from (after, at 0.85 / 0.10) points at the next fix:

| hint | same name, other city | differently named practice |
| --- | --- | --- |
| with a location | 0 | 24 |
| without a location | 18 | 0 |

A location hint removes every same-name false match, because the directory narrows to that city. But every
differently named false match happens *with* a location: `rank_candidates` adds 0.08 for a location match, which lifts a
similar name in the same city over 0.85 ("Merit Medical Practice, PC" matched "Guler Medical Practice PLLC" in
Rochester at 0.857; "D & R Physical Therapy" matched "DG Physical Therapy PC" in Worcester at 0.933). Some of the 24 are
arguably the same business (one physician's two registrations, MDLive's per-state entities). Once the directory has
already filtered by city, the bonus no longer separates candidates; it only raises them all. Dropping the bonus when
the lookup strategy already used the location is the change to test next, together with Ashwin, who uses the same
scores. On Hanna's simulated evaluation the
same change raised correct matches for legal-suffix hints from 67.2% to 70.3% (at the proposed 0.07 margin) with no
wrong matches; that report was regenerated.

## Limits

- **Hints are generated, not observed.** The evaluation's hints come from registry names (exact, without legal form,
  casual, one typo). How often a real visit-identity source supplies a usable name, and in what form, is unmeasured.
- **Provider counts are estimates.** 307 of 742 practices had no clinician registered at their address with their
  specialty and are floored at 1, so the table over-represents solo practices (462). Thirty practices estimate 26+,
  mostly groups whose address is a hospital building with many clinicians of that specialty.
- **"Independent" is a heuristic.** Subparts and institution names are dropped (304 subparts, 143 names), but a
  system-owned practice registered under its own name gets through, and some registrations are odd (consulting LLCs
  with a family-medicine taxonomy).
- **Snapshot.** The table is as of its retrieval date. Practices that registered or closed since are not reflected
  until it is rebuilt.
- **Coverage.** Ten cities in seven states and eight specialties; pediatrics is thin (17 practices) because most
  pediatric groups in these cities are hospital or system subparts.

## Privacy and responsible use

Everything in the table is public business information that CMS publishes for every provider; there is no patient
data. The authorized official's name and title are kept as the listed contact; their phone number is not. The
registry has no email addresses and none are invented: the lead page says "no email on record". Using the table to
contact practices is still subject to CAN-SPAM for email and TCPA for calls, and sending stays disabled in the app.

## Reproduce

```bash
.venv/bin/python -m leadfinder.npi build                 # about 480 requests, 2 to 3 minutes; rewrites the table
.venv/bin/python -m leadfinder.evaluation.npi_match      # rewrites docs/npi-match-report.md and .json
LEADFINDER_PRACTICES=npi .venv/bin/uvicorn asgi:app --port 8010   # on a fresh database
.venv/bin/python -m leadfinder.orchestration "linden oaks"  # one simulated visit for a real practice, through the graph
.venv/bin/python -m pytest -q tests/test_npi.py          # offline: uses the committed table and mocked HTTP
```

The builder needs outbound HTTPS to `npiregistry.cms.hhs.gov`; nothing else in this work calls the network.
