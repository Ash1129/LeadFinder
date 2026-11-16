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

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn asgi:app --port 8010
```

Open http://127.0.0.1:8010. The first start takes a few seconds while the model is trained; after that it is read
from the database. On first start an empty `data/leadfinder.db` is also filled with 40 synthetic practices, their
visits, their scored leads and call briefs (no emails: those are written on request), and the demo logins listed on the login page (`riley` and `jo` are SDRs,
`morgan` is the sales manager; the password is `leadfinder`). Set `LEADFINDER_DEMO=0` to start empty instead.

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
| `src/leadfinder/auth.py` | accounts, roles (SDR, sales manager), scrypt passwords, sessions |
| `src/leadfinder/service/` | the app, the security middleware and headers, the templates, `static/app.js` |

## Not built yet (from the proposal)

- Enrichment from the real NPPES registry instead of `seed.py`.
- Real outcomes: the model's labels are simulated, and SDR approvals and rejections do not retrain it yet.
- The LLM-judgment row of the benchmark (Claude ranking the same held-out leads).
- An LLM research agent that reads the practice's public web presence.
