# LeadFinder

An AI lead finder for health-tech sales to independent medical practices. Website visits and an NPI-style practice
record go into a transparent points score, each point shown with its reason and source. A first email is drafted from
public business facts only, and an SDR approves, edits or rejects it, with every decision logged. Nothing is sent from
the app; an approval marks the email ready to send.

It is a FastAPI service with server-rendered Jinja2 pages and SQLite storage. The pages use the same broadsheet style
as AuxoCase (the palette and type of ashwinprajeeth.com, light and dark themes), and their one script is
`/assets/app.js` under a `script-src 'self'` policy.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn asgi:app --port 8010
```

Open http://127.0.0.1:8010. On first start an empty `data/leadfinder.db` is filled with 40 synthetic practices, their
visits, their scored leads and drafts, and the demo logins listed on the login page (`riley` and `jo` are SDRs,
`morgan` is the sales manager; the password is `leadfinder`). Set `LEADFINDER_DEMO=0` to start empty instead.

```bash
.venv/bin/python -m pytest -q
```

## Layout

| Path | What it is |
| --- | --- |
| `src/leadfinder/scoring.py` | the explainable points table |
| `src/leadfinder/outreach.py` | the drafted email and the guardrail (no patient data, no claims we can't back, the CAN-SPAM footer) |
| `src/leadfinder/seed.py` | synthetic practices with valid-format NPIs, and their visits |
| `src/leadfinder/auth.py` | accounts, roles (SDR, sales manager), scrypt passwords, sessions |
| `src/leadfinder/service/` | the app, the security middleware and headers, the templates, `static/app.js` |

## Not built yet (from the proposal)

- Enrichment from the real NPPES registry instead of `seed.py`.
- An LLM research and drafting agent, and the benchmark of the points table against a trained model and LLM judgment.
- Moving the weights based on SDR rejections and edits.
