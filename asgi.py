"""The service as an ASGI app: `uvicorn asgi:app` from this folder.

The database is LEADFINDER_DB (default data/leadfinder.db). On first start an empty one is filled with synthetic
practices and the demo logins (set LEADFINDER_DEMO=0 to start empty and make the first account from
LEADFINDER_ADMIN_USER and LEADFINDER_ADMIN_PASSWORD instead).

On Vercel (which finds `app` here and runs it as one Python function) the only folder that can be written is /tmp,
so the database is made there on a cold start: the same synthetic practices and the same trained model every time,
in about two seconds. What the pages change (decisions, written emails, logins) lives in that copy only: a new
instance starts again from the seed. The host's own names are the allowed hosts.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# A .env beside this file (never committed) sets what the environment does not already: ANTHROPIC_API_KEY for the
# Claude writer, LEADFINDER_* for the rest. Lines are KEY=value; blank lines and # comments are skipped.
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

ON_VERCEL = bool(os.environ.get("VERCEL"))
if ON_VERCEL:
    os.environ.setdefault("LEADFINDER_DB", "/tmp/leadfinder/leadfinder.db")
    if not os.environ.get("LEADFINDER_ALLOWED_HOSTS"):   # this deployment's own names, as Vercel gives them
        names = [os.environ.get(k, "") for k in ("VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL")]
        os.environ["LEADFINDER_ALLOWED_HOSTS"] = ",".join(n for n in names if n)

from leadfinder.service.app import create_app  # noqa: E402

app = create_app(os.environ.get("LEADFINDER_DB", ROOT / "data" / "leadfinder.db"))
if ON_VERCEL:
    # The host ends TLS in front of the app: take the scheme from X-Forwarded-Proto, so the origin check on every
    # form post and the Secure flag on the session cookie see https.
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
