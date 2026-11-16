"""The service as an ASGI app: `uvicorn asgi:app` from this folder.

The database is LEADFINDER_DB (default data/leadfinder.db). On first start an empty one is filled with synthetic
practices and the demo logins (set LEADFINDER_DEMO=0 to start empty and make the first account from
LEADFINDER_ADMIN_USER and LEADFINDER_ADMIN_PASSWORD instead).
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

from leadfinder.service.app import create_app  # noqa: E402

app = create_app(os.environ.get("LEADFINDER_DB", ROOT / "data" / "leadfinder.db"))
