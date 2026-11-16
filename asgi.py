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

from leadfinder.service.app import create_app  # noqa: E402

app = create_app(os.environ.get("LEADFINDER_DB", ROOT / "data" / "leadfinder.db"))
