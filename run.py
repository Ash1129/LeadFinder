"""LeadFinder command line.

  python run.py seed [--n 40] [--force]          fill the database with synthetic practices, visits and scored leads
  python run.py staff add NAME --role sdr|manager [--name "Full Name"]   add an account (the password is prompted for)
  python run.py serve [--port 8000]              the pages (same as `uvicorn asgi:app`)
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from leadfinder import auth                    # noqa: E402
from leadfinder.db import DEFAULT_DB, connect  # noqa: E402
from leadfinder.seed import seed               # noqa: E402


def cmd_seed(a):
    db = Path(a.db)
    if db.exists() and a.force:
        for ext in ("", "-wal", "-shm"):
            Path(str(db) + ext).unlink(missing_ok=True)
    conn = connect(db)
    out = seed(conn, n=a.n, seed=a.seed)
    auth.seed_demo_accounts(conn)
    print(json.dumps(out) if out["practices"] else f"{db} already has practices; pass --force to start again")


def cmd_staff(a):
    conn = connect(a.db)
    password = getpass.getpass("Password (8 characters or more): ")
    print("added", auth.create_staff(conn, a.username, password, a.role, name=a.name or a.username))


def cmd_serve(a):
    import uvicorn
    from leadfinder.service.app import create_app
    uvicorn.run(create_app(a.db), host="127.0.0.1", port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--n", type=int, default=40); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("staff"); s.add_argument("what", choices=["add"]); s.add_argument("username")
    s.add_argument("--role", choices=list(auth.ROLES), default="sdr"); s.add_argument("--name", default=""); s.set_defaults(fn=cmd_staff)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
