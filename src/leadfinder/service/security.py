"""What keeps the pages and their writes safe from other sites, and an SDR's decision from going stale.

A browser will send a form to any site it is told to, so every write checks where it came from:

- Host. Only names in `LEADFINDER_ALLOWED_HOSTS` (default: this machine's loopback names, plus Starlette's test
  client) are served, so a hostile domain that re-points its DNS at 127.0.0.1 cannot reach the pages.
- Origin. A write whose Origin (or, failing that, Referer) is another site is refused. Requests with neither are
  not from a browser page (curl, a script) and pass.
- Headers. Every response says nosniff, refuses framing, and carries a content security policy that allows scripts
  only from this service (the one script is /assets/app.js; the pages have no inline script).
- Stale decisions. The approve and reject forms carry a token for the lead as the SDR saw it; if someone else
  decided it since, the write is refused and the page says why, instead of two decisions both landing.
- Login. Every page and write but the login page, the script and the theme needs a session (auth.py).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import Optional
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from .. import auth

DEFAULT_HOSTS = ("127.0.0.1", "localhost", "[::1]", "testserver")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
UNPOLICED_PATHS = ("/docs", "/redoc")          # FastAPI's API docs load their scripts from a CDN
SESSION_COOKIE = "leadfinder_session"
THEME_COOKIE = "leadfinder_theme"
OPEN_PATHS = ("/login", "/logout", "/theme", "/favicon.ico")   # reachable without a session
OPEN_PREFIXES = ("/assets/",)
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; "
       "frame-ancestors 'none'")
HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin"}


def allowed_hosts(env: Optional[dict] = None) -> set[str]:
    env = os.environ if env is None else env
    raw = (env.get("LEADFINDER_ALLOWED_HOSTS") or "").strip()
    return {h.strip().lower() for h in raw.split(",") if h.strip()} if raw else set(DEFAULT_HOSTS)


def host_name(host_header: str) -> str:
    """The host without its port; IPv6 literals keep their brackets."""
    h = (host_header or "").strip().lower()
    if h.startswith("["):
        return h[:h.find("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if ":" in h else h


def same_origin(request: Request) -> bool:
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}".lower()
    origin = request.headers.get("origin")
    if origin is not None:
        return origin.lower() == expected               # "null" (sandboxed frames, files) never matches
    referer = request.headers.get("referer")
    if referer is not None:
        r = referer.lower()
        return r == expected or r.startswith(expected + "/")
    return True


async def guard(request: Request, call_next) -> Response:
    if host_name(request.headers.get("host", "")) not in allowed_hosts():
        return PlainTextResponse("Host not allowed. Add it to LEADFINDER_ALLOWED_HOSTS if this deployment should answer to it.",
                                 status_code=400)
    if request.method in UNSAFE_METHODS and not same_origin(request):
        return PlainTextResponse("Writes from another site are refused.", status_code=403)
    response = await call_next(request)
    for k, v in HEADERS.items():
        response.headers.setdefault(k, v)
    if not request.url.path.startswith(UNPOLICED_PATHS):
        response.headers.setdefault("Content-Security-Policy", CSP)
    return response


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=auth.MAX_SECONDS, httponly=True, samesite="lax", path="/",
                        secure=request.url.scheme == "https")


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def safe_next(path: str) -> str:
    """A path on this service to come back to after logging in; anything else is the front page."""
    return path if path.startswith("/") and not path.startswith("//") and "\\" not in path else "/"


async def login_guard(request: Request, call_next) -> Response:
    """Turn the session cookie into `request.state.me`; without one, a page goes to the login page and a write is refused."""
    conn = request.app.state.conn
    me = auth.load_session(conn, request.cookies.get(SESSION_COOKIE))
    request.state.me = me
    path = request.url.path
    if me is None and path not in OPEN_PATHS and not path.startswith(OPEN_PREFIXES):
        if request.method == "GET":
            back = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(f"/login?next={quote(back, safe='')}", status_code=303)
        return PlainTextResponse("Log in first.", status_code=401)
    return await call_next(request)


def lead_token(conn: sqlite3.Connection, lead_id: int) -> str:
    """Changes when the lead's status changes or anyone records a decision on it."""
    row = conn.execute("SELECT status FROM leads WHERE id=?", (lead_id,)).fetchone()
    n = conn.execute("SELECT COUNT(*) n FROM audit WHERE lead_id=?", (lead_id,)).fetchone()["n"]
    return hashlib.sha256(f"{lead_id}|{row['status'] if row else ''}|{n}".encode()).hexdigest()[:16]
