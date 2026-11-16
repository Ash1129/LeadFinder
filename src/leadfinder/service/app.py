"""The web service: the pages an SDR and the sales manager use.

- Overview: what is waiting, the best leads not yet decided, and how the agents hand a lead along.
- Leads: every scored practice, by status, sortable.
- A lead: the practice's facts with their source, the score's reasons, the visits, and the drafted email, which the
  SDR may edit and then approve, or reject with a reason. Nothing is sent from here; an approval marks the email
  ready to send and is logged with the score and reasons that were on screen.
- Audit log (manager): every approval and rejection, who made it and what they saw.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import auth, outreach
from ..db import DEFAULT_DB, connect
from ..seed import seed
from . import security
from .security import SESSION_COOKIE, THEME_COOKIE, clear_session_cookie, lead_token, safe_next, set_session_cookie

HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))
REJECT_REASONS = {"fit": "not our kind of practice", "size": "too large: a health system buys this",
                  "customer": "already a customer", "contact": "wrong or missing contact", "other": "another reason"}
STATUSES = ("new", "approved", "rejected")


def when(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%d %b %Y %H:%M") if ts else ""


def ago(ts: float, now: Optional[float] = None) -> str:
    d = ((now or time.time()) - ts) / 86400
    return "today" if d < 1 else ("yesterday" if d < 2 else f"{int(d)} days ago")


TEMPLATES.env.filters["when"] = when
TEMPLATES.env.filters["ago"] = ago


def create_app(db_path=DEFAULT_DB, demo: Optional[bool] = None) -> FastAPI:
    """`demo`: on an empty database, fill it with synthetic practices and the demo logins. None reads
    LEADFINDER_DEMO (on unless set to 0)."""
    conn = connect(db_path)
    if demo if demo is not None else os.environ.get("LEADFINDER_DEMO", "1") != "0":
        seed(conn)
        auth.seed_demo_accounts(conn)
    auth.bootstrap_from_env(conn)

    app = FastAPI(title="LeadFinder", docs_url=None, redoc_url=None)
    app.middleware("http")(security.login_guard)  # the session cookie becomes request.state.me
    app.middleware("http")(security.guard)        # host and origin checks, security headers (runs first)
    app.state.conn = conn

    def theme(request: Request) -> str:
        return request.cookies.get(THEME_COOKIE, "system")

    @app.exception_handler(HTTPException)
    async def refused(request: Request, exc: HTTPException):
        me = getattr(request.state, "me", None)
        if exc.status_code in (403, 404) and me is not None and "text/html" in request.headers.get("accept", ""):
            return TEMPLATES.TemplateResponse(request, "refused.html", {"why": exc.detail if exc.status_code == 403 else "there is no such page",
                                                                        "name": me.name, "role": me.role_word, "theme": theme(request),
                                                                        "title": "Not your page" if exc.status_code == 403 else "Not found"},
                                              status_code=exc.status_code)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    def ctx(request: Request, **kw):
        me: auth.Principal = request.state.me
        new = conn.execute("SELECT COUNT(*) n FROM leads WHERE status='new'").fetchone()["n"]
        return {"request": request, "me": me, "can": {p: me.can(p) for p in auth.PERMISSIONS}, "theme": theme(request),
                "nav": "", "new_count": new, **kw}

    def allow(request: Request, permission: str) -> auth.Principal:
        me = request.state.me
        if not me.can(permission):
            raise HTTPException(403, f"your role ({me.role_word}) cannot do this")
        return me

    def lead_row(lead_id: int):
        row = conn.execute("SELECT l.*, p.npi, p.name, p.specialty, p.providers, p.city, p.state, p.ehr, p.contact_name, "
                           "p.contact_email, p.source FROM leads l JOIN practices p ON p.id = l.practice_id WHERE l.id=?",
                           (lead_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such lead")
        d = dict(row)
        d["reasons"] = json.loads(d["reasons"])
        return d

    # ---------------- logging in ----------------
    def login_view(request: Request, next_: str, error: str = "", status: int = 200):
        return TEMPLATES.TemplateResponse(request, "login.html", {"next": safe_next(next_), "demo": auth.demo_logins(conn),
                                                                  "demo_password": auth.DEMO_PASSWORD, "no_staff": not auth.staff_count(conn),
                                                                  "theme": theme(request), "error": error}, status_code=status)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/"):
        return login_view(request, next)

    @app.post("/login")
    def login(request: Request, username: str = Form(""), password: str = Form(""), next: str = Form("/")):
        me = auth.authenticate(conn, username, password)
        if me is None:
            return login_view(request, next, "That username and password do not match an active account "
                                             "(after five wrong tries an account is locked for fifteen minutes).", 401)
        auth.end_session(conn, request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse(safe_next(next), status_code=303)
        set_session_cookie(response, request, auth.start_session(conn, me))
        return response

    @app.post("/logout")
    def logout(request: Request):
        auth.end_session(conn, request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse("/login", status_code=303)
        clear_session_cookie(response)
        return response

    # ---------------- pages ----------------
    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request):
        allow(request, "view_leads")
        counts = {s: 0 for s in STATUSES}
        for r in conn.execute("SELECT status, COUNT(*) n FROM leads GROUP BY status"):
            counts[r["status"]] = r["n"]
        hot = conn.execute("SELECT COUNT(*) n FROM leads WHERE status='new' AND band='hot'").fetchone()["n"]
        top = [dict(r) | {"reasons": json.loads(r["reasons"])} for r in conn.execute(
            "SELECT l.id, l.score, l.band, l.reasons, p.name, p.specialty, p.providers, p.city, p.state FROM leads l "
            "JOIN practices p ON p.id = l.practice_id WHERE l.status='new' ORDER BY l.score DESC, l.id LIMIT 8")]
        stages = {
            "practices": conn.execute("SELECT COUNT(*) n FROM practices").fetchone()["n"],
            "visits": conn.execute("SELECT COUNT(*) n FROM visits").fetchone()["n"],
            "visited": conn.execute("SELECT COUNT(DISTINCT practice_id) n FROM visits").fetchone()["n"],
            "scored": sum(counts.values()),
        }
        return TEMPLATES.TemplateResponse(request, "overview.html", ctx(request, nav="overview", counts=counts, hot=hot, top=top,
                                                                        stages=stages, now=time.time()))

    @app.get("/leads", response_class=HTMLResponse)
    def leads(request: Request, status: str = "new"):
        allow(request, "view_leads")
        status = status if status in STATUSES + ("all",) else "new"
        where = "" if status == "all" else "WHERE l.status=?"
        rows = [dict(r) | {"reasons": json.loads(r["reasons"])} for r in conn.execute(
            f"SELECT l.id, l.score, l.band, l.status, l.reasons, p.name, p.specialty, p.providers, p.city, p.state, p.ehr, "
            f"(SELECT MAX(at) FROM visits v WHERE v.practice_id = p.id) last_visit FROM leads l "
            f"JOIN practices p ON p.id = l.practice_id {where} ORDER BY l.score DESC, l.id", () if status == "all" else (status,))]
        counts = {s: 0 for s in STATUSES}
        for r in conn.execute("SELECT status, COUNT(*) n FROM leads GROUP BY status"):
            counts[r["status"]] = r["n"]
        counts["all"] = sum(counts.values())
        return TEMPLATES.TemplateResponse(request, "leads.html", ctx(request, nav="leads", rows=rows, status=status, counts=counts,
                                                                     now=time.time()))

    @app.get("/lead/{lead_id}", response_class=HTMLResponse)
    def lead(request: Request, lead_id: int, stale: str = "", done: str = ""):
        allow(request, "view_leads")
        d = lead_row(lead_id)
        visits = [dict(r) for r in conn.execute("SELECT page, at FROM visits WHERE practice_id=? ORDER BY at DESC", (d["practice_id"],))]
        history = [dict(r) for r in conn.execute("SELECT a.*, s.name who FROM audit a LEFT JOIN staff s ON s.username = a.username "
                                                 "WHERE a.lead_id=? ORDER BY a.at DESC", (lead_id,))]
        return TEMPLATES.TemplateResponse(request, "lead.html", ctx(request, nav="leads", d=d, visits=visits, history=history,
                                                                    problems=outreach.check(d["subject"], d["body"]), token=lead_token(conn, lead_id),
                                                                    reject_reasons=REJECT_REASONS, stale=stale == "1", done=done,
                                                                    now=time.time()))

    def decide(request: Request, lead_id: int, seen: str, action: str, **kw):
        me = allow(request, "act_on_leads")
        d = lead_row(lead_id)
        if seen != lead_token(conn, lead_id) or d["status"] != "new":
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        conn.execute("INSERT INTO audit(lead_id, username, action, reason, note, score, reasons, subject, body, edited, at) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (lead_id, me.username, action, kw.get("reason"), kw.get("note") or None, d["score"], json.dumps(d["reasons"]),
                      kw.get("subject"), kw.get("body"), int(bool(kw.get("edited"))), time.time()))
        conn.execute("UPDATE leads SET status=?, subject=COALESCE(?, subject), body=COALESCE(?, body) WHERE id=?",
                     (action, kw.get("subject"), kw.get("body"), lead_id))
        return RedirectResponse(f"/lead/{lead_id}?done={action}", status_code=303)

    @app.post("/lead/{lead_id}/approve")
    def approve(request: Request, lead_id: int, subject: str = Form(""), body: str = Form(""), seen: str = Form("")):
        allow(request, "act_on_leads")
        subject, body = subject.strip(), body.replace("\r\n", "\n").strip()
        d = lead_row(lead_id)
        problems = outreach.check(subject, body)
        if problems:
            visits = [dict(r) for r in conn.execute("SELECT page, at FROM visits WHERE practice_id=? ORDER BY at DESC", (d["practice_id"],))]
            d = d | {"subject": subject, "body": body}
            return TEMPLATES.TemplateResponse(request, "lead.html", ctx(request, nav="leads", d=d, visits=visits, history=[],
                                                                        problems=problems, token=seen, reject_reasons=REJECT_REASONS,
                                                                        stale=False, done="", blocked=True, now=time.time()),
                                              status_code=422)
        edited = subject != d["subject"] or body != d["body"]
        return decide(request, lead_id, seen, "approved", subject=subject, body=body, edited=edited)

    @app.post("/lead/{lead_id}/reject")
    def reject(request: Request, lead_id: int, reason: str = Form(...), note: str = Form(""), seen: str = Form("")):
        allow(request, "act_on_leads")
        if reason not in REJECT_REASONS:
            raise HTTPException(400, "pick one of the reasons")
        return decide(request, lead_id, seen, "rejected", reason=reason, note=note.strip()[:500])

    @app.get("/audit", response_class=HTMLResponse)
    def audit(request: Request):
        allow(request, "view_audit")
        rows = [dict(r) for r in conn.execute("SELECT a.*, s.name who, p.name practice FROM audit a LEFT JOIN staff s ON s.username = a.username "
                                              "JOIN leads l ON l.id = a.lead_id JOIN practices p ON p.id = l.practice_id ORDER BY a.at DESC LIMIT 500")]
        return TEMPLATES.TemplateResponse(request, "audit.html", ctx(request, nav="audit", rows=rows, reject_reasons=REJECT_REASONS))

    @app.post("/theme")
    def set_theme(theme: str = Form("system"), next: str = Form("/")):
        """Light, dark or the system's, kept in a cookie so the server writes it into the page: no flash, no script."""
        r = RedirectResponse(safe_next(next), status_code=303)
        r.set_cookie(THEME_COOKIE, theme if theme in ("light", "dark") else "system", max_age=365 * 24 * 3600, samesite="lax")
        return r

    @app.get("/assets/app.js")
    def app_js():
        return FileResponse(HERE / "static" / "app.js", media_type="application/javascript")

    return app
