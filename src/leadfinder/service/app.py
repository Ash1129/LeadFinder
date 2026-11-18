"""The web service: the pages an SDR and the sales manager use.

- Overview: what is waiting, the best leads not yet decided, and how the agents hand a lead along.
- Leads: every scored practice, by status, sortable.
- A lead: the practice's facts with their source, the score's reasons, the visits, the call brief, and the first
  email. A lead starts with no email: the SDR asks for one with a button, may edit it and then approve it, or
  rejects the lead with a reason. The brief starts as a template. With a Claude credential set, Claude writes the
  email and can rewrite the brief, and code checks what comes back before it is kept (writer.py); without one the
  button fills the template. Nothing is sent from here; an approval marks the email ready to send and is logged
  with the score and reasons that were on screen and with who wrote the words.
- How it works: the path of a lead in five steps, who does each, and what is real today.
- Scoring model: the benchmark, every version of the model side by side with the scorecard that decided which is live,
  and what the live one learned.
- Audit log (manager): every approval and rejection, who made it and what they saw.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from fastapi.templating import Jinja2Templates

from .. import auth, feedback, ml, outreach, revision
from ..db import DEFAULT_DB, connect
from ..seed import apply_model, process_local_visit_handoff, seed, seed_demo_review_cases
from ..visit_analysis import PageView, VisitAnalysisInput, analyze_visits
from ..sender import Message, SendError, Sender, make_sender
from ..writer import Writer, WriterError, make_writer
from . import security
from .security import SESSION_COOKIE, THEME_COOKIE, clear_session_cookie, lead_token, safe_next, set_session_cookie

HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))
REJECT_REASONS = {"fit": "not our kind of practice", "size": "too large: a health system buys this",
                  "customer": "already a customer", "contact": "wrong or missing contact", "other": "another reason"}
STATUSES = ("new", "approved", "rejected")
REVIEW_STATUSES = ("pending", "identity_verified", "closed", "all")
CLOSE_REASONS = {"no_match": "no reliable practice match", "not_qualified": "not a qualified prospect",
                 "duplicate": "duplicate review", "other": "another reason"}
# Leads are ranked by the model's chance; the points table breaks ties and stands in for a lead the model has not scored.
RANK = "COALESCE(l.chance, l.score / 100.0) DESC, l.score DESC, l.id"
SHOWN = "COALESCE(l.chance_band, l.band) AS band"


class PageViewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, max_length=200)
    page_url: str = Field(min_length=1, max_length=2048)
    viewed_at: FiniteFloat
    seconds_on_page: FiniteFloat = Field(default=0.0, ge=0)


class VisitAnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    lead_id: str = Field(min_length=1, max_length=200)
    practice_name_raw: Optional[str] = Field(default=None, max_length=300)
    domain_or_org_hint: Optional[str] = Field(default=None, max_length=300)
    location_hint: Optional[str] = Field(default=None, max_length=300)
    company_identified: bool
    identification_confidence: FiniteFloat = Field(ge=0, le=1)
    page_views: list[PageViewPayload] = Field(max_length=1000)
    window_end: FiniteFloat
    patient_data_stripped: bool


def when(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%d %b %Y %H:%M") if ts else ""


def ago(ts: float, now: Optional[float] = None) -> str:
    d = ((now or time.time()) - ts) / 86400
    return "today" if d < 1 else ("yesterday" if d < 2 else f"{int(d)} days ago")


def review_token(row) -> str:
    return hashlib.sha256(f"{row['id']}|{row['review_status']}|{row['created_at']}".encode()).hexdigest()[:16]


TEMPLATES.env.filters["when"] = when
TEMPLATES.env.filters["ago"] = ago


def why(row) -> dict:
    """A list row with the one reason to show beside it: the model's strongest push up, else the points table's."""
    d = dict(row) | {"reasons": json.loads(row["reasons"])}
    ups = [r for r in json.loads(row["chance_reasons"] or "[]") if r["effect"] > 0]
    d["why"] = ups[0]["label"] if ups else d["reasons"][0]["label"]
    return d


def create_app(db_path=DEFAULT_DB, demo: Optional[bool] = None, writer: Optional[Writer] = None,
               sender: Optional[Sender] = None, revisions: Optional[int] = None) -> FastAPI:
    """`demo`: on an empty database, fill it with synthetic practices and the demo logins. None reads
    LEADFINDER_DEMO (on unless set to 0). `writer`: who writes emails and call briefs on request; None picks Claude
    when a credential is set and the templates otherwise (writer.make_writer). `sender`: the send step; None reads
    LEADFINDER_SENDER, which is off unless set (sender.make_sender). `revisions`: how many times a draft that fails
    the checks is rewritten, 0 to 2; None reads LEADFINDER_REVISIONS, 0 unless set (revision.py)."""
    conn = connect(db_path)
    writer = writer or make_writer()
    sender = sender if sender is not None else make_sender()
    revisions = revision.revisions_from_env() if revisions is None else max(0, min(int(revisions), revision.MAX_REVISIONS))
    ml.ensure(conn)                     # the trained scorer: fitted on first start (a few seconds), then read from the database
    demo_mode = demo if demo is not None else os.environ.get("LEADFINDER_DEMO", "1") != "0"
    if demo_mode:
        seed(conn)
        auth.seed_demo_accounts(conn)
        seed_demo_review_cases(conn)
    apply_model(conn)                   # leads from before the model, or scored by an older one
    auth.bootstrap_from_env(conn)

    app = FastAPI(title="LeadFinder", docs_url=None, redoc_url=None)
    app.middleware("http")(security.login_guard)  # the session cookie becomes request.state.me
    app.middleware("http")(security.guard)        # host and origin checks, security headers (runs first)
    app.state.conn = conn
    app.state.writer = writer
    app.state.sender = sender

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
        review_count = conn.execute(
            "SELECT COUNT(*) n FROM visit_analyses WHERE routing_decision='human_review' AND review_status='pending'"
        ).fetchone()["n"]
        return {"request": request, "me": me, "can": {p: me.can(p) for p in auth.PERMISSIONS}, "theme": theme(request),
                "nav": "", "new_count": new, "review_count": review_count, "writer": writer, "model": ml.load(conn), **kw}

    def allow(request: Request, permission: str) -> auth.Principal:
        me = request.state.me
        if not me.can(permission):
            raise HTTPException(403, f"your role ({me.role_word}) cannot do this")
        return me

    def stages() -> dict:
        """How far along the path the leads are: the counts beside each step on the overview and on How it works."""
        one = lambda sql: conn.execute(sql).fetchone()["n"]                                             # noqa: E731
        return {"visits": one("SELECT COUNT(*) n FROM visits"), "visited": one("SELECT COUNT(DISTINCT practice_id) n FROM visits"),
                "practices": one("SELECT COUNT(*) n FROM practices"), "scored": one("SELECT COUNT(*) n FROM leads"),
                "emails": one("SELECT COUNT(*) n FROM leads WHERE body != ''"), "briefs": one("SELECT COUNT(*) n FROM leads WHERE brief IS NOT NULL"),
                "decided": one("SELECT COUNT(*) n FROM leads WHERE status != 'new'")}

    def lead_row(lead_id: int):
        row = conn.execute("SELECT l.*, p.npi, p.name, p.specialty, p.providers, p.city, p.state, p.ehr, p.contact_name, "
                           "p.contact_email, p.source, l.band AS points_band, COALESCE(l.chance_band, l.band) AS shown_band "
                           "FROM leads l JOIN practices p ON p.id = l.practice_id WHERE l.id=?",
                           (lead_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such lead")
        d = dict(row)
        d["reasons"] = json.loads(d["reasons"])
        d["brief"] = json.loads(d["brief"]) if d["brief"] else None
        d["chance_reasons"] = json.loads(d["chance_reasons"]) if d["chance_reasons"] else []
        return d

    def visit_pages(d: dict) -> list[str]:
        return [r["page"] for r in conn.execute("SELECT page FROM visits WHERE practice_id=? ORDER BY at DESC", (d["practice_id"],))]

    def log_attempts(request: Request, lead_id: int, purpose: str, attempts: list, kept: bool) -> None:
        """Every draft the checks saw, passed or not; the one kept on the lead is marked (feedback.py reads these)."""
        if not attempts:
            return
        round_id, now = uuid.uuid4().hex, time.time()
        for a in attempts:
            conn.execute("INSERT INTO draft_attempts(lead_id, round, purpose, attempt, writer, subject, body, problems, passed, kept, "
                         "username, at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (lead_id, round_id, purpose, a.n, writer.name, a.subject, a.body, json.dumps(a.problems), int(a.passed),
                          int(kept and a is attempts[-1]), request.state.me.username, now))

    def lead_page(request: Request, d: dict, status_code: int = 200, **kw):
        """The lead's page, for the plain view and for every way a write on it can come back refused."""
        visits = [dict(r) for r in conn.execute("SELECT page, at FROM visits WHERE practice_id=? ORDER BY at DESC", (d["practice_id"],))]
        history = [dict(r) for r in conn.execute("SELECT a.*, s.name who FROM audit a LEFT JOIN staff s ON s.username = a.username "
                                                 "WHERE a.lead_id=? ORDER BY a.at DESC", (d["id"],))]
        drafted = bool(d["subject"] or d["body"])       # a lead has no email until someone asks for one
        send_backs = [dict(r) for r in conn.execute("SELECT s.*, st.name who FROM send_backs s LEFT JOIN staff st ON st.username = s.username "
                                                    "WHERE s.lead_id=? ORDER BY s.at DESC", (d["id"],))]
        outbox = conn.execute("SELECT * FROM outbox WHERE lead_id=?", (d["id"],)).fetchone()
        kw.setdefault("send_backs", send_backs)
        kw.setdefault("send_backs_left", max(0, revision.MAX_SEND_BACKS - len(send_backs)))
        kw.setdefault("can_send_back", revision.can_revise(writer))
        kw.setdefault("sender", sender)
        kw.setdefault("outbox", dict(outbox) if outbox else None)
        base = dict(nav="leads", d=d, visits=visits, history=history, drafted=drafted, sent_back=False, queued=False, send_error="",
                    problems=outreach.check(d["subject"], d["body"]) if drafted else [],
                    token=lead_token(conn, d["id"]), reject_reasons=REJECT_REASONS, stale=False, done="", wrote="",
                    write_error="", blocked=False, now=time.time())
        return TEMPLATES.TemplateResponse(request, "lead.html", ctx(request, **(base | kw)), status_code=status_code)

    def review_row(review_id: int):
        row = conn.execute(
            "SELECT v.*, p.name AS practice_name, p.city, p.state, s.name AS reviewer, "
            "h.practice_record_json, h.identity_method "
            "FROM visit_analyses v LEFT JOIN practices p ON p.id=v.practice_id "
            "LEFT JOIN staff s ON s.username=v.resolved_by "
            "LEFT JOIN lead_handoffs h ON h.request_id=v.request_id "
            "WHERE v.id=? AND v.routing_decision='human_review'",
            (review_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no such visit review")
        d = dict(row)
        d["analysis"] = json.loads(d["result_json"])
        d["handoff"] = json.loads(d["practice_record_json"]) if d["practice_record_json"] else None
        d["lead"] = conn.execute("SELECT id, score, chance, model_version FROM leads WHERE practice_id=?", (d["practice_id"],)).fetchone() \
            if d["practice_id"] is not None else None
        return d

    def exact_practice_match(data: VisitAnalysisInput) -> Optional[int]:
        """Only auto-link an exact local name match, narrowed by location when supplied."""
        if not data.practice_name_raw:
            return None
        rows = conn.execute("SELECT id, name, city, state FROM practices ORDER BY id").fetchall()
        matches = [row for row in rows if row["name"].casefold() == data.practice_name_raw.strip().casefold()]
        if data.location_hint:
            parts = [part.strip().casefold() for part in data.location_hint.split(",")]
            if len(parts) >= 2:
                matches = [row for row in matches
                           if row["city"].casefold() == parts[0] and row["state"].casefold() == parts[1]]
            else:
                matches = [row for row in matches if row["city"].casefold() == parts[0]]
        return matches[0]["id"] if len(matches) == 1 else None

    def run_handoff(request: Request, review, practice_id: int, identity_method: str) -> dict:
        return process_local_visit_handoff(
            conn,
            request_id=review["request_id"],
            visitor_ref=review["lead_id"],
            practice_id=practice_id,
            analysis=review["analysis"],
            identity_method=identity_method,
            created_by=request.state.me.username,
            is_demo=bool(review["is_demo"]),
        )

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
        hot = conn.execute("SELECT COUNT(*) n FROM leads WHERE status='new' AND COALESCE(chance_band, band)='hot'").fetchone()["n"]
        top = [why(r) for r in conn.execute(
            f"SELECT l.id, l.score, {SHOWN}, l.chance, l.chance_reasons, l.reasons, p.name, p.specialty, p.providers, p.city, p.state "
            f"FROM leads l JOIN practices p ON p.id = l.practice_id WHERE l.status='new' ORDER BY {RANK} LIMIT 8")]
        return TEMPLATES.TemplateResponse(request, "overview.html", ctx(request, nav="overview", counts=counts, hot=hot, top=top,
                                                                        stages=stages(), now=time.time()))

    @app.get("/how", response_class=HTMLResponse)
    def how(request: Request):
        allow(request, "view_leads")
        return TEMPLATES.TemplateResponse(request, "how.html", ctx(request, nav="how", stages=stages(),
                                                                   reject_reasons=REJECT_REASONS))

    @app.get("/leads", response_class=HTMLResponse)
    def leads(request: Request, status: str = "new"):
        allow(request, "view_leads")
        status = status if status in STATUSES + ("all",) else "new"
        where = "" if status == "all" else "WHERE l.status=?"
        rows = [why(r) for r in conn.execute(
            f"SELECT l.id, l.score, {SHOWN}, l.chance, l.chance_reasons, l.status, l.reasons, p.name, p.specialty, p.providers, p.city, "
            f"p.state, p.ehr, (SELECT MAX(at) FROM visits v WHERE v.practice_id = p.id) last_visit FROM leads l "
            f"JOIN practices p ON p.id = l.practice_id {where} ORDER BY {RANK}", () if status == "all" else (status,))]
        counts = {s: 0 for s in STATUSES}
        for r in conn.execute("SELECT status, COUNT(*) n FROM leads GROUP BY status"):
            counts[r["status"]] = r["n"]
        counts["all"] = sum(counts.values())
        return TEMPLATES.TemplateResponse(request, "leads.html", ctx(request, nav="leads", rows=rows, status=status, counts=counts,
                                                                     now=time.time()))

    @app.get("/review-queue", response_class=HTMLResponse)
    def review_queue(request: Request, status: str = "pending"):
        allow(request, "view_leads")
        status = status if status in REVIEW_STATUSES else "pending"
        where = "AND v.review_status=?" if status != "all" else ""
        params = (status,) if status != "all" else ()
        rows = []
        for row in conn.execute(
            "SELECT v.id, v.lead_id, v.created_at, v.review_status, v.is_demo, v.resolution_note, v.result_json, "
            "v.practice_id, p.name AS practice_name, p.city, p.state "
            "FROM visit_analyses v LEFT JOIN practices p ON p.id=v.practice_id "
            f"WHERE v.routing_decision='human_review' {where} ORDER BY "
            "CASE v.review_status WHEN 'pending' THEN 0 ELSE 1 END, v.created_at DESC",
            params,
        ):
            item = dict(row)
            item["analysis"] = json.loads(item.pop("result_json"))
            rows.append(item)
        counts = {s: 0 for s in REVIEW_STATUSES if s != "all"}
        for row in conn.execute(
            "SELECT review_status, COUNT(*) n FROM visit_analyses "
            "WHERE routing_decision='human_review' GROUP BY review_status"
        ):
            counts[row["review_status"]] = row["n"]
        counts["all"] = sum(counts.values())
        return TEMPLATES.TemplateResponse(
            request, "review_queue.html",
            ctx(request, nav="review", rows=rows, status=status, counts=counts),
        )

    @app.get("/review/{review_id}", response_class=HTMLResponse)
    def visit_review(request: Request, review_id: int, stale: str = "", done: str = ""):
        allow(request, "view_leads")
        d = review_row(review_id)
        practices = [dict(row) for row in conn.execute(
            "SELECT id, name, specialty, city, state FROM practices ORDER BY name, city"
        )]
        return TEMPLATES.TemplateResponse(
            request, "visit_review.html",
            ctx(
                request, nav="review", d=d, practices=practices, token=review_token(d),
                stale=stale == "1", done=done if done in ("identity_verified", "closed") else "",
                close_reasons=CLOSE_REASONS,
            ),
        )

    @app.post("/api/visit-analysis")
    def submit_visit_analysis(request: Request, payload: VisitAnalysisPayload):
        """Run the deterministic gate for an authenticated upstream caller and persist scrubbed review cases."""
        allow(request, "act_on_leads")
        if not payload.request_id.strip() or not payload.lead_id.strip():
            raise HTTPException(422, "request_id and lead_id cannot be blank")
        request_id = payload.request_id.strip()
        lead_id = payload.lead_id.strip()
        data = VisitAnalysisInput(
            lead_id=lead_id,
            practice_name_raw=payload.practice_name_raw,
            domain_or_org_hint=payload.domain_or_org_hint,
            location_hint=payload.location_hint,
            company_identified=payload.company_identified,
            identification_confidence=payload.identification_confidence,
            page_views=tuple(PageView(**view.model_dump()) for view in payload.page_views),
            window_end=payload.window_end,
            patient_data_stripped=payload.patient_data_stripped,
        )
        result = analyze_visits(data)
        result_json = json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        review_id = None
        persisted = False
        review_status = None
        if payload.patient_data_stripped and result.routing_decision == "research":
            practice_id = exact_practice_match(data)
            if practice_id is None:
                return {
                    "analysis_id": None,
                    "persisted": False,
                    "routing_decision": result.routing_decision,
                    "review_status": None,
                    "handoff_status": "local_practice_not_found",
                    "message": "No unique exact match was found in the local practice table; no lead handoff occurred.",
                    "analysis": result.as_dict(),
                }
            try:
                handoff = process_local_visit_handoff(
                    conn,
                    request_id=request_id,
                    visitor_ref=lead_id,
                    practice_id=practice_id,
                    analysis=result.as_dict(),
                    identity_method="automatic",
                    created_by=request.state.me.username,
                )
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {
                "analysis_id": handoff["id"],
                "persisted": True,
                "routing_decision": result.routing_decision,
                "review_status": None,
                "handoff_status": "complete",
                "handoff": {
                    "identity_method": "automatic",
                    "practice_id": practice_id,
                    "lead_record_id": handoff["lead_record_id"],
                    "score": handoff["score"],
                    "chance": handoff["chance"],
                    "model_version": handoff["model_version"],
                    "practice_record": json.loads(handoff["practice_record_json"]),
                },
                "analysis": result.as_dict(),
            }
        if payload.patient_data_stripped and result.routing_decision == "human_review":
            existing = conn.execute(
                "SELECT id, result_json, review_status FROM visit_analyses WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                if existing["result_json"] != result_json:
                    raise HTTPException(409, "request_id was already used for a different visit analysis")
                review_id, persisted, review_status = existing["id"], True, existing["review_status"]
            else:
                review_status = "pending"
                try:
                    cur = conn.execute(
                        "INSERT INTO visit_analyses(request_id, lead_id, routing_decision, review_status, result_json, "
                        "created_by, created_at) VALUES(?,?,?,?,?,?,?)",
                        (request_id, lead_id, result.routing_decision, review_status, result_json,
                         request.state.me.username, time.time()),
                    )
                    review_id, persisted = cur.lastrowid, True
                except sqlite3.IntegrityError:
                    # The unique key makes concurrent retries safe; surface unrelated storage failures.
                    existing = conn.execute(
                        "SELECT id, result_json, review_status FROM visit_analyses WHERE request_id=?", (request_id,)
                    ).fetchone()
                    if existing is None:
                        raise
                    if existing["result_json"] != result_json:
                        raise HTTPException(409, "request_id was already used for a different visit analysis")
                    review_id, persisted, review_status = existing["id"], True, existing["review_status"]
        return {
            "analysis_id": review_id,
            "persisted": persisted,
            "routing_decision": result.routing_decision,
            "review_status": review_status,
            "analysis": result.as_dict(),
        }

    def decide_visit_review(request: Request, review_id: int, seen: str, decision: str, **kw):
        me = allow(request, "act_on_leads")
        d = review_row(review_id)
        if seen != review_token(d) or d["review_status"] != "pending":
            return RedirectResponse(f"/review/{review_id}?stale=1", status_code=303)
        cur = conn.execute(
            "UPDATE visit_analyses SET review_status=?, practice_id=?, resolved_by=?, resolved_at=?, "
            "resolution_note=? WHERE id=? AND routing_decision='human_review' AND review_status='pending'",
            (decision, kw.get("practice_id"), me.username, time.time(), kw.get("note"), review_id),
        )
        if cur.rowcount != 1:
            return RedirectResponse(f"/review/{review_id}?stale=1", status_code=303)
        if decision == "identity_verified":
            try:
                handoff = run_handoff(request, d, kw["practice_id"], "human_review")
            except (ValueError, LookupError, RuntimeError, sqlite3.Error) as exc:
                conn.execute(
                    "UPDATE visit_analyses SET handoff_status='failed', handoff_note=? WHERE id=?",
                    (str(exc)[:500], review_id),
                )
            else:
                conn.execute(
                    "UPDATE visit_analyses SET handoff_status='complete', lead_record_id=?, handoff_note=NULL WHERE id=?",
                    (handoff["lead_record_id"], review_id),
                )
        return RedirectResponse(f"/review/{review_id}?done={decision}", status_code=303)

    @app.post("/review/{review_id}/retry-handoff")
    def retry_visit_handoff(request: Request, review_id: int):
        allow(request, "act_on_leads")
        d = review_row(review_id)
        if d["review_status"] != "identity_verified" or d["handoff_status"] != "failed":
            return RedirectResponse(f"/review/{review_id}?stale=1", status_code=303)
        try:
            handoff = run_handoff(request, d, d["practice_id"], "human_review")
        except (ValueError, LookupError, RuntimeError, sqlite3.Error) as exc:
            conn.execute(
                "UPDATE visit_analyses SET handoff_note=? WHERE id=?", (str(exc)[:500], review_id)
            )
            return RedirectResponse(f"/review/{review_id}?handoff=failed", status_code=303)
        conn.execute(
            "UPDATE visit_analyses SET handoff_status='complete', lead_record_id=?, handoff_note=NULL WHERE id=?",
            (handoff["lead_record_id"], review_id),
        )
        return RedirectResponse(f"/review/{review_id}?handoff=complete", status_code=303)

    @app.post("/review/{review_id}/verify")
    def verify_visit_identity(
        request: Request,
        review_id: int,
        practice_id: int = Form(...),
        note: str = Form(""),
        seen: str = Form(""),
    ):
        allow(request, "act_on_leads")
        if conn.execute("SELECT 1 FROM practices WHERE id=?", (practice_id,)).fetchone() is None:
            raise HTTPException(400, "choose an existing practice")
        return decide_visit_review(
            request, review_id, seen, "identity_verified", practice_id=practice_id, note=note.strip()[:500] or None
        )

    @app.post("/review/{review_id}/close")
    def close_visit_review(
        request: Request,
        review_id: int,
        reason: str = Form(...),
        note: str = Form(""),
        seen: str = Form(""),
    ):
        allow(request, "act_on_leads")
        if reason not in CLOSE_REASONS:
            raise HTTPException(400, "pick one of the close reasons")
        return decide_visit_review(
            request, review_id, seen, "closed", note=f"{CLOSE_REASONS[reason]}: {note.strip()[:400]}".rstrip(": ")
        )

    @app.get("/lead/{lead_id}", response_class=HTMLResponse)
    def lead(request: Request, lead_id: int, stale: str = "", done: str = "", wrote: str = "", sentback: str = "", queued: str = ""):
        allow(request, "view_leads")
        return lead_page(request, lead_row(lead_id), stale=stale == "1", done=done, wrote=wrote if wrote in ("email", "brief") else "",
                         sent_back=sentback == "1", queued=queued == "1")

    @app.post("/lead/{lead_id}/write")
    def write(request: Request, lead_id: int, what: str = Form(...)):
        """Have the writer (Claude, or the templates) write this lead's first email or its call brief. The email
        only while the lead is undecided; what was approved or rejected stays as it was."""
        allow(request, "act_on_leads")
        d = lead_row(lead_id)
        if what not in ("email", "brief"):
            raise HTTPException(400, "write the email or the brief")
        if what == "email" and d["status"] != "new":
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        pages = visit_pages(d)
        try:
            if what == "email":
                try:
                    drafted = revision.write_email(writer, d, d["reasons"], pages, revisions=revisions)
                except WriterError as e:
                    log_attempts(request, lead_id, "write", getattr(e, "attempts", []), kept=False)
                    raise
                log_attempts(request, lead_id, "write", drafted.attempts, kept=True)
                conn.execute("UPDATE leads SET subject=?, body=?, written_by=? WHERE id=? AND status='new'",
                             (drafted.subject, drafted.body, writer.name, lead_id))
            else:
                brief = writer.brief(d, d["reasons"], pages)
                conn.execute("UPDATE leads SET brief=?, brief_by=? WHERE id=?", (json.dumps(brief), writer.name, lead_id))
        except WriterError as e:
            return lead_page(request, d, status_code=502, write_error=f"{str(e)[:1].upper()}{str(e)[1:]}. The {what} is unchanged.")
        return RedirectResponse(f"/lead/{lead_id}?wrote={what}", status_code=303)

    def decide(request: Request, lead_id: int, seen: str, action: str, **kw):
        me = allow(request, "act_on_leads")
        d = lead_row(lead_id)
        if seen != lead_token(conn, lead_id) or d["status"] != "new":
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        conn.execute("INSERT INTO audit(lead_id, username, action, reason, note, score, reasons, subject, body, edited, at, written_by, "
                     "chance, model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (lead_id, me.username, action, kw.get("reason"), kw.get("note") or None, d["score"], json.dumps(d["reasons"]),
                      kw.get("subject"), kw.get("body"), int(bool(kw.get("edited"))), time.time(), d["written_by"], d["chance"],
                      d["model_version"]))
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
            return lead_page(request, d | {"subject": subject, "body": body}, status_code=422, problems=problems, token=seen, blocked=True)
        edited = subject != d["subject"] or body != d["body"]
        return decide(request, lead_id, seen, "approved", subject=subject, body=body, edited=edited)

    @app.post("/lead/{lead_id}/reject")
    def reject(request: Request, lead_id: int, reason: str = Form(...), note: str = Form(""), seen: str = Form("")):
        allow(request, "act_on_leads")
        if reason not in REJECT_REASONS:
            raise HTTPException(400, "pick one of the reasons")
        return decide(request, lead_id, seen, "rejected", reason=reason, note=note.strip()[:500])

    @app.post("/lead/{lead_id}/send-back")
    def send_back(request: Request, lead_id: int, note: str = Form(""), seen: str = Form("")):
        """Send the drafted email back with a note: Claude rewrites it from the note, the rewrite passes the same checks
        (and the same bounded revision loop), and it replaces the draft. The lead stays undecided. At most
        MAX_SEND_BACKS per lead; each one is logged, rewritten or not. Not an approval or a rejection, so the audit
        log and its counts are untouched."""
        me = allow(request, "act_on_leads")
        d = lead_row(lead_id)
        if seen != lead_token(conn, lead_id) or d["status"] != "new":
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        if not revision.can_revise(writer):
            raise HTTPException(400, "the template cannot be sent back; edit the draft yourself")
        if not (d["subject"] or d["body"]):
            raise HTTPException(400, "there is no draft to send back yet")
        try:
            note = revision.clean_note(note)
        except ValueError as e:
            return lead_page(request, d, status_code=422, write_error=f"{str(e)[:1].upper()}{str(e)[1:]}.")
        used = conn.execute("SELECT COUNT(*) n FROM send_backs WHERE lead_id=?", (lead_id,)).fetchone()["n"]
        if used >= revision.MAX_SEND_BACKS:
            return lead_page(request, d, status_code=409, write_error=f"This draft has been sent back {used} times, the most allowed. "
                                                                    "Edit it yourself, approve it or reject the lead.")
        try:
            drafted = revision.write_email(writer, d, d["reasons"], visit_pages(d), revisions=revisions,
                                           note=note, previous=(d["subject"], d["body"]))
        except WriterError as e:
            log_attempts(request, lead_id, "send_back", getattr(e, "attempts", []), kept=False)
            conn.execute("INSERT INTO send_backs(lead_id, username, note, subject, body, outcome, detail, at) VALUES(?,?,?,?,?,?,?,?)",
                         (lead_id, me.username, note, d["subject"], d["body"], "failed", str(e)[:500], time.time()))
            return lead_page(request, lead_row(lead_id), status_code=502,
                             write_error=f"{str(e)[:1].upper()}{str(e)[1:]}. The draft is unchanged.")
        log_attempts(request, lead_id, "send_back", drafted.attempts, kept=True)
        updated = conn.execute("UPDATE leads SET subject=?, body=?, written_by=? WHERE id=? AND status='new'",
                               (drafted.subject, drafted.body, writer.name, lead_id)).rowcount
        if not updated:                                  # decided while Claude was writing: the decision stands
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        conn.execute("INSERT INTO send_backs(lead_id, username, note, subject, body, outcome, detail, at) VALUES(?,?,?,?,?,?,?,?)",
                     (lead_id, me.username, note, d["subject"], d["body"], "rewritten",
                      f"{drafted.revisions} revision{'s' if drafted.revisions != 1 else ''} after the checks", time.time()))
        return RedirectResponse(f"/lead/{lead_id}?sentback=1", status_code=303)

    @app.post("/lead/{lead_id}/send")
    def send(request: Request, lead_id: int):
        """Hand an approved email to the send step. Off unless LEADFINDER_SENDER is set; the only sender today is the
        dry run, which records the message in the outbox and sends nothing. Once per lead."""
        me = allow(request, "act_on_leads")
        if sender is None:
            raise HTTPException(404, "sending is off")
        d = lead_row(lead_id)
        if d["status"] != "approved":
            return RedirectResponse(f"/lead/{lead_id}?stale=1", status_code=303)
        if conn.execute("SELECT 1 FROM outbox WHERE lead_id=?", (lead_id,)).fetchone():
            return RedirectResponse(f"/lead/{lead_id}?queued=1", status_code=303)
        message = Message(d["contact_name"], d["contact_email"] or "", d["subject"], d["body"])
        try:
            receipt = sender.send(message)
        except SendError as e:
            return lead_page(request, d, status_code=422, send_error=f"{str(e)[:1].upper()}{str(e)[1:]}.")
        conn.execute("INSERT OR IGNORE INTO outbox(lead_id, to_name, to_email, subject, body, sender, status, detail, queued_by, queued_at) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (lead_id, message.to_name, message.to_email, message.subject, message.body, sender.name, receipt.status,
                      receipt.detail, me.username, time.time()))
        return RedirectResponse(f"/lead/{lead_id}?queued=1", status_code=303)

    @app.get("/api/outreach-feedback")
    def outreach_feedback(request: Request):
        """The feedback summary as JSON, for evaluation (feedback.py). Manager only, like the audit log."""
        allow(request, "view_audit")
        return JSONResponse(feedback.summary(conn))

    @app.get("/audit", response_class=HTMLResponse)
    def audit(request: Request):
        allow(request, "view_audit")
        rows = [dict(r) for r in conn.execute("SELECT a.*, s.name who, p.name practice FROM audit a LEFT JOIN staff s ON s.username = a.username "
                                              "JOIN leads l ON l.id = a.lead_id JOIN practices p ON p.id = l.practice_id ORDER BY a.at DESC LIMIT 500")]
        return TEMPLATES.TemplateResponse(request, "audit.html", ctx(request, nav="audit", rows=rows, reject_reasons=REJECT_REASONS,
                                                                      fb=feedback.summary(conn)))

    @app.get("/model", response_class=HTMLResponse)
    def model_page(request: Request):
        allow(request, "view_leads")
        model = ml.load(conn)
        allowances = {key: (f"{a * 100:g} points" if key in ("top", "ece") else f"{a:g}").replace("1 points", "1 point")
                      for key, _, _, a, _ in ml.MEASURES}
        return TEMPLATES.TemplateResponse(request, "model.html", ctx(
            request, nav="model", learned=ml.learned(model) if model else [], runs=ml.side_by_side(conn), measures=ml.MEASURES,
            allowances=allowances, first=ml.FIRST, per_month=ml.MONTH,
            marks={"better": "better", "same": "the same", "worse": "worse, allowed", "blocked": "worse, too far"}))

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
