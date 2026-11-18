"""What the SDRs' decisions and the draft checks say about the outreach, in one summary.

It reads what is already logged (the audit table, plus draft_attempts, send_backs and outbox) and changes
nothing. It is evidence for the course rubric (draft-check failures, how often a draft is approved as written,
edited, sent back or rejected, and why) and the feedback format the team can evaluate against.

It does not retrain or re-weight anything. The model learning from SDR feedback is still open (README).
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter


def _problem_kind(problem: str) -> str:
    """'it makes a claim the seller cannot back (“guarantee”)' -> 'it makes a claim the seller cannot back'."""
    return re.split(r"\s*\(", problem, maxsplit=1)[0].strip()


def _rate(part: int, whole: int):
    return round(part / whole, 3) if whole else None


def summary(conn: sqlite3.Connection) -> dict:
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]                                              # noqa: E731

    approved = one("SELECT COUNT(*) FROM audit WHERE action='approved'")
    edited = one("SELECT COUNT(*) FROM audit WHERE action='approved' AND edited=1")
    rejected = one("SELECT COUNT(*) FROM audit WHERE action='rejected'")
    decided = approved + rejected
    by_reason = {r["reason"]: r["n"] for r in conn.execute(
        "SELECT reason, COUNT(*) n FROM audit WHERE action='rejected' GROUP BY reason ORDER BY n DESC")}
    by_writer = {}
    for r in conn.execute("SELECT COALESCE(written_by, 'none') w, action, edited, COUNT(*) n FROM audit GROUP BY w, action, edited"):
        w = by_writer.setdefault(r["w"], {"approved_as_drafted": 0, "approved_edited": 0, "rejected": 0})
        key = "rejected" if r["action"] == "rejected" else ("approved_edited" if r["edited"] else "approved_as_drafted")
        w[key] += r["n"]

    sent_back = one("SELECT COUNT(*) FROM send_backs")
    rewritten = one("SELECT COUNT(*) FROM send_backs WHERE outcome='rewritten'")
    leads_sent_back = one("SELECT COUNT(DISTINCT lead_id) FROM send_backs")
    approved_after = one("SELECT COUNT(DISTINCT s.lead_id) FROM send_backs s JOIN leads l ON l.id = s.lead_id WHERE l.status='approved'")

    rounds = conn.execute(
        "SELECT round, purpose, MAX(attempt) tries, MAX(passed) any_pass, "
        "MAX(CASE WHEN attempt=1 THEN passed END) first_pass FROM draft_attempts GROUP BY round, purpose").fetchall()
    attempts = one("SELECT COUNT(*) FROM draft_attempts")
    failed_attempts = one("SELECT COUNT(*) FROM draft_attempts WHERE passed=0")
    kinds: Counter = Counter()
    for r in conn.execute("SELECT problems FROM draft_attempts WHERE passed=0"):
        kinds.update({_problem_kind(p) for p in json.loads(r["problems"])})

    outbox = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM outbox GROUP BY status")}

    return {
        "decisions": {
            "decided": decided, "approved": approved, "approved_as_drafted": approved - edited, "approved_edited": edited,
            "rejected": rejected, "rejected_by_reason": by_reason, "by_writer": by_writer,
            "approval_rate": _rate(approved, decided), "edit_rate_of_approvals": _rate(edited, approved),
        },
        "send_backs": {
            "total": sent_back, "rewritten": rewritten, "failed": sent_back - rewritten,
            "leads": leads_sent_back, "leads_approved_after": approved_after,
        },
        "draft_checks": {
            "rounds": len(rounds), "attempts": attempts, "failed_attempts": failed_attempts,
            "passed_first_try": sum(1 for r in rounds if r["first_pass"]),
            "passed_after_revision": sum(1 for r in rounds if r["any_pass"] and not r["first_pass"]),
            "never_passed": sum(1 for r in rounds if not r["any_pass"]),
            "first_try_pass_rate": _rate(sum(1 for r in rounds if r["first_pass"]), len(rounds)),
            "problems": dict(kinds.most_common()),
        },
        "outbox": outbox,
    }
