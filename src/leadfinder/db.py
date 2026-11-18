"""The SQLite store: practices (the NPI-style facts), their website visits, the leads scored from them (each with
its points, the model's chance, its drafted email and call brief), staff and sessions, the trained model (one JSON
value in settings, every version ever fitted in model_runs), and the audit log of every approval and rejection.

One file, opened once per process with `check_same_thread=False` (FastAPI runs plain `def` routes in a thread pool)
and autocommit, so each statement lands on its own; WAL keeps readers and the one writer out of each other's way.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path(os.environ.get("LEADFINDER_DB", ROOT / "data" / "leadfinder.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS practices(
  id INTEGER PRIMARY KEY,
  npi TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  specialty TEXT NOT NULL,
  providers INTEGER NOT NULL,
  city TEXT NOT NULL,
  state TEXT NOT NULL,
  ehr TEXT,
  contact_name TEXT,
  contact_email TEXT,
  source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visits(
  id INTEGER PRIMARY KEY,
  practice_id INTEGER NOT NULL REFERENCES practices(id),
  page TEXT NOT NULL,
  at REAL NOT NULL,
  source_request_id TEXT,
  source_view_id TEXT
);
CREATE INDEX IF NOT EXISTS visits_practice ON visits(practice_id);
CREATE TABLE IF NOT EXISTS leads(
  id INTEGER PRIMARY KEY,
  practice_id INTEGER UNIQUE NOT NULL REFERENCES practices(id),
  score INTEGER NOT NULL,
  band TEXT NOT NULL,
  reasons TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',
  scored_at REAL NOT NULL,
  written_by TEXT NOT NULL DEFAULT 'template',
  brief TEXT,
  brief_by TEXT,
  chance REAL,
  chance_band TEXT,
  chance_reasons TEXT,
  model_version TEXT
);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY,
  lead_id INTEGER NOT NULL REFERENCES leads(id),
  username TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT,
  note TEXT,
  score INTEGER,
  reasons TEXT,
  subject TEXT,
  body TEXT,
  edited INTEGER NOT NULL DEFAULT 0,
  at REAL NOT NULL,
  written_by TEXT,
  chance REAL,
  model_version TEXT
);
CREATE TABLE IF NOT EXISTS staff(
  username TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  pw_hash TEXT NOT NULL,
  disabled INTEGER NOT NULL DEFAULT 0,
  failed INTEGER NOT NULL DEFAULT 0,
  locked_until REAL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_runs(
  version TEXT PRIMARY KEY,
  month INTEGER NOT NULL,
  status TEXT NOT NULL,
  verdict TEXT NOT NULL,
  against TEXT,
  card TEXT NOT NULL,
  trained_at REAL NOT NULL,
  model TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
  token_hash TEXT PRIMARY KEY,
  username TEXT NOT NULL REFERENCES staff(username),
  created_at REAL NOT NULL,
  last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS visit_analyses(
  id INTEGER PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  lead_id TEXT NOT NULL,
  routing_decision TEXT NOT NULL CHECK(routing_decision='human_review'),
  review_status TEXT NOT NULL CHECK(review_status IN ('pending', 'identity_verified', 'closed')),
  is_demo INTEGER NOT NULL DEFAULT 0,
  result_json TEXT NOT NULL,
  created_by TEXT NOT NULL REFERENCES staff(username),
  created_at REAL NOT NULL,
  practice_id INTEGER REFERENCES practices(id),
  resolved_by TEXT REFERENCES staff(username),
  resolved_at REAL,
  resolution_note TEXT,
  handoff_status TEXT NOT NULL DEFAULT 'not_started',
  lead_record_id INTEGER REFERENCES leads(id),
  handoff_note TEXT
);
CREATE INDEX IF NOT EXISTS visit_analyses_review ON visit_analyses(routing_decision, review_status, created_at);
CREATE TABLE IF NOT EXISTS lead_handoffs(
  id INTEGER PRIMARY KEY,
  request_id TEXT UNIQUE NOT NULL,
  visitor_ref TEXT NOT NULL,
  identity_method TEXT NOT NULL CHECK(identity_method IN ('automatic', 'human_review')),
  practice_id INTEGER NOT NULL REFERENCES practices(id),
  lead_record_id INTEGER NOT NULL REFERENCES leads(id),
  analysis_json TEXT NOT NULL,
  practice_record_json TEXT NOT NULL,
  created_by TEXT NOT NULL REFERENCES staff(username),
  created_at REAL NOT NULL,
  is_demo INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS lead_handoffs_created ON lead_handoffs(created_at);
-- Outreach (revision.py, sender.py, feedback.py): every draft the checks saw, every send-back, and the outbox.
CREATE TABLE IF NOT EXISTS draft_attempts(
  id INTEGER PRIMARY KEY,
  lead_id INTEGER NOT NULL REFERENCES leads(id),
  round TEXT NOT NULL,
  purpose TEXT NOT NULL CHECK(purpose IN ('write', 'send_back')),
  attempt INTEGER NOT NULL,
  writer TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  problems TEXT NOT NULL,
  passed INTEGER NOT NULL,
  kept INTEGER NOT NULL DEFAULT 0,
  username TEXT NOT NULL,
  at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS draft_attempts_lead ON draft_attempts(lead_id, at);
CREATE TABLE IF NOT EXISTS send_backs(
  id INTEGER PRIMARY KEY,
  lead_id INTEGER NOT NULL REFERENCES leads(id),
  username TEXT NOT NULL,
  note TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK(outcome IN ('rewritten', 'failed')),
  detail TEXT,
  at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS send_backs_lead ON send_backs(lead_id, at);
CREATE TABLE IF NOT EXISTS outbox(
  id INTEGER PRIMARY KEY,
  lead_id INTEGER UNIQUE NOT NULL REFERENCES leads(id),
  to_name TEXT,
  to_email TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  sender TEXT NOT NULL,
  status TEXT NOT NULL,
  detail TEXT,
  queued_by TEXT NOT NULL,
  queued_at REAL NOT NULL
);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Columns added after the first version: a database made before them gets them here.
ADDED = {"leads": {"written_by": "TEXT NOT NULL DEFAULT 'template'", "brief": "TEXT", "brief_by": "TEXT",
                   "chance": "REAL", "chance_band": "TEXT", "chance_reasons": "TEXT", "model_version": "TEXT"},
         "audit": {"written_by": "TEXT", "chance": "REAL", "model_version": "TEXT"},
         "visits": {"source_request_id": "TEXT", "source_view_id": "TEXT"},
         "visit_analyses": {"is_demo": "INTEGER NOT NULL DEFAULT 0",
                            "handoff_status": "TEXT NOT NULL DEFAULT 'not_started'",
                            "lead_record_id": "INTEGER REFERENCES leads(id)", "handoff_note": "TEXT"}}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS visits_gate_view ON visits(source_request_id, source_view_id)"
    )


def get_setting(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, json.dumps(value)))
