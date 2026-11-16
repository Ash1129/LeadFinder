"""The SQLite store: practices (the NPI-style facts), their website visits, the leads scored from them, staff and
sessions, and the audit log of every approval and rejection.

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
  at REAL NOT NULL
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
  scored_at REAL NOT NULL
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
  at REAL NOT NULL
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
CREATE TABLE IF NOT EXISTS sessions(
  token_hash TEXT PRIMARY KEY,
  username TEXT NOT NULL REFERENCES staff(username),
  created_at REAL NOT NULL,
  last_seen REAL NOT NULL
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
    return conn


def get_setting(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, json.dumps(value)))
