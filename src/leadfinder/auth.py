"""Who may do what: staff accounts, roles and sessions.

Everyone who opens the pages logs in with a username and a password and gets a role:
- an SDR sees the overview, the leads and each lead, and approves or rejects drafted emails;
- the sales manager sees all that and the audit log of every approval and rejection.
A write records who made it from the session, never from a typed name.

Passwords are hashed with scrypt; five wrong passwords lock an account for fifteen minutes. A session is a random
256-bit token in an HttpOnly cookie, of which the database keeps only a hash, so a copy of the file opens no session;
it ends after twelve idle hours or seven days. The HTTP side (the cookie, the middleware that turns a request into a
principal) is in service/security.py.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

PERMISSIONS = frozenset({"view_leads", "act_on_leads", "view_audit"})
ROLES: dict[str, frozenset[str]] = {
    "sdr": frozenset({"view_leads", "act_on_leads"}),
    "manager": PERMISSIONS,
}
ROLE_WORDS = {"sdr": "SDR", "manager": "Sales manager"}
IDLE_SECONDS = 12 * 3600
MAX_SECONDS = 7 * 24 * 3600
LOCK_AFTER, LOCK_SECONDS = 5, 15 * 60
MIN_PASSWORD = 8
DEMO_PASSWORD = "leadfinder"            # every account `seed_demo_accounts` makes, on synthetic data only


class AuthError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    username: str
    name: str
    role: str

    def can(self, permission: str) -> bool:
        return permission in ROLES.get(self.role, frozenset())

    @property
    def role_word(self) -> str:
        return ROLE_WORDS.get(self.role, self.role)


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        want = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=2 ** 14, r=8, p=1, dklen=len(want))
        return hmac.compare_digest(got, want)
    except (ValueError, TypeError):
        return False


def normalise_username(username: str) -> str:
    u = username.strip().lower()
    if not u or len(u) > 40 or not all(ch.isalnum() or ch in "._-" for ch in u):
        raise AuthError("a username is 1 to 40 letters, digits, dots, dashes or underscores")
    return u


# ---------------------------------------------------------------- accounts
def create_staff(conn: sqlite3.Connection, username: str, password: str, role: str, name: str = "") -> str:
    u = normalise_username(username)
    if role not in ROLES:
        raise AuthError(f"role must be one of {', '.join(ROLES)}")
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"a password needs at least {MIN_PASSWORD} characters")
    if conn.execute("SELECT 1 FROM staff WHERE username=?", (u,)).fetchone():
        raise AuthError(f"{u} already has an account")
    conn.execute("INSERT INTO staff(username, name, role, pw_hash, created_at) VALUES(?,?,?,?,?)",
                 (u, name.strip() or u, role, hash_password(password), time.time()))
    return u


def staff_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) n FROM staff").fetchone()["n"]


def bootstrap_from_env(conn: sqlite3.Connection, env: Optional[dict] = None) -> bool:
    """The first account, a sales manager, from LEADFINDER_ADMIN_USER and LEADFINDER_ADMIN_PASSWORD, when none exists."""
    env = os.environ if env is None else env
    user, password = (env.get("LEADFINDER_ADMIN_USER") or "").strip(), env.get("LEADFINDER_ADMIN_PASSWORD") or ""
    if not user or not password or staff_count(conn):
        return False
    create_staff(conn, user, password, "manager", name=user)
    return True


DEMO_PEOPLE = [("riley", "sdr", "Riley Chen"), ("jo", "sdr", "Jo Alvarez"), ("morgan", "manager", "Morgan Ellis")]


def seed_demo_accounts(conn: sqlite3.Connection) -> list[str]:
    """The synthetic data's logins, all with DEMO_PASSWORD. Accounts that exist are left as they are."""
    made = []
    for username, role, name in DEMO_PEOPLE:
        try:
            made.append(create_staff(conn, username, DEMO_PASSWORD, role, name=name))
        except AuthError:
            pass
    conn.execute("INSERT INTO settings(key, value) VALUES('demo_logins', '1') ON CONFLICT(key) DO UPDATE SET value='1'")
    return made


def demo_logins(conn: sqlite3.Connection) -> list[dict]:
    row = conn.execute("SELECT value FROM settings WHERE key='demo_logins'").fetchone()
    if not row or row["value"] != "1":
        return []
    return [dict(r) for r in conn.execute("SELECT username, name, role FROM staff WHERE disabled=0 ORDER BY "
                                          "CASE role WHEN 'sdr' THEN 0 ELSE 1 END, username")]


# ---------------------------------------------------------------- logging in
def authenticate(conn: sqlite3.Connection, username: str, password: str) -> Optional[Principal]:
    """The principal for a right username and password; None for a wrong pair, a disabled account or a locked one."""
    try:
        u = normalise_username(username)
    except AuthError:
        return None
    row = conn.execute("SELECT * FROM staff WHERE username=?", (u,)).fetchone()
    now = time.time()
    if row is None:
        verify_password(password, hash_password("x" * MIN_PASSWORD))      # the same time whether or not the account exists
        return None
    if row["disabled"] or (row["locked_until"] and row["locked_until"] > now):
        return None
    if not verify_password(password, row["pw_hash"]):
        failed = (row["failed"] or 0) + 1
        locked = now + LOCK_SECONDS if failed >= LOCK_AFTER else None
        conn.execute("UPDATE staff SET failed=?, locked_until=? WHERE username=?", (0 if locked else failed, locked, u))
        return None
    conn.execute("UPDATE staff SET failed=0, locked_until=NULL WHERE username=?", (u,))
    return Principal(username=u, name=row["name"], role=row["role"])


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(conn: sqlite3.Connection, p: Principal) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn.execute("DELETE FROM sessions WHERE last_seen < ? OR created_at < ?", (now - IDLE_SECONDS, now - MAX_SECONDS))
    conn.execute("INSERT INTO sessions(token_hash, username, created_at, last_seen) VALUES(?,?,?,?)",
                 (_token_hash(token), p.username, now, now))
    return token


def load_session(conn: sqlite3.Connection, token: Optional[str]) -> Optional[Principal]:
    if not token:
        return None
    now = time.time()
    row = conn.execute("SELECT s.created_at, s.last_seen, t.username, t.name, t.role, t.disabled FROM sessions s "
                       "JOIN staff t ON t.username = s.username WHERE s.token_hash=?", (_token_hash(token),)).fetchone()
    if row is None:
        return None
    if row["disabled"] or row["last_seen"] < now - IDLE_SECONDS or row["created_at"] < now - MAX_SECONDS:
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))
        return None
    if now - row["last_seen"] > 60:
        conn.execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (now, _token_hash(token)))
    return Principal(username=row["username"], name=row["name"], role=row["role"])


def end_session(conn: sqlite3.Connection, token: Optional[str]) -> None:
    if token:
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))
