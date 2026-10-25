"""SQLite storage: users, whitelist, interaction log, runtime settings."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    username    TEXT,
    first_name  TEXT,
    last_name   TEXT,
    status      TEXT NOT NULL DEFAULT 'unknown',   -- unknown | allowed | blocked
    engine      TEXT,                              -- personal engine override
    notified    INTEGER NOT NULL DEFAULT 0,        -- owner was told about this unknown user
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS users_username ON users (lower(username));

CREATE TABLE IF NOT EXISTS pending_usernames (
    username    TEXT PRIMARY KEY,                  -- lower-case, without '@'
    added_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    user_id     INTEGER,
    chat_id     INTEGER,
    chat_type   TEXT,
    source      TEXT,      -- private | group | guest | business | command
    kind        TEXT,      -- transcribe | error | denied | limited | too_big | command
    duration    REAL,
    language    TEXT,
    engine      TEXT,
    latency     REAL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS interactions_user_ts ON interactions (user_id, ts);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS business_connections (
    id           TEXT PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    user_chat_id INTEGER NOT NULL,
    can_reply    INTEGER NOT NULL DEFAULT 0,
    is_enabled   INTEGER NOT NULL DEFAULT 1,
    updated_at   REAL NOT NULL
);
"""

# Kinds that count against a user's rate limit.
COUNTED_KINDS = ("transcribe", "error")


@dataclass
class UserRow:
    id: int
    username: str | None
    first_name: str | None
    last_name: str | None
    status: str
    engine: str | None
    notified: bool
    first_seen: float
    last_seen: float

    @property
    def display(self) -> str:
        name = " ".join(p for p in (self.first_name, self.last_name) if p) or "?"
        if self.username:
            return f"{name} (@{self.username})"
        return name


@dataclass
class InteractionRow:
    id: int
    ts: float
    user_id: int | None
    chat_id: int | None
    chat_type: str | None
    source: str | None
    kind: str | None
    duration: float | None
    language: str | None
    engine: str | None
    latency: float | None
    detail: str | None


@dataclass
class BusinessConnectionRow:
    id: str
    user_id: int
    user_chat_id: int
    can_reply: bool
    is_enabled: bool


def normalize_username(username: str) -> str:
    return username.strip().lstrip("@").lower()


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> "Database":
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("database is not open")
        return self._db

    # ---- users -------------------------------------------------------------------------

    async def touch_user(
        self,
        user_id: int,
        username: str | None,
        first_name: str | None,
        last_name: str | None,
    ) -> tuple[UserRow, bool]:
        """Insert or update a user seen in an update.

        Returns the row and whether the user was auto-approved now because their username
        was pre-authorized by the owner.
        """
        now = time.time()
        await self.db.execute(
            """INSERT INTO users (id, username, first_name, last_name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET username=excluded.username,
                   first_name=excluded.first_name, last_name=excluded.last_name,
                   last_seen=excluded.last_seen""",
            (user_id, username, first_name, last_name, now, now),
        )
        approved = False
        if username:
            uname = normalize_username(username)
            # Usernames move between accounts: forget it on any other (stale) row.
            await self.db.execute(
                "UPDATE users SET username=NULL WHERE lower(username)=? AND id!=?", (uname, user_id)
            )
            cur = await self.db.execute("DELETE FROM pending_usernames WHERE username=?", (uname,))
            if cur.rowcount == 1:  # only the first concurrent caller wins
                await self.db.execute(
                    "UPDATE users SET status='allowed' WHERE id=? AND status!='blocked'", (user_id,)
                )
                approved = True
        await self.db.commit()
        user = await self.get_user(user_id)
        assert user is not None
        return user, approved and user.status == "allowed"

    async def get_user(self, user_id: int) -> UserRow | None:
        cur = await self.db.execute("SELECT * FROM users WHERE id=?", (user_id,))
        row = await cur.fetchone()
        return _user(row) if row else None

    async def find_user_by_username(self, username: str) -> UserRow | None:
        cur = await self.db.execute(
            "SELECT * FROM users WHERE lower(username)=? ORDER BY last_seen DESC LIMIT 1",
            (normalize_username(username),),
        )
        row = await cur.fetchone()
        return _user(row) if row else None

    async def set_status(self, user_id: int, status: str) -> None:
        assert status in ("unknown", "allowed", "blocked")
        now = time.time()
        await self.db.execute(
            """INSERT INTO users (id, status, first_seen, last_seen) VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status""",
            (user_id, status, now, now),
        )
        await self.db.commit()

    async def ensure_user(
        self, user_id: int, username: str | None, first_name: str | None, last_name: str | None
    ) -> None:
        """Remember a user we learned about indirectly (e.g. shared contact) without touching last_seen."""
        now = time.time()
        await self.db.execute(
            """INSERT INTO users (id, username, first_name, last_name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   username=coalesce(excluded.username, users.username),
                   first_name=coalesce(excluded.first_name, users.first_name),
                   last_name=coalesce(excluded.last_name, users.last_name)""",
            (user_id, username, first_name, last_name, now, now),
        )
        await self.db.commit()

    async def claim_notification(self, user_id: int) -> bool:
        """Atomically mark that the owner was told about this user; True only for the first caller."""
        cur = await self.db.execute("UPDATE users SET notified=1 WHERE id=? AND notified=0", (user_id,))
        await self.db.commit()
        return cur.rowcount == 1

    async def set_user_engine(self, user_id: int, engine: str | None) -> None:
        await self.db.execute("UPDATE users SET engine=? WHERE id=?", (engine, user_id))
        await self.db.commit()

    async def list_users(self, status: str | None = None, limit: int = 50, offset: int = 0) -> list[UserRow]:
        if status:
            cur = await self.db.execute(
                "SELECT * FROM users WHERE status=? ORDER BY last_seen DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM users ORDER BY last_seen DESC LIMIT ? OFFSET ?", (limit, offset)
            )
        return [_user(r) for r in await cur.fetchall()]

    async def count_users(self, status: str | None = None) -> int:
        if status:
            cur = await self.db.execute("SELECT count(*) FROM users WHERE status=?", (status,))
        else:
            cur = await self.db.execute("SELECT count(*) FROM users")
        row = await cur.fetchone()
        return int(row[0])

    # ---- pre-authorized usernames ------------------------------------------------------

    async def add_pending_username(self, username: str) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO pending_usernames (username, added_at) VALUES (?, ?)",
            (normalize_username(username), time.time()),
        )
        await self.db.commit()

    async def remove_pending_username(self, username: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM pending_usernames WHERE username=?", (normalize_username(username),)
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def list_pending_usernames(self) -> list[str]:
        cur = await self.db.execute("SELECT username FROM pending_usernames ORDER BY added_at")
        return [r[0] for r in await cur.fetchall()]

    # ---- interactions ------------------------------------------------------------------

    async def log_interaction(
        self,
        *,
        user_id: int | None,
        chat_id: int | None,
        chat_type: str | None,
        source: str,
        kind: str,
        duration: float | None = None,
        language: str | None = None,
        engine: str | None = None,
        latency: float | None = None,
        detail: str | None = None,
        ts: float | None = None,
    ) -> int:
        cur = await self.db.execute(
            """INSERT INTO interactions
               (ts, user_id, chat_id, chat_type, source, kind, duration, language, engine, latency, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts or time.time(), user_id, chat_id, chat_type, source, kind, duration, language,
             engine, latency, detail),
        )
        await self.db.commit()
        return int(cur.lastrowid)

    async def update_interaction(self, interaction_id: int, **fields: Any) -> None:
        allowed = {"kind", "duration", "language", "engine", "latency", "detail"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if not cols:
            return
        await self.db.execute(
            f"UPDATE interactions SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
            (*cols.values(), interaction_id),
        )
        await self.db.commit()

    async def count_requests(self, user_id: int, since: float) -> int:
        cur = await self.db.execute(
            f"""SELECT count(*) FROM interactions WHERE user_id=? AND ts>=?
                AND kind IN ({",".join("?" * len(COUNTED_KINDS))})""",
            (user_id, since, *COUNTED_KINDS),
        )
        row = await cur.fetchone()
        return int(row[0])

    async def oldest_request_since(self, user_id: int, since: float) -> float | None:
        cur = await self.db.execute(
            f"""SELECT min(ts) FROM interactions WHERE user_id=? AND ts>=?
                AND kind IN ({",".join("?" * len(COUNTED_KINDS))})""",
            (user_id, since, *COUNTED_KINDS),
        )
        row = await cur.fetchone()
        return row[0]

    async def recent_interactions(self, limit: int = 20, user_id: int | None = None) -> list[InteractionRow]:
        if user_id is None:
            cur = await self.db.execute("SELECT * FROM interactions ORDER BY ts DESC LIMIT ?", (limit,))
        else:
            cur = await self.db.execute(
                "SELECT * FROM interactions WHERE user_id=? ORDER BY ts DESC LIMIT ?", (user_id, limit)
            )
        return [InteractionRow(**dict(r)) for r in await cur.fetchall()]

    async def stats(self, since: float) -> dict[str, Any]:
        cur = await self.db.execute(
            """SELECT count(*), coalesce(sum(duration), 0), coalesce(avg(latency), 0),
                      count(DISTINCT user_id)
               FROM interactions WHERE kind='transcribe' AND ts>=?""",
            (since,),
        )
        n, audio, latency, users = await cur.fetchone()
        cur = await self.db.execute(
            """SELECT engine, count(*) FROM interactions WHERE kind='transcribe' AND ts>=?
               GROUP BY engine ORDER BY 2 DESC""",
            (since,),
        )
        engines = {r[0]: r[1] for r in await cur.fetchall()}
        cur = await self.db.execute(
            """SELECT language, count(*) FROM interactions WHERE kind='transcribe' AND ts>=?
               GROUP BY language ORDER BY 2 DESC""",
            (since,),
        )
        languages = {r[0]: r[1] for r in await cur.fetchall()}
        cur = await self.db.execute(
            "SELECT kind, count(*) FROM interactions WHERE ts>=? GROUP BY kind", (since,)
        )
        kinds = {r[0]: r[1] for r in await cur.fetchall()}
        return {"count": n, "audio_seconds": audio, "avg_latency": latency, "users": users,
                "engines": engines, "languages": languages, "kinds": kinds}

    # ---- settings ----------------------------------------------------------------------

    async def get_settings(self) -> dict[str, Any]:
        cur = await self.db.execute("SELECT key, value FROM settings")
        return {r[0]: json.loads(r[1]) for r in await cur.fetchall()}

    async def set_setting(self, key: str, value: Any) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value))
        )
        await self.db.commit()

    async def delete_setting(self, key: str) -> None:
        await self.db.execute("DELETE FROM settings WHERE key=?", (key,))
        await self.db.commit()

    # ---- business connections ----------------------------------------------------------

    async def upsert_business_connection(
        self, conn_id: str, user_id: int, user_chat_id: int, can_reply: bool, is_enabled: bool
    ) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO business_connections
               (id, user_id, user_chat_id, can_reply, is_enabled, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conn_id, user_id, user_chat_id, int(can_reply), int(is_enabled), time.time()),
        )
        await self.db.commit()

    async def get_business_connection(self, conn_id: str) -> BusinessConnectionRow | None:
        cur = await self.db.execute(
            "SELECT id, user_id, user_chat_id, can_reply, is_enabled FROM business_connections WHERE id=?",
            (conn_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return BusinessConnectionRow(row[0], row[1], row[2], bool(row[3]), bool(row[4]))


def _user(row: aiosqlite.Row) -> UserRow:
    d = dict(row)
    d["notified"] = bool(d["notified"])
    return UserRow(**d)
