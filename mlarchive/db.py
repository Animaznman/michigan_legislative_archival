"""SQLite-backed state store and work queue.

One row per recording. A row moves through stages:

    fetch  ->  transcribe  ->  done

Workers *claim* a row by taking a time-limited lease in a single atomic
UPDATE ... RETURNING. A worker that dies simply stops renewing its lease;
once it expires another worker picks the row up. Every write that finishes
or fails a job checks `lease_owner = me` (a fencing check), so a worker
whose lease was taken over can't clobber the new owner's progress.

Why SQLite: one file, transactional, zero ops, survives power loss (WAL),
and the workload is tiny (a few hundred new rows a year). Discovery,
fetch and transcribe can run as threads in one process or as separate
processes on the same machine and they coordinate only through this file.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id               INTEGER PRIMARY KEY,
    source           TEXT NOT NULL,          -- 'house' | 'senate'
    source_id        TEXT NOT NULL,          -- portal's own stable id
    title            TEXT NOT NULL,
    committee        TEXT,
    recorded_on      TEXT NOT NULL,          -- ISO date, used for ordering
    page_url         TEXT,
    media_url        TEXT,
    meta             TEXT NOT NULL DEFAULT '{}',
    stage            TEXT NOT NULL DEFAULT 'fetch',  -- fetch|transcribe|done
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  REAL NOT NULL DEFAULT 0,
    lease_owner      TEXT,
    lease_expires_at REAL,
    last_error       TEXT,
    audio_path       TEXT,
    transcript_path  TEXT,
    discovered_at    REAL NOT NULL,
    updated_at       REAL NOT NULL,
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS ix_queue ON recordings (stage, next_attempt_at, recorded_on);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    recording_id INTEGER REFERENCES recordings(id),
    at           REAL NOT NULL,
    kind         TEXT NOT NULL,     -- discovered|claimed|advanced|retry|parked|...
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_rec ON events (recording_id);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

STAGES = ("fetch", "transcribe", "done")


@dataclass
class Recording:
    id: int
    source: str
    source_id: str
    title: str
    committee: str | None
    recorded_on: str
    page_url: str | None
    media_url: str | None
    meta: dict
    stage: str
    attempts: int
    audio_path: str | None
    transcript_path: str | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Recording":
        return cls(r["id"], r["source"], r["source_id"], r["title"], r["committee"],
                   r["recorded_on"], r["page_url"], r["media_url"],
                   json.loads(r["meta"] or "{}"), r["stage"], r["attempts"],
                   r["audio_path"], r["transcript_path"])


class LeaseLost(Exception):
    """Our lease expired and someone else owns the job now."""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)   # idempotent (IF NOT EXISTS); manages its own tx

    # One connection per thread: sqlite3 connections must not be shared.
    def _conn(self) -> sqlite3.Connection: # If the thread doesn't have a connection yet, create it and configure it
        c = getattr(self._local, "conn", None) # Check connection
        if c is None: # All threads share same Database object, but not the same SQLite connection
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None) # Allows issuing of 'BEGIN' 'UPDATE' etc. manually
            c.row_factory = sqlite3.Row # Returns dict-like access instead of tuples
            c.execute("PRAGMA journal_mode=WAL") # Write-Ahead Log to allow surviving power loss
            c.execute("PRAGMA synchronous=NORMAL") # How aggressively SQLite flushes data to disk
            c.execute("PRAGMA busy_timeout=30000") # If db is blocked, wait 30 seconds
            c.execute("PRAGMA foreign_keys=ON") # Enforces foreign keys
            self._local.conn = c
        return c

    class _Tx:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            # IMMEDIATE takes the write lock up front -> no upgrade deadlocks.
            self.conn.execute("BEGIN IMMEDIATE")
            return self.conn

        def __exit__(self, exc_type, *_):
            self.conn.execute("ROLLBACK" if exc_type else "COMMIT")

    def _tx(self):
        return self._Tx(self._conn())

    def _event(self, c, rec_id: int | None, kind: str, detail: str | None = None):
        c.execute("INSERT INTO events (recording_id, at, kind, detail) VALUES (?,?,?,?)",
                  (rec_id, time.time(), kind, detail))

    # ---------------------------------------------------------------- discovery
    def upsert_discovered(self, *, source: str, source_id: str, title: str,
                          committee: str | None, recorded_on: str,
                          page_url: str | None, media_url: str | None,
                          meta: dict[str, Any]) -> bool:
        """Insert a newly seen recording. Returns True if it was new.

        Idempotent: the (source, source_id) UNIQUE key means re-discovering a
        recording is a no-op apart from refreshing descriptive fields. Its
        stage/progress is never touched.
        """
        now = time.time()
        with self._tx() as c:
            cur = c.execute(
                """INSERT INTO recordings (source, source_id, title, committee, recorded_on,
                       page_url, media_url, meta, discovered_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (source, source_id) DO NOTHING""",
                (source, source_id, title, committee, recorded_on, page_url, media_url,
                 json.dumps(meta), now, now))
            if cur.rowcount:
                self._event(c, cur.lastrowid, "discovered", title)
                return True
            c.execute(
                """UPDATE recordings SET title=?, committee=?, page_url=?, media_url=?, meta=?
                   WHERE source=? AND source_id=?""",
                (title, committee, page_url, media_url, json.dumps(meta), source, source_id))
            return False

    def known_ids(self, source: str) -> set[str]:
        rows = self._conn().execute("SELECT source_id FROM recordings WHERE source=?", (source,))
        return {r[0] for r in rows}

    # -------------------------------------------------------------------- queue
    def claim(self, stage: str, owner: str, lease_s: float, since: str,
              sources: list[str] | None = None, rec_id: int | None = None) -> Recording | None:
        """Atomically lease the newest ready job in `stage`, or return None.

        Newest first: a hearing that just ended jumps ahead of the backlog.
        """
        now = time.time()
        srcs = list(sources or [])
        src_sql = f"AND source IN ({','.join('?' * len(srcs))})" if srcs else ""
        if rec_id is not None:   # operator asked for this one: ignore window/backoff
            src_sql, srcs, since, gate = "AND id=?", [rec_id], "", now + 10 * 365 * 86400
        else:
            gate = now
        with self._tx() as c:
            row = c.execute(
                f"""UPDATE recordings
                      SET lease_owner=?, lease_expires_at=?, updated_at=?
                    WHERE id = (
                        SELECT id FROM recordings
                         WHERE stage=? AND next_attempt_at<=? AND recorded_on>=?
                           AND (lease_expires_at IS NULL OR lease_expires_at<?) {src_sql}
                         ORDER BY recorded_on DESC, id DESC LIMIT 1)
                RETURNING *""",
                (owner, now + lease_s, now, stage, gate, since, now, *srcs)).fetchone()
            if row:
                self._event(c, row["id"], "claimed", f"{stage} by {owner}")
        return Recording.from_row(row) if row else None

    def heartbeat(self, rec_id: int, owner: str, lease_s: float) -> None:
        with self._tx() as c:
            cur = c.execute(
                "UPDATE recordings SET lease_expires_at=? WHERE id=? AND lease_owner=?",
                (time.time() + lease_s, rec_id, owner))
        if not cur.rowcount:
            raise LeaseLost(f"recording {rec_id} no longer leased by {owner}")

    def advance(self, rec_id: int, owner: str, from_stage: str, to_stage: str,
                **fields) -> None:
        """Finish a stage: move on, reset attempts, release the lease."""
        allowed = {"audio_path", "transcript_path", "media_url"}
        assert set(fields) <= allowed, fields
        sets = "".join(f", {k}=?" for k in fields)
        with self._tx() as c:
            cur = c.execute(
                f"""UPDATE recordings SET stage=?, attempts=0, next_attempt_at=0,
                        last_error=NULL, lease_owner=NULL, lease_expires_at=NULL,
                        updated_at=? {sets}
                     WHERE id=? AND lease_owner=? AND stage=?""",
                (to_stage, time.time(), *fields.values(), rec_id, owner, from_stage))
            if not cur.rowcount:
                raise LeaseLost(f"recording {rec_id}: lost lease before advancing")
            self._event(c, rec_id, "advanced", f"{from_stage} -> {to_stage}")

    def fail(self, rec_id: int, owner: str, error: str, *, base_s: float, cap_s: float,
             max_attempts: int, park_s: float, delay_s: float | None = None,
             count_attempt: bool = True) -> tuple[int, float]:
        """Record a failure and schedule the retry. Returns (attempts, delay).

        Exponential backoff (base * 2^n, capped). After max_attempts the job
        is *parked* (retried once per park_s) instead of given up on: an
        upstream outage that lasts days still heals itself.
        """
        with self._tx() as c:
            row = c.execute("SELECT attempts FROM recordings WHERE id=? AND lease_owner=?",
                            (rec_id, owner)).fetchone()
            if row is None:
                raise LeaseLost(f"recording {rec_id}: lost lease before recording failure")
            attempts = row[0] + (1 if count_attempt else 0)
            if delay_s is None:
                delay_s = park_s if attempts >= max_attempts else min(
                    base_s * 2 ** max(attempts - 1, 0), cap_s)
            c.execute(
                """UPDATE recordings SET attempts=?, next_attempt_at=?, last_error=?,
                       lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE id=?""",
                (attempts, time.time() + delay_s, error[:2000], time.time(), rec_id))
            kind = "parked" if attempts >= max_attempts else "retry"
            self._event(c, rec_id, kind, f"attempt {attempts}, in {delay_s:.0f}s: {error[:500]}")
        return attempts, delay_s

    def release(self, rec_id: int, owner: str) -> None:
        """Give a job back untouched (e.g. on shutdown), no attempt counted."""
        with self._tx() as c:
            c.execute("UPDATE recordings SET lease_owner=NULL, lease_expires_at=NULL "
                      "WHERE id=? AND lease_owner=?", (rec_id, owner))

    def retry_now(self, rec_id: int | None = None, stage: str | None = None) -> int:
        """Operator command: make parked/backed-off jobs eligible immediately."""
        q = "UPDATE recordings SET attempts=0, next_attempt_at=0 WHERE stage!='done'"
        args: list = []
        if rec_id is not None:
            q += " AND id=?"; args.append(rec_id)
        if stage is not None:
            q += " AND stage=?"; args.append(stage)
        with self._tx() as c:
            return c.execute(q, args).rowcount

    def reset_stage(self, rec_id: int, stage: str) -> None:
        assert stage in STAGES
        with self._tx() as c:
            c.execute("UPDATE recordings SET stage=?, attempts=0, next_attempt_at=0, "
                      "lease_owner=NULL, lease_expires_at=NULL WHERE id=?", (stage, rec_id))
            self._event(c, rec_id, "reset", f"-> {stage}")

    # ------------------------------------------------------------------ reports
    def counts(self, since: str) -> dict[str, int]:
        now = time.time()
        rows = self._conn().execute(
            """SELECT CASE
                        WHEN recorded_on < ? THEN 'out_of_window'
                        WHEN stage='done' THEN 'done'
                        WHEN lease_expires_at > ? THEN stage || ':running'
                        WHEN next_attempt_at > ? THEN stage || ':waiting_retry'
                        ELSE stage || ':ready' END AS s, COUNT(*)
                 FROM recordings GROUP BY s ORDER BY s""", (since, now, now))
        return {r[0]: r[1] for r in rows}

    def problems(self, limit: int = 20) -> Iterator[sqlite3.Row]:
        return self._conn().execute(
            """SELECT id, source, source_id, stage, attempts, next_attempt_at, last_error
                 FROM recordings WHERE last_error IS NOT NULL AND stage!='done'
                ORDER BY attempts DESC, updated_at DESC LIMIT ?""", (limit,))

    def get(self, rec_id: int) -> Recording | None:
        row = self._conn().execute("SELECT * FROM recordings WHERE id=?", (rec_id,)).fetchone()
        return Recording.from_row(row) if row else None

    def kv_get(self, key: str) -> str | None:
        row = self._conn().execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO kv (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
