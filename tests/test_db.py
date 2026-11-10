import time

import pytest

from mlarchive.db import Database, LeaseLost


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "state.db")


def add(db, sid, day="2026-03-01", source="house"):
    return db.upsert_discovered(source=source, source_id=sid, title=sid, committee=None,
                                recorded_on=day, page_url=None, media_url=f"http://x/{sid}",
                                meta={})


FAIL = dict(base_s=10, cap_s=100, max_attempts=3, park_s=1000)


def test_discovery_is_idempotent(db):
    assert add(db, "a") is True
    assert add(db, "a") is False
    assert db.counts("2000-01-01") == {"fetch:ready": 1}


def test_rediscovery_never_resets_progress(db):
    add(db, "a")
    rec = db.claim("fetch", "w1", 60, "2000-01-01")
    db.advance(rec.id, "w1", "fetch", "transcribe", audio_path="/a.m4a")
    add(db, "a")
    assert db.get(rec.id).stage == "transcribe"


def test_claim_is_exclusive_until_lease_expires(db):
    add(db, "a")
    assert db.claim("fetch", "w1", 60, "2000-01-01") is not None
    assert db.claim("fetch", "w2", 60, "2000-01-01") is None


def test_expired_lease_is_reclaimed(db):
    add(db, "a")
    db.claim("fetch", "w1", -1, "2000-01-01")          # lease already expired: w1 "crashed"
    rec = db.claim("fetch", "w2", 60, "2000-01-01")
    assert rec is not None


def test_fencing_old_owner_cannot_advance(db):
    add(db, "a")
    rec = db.claim("fetch", "w1", -1, "2000-01-01")
    db.claim("fetch", "w2", 60, "2000-01-01")
    with pytest.raises(LeaseLost):
        db.advance(rec.id, "w1", "fetch", "transcribe")
    with pytest.raises(LeaseLost):
        db.heartbeat(rec.id, "w1", 60)
    db.advance(rec.id, "w2", "fetch", "transcribe")


def test_newest_first(db):
    add(db, "old", "2026-01-05")
    add(db, "new", "2026-10-05")
    add(db, "mid", "2026-05-05")
    order = [db.claim("fetch", "w", 60, "2000-01-01").source_id for _ in range(3)]
    assert order == ["new", "mid", "old"]


def test_process_since_window_and_source_filter(db):
    add(db, "ancient", "2019-01-01")
    add(db, "s", "2026-02-01", source="senate")
    assert db.claim("fetch", "w", 60, "2026-01-01", ["house"]) is None
    assert db.claim("fetch", "w", 60, "2026-01-01", ["senate"]).source_id == "s"
    assert db.claim("fetch", "w", 60, "2026-01-01") is None


def test_backoff_grows_then_parks(db):
    add(db, "a")
    delays = []
    rec_id = db.claim("fetch", "w", 60, "2000-01-01").id
    db.release(rec_id, "w")
    for _ in range(3):
        with db._tx() as c:
            c.execute("UPDATE recordings SET next_attempt_at=0")
        rec = db.claim("fetch", "w", 60, "2000-01-01")
        _, d = db.fail(rec.id, "w", "boom", **FAIL)
        delays.append(d)
    assert delays == [10, 20, 1000]          # 10, 20, then parked
    assert db.claim("fetch", "w", 60, "2000-01-01") is None   # waiting
    assert db.get(rec_id).stage == "fetch"                    # never dropped


def test_not_ready_does_not_count_attempt(db):
    add(db, "a")
    rec = db.claim("fetch", "w", 60, "2000-01-01")
    attempts, d = db.fail(rec.id, "w", "not ready", delay_s=5, count_attempt=False, **FAIL)
    assert (attempts, d) == (0, 5)


def test_retry_now_makes_job_eligible(db):
    add(db, "a")
    rec = db.claim("fetch", "w", 60, "2000-01-01")
    db.fail(rec.id, "w", "boom", **FAIL)
    assert db.claim("fetch", "w", 60, "2000-01-01") is None
    assert db.retry_now(rec.id) == 1
    assert db.claim("fetch", "w", 60, "2000-01-01") is not None


def test_claim_specific_id_ignores_backoff(db):
    add(db, "a")
    rec = db.claim("fetch", "w", 60, "2000-01-01")
    db.fail(rec.id, "w", "boom", **FAIL)
    assert db.claim("fetch", "op", 60, "2099-01-01", rec_id=rec.id) is not None


def test_events_audit_trail(db):
    add(db, "a")
    rec = db.claim("fetch", "w", 60, "2000-01-01")
    db.advance(rec.id, "w", "fetch", "transcribe")
    kinds = [r[0] for r in db._conn().execute(
        "SELECT kind FROM events WHERE recording_id=? ORDER BY id", (rec.id,))]
    assert kinds == ["discovered", "claimed", "advanced"]
