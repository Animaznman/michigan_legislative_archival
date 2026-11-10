import threading

import pytest

from conftest import make_media
from mlarchive import pipeline as pl
from mlarchive.config import Settings
from mlarchive.db import Database
from mlarchive.fetcher import FetchResult, NotReady


@pytest.fixture
def pipe(tmp_path):
    s = Settings(data_dir=tmp_path / "data", process_since="2000-01-01", lease_s=60,
                 sources=["house"])
    p = pl.Pipeline(s, Database(s.db_path), threading.Event())
    p.db.upsert_discovered(source="house", source_id="AGRR-022526.mp4", title="t",
                           committee="c", recorded_on="2026-02-25", page_url=None,
                           media_url="https://house.test/AGRR-022526.mp4", meta={})
    return p


def test_safe_name():
    assert pl.safe_name("AGRR-022526.mp4") == "AGRR-022526"
    assert pl.safe_name("../x y") == ".._x_y"


def test_fetch_then_transcribe_then_done(pipe, tmp_path, monkeypatch):
    def fake_fetch(ctx, strategy, url, dst):
        assert strategy == "mp4-ranges"
        dst.parent.mkdir(parents=True, exist_ok=True)
        make_media(dst, seconds=2, fmt="ipod", video=False)
        return FetchResult(dst, strategy, 1, 2.0)
    monkeypatch.setattr(pl, "fetch_audio", fake_fetch)

    def fake_tx(tr, audio, work, oj, ot, meta, check):
        oj.parent.mkdir(parents=True, exist_ok=True)
        oj.write_text("{}"); ot.write_text("")
        return {"segments": [], "audio": {"duration_s": 2}, "elapsed_s": 0}
    monkeypatch.setattr(pl, "transcribe_file", fake_tx)

    assert pipe.run_one("fetch", "w")
    assert pipe.db.get(1).stage == "transcribe"
    assert pipe.run_one("transcribe", "w")
    assert pipe.db.get(1).stage == "done"
    assert not pipe.run_one("fetch", "w")


def test_existing_audio_is_not_downloaded_again(pipe, monkeypatch):
    rec = pipe.db.get(1)
    audio = pipe._paths(rec)["audio"]
    audio.parent.mkdir(parents=True)
    make_media(audio, seconds=2, fmt="ipod", video=False)
    monkeypatch.setattr(pl, "fetch_audio", lambda *a: pytest.fail("should not download"))
    pipe.run_one("fetch", "w")
    assert pipe.db.get(1).stage == "transcribe"


def test_not_ready_waits_without_burning_attempts(pipe, monkeypatch):
    def nr(*a):
        raise NotReady("still uploading")
    monkeypatch.setattr(pl, "fetch_audio", nr)
    pipe.run_one("fetch", "w")
    r = pipe.db._conn().execute("SELECT attempts, next_attempt_at, last_error FROM recordings").fetchone()
    assert r["attempts"] == 0 and r["next_attempt_at"] > 0 and "uploading" in r["last_error"]


def test_crash_is_recorded_and_retried(pipe, monkeypatch):
    def boom(*a):
        raise ValueError("disk full")
    monkeypatch.setattr(pl, "fetch_audio", boom)
    pipe.run_one("fetch", "w")
    r = pipe.db._conn().execute("SELECT stage, attempts, lease_owner FROM recordings").fetchone()
    assert (r["stage"], r["attempts"], r["lease_owner"]) == ("fetch", 1, None)


def test_shutdown_releases_job_without_counting(pipe, monkeypatch):
    def slow(ctx, *a):
        pipe.stop.set()
        ctx.check()
    monkeypatch.setattr(pl, "fetch_audio", slow)
    pipe.run_one("fetch", "w")
    r = pipe.db._conn().execute("SELECT attempts, lease_owner FROM recordings").fetchone()
    assert (r["attempts"], r["lease_owner"]) == (0, None)


def test_missing_audio_sends_job_back_to_fetch(pipe):
    rec = pipe.db.claim("fetch", "w", 60, "2000-01-01")
    pipe.db.advance(rec.id, "w", "fetch", "transcribe", audio_path="/nope.m4a")
    pipe.run_one("transcribe", "w")
    assert pipe.db.get(1).stage == "fetch"


def test_one_source_failing_does_not_block_other(pipe, monkeypatch):
    pipe.s.sources = ["house", "senate"]
    def bad(client, **kw):
        raise RuntimeError("portal down")
        yield
    monkeypatch.setitem(pl.sources.SOURCES, "house", bad)
    monkeypatch.setitem(pl.sources.SOURCES, "senate", lambda client, **kw: iter([]))
    assert pipe.discover_once() == {"house": -1, "senate": 0}
