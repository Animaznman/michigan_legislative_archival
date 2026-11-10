"""Orchestration: discovery loop + stage workers, coordinated through the DB.

    [discovery] --insert--> recordings(stage=fetch)
    [fetch worker xN] --claim fetch--> audio .m4a --advance--> stage=transcribe
    [transcribe worker] --claim transcribe--> transcript .json/.txt --advance--> done

Workers never call each other. The only coupling is the recordings table,
so any stage can crash, restart, or run in a separate process
(`mlarchive run --stages fetch`) without losing or duplicating work.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from . import net, sources
from .config import Settings
from .db import Database, LeaseLost, Recording
from .fetcher import FetchContext, NotReady, fetch_audio, probe_audio
from .transcriber import TranscribeOptions, Transcriber, transcribe_file

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Shutdown requested or lease lost: stop work, don't count as a failure."""


def safe_name(source_id: str) -> str:
    stem = re.sub(r"\.mp4$", "", source_id, flags=re.I)
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:120]


class Heartbeat:
    """Keeps a job's lease alive while it runs; flags if the lease is lost."""

    def __init__(self, db: Database, rec_id: int, owner: str, lease_s: float):
        self.db, self.rec_id, self.owner, self.lease_s = db, rec_id, owner, lease_s
        self.lost = threading.Event()
        self._done = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name=f"hb-{rec_id}")

    def _run(self):
        while not self._done.wait(self.lease_s / 3):
            try:
                self.db.heartbeat(self.rec_id, self.owner, self.lease_s)
            except LeaseLost:
                log.error("lease lost on recording %s; abandoning job", self.rec_id)
                self.lost.set()
                return
            except Exception as e:   # DB busy etc.: try again next beat
                log.warning("heartbeat failed for %s: %s", self.rec_id, e)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._done.set()
        self._t.join(timeout=5)


class Pipeline:
    def __init__(self, settings: Settings, db: Database | None = None,
                 stop: threading.Event | None = None):
        self.s = settings
        self.db = db or Database(settings.db_path)
        self.stop = stop or threading.Event()
        self.client = net.make_client(settings.user_agent)
        self._transcribers = threading.local()
        self._limit_lock = threading.Lock()
        self._fetch_budget: int | None = None

    # -------------------------------------------------------------- discovery
    def discover_once(self, only: list[str] | None = None) -> dict[str, int]:
        new_counts = {}
        for name in only or self.s.sources:
            fn = sources.SOURCES[name]
            t0 = time.time()
            try:
                kwargs = {}
                if name == "senate":
                    last_full = float(self.db.kv_get("senate:last_full_sync") or 0)
                    full = time.time() - last_full > self.s.senate_full_sync_s
                    kwargs = {"known": self.db.known_ids("senate"), "full": full}
                new = seen = 0
                for item in fn(self.client, **kwargs):
                    seen += 1
                    new += self.db.upsert_discovered(
                        source=item.source, source_id=item.source_id, title=item.title,
                        committee=item.committee, recorded_on=item.recorded_on,
                        page_url=item.page_url, media_url=item.media_url, meta=item.meta)
                if name == "senate" and kwargs.get("full"):
                    self.db.kv_set("senate:last_full_sync", str(time.time()))
                self.db.kv_set(f"{name}:last_discovery_ok", str(time.time()))
                new_counts[name] = new
                log.info("discovery %s: %d listed, %d new (%.1fs)", name, seen, new, time.time() - t0)
            except Exception as e:
                # One portal being down must not stop the other or the workers.
                log.error("discovery %s failed: %s", name, e, exc_info=not isinstance(
                    e, (net.TransientHTTPError, net.PermanentHTTPError)))
                new_counts[name] = -1
        return new_counts

    def discovery_loop(self):
        while not self.stop.is_set():
            self.discover_once()
            self.stop.wait(self.s.poll_interval_s)

    # ----------------------------------------------------------------- stages
    def _paths(self, rec: Recording) -> dict[str, Path]:
        name = safe_name(rec.source_id)
        year = rec.recorded_on[:4]
        tdir = self.s.transcript_dir / rec.source / year
        return {
            "audio": self.s.audio_dir / rec.source / year / f"{name}.m4a",
            "json": tdir / f"{name}.json",
            "txt": tdir / f"{name}.txt",
            "work": self.s.work_dir / rec.source / name,
        }

    def do_fetch(self, rec: Recording, owner: str, check: Callable[[], None]) -> None:
        p = self._paths(rec)
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.s.data_dir).free / 2**30
        if free_gb < self.s.min_free_gb:
            # Waiting is the right call: freeing space is the operator's job,
            # and a half-written download is worse than a delayed one.
            log.error("only %.1f GB free under %s (min %.1f); pausing downloads",
                      free_gb, self.s.data_dir, self.s.min_free_gb)
            raise NotReady(f"low disk space: {free_gb:.1f} GB free")
        if p["audio"].exists():
            # Idempotency: a previous run finished the file (renamed into place)
            # but crashed before updating the DB. Verify and move on.
            try:
                probe_audio(p["audio"])
                log.info("[%s] audio already present, skipping download", rec.id)
                self.db.advance(rec.id, owner, "fetch", "transcribe", audio_path=str(p["audio"]))
                return
            except Exception as e:
                log.warning("[%s] existing audio unreadable (%s); refetching", rec.id, e)
                p["audio"].unlink()

        if rec.source == "senate":
            try:
                url = sources.resolve_senate_media(self.client, rec.source_id, "HLS")
            except (net.TransientHTTPError, RuntimeError) as e:
                url = f"https://dlttx48mxf9m3.cloudfront.net/outputs/{rec.source_id}/Default/HLS/out.m3u8"
                log.warning("[%s] media API failed (%s); using known URL pattern", rec.id, e)
            strategy = "hls"
        else:
            url, strategy = rec.media_url, "mp4-ranges"

        ctx = FetchContext(self.client, p["work"], self.s.download_connections, check)
        try:
            res = fetch_audio(ctx, strategy, url, p["audio"])
        except net.PermanentHTTPError as e:
            if rec.source != "senate" or e.status not in (403, 404):
                raise
            # Some (older) Senate uploads have no HLS rendition; the MP4 always exists.
            url = sources.resolve_senate_media(self.client, rec.source_id, "MP4")
            log.warning("[%s] no HLS (%s); falling back to MP4 %s", rec.id, e, url)
            shutil.rmtree(p["work"], ignore_errors=True)
            res = fetch_audio(ctx, "full", url, p["audio"])
        self.db.advance(rec.id, owner, "fetch", "transcribe",
                        audio_path=str(res.path), media_url=url)
        shutil.rmtree(p["work"], ignore_errors=True)

    def _transcriber(self) -> Transcriber:
        tr = getattr(self._transcribers, "tr", None)
        if tr is None:
            tr = Transcriber(TranscribeOptions(
                model=self.s.whisper_model, device=self.s.whisper_device,
                compute_type=self.s.whisper_compute_type, language=self.s.language,
                window_s=self.s.window_s))
            self._transcribers.tr = tr
        return tr

    def do_transcribe(self, rec: Recording, owner: str, check: Callable[[], None]) -> None:
        p = self._paths(rec)
        if p["json"].exists() and p["txt"].exists():
            log.info("[%s] transcript already present, marking done", rec.id)
            self.db.advance(rec.id, owner, "transcribe", "done", transcript_path=str(p["json"]))
            return
        audio = Path(rec.audio_path) if rec.audio_path else p["audio"]
        if not audio.exists():
            # Self-heal: audio was deleted out from under us -> go fetch it again.
            log.warning("[%s] audio missing; sending back to fetch", rec.id)
            self.db.advance(rec.id, owner, "transcribe", "fetch")
            return
        meta = {"id": rec.id, "source": rec.source, "source_id": rec.source_id,
                "title": rec.title, "committee": rec.committee,
                "recorded_on": rec.recorded_on, "page_url": rec.page_url}
        doc = transcribe_file(self._transcriber(), audio, p["work"], p["json"], p["txt"],
                              meta, check)
        self.db.advance(rec.id, owner, "transcribe", "done", transcript_path=str(p["json"]))
        log.info("[%s] transcribed %s: %d segments, %.0fs audio in %.0fs",
                 rec.id, rec.title, len(doc["segments"]), doc["audio"]["duration_s"],
                 doc["elapsed_s"])
        shutil.rmtree(p["work"], ignore_errors=True)
        if not self.s.keep_audio:
            audio.unlink(missing_ok=True)

    # ----------------------------------------------------------------- runner
    def run_one(self, stage: str, owner: str, rec_id: int | None = None) -> bool:
        """Claim and process one job. Returns False if the queue was empty."""
        if stage == "fetch" and self._fetch_budget is not None:
            with self._limit_lock:
                if self._fetch_budget <= 0:
                    return False
                self._fetch_budget -= 1
        rec = self.db.claim(stage, owner, self.s.lease_s, self.s.process_since,
                            None if rec_id else self.s.sources, rec_id=rec_id)
        if rec is None:
            if stage == "fetch" and self._fetch_budget is not None:
                with self._limit_lock:
                    self._fetch_budget += 1
            return False
        handler = self.do_fetch if stage == "fetch" else self.do_transcribe
        log.info("[%s] %s start: %s %s (%s) attempt %d", rec.id, stage, rec.source,
                 rec.source_id, rec.title, rec.attempts + 1)
        with Heartbeat(self.db, rec.id, owner, self.s.lease_s) as hb:
            def check():
                if self.stop.is_set() or hb.lost.is_set():
                    raise Cancelled()
            try:
                handler(rec, owner, check)
            except (Cancelled, InterruptedError):
                if not hb.lost.is_set():
                    self.db.release(rec.id, owner)
                log.info("[%s] %s interrupted; will resume later", rec.id, stage)
            except KeyboardInterrupt:
                self.db.release(rec.id, owner)
                raise
            except LeaseLost as e:
                log.error("[%s] %s", rec.id, e)
            except Exception as e:
                self._record_failure(rec, owner, stage, e)
        return True

    def _record_failure(self, rec: Recording, owner: str, stage: str, e: Exception):
        kw = dict(base_s=self.s.backoff_base_s, cap_s=self.s.backoff_cap_s,
                  max_attempts=self.s.max_attempts, park_s=self.s.park_s)
        if isinstance(e, NotReady):
            # Normal while a recording is still being published; don't burn attempts.
            kw.update(delay_s=self.s.poll_interval_s, count_attempt=False)
            log.info("[%s] %s not ready: %s", rec.id, stage, e)
        elif isinstance(e, net.PermanentHTTPError) and e.status in (404, 410):
            kw.update(delay_s=self.s.park_s)
            log.error("[%s] %s: media gone (%s); parking", rec.id, stage, e)
        else:
            log.error("[%s] %s failed: %s", rec.id, stage, e,
                      exc_info=not isinstance(e, net.TransientHTTPError))
        msg = f"{type(e).__name__}: {e}"
        if not isinstance(e, (NotReady, net.TransientHTTPError, net.PermanentHTTPError)):
            msg += "\n" + "".join(traceback.format_exception(e)[-3:])
        try:
            attempts, delay = self.db.fail(rec.id, owner, msg, **kw)
            log.info("[%s] %s retry scheduled in %.0fs (attempt %d)", rec.id, stage, delay, attempts)
        except LeaseLost as le:
            log.error("[%s] %s", rec.id, le)

    def worker_loop(self, stage: str, idle_s: float = 10.0, exit_when_idle: Callable[[], bool] | None = None):
        owner = f"{socket.gethostname()}:{os.getpid()}:{threading.current_thread().name}"
        while not self.stop.is_set():
            try:
                worked = self.run_one(stage, owner)
            except Exception as e:   # DB locked for 30s etc.: never let a worker die
                log.exception("worker %s error: %s", owner, e)
                worked = False
            if not worked:
                if exit_when_idle and exit_when_idle():
                    return
                self.stop.wait(idle_s)

    def process(self, rec_id: int) -> None:
        """Run the remaining stages for one recording in the foreground."""
        owner = f"{socket.gethostname()}:{os.getpid()}:manual"
        for stage in ("fetch", "transcribe"):
            rec = self.db.get(rec_id)
            if rec is None:
                raise SystemExit(f"no recording {rec_id}")
            if rec.stage == stage and not self.run_one(stage, owner, rec_id=rec_id):
                raise SystemExit(f"recording {rec_id} is currently leased by another worker; "
                                 f"try again in up to {self.s.lease_s}s")

    def run(self, stages: list[str], once: bool = False, limit: int | None = None) -> None:
        """Run the selected stages until stop is set (or, with once, until drained)."""
        self._fetch_budget = limit
        threads: list[threading.Thread] = []
        fetchers: list[threading.Thread] = []

        def fetchers_idle():
            return once and not any(t.is_alive() for t in fetchers)

        if "discover" in stages:
            if once:
                self.discover_once()
            else:
                threads.append(threading.Thread(target=self.discovery_loop, name="discovery", daemon=True))
        if "fetch" in stages:
            for i in range(self.s.fetch_workers):
                t = threading.Thread(target=self.worker_loop, name=f"fetch-{i}", daemon=True,
                                     args=("fetch", 1.0 if once else 10.0, (lambda: True) if once else None))
                fetchers.append(t)
            threads += fetchers
        if "transcribe" in stages:
            for i in range(self.s.transcribe_workers):
                threads.append(threading.Thread(
                    target=self.worker_loop, name=f"transcribe-{i}", daemon=True,
                    args=("transcribe", 1.0 if once else 10.0, fetchers_idle if once else None)))
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            log.info("interrupt: finishing current step and shutting down...")
            self.stop.set()
            for t in threads:
                t.join(timeout=60)
