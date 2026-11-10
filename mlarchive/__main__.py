"""Command line entry point:  python -m mlarchive <command>

  run        long-running service: poll portals, fetch audio, transcribe
  run --once discover, drain the queues, exit (cron-friendly / demo)
  discover   poll the portals once and record new recordings
  process N  run fetch+transcribe for recording N now
  status     queue counts and recent problems
  retry      make waiting/parked jobs eligible now
  reset      push one recording back to an earlier stage
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import signal
import sys
import threading
import time

from .config import load_settings
from .db import Database


def setup_logging(log_dir, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-12s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    fileh = logging.handlers.RotatingFileHandler(
        log_dir / "mlarchive.log", maxBytes=10 * 2**20, backupCount=5, encoding="utf-8")
    fileh.setFormatter(fmt)
    root.handlers[:] = [console, fileh]
    for noisy in ("httpx", "httpx2", "httpcore", "faster_whisper", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlarchive", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="TOML settings file (default: $MLA_CONFIG)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--once", action="store_true", help="drain queues then exit")
    r.add_argument("--stages", default="discover,fetch,transcribe")
    r.add_argument("--limit", type=int, help="with --once: fetch at most N recordings")
    d = sub.add_parser("discover")
    d.add_argument("--source", action="append", choices=["house", "senate"])
    pr = sub.add_parser("process", help="fetch+transcribe one recording now")
    pr.add_argument("id", type=int)
    sub.add_parser("status")
    rt = sub.add_parser("retry")
    rt.add_argument("id", nargs="?", type=int)
    rt.add_argument("--stage", choices=["fetch", "transcribe"])
    rs = sub.add_parser("reset")
    rs.add_argument("id", type=int)
    rs.add_argument("stage", choices=["fetch", "transcribe"])

    args = ap.parse_args(argv)
    s = load_settings(args.config)
    setup_logging(s.log_dir, args.verbose)
    db = Database(s.db_path)

    if args.cmd == "status":
        print(f"data dir: {s.data_dir.resolve()}   processing recordings since {s.process_since}")
        for k, v in db.counts(s.process_since).items():
            print(f"  {k:28s} {v}")
        for src in s.sources:
            ok = db.kv_get(f"{src}:last_discovery_ok")
            print(f"  last successful {src} discovery: "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(ok))) if ok else 'never'}")
        rows = list(db.problems())
        if rows:
            print("\nrecent problems:")
            for row in rows:
                when = max(0, row["next_attempt_at"] - time.time())
                err = (row["last_error"] or "").splitlines()[0][:110]
                print(f"  #{row['id']} {row['source']}/{row['source_id']} [{row['stage']}] "
                      f"attempts={row['attempts']} retry_in={when:.0f}s  {err}")
        return 0
    if args.cmd == "retry":
        n = db.retry_now(args.id, args.stage)
        print(f"{n} job(s) made eligible now")
        return 0
    if args.cmd == "reset":
        db.reset_stage(args.id, args.stage)
        print(f"recording {args.id} -> {args.stage}")
        return 0

    from .pipeline import Pipeline   # heavy imports only when needed
    stop = threading.Event()
    pipe = Pipeline(s, db, stop)

    def on_signal(signum, _frame):
        logging.getLogger("mlarchive").info("signal %s: shutting down gracefully", signum)
        stop.set()
    for sig in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), on_signal)

    if args.cmd == "discover":
        pipe.discover_once(args.source)
        return 0
    if args.cmd == "process":
        pipe.process(args.id)
        rec = db.get(args.id)
        print(f"recording {args.id}: stage={rec.stage} transcript={rec.transcript_path}")
        return 0
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    pipe.run(stages, once=args.once, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
