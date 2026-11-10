# Michigan Legislative Archival

Watches the Michigan House and Senate video archives, pulls the **audio** of every new
hearing, and transcribes it **locally** with Whisper. It runs unattended: network blips,
dropped streams, crashes, half-uploaded files and full disks are retried and recovered
automatically.

| Portal | What it really is | How we get audio |
|---|---|---|
| House `house.mi.gov/VideoArchive` | Server-rendered HTML linking to plain MP4 files (`/ArchiveVideoFiles/<CODE-MMDDYY>.mp4`, ~1–4 GB, ~1.3 MB/s) | **Byte-range extraction**: read the MP4 index, download only the audio bytes (~5% of the file) |
| Senate `cloud.castus.tv/vod/misenate` | React app over a JSON API on AWS Lambda; media on CloudFront (MP4 + HLS) | **Lowest-bitrate HLS rendition**, audio copied out of the 6 s segments |

Measured on real recordings:

| Recording | Remote size | Downloaded | Wall time |
|---|---|---|---|
| House Appropriations, 78 min | 3,531 MB | 178 MB | ~30 s fetch, 67 s transcribe (RTX 5070 Ti) |
| House Session, 6 min | 242 MB | 13 MB | 4 s |
| Senate Session, 22 s | – | 2.4 MB | 2 s fetch, 3 s transcribe |

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt      # (bin/ instead of Scripts/ on Linux/macOS)
.venv/Scripts/python -m pytest                          # 50+ offline tests, ~4 s
.venv/Scripts/python -m mlarchive run --once --limit 2  # discover, fetch+transcribe 2 newest, exit
.venv/Scripts/python -m mlarchive status
.venv/Scripts/python -m mlarchive run                   # the long-running service
```

No ffmpeg install, no API keys, no paid services. PyAV bundles FFmpeg's libraries; Whisper
weights download once from Hugging Face and are cached.

Output:

```
data/
  state.db                                   SQLite: recordings, events (audit log), kv
  audio/house/2026/AGRR-022526.m4a           audio only, original AAC, not re-encoded
  transcripts/house/2026/AGRR-022526.json    segments with timestamps + metadata
  transcripts/house/2026/AGRR-022526.txt     "[00:11:44] ... will come to order."
  work/...                                   resumable partial downloads / checkpoints
  logs/mlarchive.log                         rotating, 10 MB x 5
```

### Commands

| Command | Purpose |
|---|---|
| `run` | Service: discovery every 15 min + fetch workers + transcribe worker |
| `run --once [--limit N]` | Discover, process the queue (or N items), exit. Cron-friendly |
| `run --stages fetch` | Run only some stages (e.g. transcription on a GPU box) |
| `discover [--source house]` | Poll the portals once |
| `status` | Queue counts, last successful poll per portal, recent errors |
| `process ID` | Fetch + transcribe one recording now, ignoring backoff |
| `retry [ID] [--stage S]` | Make waiting/parked jobs eligible immediately |
| `reset ID fetch\|transcribe` | Send a recording back to an earlier stage (e.g. re-transcribe with a new model) |

Configuration: `config.example.toml`, or env vars `MLA_<SETTING>` (e.g. `MLA_WHISPER_MODEL=small`).

---

## Architecture

```
            ┌──────────────┐  every 15 min
            │  discovery   │──── House HTML / Senate JSON API
            └──────┬───────┘
                   │ INSERT ... ON CONFLICT DO NOTHING
                   ▼
   ┌─────────────────────────────────────────────┐
   │ SQLite  recordings(stage, lease, attempts…) │◄── status / retry / reset (CLI)
   └───────┬──────────────────────────▲──────────┘
     claim │ stage=fetch      advance │ stage=transcribe → done
           ▼                          │
   ┌───────────────┐   .m4a   ┌───────┴────────┐
   │ fetch workers │ ───────► │ transcribe     │──► transcripts/*.json, *.txt
   │  (x2 threads) │  on disk │ worker (GPU)   │
   └───────────────┘          └────────────────┘
```

**How the services talk to each other:** they don't call each other at all. The SQLite
database is the only interface:

* Discovery inserts rows (`stage='fetch'`).
* A worker *claims* a row with one atomic statement:
  `UPDATE recordings SET lease_owner=?, lease_expires_at=now+120s WHERE id=(SELECT … newest ready row …) RETURNING *`.
  SQLite runs that under its write lock, so two workers can never get the same row.
* While working, a heartbeat thread extends the lease every 40 s.
* On success the worker *advances* the row (`fetch → transcribe → done`) and releases the
  lease, in the same statement that checks `lease_owner = me`. On failure it records the
  error and schedules a retry.
* Files are handed between stages on disk (`audio_path` column). No large data passes
  through the DB.

Because of this, each stage can run as a thread in one process (the default) or as a separate
process (`run --stages discover,fetch` on one terminal, `run --stages transcribe` on another),
and you can start two copies of the service by accident without any duplicate work.
Ordering is **newest first**, so a hearing that just ended jumps ahead of any backlog.

### Modules

| File | Responsibility |
|---|---|
| `mlarchive/sources.py` | Portal scrapers → `Discovered` records (House HTML parser, Senate API pager) |
| `mlarchive/mp4audio.py` | MP4 box parser; plans the audio-only download and builds the new `.m4a` header |
| `mlarchive/fetcher.py` | The three download strategies, all resumable; audio remux and validation |
| `mlarchive/transcriber.py` | PCM decoding, windowing at quiet points, faster-whisper, checkpoints |
| `mlarchive/db.py` | Schema, claim/lease/heartbeat/advance/fail, backoff, audit events |
| `mlarchive/net.py` | HTTP client, retry policy, verified range reads |
| `mlarchive/pipeline.py` | Discovery loop, worker loops, error classification, shutdown |
| `mlarchive/__main__.py` | CLI, logging |

---

## Idempotency: why it's safe to run again (and again)

The rule everywhere: **a file only appears under its final name once it is complete**
(write `x.part`, `fsync`, `os.replace` — atomic on Windows and POSIX). So "file exists" means
"file is complete," and every step can check before doing work.

| Layer | Mechanism |
|---|---|
| Discovery | `UNIQUE(source, source_id)` with `ON CONFLICT DO NOTHING`. Ids are the portal's own (House filename, Senate `_id`). Re-discovering refreshes titles but never touches progress. |
| Claiming | Atomic lease + fencing: a worker whose lease expired can't advance or fail a job someone else now owns (`LeaseLost`). |
| Fetch | If the final `.m4a` exists and decodes, skip straight to transcribe (covers "crashed after the rename but before the DB update"). |
| Partial downloads | Work is split into units (4 MB batches of audio chunks / HLS segments / one whole file). Each finished unit is its own file. A restart re-downloads **only missing units**. Tested by hard-killing mid-download: the rerun fetched 120 MB instead of 178 MB. |
| Remote changed | A `manifest.json` records size+ETag (or playlist) for the pieces. If it differs, the pieces are discarded. Range requests send `If-Range: <etag>`, so a file replaced mid-download is detected (`SourceChanged`) instead of being stitched from two versions. |
| Transcribe | Final `.json` + `.txt` exist → mark done. Long recordings are checkpointed per 30-minute window and resume at the last finished window. |

The big one, "don't re-download the massive video," is handled twice over: we never download
the video in the first place (only audio bytes), and the audio we do download is never fetched
twice.

---

## Failure handling ("Survive")

Two retry layers:

1. **In-process** (`net.request`): connection errors, timeouts, 408/425/429/5xx get up to 6
   tries with jittered exponential backoff (1 s, 2 s, 4 s… capped at 60 s; honours `Retry-After`).
   This absorbs blips without failing the job.
2. **Job queue** (`db.fail`): if a job still fails, the error is stored on the row and it's
   retried later: 30 s, 60 s, 2 min, … capped at 6 h. After 8 attempts it is *parked*
   (retried once a day) rather than dropped, so a multi-day portal outage heals itself.
   `status` shows exactly what is failing and why.

| Situation | Behaviour |
|---|---|
| Wi-Fi drops / connection reset mid-file | Range resume from the last byte written (full download) or last finished batch/segment |
| Stream cut, server returns short body | Verified: range length checked, retried |
| Process killed, machine reboots | Lease expires after 120 s, job reclaimed by the next run, partial work reused |
| Ctrl+C / SIGTERM / SIGBREAK | Workers stop at the next checkpoint, release their lease without counting an attempt |
| House MP4 still uploading (no `moov` yet) | `NotReady`: wait one poll interval, attempts not counted |
| Senate video still transcoding / HLS without `ENDLIST` | Not listed yet / `NotReady` |
| Senate API returns 502 (Lambda 6 MB response cap) | Page is re-requested as two half-size pages, down to a single record |
| One portal down | Other portal and all workers keep going; discovery error logged |
| Video deleted (404/410) | Parked for a day, not retried hot |
| MP4 layout we can't plan (fragmented etc.) | Falls back to resumable full download + local audio copy |
| CUDA libraries missing / GPU error | Falls back to CPU Whisper (`small`, int8) automatically |
| Disk below 5 GB free | Downloads pause (logged as error), resume when space is freed |
| Audio deleted after fetch | Transcribe stage sends the job back to fetch |
| Portal markup changes (zero videos parsed) | Discovery raises and logs loudly instead of silently finding nothing |

---

## Why these choices

**Python.** The two hard parts, media containers and speech recognition, have their best
libraries in Python (PyAV = FFmpeg's libraries; faster-whisper). The rest is I/O-bound glue,
where Python threads are fine.

**SQLite, not Redis/RabbitMQ/Postgres.** The workload is a few hundred recordings a year, all
on one machine. SQLite gives transactions, crash safety (WAL), a queryable audit log and zero
operations (one file you can open and inspect). Leases give the "at-least-once, never twice
concurrently" guarantee you would otherwise need a broker for. If this ever needed several
machines, the same schema and claim query port directly to Postgres (`FOR UPDATE SKIP LOCKED`).

**Threads in one process, not microservices.** Fewer moving parts to keep alive. The stages
are still decoupled (they only share the DB), so splitting them into processes is a CLI flag,
not a rewrite.

**Byte-range audio extraction (House).** The House server delivers ~1.3 MB/s, so a 3.5 GB
hearing takes about 45 minutes, and the archive holds hundreds. The audio is ~5% of the bytes.
Parsing `moov` (~2 MB) gives exact byte offsets of every audio chunk, so we fetch just those
in parallel and write a new `.m4a` whose chunk-offset table points into a compact `mdat`.
No re-encode: the audio is bit-identical (a test asserts this). Costs about 250 lines and
saves ~95% bandwidth, time and disk.

**HLS 360p (Senate).** The Senate MP4s interleave audio in ~200k tiny pieces (one AAC frame
each), so range requests would be ~200k requests or would drag in most of the video. The 360p
HLS rendition is ~6× smaller than the 1080p MP4. Speech audio doesn't need 1080p video, and the
6-second segments are natural resume points.

**faster-whisper `large-v3-turbo`.** Near-`large-v3` accuracy at several times the speed.
On an RTX 5070 Ti: ~70× realtime. VAD skips the long silences (recesses, roll-call pauses).
`condition_on_previous_text=False` prevents Whisper's known repetition loops on multi-hour
audio. Audio is decoded by PyAV straight to 16 kHz arrays, in 30-minute windows cut at the
quietest moment near the boundary. That keeps memory bounded on 5-hour sessions and gives a
checkpoint per window.

**`truststore`.** `house.mi.gov` serves an incomplete TLS certificate chain. Browsers and the
Windows trust store fetch the missing intermediate; Python's bundled CA list doesn't. Rather
than disable verification, we verify against the OS store.

**Lean dependencies.** httpx, beautifulsoup4, PyAV, faster-whisper, truststore. No ffmpeg
binary, no scheduler framework, no ORM, no broker.

---

## Running it as a service

The process is meant to run forever and be restarted by the OS if it ever exits.

* **Windows:** Task Scheduler → "At startup", action `…\.venv\Scripts\python.exe -m mlarchive run`,
  start in the repo folder, "If the task fails, restart every 1 minute".
* **Linux:** a systemd unit with `Restart=always`; `systemctl stop` sends SIGTERM, which shuts down gracefully.
* **Cron style:** `run --once` every 15 minutes is also safe, because leases prevent overlapping runs from colliding.

## Tests

`python -m pytest` runs 50+ tests offline in a few seconds. Real MP4 / MPEG-TS files are
generated with PyAV and served by a fake HTTP server (`httpx.MockTransport`) with Range/ETag
support and scriptable failures. Covered: bit-identical audio extraction, resume after a
crash, remote file changes, dropped streams, transient 5xx/timeouts, Senate 502 page
splitting, lease exclusivity/expiry/fencing, backoff and parking, transcription checkpoint
resume, shutdown semantics.

## Known limits / next steps

* Polling (15 min) rather than push. Neither portal offers notifications, and a recording
  appears only after the session ends, so this is near-real-time for the use case.
* Live streams are ignored; only finished recordings are archived.
* No speaker diarization (would add pyannote + a Hugging Face token).
* Audio is kept as original AAC (~140 MB/hour for House). `keep_audio = false` deletes it after
  transcription; re-encoding to Opus would cut it ~8×.
