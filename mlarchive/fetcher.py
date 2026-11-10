"""Get a recording's audio onto disk as an .m4a, as cheaply and resumably as possible.

Strategies (picked per source, with fallback):

  mp4-ranges  House. Range-request only the audio chunks of the remote MP4
              (see mp4audio.py): ~180 MB instead of ~3.5 GB per hearing.
  hls         Senate. Its MP4s interleave audio in ~200k tiny pieces, so range
              tricks don't pay off; instead we take the lowest-bitrate HLS
              rendition (6-second .ts segments) and copy its audio track out.
  full        Fallback for an MP4 layout we can't plan (e.g. fragmented):
              resumable whole-file download, then copy the audio track out.

Idempotency / resumability, the same rule everywhere: every unit of work
(a batch of audio chunks, an HLS segment, the whole-file download) is
written to `<name>.part` and atomically renamed when complete. On restart,
completed units are skipped and only missing ones are fetched. A manifest
records which remote version (size/ETag or playlist) the pieces belong to;
if the remote changed, the pieces are discarded. The final .m4a also
appears via rename, so "file exists" means "file is complete".
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin

import av
import httpx

from . import mp4audio, net
from .mp4audio import UnsupportedMp4

log = logging.getLogger(__name__)

BATCH_BYTES = 4 * 1024 * 1024


class NotReady(Exception):
    """The media isn't fully published yet (still uploading / live). Retry later."""


@dataclass
class FetchContext:
    client: httpx.Client
    work_dir: Path                     # scratch dir for this recording
    connections: int
    check: Callable[[], None]          # raises if we should stop (shutdown / lease lost)


@dataclass
class FetchResult:
    path: Path
    strategy: str
    downloaded_bytes: int
    duration_s: float


# ---------------------------------------------------------------- utilities
def atomic_write(path: Path, data: bytes | Iterable[bytes]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        if isinstance(data, (bytes, bytearray)):
            f.write(data)
        else:
            for piece in data:
                f.write(piece)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_manifest(work: Path, expect: dict) -> None:
    """Wipe scratch pieces if they belong to a different remote version."""
    mf = work / "manifest.json"
    if mf.exists():
        try:
            if json.loads(mf.read_text()) == expect:
                return
        except (OSError, ValueError):
            pass
        log.info("remote changed or manifest unreadable; discarding partial pieces in %s", work)
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    atomic_write(mf, json.dumps(expect).encode())


def _run_parallel(tasks: list[Callable[[], int]], n: int, check: Callable[[], None]) -> int:
    """Run download tasks on n threads; first error cancels the rest."""
    if not tasks:
        return 0
    total = 0
    abort = threading.Event()

    def wrap(t):
        if abort.is_set():
            return 0
        check()
        return t()

    with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
        futures = [pool.submit(wrap, t) for t in tasks]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            if f.exception():
                abort.set()
                for p in pending:
                    p.cancel()
                raise f.exception()
        for f in futures:
            total += f.result()
    return total


def probe_audio(path: Path) -> float:
    """Open the file and return the audio duration in seconds (validates it)."""
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise ValueError(f"{path} has no audio stream")
        s = c.streams.audio[0]
        if s.duration and s.time_base:
            return float(s.duration * s.time_base)
        if c.duration:
            return c.duration / 1_000_000
        # Fall back to the last packet timestamp.
        last = 0.0
        for p in c.demux(s):
            if p.pts is not None:
                last = float(p.pts * s.time_base)
        return last


def remux_audio(src, dst: Path, check: Callable[[], None]) -> None:
    """Copy (no re-encode) the first audio track of `src` into an .m4a at dst.

    `src` is a path or a readable file object (non-seekable is fine).
    """
    tmp = dst.with_name(dst.name + ".part")
    with av.open(src, mode="r") as inp:
        if not inp.streams.audio:
            raise ValueError("source has no audio stream")
        a_in = inp.streams.audio[0]
        with av.open(str(tmp), mode="w", format="ipod") as out:
            a_out = out.add_stream_from_template(a_in)
            n = 0
            shift = 0          # applied after a timestamp discontinuity
            last_dts = None
            for pkt in inp.demux(a_in):
                if pkt.dts is None:
                    continue
                # HLS discontinuities (or an encoder restart) make timestamps jump
                # backwards; the muxer requires them to increase, so shift the
                # rest of the stream to continue where the previous part ended.
                dts = pkt.dts + shift
                if last_dts is not None and dts <= last_dts:
                    shift += last_dts + (pkt.duration or 1) - dts
                    dts = pkt.dts + shift
                pkt.dts, pkt.pts = dts, (pkt.pts + shift if pkt.pts is not None else dts)
                last_dts = dts
                pkt.stream = a_out
                out.mux(pkt)
                n += 1
                if n % 5000 == 0:
                    check()
    os.replace(tmp, dst)


class ChainReader(io.RawIOBase):
    """Read several files as one stream (MPEG-TS segments are concatenable)."""

    def __init__(self, paths: list[Path]):
        self._paths = list(paths)
        self._f = None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self._f is None:
                if not self._paths:
                    return 0
                self._f = open(self._paths.pop(0), "rb")
            n = self._f.readinto(b)
            if n:
                return n
            self._f.close()
            self._f = None

    def close(self):
        if self._f:
            self._f.close()
        super().close()


# ------------------------------------------------------------- mp4-ranges
def fetch_mp4_ranges(ctx: FetchContext, url: str, dst: Path) -> FetchResult:
    head = net.request(ctx.client, "HEAD", url)
    size = int(head.headers.get("content-length") or 0)
    etag = head.headers.get("etag")
    if not size:
        raise NotReady(f"no content-length for {url}")
    if head.headers.get("accept-ranges", "").lower() != "bytes":
        raise UnsupportedMp4("server does not advertise byte ranges")

    def read(off: int, n: int) -> bytes:
        ctx.check()
        return net.get_range(ctx.client, url, off, n, etag)

    try:
        plan = mp4audio.plan_remote(read, size)
    except UnsupportedMp4 as e:
        if "missing moov" in str(e):
            # moov is written last; an MP4 without it is still being uploaded.
            raise NotReady(str(e)) from e
        raise

    _load_manifest(ctx.work_dir, {"url": url, "size": size, "etag": etag,
                                  "strategy": "mp4-ranges", "batch": BATCH_BYTES})

    # Group chunks into ~4 MB batches; each batch becomes one piece file.
    batches: list[list[mp4audio.Chunk]] = [[]]
    acc = 0
    for ch in plan.chunks:
        if acc >= BATCH_BYTES:
            batches.append([])
            acc = 0
        batches[-1].append(ch)
        acc += ch.size

    def task(i: int, chunks: list[mp4audio.Chunk]) -> Callable[[], int]:
        piece = ctx.work_dir / f"b{i:05d}.bin"
        expected = sum(c.size for c in chunks)

        def run() -> int:
            if piece.exists() and piece.stat().st_size == expected:
                return 0
            out = bytearray()
            fetched = 0
            ci = 0
            for off, length in mp4audio.coalesce(chunks):
                data = read(off, length)
                fetched += length
                # Pull our chunks back out of the (possibly gap-including) range.
                while ci < len(chunks) and chunks[ci].src_offset + chunks[ci].size <= off + length:
                    c = chunks[ci]
                    out += data[c.src_offset - off: c.src_offset - off + c.size]
                    ci += 1
            if len(out) != expected:
                raise RuntimeError(f"batch {i}: assembled {len(out)} != {expected}")
            atomic_write(piece, bytes(out))
            return fetched

        return run

    fetched = _run_parallel([task(i, b) for i, b in enumerate(batches)], ctx.connections, ctx.check)

    def pieces():
        yield plan.header
        for i in range(len(batches)):
            with open(ctx.work_dir / f"b{i:05d}.bin", "rb") as f:
                yield f.read()

    tmp = ctx.work_dir / "assembled.m4a"
    atomic_write(tmp, pieces())
    if tmp.stat().st_size != plan.total_size:
        raise RuntimeError(f"assembled size {tmp.stat().st_size} != planned {plan.total_size}")
    duration = probe_audio(tmp)
    if duration <= 0:
        raise RuntimeError("assembled audio has zero duration")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dst)
    log.info("mp4-ranges: %s -> %s (%.1f MB fetched of %.1f MB remote, %.0fs audio)",
             url, dst, fetched / 2**20, size / 2**20, duration)
    return FetchResult(dst, "mp4-ranges", fetched, duration)


# -------------------------------------------------------------------- hls
_ATTR = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')


def parse_master(text: str, base: str) -> list[tuple[int, str]]:
    """[(bandwidth, absolute_variant_url)] from a master playlist."""
    out, lines = [], [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:"):
            attrs = dict(_ATTR.findall(line.split(":", 1)[1]))
            uri = next((l for l in lines[i + 1:] if l and not l.startswith("#")), None)
            if uri:
                out.append((int(attrs.get("BANDWIDTH", "0") or 0), urljoin(base, uri)))
    return out


def parse_media(text: str, base: str) -> tuple[list[tuple[float, str]], bool]:
    """([(duration, absolute_segment_url)], has_endlist) from a media playlist."""
    segs, dur, ended = [], None, False
    for line in (l.strip() for l in text.splitlines()):
        if line.startswith("#EXTINF:"):
            dur = float(line.split(":", 1)[1].split(",")[0])
        elif line.startswith("#EXT-X-KEY:") and "METHOD=NONE" not in line:
            raise UnsupportedMp4("encrypted HLS not supported")
        elif line == "#EXT-X-ENDLIST":
            ended = True
        elif line and not line.startswith("#"):
            segs.append((dur or 0.0, urljoin(base, line)))
            dur = None
    return segs, ended


def fetch_hls(ctx: FetchContext, master_url: str, dst: Path) -> FetchResult:
    text = net.request(ctx.client, "GET", master_url).text
    variants = parse_master(text, master_url)
    media_url = min(variants)[1] if variants else master_url   # lowest bitrate is plenty for speech
    segs, ended = parse_media(net.request(ctx.client, "GET", media_url).text, media_url)
    if not ended:
        raise NotReady("HLS playlist has no ENDLIST (still live?)")
    if not segs:
        raise RuntimeError("HLS playlist has no segments")
    _load_manifest(ctx.work_dir, {"url": media_url, "segments": len(segs), "strategy": "hls"})

    def task(i: int, url: str) -> Callable[[], int]:
        piece = ctx.work_dir / f"s{i:05d}.ts"

        def run() -> int:
            if piece.exists() and piece.stat().st_size > 0:
                return 0
            data = net.request(ctx.client, "GET", url).content
            if not data or data[0] != 0x47:   # every TS packet starts with sync byte 0x47
                raise RuntimeError(f"segment {i} is not MPEG-TS ({len(data)} bytes)")
            atomic_write(piece, data)
            return len(data)

        return run

    fetched = _run_parallel([task(i, u) for i, (_, u) in enumerate(segs)], ctx.connections, ctx.check)

    paths = [ctx.work_dir / f"s{i:05d}.ts" for i in range(len(segs))]
    tmp = ctx.work_dir / "assembled.m4a"
    with ChainReader(paths) as stream:
        remux_audio(stream, tmp, ctx.check)
    duration = probe_audio(tmp)
    expected = sum(d for d, _ in segs)
    if expected and abs(duration - expected) > max(5.0, 0.02 * expected):
        raise RuntimeError(f"HLS audio {duration:.0f}s vs playlist {expected:.0f}s")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dst)
    log.info("hls: %s -> %s (%d segments, %.1f MB, %.0fs audio)",
             media_url, dst, len(segs), fetched / 2**20, duration)
    return FetchResult(dst, "hls", fetched, duration)


# ------------------------------------------------------------------- full
def download_resumable(ctx: FetchContext, url: str, dst: Path) -> int:
    """Whole-file download that resumes from the .part file's length."""
    head = net.request(ctx.client, "HEAD", url)
    size = int(head.headers.get("content-length") or 0)
    etag = head.headers.get("etag")
    _load_manifest(ctx.work_dir, {"url": url, "size": size, "etag": etag, "strategy": "full"})
    part = dst.with_name(dst.name + ".part")
    if dst.exists() and (not size or dst.stat().st_size == size):
        return 0
    fetched = 0
    stalls = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        if stalls >= 5:
            raise net.TransientHTTPError(f"download of {url} stalled at {have} bytes")
        headers = {"Range": f"bytes={have}-"} if have else {}
        if have and etag:
            headers["If-Range"] = etag
        try:
            with ctx.client.stream("GET", url, headers=headers) as r:
                if r.status_code >= 400:
                    raise net.TransientHTTPError(f"HTTP {r.status_code}")
                mode = "ab" if r.status_code == 206 else "wb"   # 200 -> start over
                with open(part, mode) as f:
                    for block in r.iter_bytes():   # unbuffered: a cut loses nothing written so far
                        f.write(block)
                        fetched += len(block)
                        ctx.check()
            if not size:
                break
        except httpx.TransportError as e:
            # Stream cut out: loop and resume from what we have.
            now_have = part.stat().st_size if part.exists() else 0
            stalls = 0 if now_have > have else stalls + 1
            log.warning("download interrupted at %d bytes (%s); resuming", now_have, e)
            ctx.check()
    os.replace(part, dst)
    return fetched


def fetch_full(ctx: FetchContext, url: str, dst: Path) -> FetchResult:
    src = ctx.work_dir / "source.bin"
    fetched = download_resumable(ctx, url, src)
    tmp = ctx.work_dir / "assembled.m4a"
    remux_audio(str(src), tmp, ctx.check)
    duration = probe_audio(tmp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dst)
    return FetchResult(dst, "full", fetched, duration)


def fetch_audio(ctx: FetchContext, strategy: str, url: str, dst: Path) -> FetchResult:
    if strategy == "hls":
        return fetch_hls(ctx, url, dst)
    if strategy == "mp4-ranges":
        try:
            return fetch_mp4_ranges(ctx, url, dst)
        except UnsupportedMp4 as e:
            log.warning("mp4-ranges not possible for %s (%s); falling back to full download", url, e)
            return fetch_full(ctx, url, dst)
    if strategy == "full":
        return fetch_full(ctx, url, dst)
    raise ValueError(f"unknown strategy {strategy}")
