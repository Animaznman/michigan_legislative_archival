"""Extract the audio track of a remote MP4 using HTTP range requests only.

Why this exists: a House hearing is a ~3.7 GB MP4 served at ~1.3 MB/s
(~45 minutes to download). The AAC audio is only ~5% of those bytes. An MP4
keeps a byte-offset table for every track in its `moov` box, so we can:

  1. Walk the top-level boxes (a few 16-byte range reads) to find `moov`.
  2. Download just `moov` (a couple of MB) and parse the audio track's
     sample tables (stsc / stsz / stco|co64) into a list of audio "chunks"
     (offset, size) inside `mdat`.
  3. Range-download only those chunks.
  4. Write a new, small `.m4a`: ftyp + moov (audio track only, chunk offsets
     rewritten) + mdat (the audio chunks back to back).

No audio is decoded or re-encoded; the output is bit-identical audio.
This module only parses and plans. Downloading lives in fetcher.py.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable

# Boxes whose payload is just more boxes (the ones we need to descend into).
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}

RangeReader = Callable[[int, int], bytes]  # (offset, length) -> bytes


class UnsupportedMp4(Exception):
    """The file is not a layout we can extract from (e.g. fragmented MP4)."""


@dataclass(frozen=True) # Fields cannot be modified after instantiation
class Box:
    type: bytes
    start: int        # offset of the box header within the buffer/file
    header_len: int   # 8, or 16 for 64-bit sizes
    size: int         # total size including header

    @property
    def payload_start(self) -> int:
        return self.start + self.header_len

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class Chunk:
    src_offset: int   # absolute offset in the remote file
    size: int


@dataclass
class AudioPlan:
    """Everything needed to build the audio-only file."""
    chunks: list[Chunk]
    header: bytes                # ftyp + rewritten moov + mdat header
    total_size: int              # final .m4a size in bytes
    remote_size: int

    @property
    def audio_bytes(self) -> int:
        return sum(c.size for c in self.chunks)


def _parse_header(buf: bytes, pos: int, limit: int) -> Box:
    if pos + 8 > limit:
        raise UnsupportedMp4(f"truncated box header at {pos}")
    size, typ = struct.unpack_from(">I4s", buf, pos)
    hl = 8
    if size == 1:
        if pos + 16 > limit:
            raise UnsupportedMp4(f"truncated large box header at {pos}")
        size = struct.unpack_from(">Q", buf, pos + 8)[0]
        hl = 16
    elif size == 0:  # box extends to end of buffer/file
        size = limit - pos
    if size < hl:
        raise UnsupportedMp4(f"invalid box size {size} at {pos}")
    return Box(typ, pos, hl, size)


def iter_boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        box = _parse_header(buf, pos, end)
        yield box
        pos = box.end


def find_top_level(read: RangeReader, file_size: int) -> dict[bytes, Box]:
    """Locate top-level boxes with tiny range reads (no need to scan mdat)."""
    boxes: dict[bytes, Box] = {}
    pos = 0
    while pos + 8 <= file_size:
        head = read(pos, min(16, file_size - pos))
        size, typ = struct.unpack_from(">I4s", head, 0)
        hl = 8
        if size == 1:
            size, hl = struct.unpack_from(">Q", head, 8)[0], 16
        elif size == 0:  # "extends to end of file"
            size = file_size - pos
        if size < hl or pos + size > file_size:
            raise UnsupportedMp4(f"bad top-level box {typ!r} size {size} at {pos}")
        boxes.setdefault(typ, Box(typ, pos, hl, size))
        pos += size
    if b"moof" in boxes:
        raise UnsupportedMp4("fragmented MP4 (moof) not supported")
    if b"moov" not in boxes or b"mdat" not in boxes:
        raise UnsupportedMp4(f"missing moov/mdat; top-level boxes: {list(boxes)}")
    return boxes


def _child(buf: bytes, parent: Box, typ: bytes) -> Box | None:
    for b in iter_boxes(buf, parent.payload_start, parent.end):
        if b.type == typ:
            return b
    return None


def _path(buf: bytes, parent: Box, *types: bytes) -> Box | None:
    cur: Box | None = parent
    for t in types:
        if cur is None:
            return None
        cur = _child(buf, cur, t)
    return cur


def _handler_type(buf: bytes, trak: Box) -> bytes:
    hdlr = _path(buf, trak, b"mdia", b"hdlr")
    if hdlr is None:
        return b""
    # FullBox: version(1) flags(3) pre_defined(4) handler_type(4)
    return buf[hdlr.payload_start + 8: hdlr.payload_start + 12]


def _chunk_table(buf: bytes, stbl: Box) -> tuple[list[int], list[int], Box]:
    """Return (chunk_offsets, chunk_sizes, offset_box) for one track."""
    stco = _child(buf, stbl, b"stco") or _child(buf, stbl, b"co64")
    stsc = _child(buf, stbl, b"stsc")
    stsz = _child(buf, stbl, b"stsz")
    if not (stco and stsc and stsz):
        raise UnsupportedMp4("audio track missing stco/stsc/stsz")

    # Chunk offsets
    p = stco.payload_start + 4
    (n_chunks,) = struct.unpack_from(">I", buf, p)
    fmt = ">%dI" % n_chunks if stco.type == b"stco" else ">%dQ" % n_chunks
    offsets = list(struct.unpack_from(fmt, buf, p + 4))

    # Sample sizes
    p = stsz.payload_start + 4
    uniform, n_samples = struct.unpack_from(">II", buf, p)
    sizes = [uniform] * n_samples if uniform else list(
        struct.unpack_from(">%dI" % n_samples, buf, p + 8))

    # Sample-to-chunk runs: (first_chunk[1-based], samples_per_chunk, desc_idx)
    p = stsc.payload_start + 4
    (n_runs,) = struct.unpack_from(">I", buf, p)
    runs = [struct.unpack_from(">III", buf, p + 4 + 12 * i)[:2] for i in range(n_runs)]

    chunk_sizes = []
    sample_i = 0
    for r, (first, per_chunk) in enumerate(runs):
        last = runs[r + 1][0] - 1 if r + 1 < len(runs) else n_chunks
        for _ in range(first, last + 1):
            chunk_sizes.append(sum(sizes[sample_i:sample_i + per_chunk]))
            sample_i += per_chunk
    if len(chunk_sizes) != n_chunks or sample_i != n_samples:
        raise UnsupportedMp4(
            f"inconsistent sample tables: chunks {len(chunk_sizes)}/{n_chunks}, "
            f"samples {sample_i}/{n_samples}")
    return offsets, chunk_sizes, stco


def _box_header(typ: bytes, payload_len: int) -> bytes:
    if payload_len + 8 <= 0xFFFFFFFF:
        return struct.pack(">I4s", payload_len + 8, typ)
    return struct.pack(">I4sQ", 1, typ, payload_len + 16)


FTYP = _box_header(b"ftyp", 16) + b"M4A " + struct.pack(">I", 0) + b"M4A isom"


def plan_from_moov(moov_buf: bytes, remote_size: int) -> AudioPlan:
    """Build the extraction plan from the raw bytes of the `moov` box."""
    moov = _parse_header(moov_buf, 0, len(moov_buf))
    if moov.type != b"moov":
        raise UnsupportedMp4("buffer does not start with moov")

    mvhd = _child(moov_buf, moov, b"mvhd")
    audio_trak = None
    for b in iter_boxes(moov_buf, moov.payload_start, moov.end):
        if b.type == b"trak" and _handler_type(moov_buf, b) == b"soun":
            audio_trak = b
            break
    if mvhd is None or audio_trak is None:
        raise UnsupportedMp4("no mvhd or no audio track")

    stbl = _path(moov_buf, audio_trak, b"mdia", b"minf", b"stbl")
    if stbl is None:
        raise UnsupportedMp4("audio track has no stbl")
    offsets, sizes, offset_box = _chunk_table(moov_buf, stbl)

    # New moov = mvhd + audio trak (video, text, udta tracks dropped).
    # Its size doesn't depend on the offset values, so we can lay out first.
    trak = bytearray(moov_buf[audio_trak.start:audio_trak.end])
    mvhd_bytes = moov_buf[mvhd.start:mvhd.end]
    new_moov_len = 8 + len(mvhd_bytes) + len(trak)
    audio_len = sum(sizes)
    mdat_hdr = _box_header(b"mdat", audio_len)
    data_start = len(FTYP) + new_moov_len + len(mdat_hdr)

    new_offsets = []
    pos = data_start
    for s in sizes:
        new_offsets.append(pos)
        pos += s

    # Patch the chunk-offset table inside our copy of the trak.
    table = offset_box.payload_start + 8 - audio_trak.start
    if offset_box.type == b"stco":
        if pos > 0xFFFFFFFF:
            raise UnsupportedMp4("audio larger than 4 GiB with 32-bit stco")
        struct.pack_into(">%dI" % len(new_offsets), trak, table, *new_offsets)
    else:
        struct.pack_into(">%dQ" % len(new_offsets), trak, table, *new_offsets)

    header = FTYP + _box_header(b"moov", new_moov_len - 8) + mvhd_bytes + bytes(trak) + mdat_hdr
    chunks = [Chunk(o, s) for o, s in zip(offsets, sizes)]
    for c in chunks:
        if c.src_offset + c.size > remote_size:
            raise UnsupportedMp4("chunk points past end of file")
    return AudioPlan(chunks=chunks, header=header, total_size=pos, remote_size=remote_size)


def plan_remote(read: RangeReader, remote_size: int) -> AudioPlan:
    boxes = find_top_level(read, remote_size)
    moov = boxes[b"moov"]
    return plan_from_moov(read(moov.start, moov.size), remote_size)


def coalesce(chunks: list[Chunk], max_gap: int = 64 * 1024,
             max_span: int = 8 * 1024 * 1024) -> list[tuple[int, int]]:
    """Merge nearby chunks into (offset, length) HTTP ranges.

    Reading a few KB of video between two audio chunks is cheaper than paying
    another request round-trip, so gaps up to `max_gap` are read through.
    """
    ranges: list[list[int]] = []
    for c in chunks:
        if ranges:
            start, end = ranges[-1]
            if 0 <= c.src_offset - end <= max_gap and c.src_offset + c.size - start <= max_span:
                ranges[-1][1] = c.src_offset + c.size
                continue
        ranges.append([c.src_offset, c.src_offset + c.size])
    return [(s, e - s) for s, e in ranges]
