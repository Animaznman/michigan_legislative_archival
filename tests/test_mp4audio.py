import struct

import pytest

from conftest import audio_packets, make_media
from mlarchive import mp4audio
from mlarchive.mp4audio import Chunk, UnsupportedMp4, coalesce, plan_remote


def reader(data: bytes, log: list | None = None):
    def read(off, n):
        if log is not None:
            log.append((off, n))
        return data[off:off + n]
    return read


def build(data: bytes, plan) -> bytes:
    return plan.header + b"".join(data[c.src_offset:c.src_offset + c.size] for c in plan.chunks)


def test_audio_only_rebuild_is_bit_identical(tmp_path):
    src = make_media(tmp_path / "in.mp4", seconds=8)
    data = src.read_bytes()
    reads = []
    plan = plan_remote(reader(data, reads), len(data))

    out = tmp_path / "out.m4a"
    out.write_bytes(build(data, plan))
    assert out.stat().st_size == plan.total_size
    # Same AAC packets, byte for byte, and no video track left.
    assert audio_packets(out) == audio_packets(src)
    import av
    with av.open(str(out)) as c:
        assert [s.type for s in c.streams] == ["audio"]
    # Planning only touched box headers + moov, never the bulk of mdat.
    assert sum(n for _, n in reads) < 0.2 * len(data)


def test_audio_bytes_are_small_fraction(tmp_path):
    src = make_media(tmp_path / "in.mp4", seconds=5)
    data = src.read_bytes()
    plan = plan_remote(reader(data), len(data))
    assert 0 < plan.audio_bytes < len(data)


def test_missing_moov_is_reported(tmp_path):
    src = make_media(tmp_path / "in.mp4", seconds=2)
    data = src.read_bytes()
    boxes = mp4audio.find_top_level(reader(data), len(data))
    truncated = data[:boxes[b"moov"].start]          # upload still in progress
    with pytest.raises(UnsupportedMp4, match="missing moov"):
        plan_remote(reader(truncated), len(truncated))


def test_fragmented_mp4_rejected():
    def box(t, payload=b""):
        return struct.pack(">I4s", 8 + len(payload), t) + payload
    data = box(b"ftyp", b"isom\0\0\0\0") + box(b"moov") + box(b"moof") + box(b"mdat", b"x" * 10)
    with pytest.raises(UnsupportedMp4, match="fragmented"):
        mp4audio.find_top_level(reader(data), len(data))


def test_bad_box_size_rejected():
    data = struct.pack(">I4s", 9999, b"ftyp") + b"\0" * 8
    with pytest.raises(UnsupportedMp4):
        mp4audio.find_top_level(reader(data), len(data))


def test_coalesce_merges_small_gaps_only():
    chunks = [Chunk(0, 10), Chunk(15, 10), Chunk(10_000, 5)]
    assert coalesce(chunks, max_gap=100) == [(0, 25), (10_000, 5)]
    assert coalesce(chunks, max_gap=0) == [(0, 10), (15, 10), (10_000, 5)]


def test_coalesce_respects_max_span():
    chunks = [Chunk(i * 10, 10) for i in range(10)]
    assert coalesce(chunks, max_gap=0, max_span=30) == [(0, 30), (30, 30), (60, 30), (90, 10)]
