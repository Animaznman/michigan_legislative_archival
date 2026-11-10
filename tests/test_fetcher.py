import threading

import httpx
import pytest

from conftest import audio_packets, make_media
from mlarchive import fetcher, net
from mlarchive.fetcher import FetchContext, NotReady


def ctx(server, tmp_path, check=lambda: None, connections=2):
    return FetchContext(server.client(), tmp_path / "work", connections, check)


# ----------------------------------------------------------------- mp4-ranges
MP4 = "https://house.test/ArchiveVideoFiles/X.mp4"


def test_mp4_ranges_end_to_end(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "BATCH_BYTES", 2048)     # many small batches
    src = make_media(tmp_path / "src.mp4", seconds=8)
    server.add(MP4, src.read_bytes())
    dst = tmp_path / "out" / "X.m4a"
    res = fetcher.fetch_audio(ctx(server, tmp_path), "mp4-ranges", MP4, dst)
    assert res.strategy == "mp4-ranges"
    assert audio_packets(dst) == audio_packets(src)
    assert not (tmp_path / "out" / "X.m4a.part").exists()


def test_mp4_ranges_resume_skips_finished_batches(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "BATCH_BYTES", 2048)
    src = make_media(tmp_path / "src.mp4", seconds=8)
    server.add(MP4, src.read_bytes())
    dst = tmp_path / "X.m4a"

    # First run dies after a few range requests (stream cut / process killed).
    calls = {"n": 0}
    def check():
        calls["n"] += 1
        if calls["n"] > 12:
            raise RuntimeError("simulated crash")
    with pytest.raises(RuntimeError):
        fetcher.fetch_audio(ctx(server, tmp_path, check, connections=1), "mp4-ranges", MP4, dst)
    done_batches = len(list((tmp_path / "work").glob("b*.bin")))
    assert done_batches > 0 and not dst.exists()

    server.requests.clear()
    res = fetcher.fetch_audio(ctx(server, tmp_path), "mp4-ranges", MP4, dst)
    first_total = fetcher.mp4audio.plan_remote(
        lambda o, n: src.read_bytes()[o:o + n], src.stat().st_size).audio_bytes
    assert res.downloaded_bytes < first_total            # only the remainder
    assert audio_packets(dst) == audio_packets(src)


def test_mp4_ranges_discard_pieces_when_remote_changes(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "BATCH_BYTES", 2048)
    src = make_media(tmp_path / "src.mp4", seconds=4)
    server.add(MP4, src.read_bytes(), etag='"v1"')
    work = tmp_path / "work"
    fetcher._load_manifest(work, {"url": MP4, "size": 1, "etag": '"old"',
                                  "strategy": "mp4-ranges", "batch": 2048})
    (work / "b00000.bin").write_bytes(b"stale")
    fetcher.fetch_audio(ctx(server, tmp_path), "mp4-ranges", MP4, tmp_path / "X.m4a")
    assert audio_packets(tmp_path / "X.m4a") == audio_packets(src)


def test_etag_change_mid_download_detected(server, tmp_path):
    server.add(MP4, b"x" * 100, etag='"v2"')
    with pytest.raises(net.SourceChanged):
        net.get_range(server.client(), MP4, 0, 10, etag='"v1"')


def test_upload_in_progress_is_not_ready(server, tmp_path):
    src = make_media(tmp_path / "src.mp4", seconds=2)
    data = src.read_bytes()
    server.add(MP4, data[: data.rfind(b"moov") - 4])      # moov not written yet
    with pytest.raises(NotReady):
        fetcher.fetch_audio(ctx(server, tmp_path), "mp4-ranges", MP4, tmp_path / "X.m4a")


def test_unsupported_layout_falls_back_to_full_download(server, tmp_path, monkeypatch):
    src = make_media(tmp_path / "src.mp4", seconds=3)
    server.add(MP4, src.read_bytes())

    def boom(*a, **k):
        raise fetcher.UnsupportedMp4("fragmented")
    monkeypatch.setattr(fetcher.mp4audio, "plan_remote", boom)
    res = fetcher.fetch_audio(ctx(server, tmp_path), "mp4-ranges", MP4, tmp_path / "X.m4a")
    assert res.strategy == "full"
    assert audio_packets(tmp_path / "X.m4a") == audio_packets(src)


# ------------------------------------------------------------------------ hls
HLS = "https://cdn.test/outputs/abc/Default/HLS/"


def hls_server(server, tmp_path, n=3, endlist=True):
    master = ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080\n1080p.m3u8\n"
              "#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\n360p.m3u8\n")
    server.add(HLS + "out.m3u8", master.encode())
    lines = ["#EXTM3U", "#EXT-X-TARGETDURATION:3"]
    for i in range(n):
        seg = make_media(tmp_path / f"seg{i}.ts", seconds=2, fmt="mpegts")
        server.add(HLS + f"360p_{i:03d}.ts", seg.read_bytes())
        lines += ["#EXTINF:2.0,", f"360p_{i:03d}.ts"]
    if endlist:
        lines.append("#EXT-X-ENDLIST")
    server.add(HLS + "360p.m3u8", "\n".join(lines).encode())


def test_hls_picks_lowest_bitrate_and_extracts_audio(server, tmp_path):
    hls_server(server, tmp_path)
    dst = tmp_path / "a.m4a"
    res = fetcher.fetch_audio(ctx(server, tmp_path), "hls", HLS + "out.m3u8", dst)
    assert res.strategy == "hls"
    assert server.count("1080p") == 0
    assert 5.0 < res.duration_s < 7.0
    import av
    with av.open(str(dst)) as c:
        assert [s.type for s in c.streams] == ["audio"]


def test_hls_retries_transient_segment_failure(server, tmp_path):
    hls_server(server, tmp_path)
    server.fail[HLS + "360p_001.ts"] = [503, httpx.ReadTimeout("blip")]
    fetcher.fetch_audio(ctx(server, tmp_path), "hls", HLS + "out.m3u8", tmp_path / "a.m4a")
    assert server.count("360p_001.ts") == 3


def test_hls_resume_skips_downloaded_segments(server, tmp_path):
    hls_server(server, tmp_path, n=4)
    server.fail[HLS + "360p_003.ts"] = [404]                 # first run fails on last seg
    with pytest.raises(net.PermanentHTTPError):
        fetcher.fetch_audio(ctx(server, tmp_path, connections=1), "hls", HLS + "out.m3u8",
                            tmp_path / "a.m4a")
    server.requests.clear()
    fetcher.fetch_audio(ctx(server, tmp_path), "hls", HLS + "out.m3u8", tmp_path / "a.m4a")
    assert server.count(".ts") == 1                           # only the missing one


def test_hls_without_endlist_is_not_ready(server, tmp_path):
    hls_server(server, tmp_path, endlist=False)
    with pytest.raises(NotReady):
        fetcher.fetch_audio(ctx(server, tmp_path), "hls", HLS + "out.m3u8", tmp_path / "a.m4a")


def test_parse_media_rejects_encryption():
    with pytest.raises(fetcher.UnsupportedMp4):
        fetcher.parse_media("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"k\"\n#EXTINF:2,\na.ts\n",
                            "http://x/")


# --------------------------------------------------------- resumable download
def test_full_download_resumes_after_stream_cut(server, tmp_path):
    url = "https://x.test/big.bin"
    data = bytes(range(256)) * 4000
    server.add(url, data)
    real = server.__call__
    state = {"cut": True}

    def handler(req):
        resp = real(req)
        if req.method == "GET" and state["cut"]:
            state["cut"] = False
            def gen():
                yield resp.content[:5000]
                raise httpx.ReadError("connection reset")
            return httpx.Response(resp.status_code, headers=resp.headers, content=gen())
        return resp

    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = FetchContext(client, tmp_path / "w", 1, lambda: None)
    dst = tmp_path / "big.bin"
    fetcher.download_resumable(c, url, dst)
    assert dst.read_bytes() == data
    ranges = [r.headers.get("range") for r in server.requests if r.method == "GET"]
    assert ranges[0] is None and ranges[1].startswith("bytes=") and ranges[1] != "bytes=0-"


def test_cancel_check_stops_parallel_download(server, tmp_path):
    hls_server(server, tmp_path, n=4)
    stop = threading.Event()
    stop.set()
    def check():
        if stop.is_set():
            raise InterruptedError()
    with pytest.raises(InterruptedError):
        fetcher.fetch_audio(ctx(server, tmp_path, check), "hls", HLS + "out.m3u8", tmp_path / "a.m4a")
