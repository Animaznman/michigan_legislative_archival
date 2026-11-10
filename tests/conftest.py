"""Shared fixtures: synthetic media files and a fake HTTP server.

No test touches the network. Media is generated with PyAV so the MP4/TS
parsing code runs against real container files.
"""
from __future__ import annotations

import fractions
import re
from pathlib import Path

import av
import httpx
import numpy as np
import pytest


def make_media(path: Path, seconds: float = 6.0, fmt: str = "mp4", video: bool = True,
               silence_at: tuple[float, float] | None = None, rate: int = 48000) -> Path:
    """Write a file with a sine-wave AAC track (+ optional MPEG-4 video)."""
    with av.open(str(path), mode="w", format=fmt) as out:   # mp4: moov written last, like the House files
        vs = None
        if video:
            vs = out.add_stream("mpeg4", rate=10)
            vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        a = out.add_stream("aac", rate=rate)
        a.layout = "mono"
        n_frames = int(seconds * rate / 1024)
        v_idx = 0
        for i in range(n_frames):
            t = np.arange(i * 1024, (i + 1) * 1024) / rate
            samples = 0.3 * np.sin(2 * np.pi * 440 * t)
            if silence_at and silence_at[0] <= i * 1024 / rate < silence_at[1]:
                samples *= 0
            fr = av.AudioFrame.from_ndarray(samples.astype(np.float32).reshape(1, -1),
                                            format="fltp", layout="mono")
            fr.sample_rate = rate
            fr.pts = i * 1024
            fr.time_base = fractions.Fraction(1, rate)
            for p in a.encode(fr):
                out.mux(p)
            while vs is not None and v_idx / 10 <= i * 1024 / rate:
                img = np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8)
                vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
                vf.pts = v_idx
                vf.time_base = fractions.Fraction(1, 10)
                for p in vs.encode(vf):
                    out.mux(p)
                v_idx += 1
        for p in a.encode(None):
            out.mux(p)
        if vs is not None:
            for p in vs.encode(None):
                out.mux(p)
    return path


def audio_packets(path) -> list[bytes]:
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        return [bytes(p) for p in c.demux(s) if p.size]


class FakeServer:
    """httpx MockTransport handler: static files with Range/ETag support,
    plus scriptable failures (`fail[url] = [503, 503]` -> first two fail)."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.fail: dict[str, list] = {}
        self.json_handlers: dict[str, callable] = {}
        self.requests: list[httpx.Request] = []

    def add(self, url: str, data: bytes, etag: str = '"v1"'):
        self.files[url] = data
        self.etags[url] = etag

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        url = str(req.url)
        script = self.fail.get(url)
        if script:
            what = script.pop(0)
            if isinstance(what, int):
                return httpx.Response(what)
            raise what
        if url in self.json_handlers:
            return self.json_handlers[url](req)
        if url not in self.files:
            return httpx.Response(404)
        data, etag = self.files[url], self.etags[url]
        headers = {"ETag": etag, "Accept-Ranges": "bytes"}
        if req.method == "HEAD":
            return httpx.Response(200, headers={**headers, "Content-Length": str(len(data))})
        rng = req.headers.get("range")
        if_range = req.headers.get("if-range")
        if rng and (not if_range or if_range == etag):
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            start = int(m[1])
            end = int(m[2]) if m[2] else len(data) - 1
            body = data[start:end + 1]
            return httpx.Response(206, content=body, headers={
                **headers, "Content-Range": f"bytes {start}-{end}/{len(data)}"})
        return httpx.Response(200, content=data, headers=headers)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def count(self, url_part: str) -> int:
        return sum(1 for r in self.requests if url_part in str(r.url))


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Make in-process retry sleeps instant."""
    import mlarchive.net as net
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    monkeypatch.setattr(net.random, "uniform", lambda a, b: 0.0)
