"""Local speech-to-text with faster-whisper (CTranslate2 Whisper).

Audio is decoded by PyAV straight to 16 kHz mono and fed to Whisper in
windows (default 30 min). Each finished window is checkpointed to disk, so
a crash during a 5-hour session resumes at the last window instead of
starting over, and memory stays bounded (~115 MB per window).
Window boundaries are moved to the quietest point near the cut so we don't
slice a word in half.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import av
import numpy as np

from .fetcher import atomic_write

log = logging.getLogger(__name__)

SR = 16000 # Sample Rate/ throughput of the file at 16kHz


def _register_cuda_dlls() -> None:
    """On Windows, CTranslate2 needs cuBLAS/cuDNN DLLs; the nvidia-* pip
    wheels ship them but don't put them on the DLL search path."""
    if os.name != "nt":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    for root in list(getattr(nvidia, "__path__", [])):
        for d in glob.glob(os.path.join(root, "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def iter_pcm(path: Path, block_s: float = 10.0) -> Iterator[np.ndarray]:
    """Yield float32 mono 16 kHz blocks of about `block_s` seconds."""
    target = int(block_s * SR)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    buf: list[np.ndarray] = []
    have = 0
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            for out in resampler.resample(frame):
                a = out.to_ndarray().reshape(-1)
                buf.append(a)
                have += a.size
            if have >= target:
                yield np.concatenate(buf).astype(np.float32) / 32768.0
                buf, have = [], 0
        for out in resampler.resample(None):
            buf.append(out.to_ndarray().reshape(-1))
    if buf:
        yield np.concatenate(buf).astype(np.float32) / 32768.0


def quiet_cut(audio: np.ndarray, search_s: float = 20.0, frame_s: float = 0.25) -> int:
    """Sample index of the quietest frame in the last `search_s` seconds."""
    n = int(frame_s * SR)
    start = max(0, audio.size - int(search_s * SR))
    tail = audio[start:]
    frames = tail.size // n
    if frames < 2:
        return audio.size
    rms = np.sqrt((tail[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    return start + int(np.argmin(rms)) * n + n // 2


def windows(path: Path, window_s: float, skip_until_s: float = 0.0
            ) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (start_seconds, samples) windows, each cut at a quiet point.

    `skip_until_s` resumes after already-checkpointed windows: we still
    decode from the beginning (decoding is ~100x realtime) but discard
    samples before that point, so boundaries are reproduced exactly.
    """
    target = int(window_s * SR)
    pos = 0                    # absolute sample index of buf[0]
    skip = int(round(skip_until_s * SR))
    buf = np.zeros(0, dtype=np.float32)
    for block in iter_pcm(path):
        if pos + buf.size + block.size <= skip:
            pos += block.size
            continue
        if pos < skip:
            block = block[skip - pos:]
            pos = skip
        buf = np.concatenate([buf, block])
        while buf.size >= target + 30 * SR:   # keep slack so the cut search has room
            cut = quiet_cut(buf[:target])
            yield pos / SR, buf[:cut]
            pos += cut
            buf = buf[cut:]
    if buf.size:
        yield pos / SR, buf


@dataclass
class TranscribeOptions:
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "en"
    window_s: int = 1800
    beam_size: int = 5


class Transcriber:
    """Holds one loaded Whisper model. Not thread-safe: one per worker."""

    def __init__(self, opts: TranscribeOptions):
        self.opts = opts
        self._model = None
        self.device = None
        self.model_name = None
        self._lock = threading.Lock()

    def _resolve(self) -> tuple[str, str, str]:
        dev = self.opts.device
        if dev == "auto":
            _register_cuda_dlls()
            try:
                import ctranslate2
                dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                dev = "cpu"
        model = self.opts.model
        if model == "auto":
            model = "large-v3-turbo" if dev == "cuda" else "small"
        ct = self.opts.compute_type
        if ct == "auto":
            ct = "float16" if dev == "cuda" else "int8"
        return dev, model, ct

    def _load(self, force_cpu: bool = False):
        from faster_whisper import WhisperModel
        dev, model, ct = self._resolve()
        if force_cpu:
            dev, ct = "cpu", "int8"
            if self.opts.model == "auto":
                model = "small"
        if dev == "cuda":
            _register_cuda_dlls()
        log.info("loading whisper model=%s device=%s compute=%s", model, dev, ct)
        self._model = WhisperModel(model, device=dev, compute_type=ct)
        self.device, self.model_name = dev, model

    def _run_window(self, audio: np.ndarray, offset: float, check: Callable[[], None]) -> list[dict]:
        segments, _info = self._model.transcribe(
            audio,
            language=self.opts.language,
            beam_size=self.opts.beam_size,
            vad_filter=True,                       # skip the long silences in sessions
            vad_parameters={"min_silence_duration_ms": 1000},
            condition_on_previous_text=False,      # avoids repetition loops on long audio
        )
        out = []
        for s in segments:                          # generator: decoding happens here
            check()
            text = s.text.strip()
            if text:
                out.append({"start": round(offset + s.start, 2), "end": round(offset + s.end, 2),
                            "text": text, "avg_logprob": round(s.avg_logprob, 3),
                            "no_speech_prob": round(s.no_speech_prob, 3)})
        return out

    def transcribe_window(self, audio: np.ndarray, offset: float,
                          check: Callable[[], None]) -> list[dict]:
        with self._lock:
            if self._model is None:
                try:
                    self._load()
                except Exception as e:
                    if self.opts.device in ("auto",):
                        log.error("GPU model load failed (%s); falling back to CPU", e)
                        self._load(force_cpu=True)
                    else:
                        raise
            try:
                return self._run_window(audio, offset, check)
            except RuntimeError as e:
                # Missing CUDA libraries often only show up at first inference.
                msg = str(e).lower()
                if self.device == "cuda" and self.opts.device == "auto" and any(
                        k in msg for k in ("cublas", "cudnn", "cuda")):
                    log.error("CUDA inference failed (%s); falling back to CPU", e)
                    self._load(force_cpu=True)
                    return self._run_window(audio, offset, check)
                raise


def transcribe_file(tr: Transcriber, audio_path: Path, work: Path, out_json: Path,
                    out_txt: Path, recording: dict, check: Callable[[], None]) -> dict:
    """Transcribe with per-window checkpoints; write JSON + TXT atomically."""
    work.mkdir(parents=True, exist_ok=True)
    ident = {"audio": str(audio_path), "size": audio_path.stat().st_size,
             "window_s": tr.opts.window_s, "language": tr.opts.language,
             "model_opt": tr.opts.model}
    mf = work / "tx_manifest.json"
    if mf.exists():
        try:
            same = json.loads(mf.read_text()) == ident
        except ValueError:
            same = False
        if not same:
            for p in work.glob("tx_w*.json"):
                p.unlink()
    atomic_write(mf, json.dumps(ident).encode()) # Alternative to regular write to ensure 
    # temporary files are made in case a crash

    done = sorted(work.glob("tx_w*.json"))
    resume_at = 0.0
    if done:
        resume_at = json.loads(done[-1].read_text())["end_s"]
        log.info("resuming transcription of %s at %.0fs (%d windows done)",
                 audio_path.name, resume_at, len(done))
    idx = len(done)
    t0 = time.time()
    for start, audio in windows(audio_path, tr.opts.window_s, skip_until_s=resume_at):
        check()
        segs = tr.transcribe_window(audio, start, check)
        end = start + audio.size / SR
        atomic_write(work / f"tx_w{idx:03d}.json", json.dumps(
            {"index": idx, "start_s": start, "end_s": end, "segments": segs}).encode())
        log.info("window %d [%.0fs-%.0fs] -> %d segments", idx, start, end, len(segs))
        idx += 1

    parts = [json.loads(p.read_text()) for p in sorted(work.glob("tx_w*.json"))]
    segments = [s for p in parts for s in p["segments"]]
    duration = parts[-1]["end_s"] if parts else 0.0
    doc = {
        "recording": recording,
        "audio": {"path": audio_path.as_posix(), "duration_s": round(duration, 2)},
        "transcriber": {"engine": "faster-whisper", "model": tr.model_name,
                        "device": tr.device, "language": tr.opts.language},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "segments": segments,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out_txt, ("\n".join(_fmt(s) for s in segments) + "\n").encode("utf-8"))
    atomic_write(out_json, json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"))
    for p in work.glob("tx_*.json"):
        p.unlink()
    return doc


def _fmt(seg: dict) -> str:
    s = int(seg["start"])
    return f"[{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}] {seg['text']}"
