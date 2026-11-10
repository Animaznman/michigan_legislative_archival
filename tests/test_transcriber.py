import json

import numpy as np
import pytest

from conftest import make_media
from mlarchive import transcriber
from mlarchive.transcriber import SR, TranscribeOptions, quiet_cut, transcribe_file, windows


def test_quiet_cut_finds_silence():
    a = np.ones(60 * SR, dtype=np.float32)
    a[50 * SR: 51 * SR] = 0
    cut = quiet_cut(a, search_s=20)
    assert 50 * SR <= cut <= 51 * SR


def test_windows_cover_audio_exactly_and_resume_reproduces_boundaries(tmp_path):
    src = make_media(tmp_path / "a.m4a", seconds=100, fmt="ipod", video=False,
                     silence_at=(55, 57))
    ws = list(windows(src, window_s=40))
    total = sum(w.size for _, w in ws)
    assert abs(total / SR - 100) < 0.5
    starts = [s for s, _ in ws]
    for (s, w), nxt in zip(ws, starts[1:]):
        assert abs(s + w.size / SR - nxt) < 1e-6        # contiguous, no gaps/overlap
    resumed = list(windows(src, window_s=40, skip_until_s=starts[1]))
    assert [round(s, 3) for s, _ in resumed] == [round(s, 3) for s in starts[1:]]


class FakeTranscriber(transcriber.Transcriber):
    def __init__(self, fail_on_window=None):
        super().__init__(TranscribeOptions(window_s=40))
        self.calls, self.fail_on = [], fail_on_window
        self.model_name, self.device = "fake", "cpu"

    def transcribe_window(self, audio, offset, check):
        self.calls.append(round(offset, 2))
        if self.fail_on is not None and len(self.calls) - 1 == self.fail_on:
            raise RuntimeError("GPU fell over")
        return [{"start": offset, "end": offset + 1, "text": f"w@{offset:.0f}"}]


def test_transcription_resumes_from_checkpoint(tmp_path):
    src = make_media(tmp_path / "a.m4a", seconds=100, fmt="ipod", video=False)
    out_json, out_txt, work = tmp_path / "t.json", tmp_path / "t.txt", tmp_path / "work"

    t1 = FakeTranscriber(fail_on_window=1)
    with pytest.raises(RuntimeError):
        transcribe_file(t1, src, work, out_json, out_txt, {"id": 1}, lambda: None)
    assert len(list(work.glob("tx_w*.json"))) == 1 and not out_json.exists()

    t2 = FakeTranscriber()
    doc = transcribe_file(t2, src, work, out_json, out_txt, {"id": 1}, lambda: None)
    assert t2.calls[0] == t1.calls[1]                     # window 0 not redone
    assert len(doc["segments"]) == len(t1.calls) - 1 + len(t2.calls)
    assert json.loads(out_json.read_text())["recording"] == {"id": 1}
    assert out_txt.read_text().startswith("[00:00:00] w@0")
    assert not list(work.glob("tx_*.json"))               # checkpoints cleaned up
