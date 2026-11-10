"""Settings: defaults, overridden by an optional TOML file, then env vars.

Env vars are named MLA_<FIELD>, e.g. MLA_WHISPER_MODEL=small.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path("data")

    # Discovery
    poll_interval_s: int = 900          # how often to check the portals
    senate_full_sync_s: int = 86400     # walk every Senate page this often
    sources: list[str] = field(default_factory=lambda: ["house", "senate"])
    # Only process recordings on/after this date (ISO). Older ones are still
    # recorded in the DB, so lowering this later picks them up automatically.
    process_since: str = "2026-01-01"

    # Workers
    fetch_workers: int = 2
    transcribe_workers: int = 1
    download_connections: int = 4       # parallel range/segment requests per job
    lease_s: int = 120                  # a crashed worker's job is reclaimed after this
    max_attempts: int = 8               # then the job is parked for park_s
    backoff_base_s: int = 30
    backoff_cap_s: int = 6 * 3600
    park_s: int = 24 * 3600

    # Transcription
    whisper_model: str = "auto"         # auto -> large-v3-turbo on GPU, small on CPU
    whisper_device: str = "auto"        # auto | cuda | cpu
    whisper_compute_type: str = "auto"
    language: str = "en"
    window_s: int = 1800                # transcribe in windows; checkpoint each

    keep_audio: bool = True             # keep .m4a after transcription
    min_free_gb: float = 5.0            # pause downloads below this much free disk

    user_agent: str = "mlarchive/1.0 (Michigan Legislature public-record archiver)"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

# Check to make sure values are what they 'seem' to be
def _coerce(value, typ, current):
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, list):
        return value if isinstance(value, list) else [v.strip() for v in str(value).split(",") if v.strip()]
    return str(value)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    s = Settings()
    path = path or os.environ.get("MLA_CONFIG")
    overrides: dict = {}
    if path and Path(path).exists():
        with open(path, "rb") as f:
            overrides.update(tomllib.load(f))
    for f in fields(Settings):
        env = os.environ.get(f"MLA_{f.name.upper()}")
        if env is not None:
            overrides[f.name] = env
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise ValueError(f"unknown setting: {k}")
        setattr(s, k, _coerce(v, None, getattr(s, k)))
    return s
