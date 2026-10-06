from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobOptions:
    enable_litter: bool = True
    enable_urination: bool = True
    require_plate_for_csv: bool = False
    heavy_frame_stride: int = 1
    max_vlm_calls_per_video: int = 2400


@dataclass
class VideoJob:
    job_id: str
    video_path: Path
    options: JobOptions
    station_id: str = "unknown"
    status: str = "pending"
    progress: float = 0.0
    elapsed_sec: float = 0.0
    violations: int = 0
    output_dir: Path | None = None
    csv_path: Path | None = None
    snap_dir: Path | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    error: str = ""


@dataclass
class ProgressUpdate:
    job_id: str
    progress: float
    stage: str
    frame: int = 0
    total_frames: int = 0
    vlm_calls_done: int = 0
    last_call_latency_ms: int = 0
    vlm_timeouts: int = 0
