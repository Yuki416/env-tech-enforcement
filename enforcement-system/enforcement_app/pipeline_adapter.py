from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

from .config import AppConfig
from .models import JobOptions, ProgressUpdate, VideoJob
from .station import UNKNOWN_STATION, sanitize_station_id
from .vlm_enforcement import VLMEnforcementPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


class PipelineEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config: AppConfig = config
        self._pipeline = VLMEnforcementPipeline(
            model_path=str(self._config.model_litter),
            prefilter_conf=self._config.prefilter_conf,
            glm_api_key=self._config.glm_api_key,
            glm_model=self._config.glm_model,
            glm_api_base=self._config.glm_api_base,
            glm_timeout_sec=self._config.glm_timeout_sec,
            vlm_provider=self._config.vlm_provider,
            vlm_retry_count=self._config.vlm_retry_count,
            vlm_retry_backoff_sec=self._config.vlm_retry_backoff_sec,
            vlm_cooldown_sec=self._config.vlm_cooldown_sec,
            event_merge_sec=self._config.event_merge_sec,
            min_violation_confidence=self._config.min_violation_confidence,
            confirm_violation_confidence=self._config.confirm_violation_confidence,
            max_candidates_per_frame=self._config.max_candidates_per_frame,
            roi_expand=self._config.roi_expand,
            max_vlm_calls_per_video=self._config.max_vlm_calls_per_video,
            max_vlm_calls_per_minute=self._config.max_vlm_calls_per_minute,
            max_vlm_inflight=self._config.max_vlm_inflight,
            litter_clip_before_sec=self._config.litter_clip_before_sec,
            litter_clip_after_sec=self._config.litter_clip_after_sec,
            glm_connect_timeout_sec=self._config.glm_connect_timeout_sec,
            vlm_hard_timeout_sec=self._config.vlm_hard_timeout_sec,
            relabel_lit_to_urin=self._config.relabel_lit_to_urin,
        )

    def warmup(self) -> None:
        self._pipeline.warmup()

    def set_max_vlm_inflight(self, n: int) -> None:
        """委派給內部 VLMEnforcementPipeline，讓 GUI「平行影片數」能同步 VLM 槽位數。"""
        self._pipeline.set_max_vlm_inflight(n)

    def process_video(
        self,
        job: VideoJob,
        update_progress: Callable[[ProgressUpdate], None],
        output_root: Path | None = None,
    ) -> VideoJob:
        started = time.perf_counter()
        effective_root = (
            output_root if output_root is not None else self._config.output_root
        )
        # 輸出路徑分層：{root}/{station}/{timestamp}_{stem}/
        # 讓使用者在檔案總管看到頂層是 CH1/CH2/CH3... 以站別歸檔（2026-04-23 需求）。
        station_dir = sanitize_station_id(job.station_id or UNKNOWN_STATION)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            effective_root
            / station_dir
            / f"{timestamp}_{job.video_path.stem}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "違規紀錄.csv"
        snap_dir = out_dir / "違規截圖"
        audit_log_path = out_dir / "vlm_decisions.csv"

        def _on_progress(payload: dict[str, object]) -> None:
            pct = _to_float(payload.get("progress", 0.0), 0.0)
            stage = str(payload.get("stage", "inference"))
            frame = _to_int(payload.get("frame", 0), 0)
            total = _to_int(payload.get("total_frames", 0), 0)
            calls_done = _to_int(payload.get("vlm_calls_done", 0), 0)
            latency_ms = _to_int(payload.get("last_call_latency_ms", 0), 0)
            timeouts = _to_int(payload.get("vlm_timeouts", 0), 0)
            update_progress(
                ProgressUpdate(
                    job_id=job.job_id,
                    progress=max(0.0, min(1.0, pct)),
                    stage=stage,
                    frame=frame,
                    total_frames=total,
                    vlm_calls_done=calls_done,
                    last_call_latency_ms=latency_ms,
                    vlm_timeouts=timeouts,
                )
            )

        # 在 pipeline 真正開始之前，先推一個 'preparing' 進度，讓 GUI 能把
        # 「排隊中」狀態翻到「準備中」。這是問題 5 修正：使用者曾回報已排隊
        # 與剛起跑的任務都顯示 0%，難以分辨，看起來像卡住。
        _on_progress({"progress": 0.0, "stage": "preparing"})

        result_obj = self._pipeline.infer_single_video(
            video_path=job.video_path,
            out_csv_path=out_csv,
            snap_dir=snap_dir,
            detect_litter=job.options.enable_litter,
            detect_urination=job.options.enable_urination,
            frame_stride=job.options.heavy_frame_stride,
            max_vlm_calls_per_video=job.options.max_vlm_calls_per_video,
            progress_callback=_on_progress,
            audit_log_path=audit_log_path,
        )
        result = result_obj if isinstance(result_obj, dict) else {}
        events_obj = result.get("events")
        events = events_obj if isinstance(events_obj, list) else []
        elapsed = time.perf_counter() - started
        snap_dir_obj = result.get("snap_dir")
        snap_dir = snap_dir_obj if isinstance(snap_dir_obj, str) else ""
        csv_obj = result.get("csv")
        csv_path = csv_obj if isinstance(csv_obj, str) and csv_obj else str(out_csv)
        output_snap_dir = Path(snap_dir) if snap_dir else None
        return replace(
            job,
            status="done",
            progress=1.0,
            elapsed_sec=elapsed,
            violations=len(events),
            output_dir=out_dir,
            csv_path=Path(csv_path),
            snap_dir=output_snap_dir,
            events=[event for event in events if isinstance(event, dict)],
        )


def build_job_options(
    enable_litter: bool,
    enable_urination: bool,
    require_plate_for_csv: bool,
    heavy_frame_stride: int,
    max_vlm_calls_per_video: int,
) -> JobOptions:
    return JobOptions(
        enable_litter=enable_litter,
        enable_urination=enable_urination,
        require_plate_for_csv=require_plate_for_csv,
        heavy_frame_stride=max(1, heavy_frame_stride),
        max_vlm_calls_per_video=max(0, int(max_vlm_calls_per_video)),
    )
