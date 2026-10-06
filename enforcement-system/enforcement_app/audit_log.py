"""Per-video VLM decision audit log.

Writes `vlm_decisions.csv` alongside the existing 違規紀錄.csv. One row per
decision event lets the operator answer "why didn't frame X get reported?"
without having to re-run the pipeline. See spec
docs/superpowers/specs/2026-04-26-vlm-observability-design.md §3 for schema.

Each video gets its own file → no cross-thread write contention → no lock.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_REASON_MAX_CHARS = 200

_HEADER = [
    "frame_index", "frame_time_sec", "overlay_time", "event_type",
    "candidate_label", "candidate_box", "vlm_called", "vlm_latency_ms",
    "vlm_has_violation", "vlm_behavior", "vlm_confidence", "vlm_plate",
    "vlm_reason", "decision", "decision_note", "snapshot_path",
]


def _truncate_reason(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    if len(cleaned) <= _REASON_MAX_CHARS:
        return cleaned
    return cleaned[:_REASON_MAX_CHARS] + "…"


def _box_to_str(box: tuple[int, int, int, int] | None) -> str:
    if box is None:
        return ""
    x1, y1, x2, y2 = box
    return f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"


class AuditLogger:
    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8-sig", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(_HEADER)
        self._fh.flush()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fh.close()
        finally:
            self._closed = True

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def log_vlm_call(
        self,
        *,
        frame_index: int,
        frame_time_sec: float,
        overlay_time: str,
        candidate_label: str,
        candidate_box: tuple[int, int, int, int] | None,
        latency_ms: int,
        timed_out: bool,
        vlm_response: dict[str, Any] | None,
        decision: str,
        decision_note: str,
        snapshot_path: str,
    ) -> None:
        if vlm_response is None:
            has_violation = ""
            behavior = ""
            confidence = "0.0"
            plate = ""
            reason = ""
        else:
            has_violation = "true" if bool(vlm_response.get("has_violation", False)) else "false"
            behavior = str(vlm_response.get("behavior", "") or "")
            conf_raw = vlm_response.get("confidence", 0.0)
            try:
                confidence = f"{float(conf_raw):.2f}"
            except (TypeError, ValueError):
                confidence = "0.0"
            plate = str(vlm_response.get("plate", "") or "")
            reason = _truncate_reason(str(vlm_response.get("reason", "") or ""))

        self._writer.writerow([
            int(frame_index),
            f"{float(frame_time_sec):.2f}",
            overlay_time or "",
            "vlm_call",
            candidate_label or "",
            _box_to_str(candidate_box),
            "1",
            int(latency_ms),
            has_violation,
            behavior,
            confidence,
            plate,
            reason,
            decision or "",
            decision_note or "",
            snapshot_path or "",
        ])
        self._fh.flush()

    def log_prefilter_empty_range(
        self,
        *,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> None:
        time_sec = float(start_frame) / fps if fps > 0 else 0.0
        self._writer.writerow([
            int(start_frame), f"{time_sec:.2f}", "",
            "prefilter_empty_range",
            "", "", "0", 0, "", "", "0.0", "", "",
            "", f"range_to={int(end_frame)}", "",
        ])
        self._fh.flush()

    def log_skip(
        self,
        *,
        frame_index: int,
        frame_time_sec: float,
        overlay_time: str,
        event_type: str,
        candidate_label: str,
        candidate_box: tuple[int, int, int, int] | None,
        decision_note: str,
    ) -> None:
        self._writer.writerow([
            int(frame_index), f"{float(frame_time_sec):.2f}", overlay_time or "",
            event_type,
            candidate_label or "", _box_to_str(candidate_box),
            "0", 0, "", "", "0.0", "", "",
            "", decision_note or "", "",
        ])
        self._fh.flush()

    def log_event_kept(
        self,
        *,
        frame_index: int,
        frame_time_sec: float,
        overlay_time: str,
        candidate_label: str,
        candidate_box: tuple[int, int, int, int] | None,
        behavior: str,
        confidence: float,
        plate: str,
        snapshot_path: str,
    ) -> None:
        self._writer.writerow([
            int(frame_index), f"{float(frame_time_sec):.2f}", overlay_time or "",
            "event_kept",
            candidate_label or "", _box_to_str(candidate_box),
            "0", 0, "", behavior or "", f"{float(confidence):.2f}",
            plate or "", "",
            "accepted", "", snapshot_path or "",
        ])
        self._fh.flush()

    def log_event_merged(
        self,
        *,
        frame_index: int,
        frame_time_sec: float,
        overlay_time: str,
        winner_frame_index: int,
    ) -> None:
        self._writer.writerow([
            int(frame_index), f"{float(frame_time_sec):.2f}", overlay_time or "",
            "event_merged",
            "", "", "0", 0, "", "", "0.0", "", "",
            f"merged_into_frame={int(winner_frame_index)}", "", "",
        ])
        self._fh.flush()

    def log_event_deduped(
        self,
        *,
        frame_index: int,
        frame_time_sec: float,
        overlay_time: str,
        winning_plate: str,
    ) -> None:
        self._writer.writerow([
            int(frame_index), f"{float(frame_time_sec):.2f}", overlay_time or "",
            "event_deduped",
            "", "", "0", 0, "", "", "0.0", "", "",
            f"deduped_by_plate={winning_plate}", "", "",
        ])
        self._fh.flush()
