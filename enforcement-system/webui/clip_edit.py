"""違規片段的人工編輯 —— 刪除 / 新增 / 改起訖時間。

## 為什麼要重新產生證物檔

這是舉證系統。如果紀錄上寫「違規區間 00:00:05–00:00:08」，附的 mp4 卻還是舊範圍，
證物就對不起來了。所以**任何改動都會重新匯出片段與截圖、重寫 `違規紀錄.csv`**。

## 但不需要重跑推論

`codetr_scan.csv` 已經逐幀存了每個命中幀的 label / score / box。改範圍時直接從那裡
查回落在新範圍內的框來畫，秒級完成，不必再花 0.42 秒/幀跑 Co-DETR。

人工新增的片段如果落在偵測器沒抓到的區段，就沒有框可畫 —— 那是誠實的呈現：
偵測器確實沒看到，是稽查人員自己判斷的。

## 稽查軌跡

每次改動都追加一列到輸出目錄的 `稽查軌跡.csv`（誰、什麼時候、改了什麼）。
執法用途上「這筆紀錄被人工調整過」本身就是必須留存的資訊。
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from .codetr_test_mode import (
    _CLIP_AFTER_SEC,
    _CLIP_BEFORE_SEC,
    _LABEL_ZH,
    _draw_boxes,
    _export_clip_h264,
    _fmt_time,
)

logger = logging.getLogger(__name__)

# 人工新增/調整時，違規行為的可選值。與偵測器的類別對齊，另加一個「其他」。
BEHAVIOR_CHOICES = ["亂丟垃圾", "垃圾堆置", "便溺", "其他"]


class ClipEditError(Exception):
    """給呼叫端轉成 400 用的預期錯誤。"""


class ClipEditor:
    def __init__(self, store: Any) -> None:
        self.store = store

    # ── 對外操作 ─────────────────────────────────────────────────
    def delete(self, job_id: str, event_index: int) -> dict[str, Any]:
        job, events = self._job_events(job_id)
        if not 0 <= event_index < len(events):
            raise ClipEditError(f"事件索引超出範圍：{event_index}")
        removed = events.pop(event_index)
        self._cleanup_files(removed)
        self._commit(job, events, "刪除",
                     f"{removed.get('behavior')} {removed.get('seg_start_time')}–{removed.get('seg_end_time')}")
        return {"events": events, "removed": removed}

    def update(self, job_id: str, event_index: int,
               start_sec: float, end_sec: float,
               behavior: str | None = None,
               plate: str | None = None,
               reason: str | None = None) -> dict[str, Any]:
        job, events = self._job_events(job_id)
        if not 0 <= event_index < len(events):
            raise ClipEditError(f"事件索引超出範圍：{event_index}")
        old = events[event_index]
        before = f"{old.get('seg_start_time')}–{old.get('seg_end_time')}"

        new = self._build(job, start_sec, end_sec,
                          behavior or str(old.get("behavior") or "亂丟垃圾"),
                          index=event_index + 1, plate=plate, reason=reason)
        # overlay_time 是從畫面 OCR 讀出來的，跟人工編輯無關，跟著事件走
        if old.get("overlay_time"):
            new["overlay_time"] = old["overlay_time"]
        self._cleanup_files(old, keep=new)
        events[event_index] = new
        self._commit(job, events, "修改起訖",
                     f"{before} → {new['seg_start_time']}–{new['seg_end_time']}")
        return {"events": events, "event": new}

    def add(self, job_id: str, start_sec: float, end_sec: float,
            behavior: str = "亂丟垃圾",
            plate: str | None = None,
            reason: str | None = None) -> dict[str, Any]:
        job, events = self._job_events(job_id)
        new = self._build(job, start_sec, end_sec, behavior,
                          index=len(events) + 1, plate=plate, reason=reason)
        events.append(new)
        # 依時間排序，讓清單順序跟影片時間軸一致
        events.sort(key=lambda e: float(e.get("seg_start_sec") or 0))
        self._commit(job, events, "人工新增",
                     f"{behavior} {new['seg_start_time']}–{new['seg_end_time']}")
        return {"events": events, "event": new}

    # ── 內部 ─────────────────────────────────────────────────────
    def _job_events(self, job_id: str) -> tuple[Any, list[dict[str, Any]]]:
        job = self.store.get(job_id)
        if job is None:
            raise ClipEditError(f"找不到工作：{job_id}")
        if not job.output_dir:
            raise ClipEditError("這個工作還沒有輸出目錄，無法編輯片段")
        return job, [dict(e) for e in job.events]

    def _video_meta(self, job: Any) -> tuple[float, int]:
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            raise ClipEditError(f"讀不到來源影片：{job.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return fps, total

    def _load_scan(self, out_dir: Path, fps: float) -> dict[int, list[dict[str, Any]]]:
        """讀回 codetr_scan.csv 的逐幀偵測框。沒有這個檔（例如完整模式）就回空。"""
        path = out_dir / "codetr_scan.csv"
        if not path.is_file():
            return {}
        boxes: dict[int, list[dict[str, Any]]] = {}
        try:
            with path.open(encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    try:
                        fi = int(row["frame_index"])
                        x1, y1, x2, y2 = (int(v) for v in row["box"].split(","))
                    except (KeyError, ValueError):
                        continue
                    boxes.setdefault(fi, []).append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "label": row.get("label", ""),
                        "score": float(row.get("score") or 0.0),
                    })
        except OSError as exc:
            logger.warning("讀 codetr_scan.csv 失敗：%s", exc)
        return boxes

    def _build(self, job: Any, start_sec: float, end_sec: float,
               behavior: str, index: int,
               plate: str | None = None,
               reason: str | None = None) -> dict[str, Any]:
        fps, total = self._video_meta(job)
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        max_sec = (total / fps) if total > 0 else end_sec
        # 先判超出範圍再夾。順序反過來的話，12–15 秒在 10 秒的影片上會被夾成
        # 10.0–10.0，然後報「時間太接近」—— 那個訊息會讓人以為是自己填太窄。
        if total > 0 and start_sec >= max_sec:
            raise ClipEditError(
                f"起始時間 {start_sec:.1f} 秒超出影片長度（本片共 {max_sec:.1f} 秒）"
            )
        start_sec = max(0.0, min(start_sec, max_sec))
        end_sec = max(0.0, min(end_sec, max_sec))
        if end_sec - start_sec < 1.0 / fps:
            raise ClipEditError(
                f"區間太短：{start_sec:.2f}–{end_sec:.2f} 秒，至少要涵蓋一幀（{1 / fps:.2f} 秒）"
            )

        out_dir = Path(job.output_dir)
        snap_dir = out_dir / "違規截圖"
        clip_dir = out_dir / "違規片段"
        snap_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)

        f0, f1 = int(round(start_sec * fps)), int(round(end_sec * fps))
        scan = self._load_scan(out_dir, fps)
        in_range = {fi: bs for fi, bs in scan.items() if f0 <= fi <= f1}

        # 代表幀：範圍內分數最高的偵測；沒有偵測就用中點
        if in_range:
            peak_fi = max(in_range, key=lambda fi: max(b["score"] for b in in_range[fi]))
            peak_boxes = in_range[peak_fi]
            conf = round(max(b["score"] for b in peak_boxes), 3)
        else:
            peak_fi, peak_boxes, conf = (f0 + f1) // 2, [], 0.0

        stamp = f"{_fmt_time(peak_fi / fps).replace(':', '-')}_第{index:02d}號"
        snap_path = snap_dir / f"違規截圖_{stamp}.jpg"
        clip_path = clip_dir / f"違規片段_{stamp}.mp4"

        cap = cv2.VideoCapture(job.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak_fi)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ClipEditError(f"讀不到第 {peak_fi} 幀")
        cv2.imwrite(str(snap_path), _draw_boxes(frame, peak_boxes),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        clip_ok, clip_first = _export_clip_h264(
            Path(job.video_path), f0, f1, fps, clip_path, in_range
        )

        label_en = peak_boxes[0]["label"] if peak_boxes else ""
        # 判定理由：稽查人員填了就照用，留空才自動產生。
        # 這樣「看到什麼就存什麼」，不會有偷偷改寫使用者文字的情況。
        manual_reason = (reason or "").strip()
        if manual_reason:
            final_reason = manual_reason
        else:
            src = (f"Co-DETR 偵測到 {label_en}，最高信心 {conf:.2f}。"
                   if peak_boxes else "此範圍內偵測器沒有偵測結果。")
            final_reason = (
                f"【人工調整】違規區間第 {f0}~{f1} 幀（{end_sec - start_sec:.1f} 秒）。{src}"
                "※ 起訖由稽查人員設定。"
            )
        return {
            "frame_index": peak_fi,
            "frame_time": _fmt_time(peak_fi / fps),
            "overlay_time": "",
            "plate": (plate or "").strip(),
            "behavior": behavior or _LABEL_ZH.get(label_en, "亂丟垃圾"),
            "confidence": conf,
            "reason": final_reason,
            "reason_manual": bool(manual_reason),
            "snapshot": str(snap_path),
            "clip": str(clip_path) if clip_ok else "",
            "candidate_box": None,
            "clip_start_sec": round(clip_first / fps, 3) if clip_ok else 0.0,
            "clip_fps": round(fps, 3),
            "seg_start_sec": round(f0 / fps, 3),
            "seg_end_sec": round(f1 / fps, 3),
            "seg_start_time": _fmt_time(f0 / fps),
            "seg_end_time": _fmt_time(f1 / fps),
            "edited": True,
        }

    def _cleanup_files(self, ev: dict[str, Any], keep: dict[str, Any] | None = None) -> None:
        """刪掉舊事件的證物檔。keep 是新事件 —— 檔名可能相同，那就別刪。"""
        keep_paths = {str(keep.get("snapshot") or ""), str(keep.get("clip") or "")} if keep else set()
        for key in ("snapshot", "clip"):
            path = str(ev.get(key) or "")
            if not path or path in keep_paths:
                continue
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("刪除 %s 失敗：%s", path, exc)

    def _commit(self, job: Any, events: list[dict[str, Any]],
                action: str, detail: str) -> None:
        out_dir = Path(job.output_dir)
        self._rewrite_csv(out_dir / "違規紀錄.csv", job.video_name, events)
        self._append_audit(out_dir / "稽查軌跡.csv", action, detail, len(events))
        self.store.update(job.job_id, events=events, violations=len(events))
        self.store.log(f"[EDIT] {job.video_name}：{action} — {detail}（現有 {len(events)} 件）",
                       level="success")

    def _rewrite_csv(self, path: Path, video_name: str,
                     events: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["來源影片", "幀時間", "違規起始", "違規結束", "畫面時間", "車牌",
                        "違規行為", "信心分數", "判定理由", "截圖路徑", "影片片段路徑",
                        "人工調整"])
            for ev in events:
                w.writerow([
                    video_name, ev.get("frame_time", ""), ev.get("seg_start_time", ""),
                    ev.get("seg_end_time", ""), ev.get("overlay_time", ""),
                    ev.get("plate", ""), ev.get("behavior", ""), ev.get("confidence", ""),
                    ev.get("reason", ""), ev.get("snapshot", ""), ev.get("clip", ""),
                    "是" if ev.get("edited") else "",
                ])

    def _append_audit(self, path: Path, action: str, detail: str, remaining: int) -> None:
        new_file = not path.is_file()
        with path.open("a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["時間", "操作", "內容", "調整後件數"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), action, detail, remaining])
