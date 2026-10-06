"""JobStore — 網頁介面的單一事實來源（single source of truth）。

Tkinter 版把狀態散在 Treeview 的 item 裡，關掉視窗就沒了。網頁版必須能承受
「瀏覽器重新整理、關掉再開、網路短斷」（見 gui_redesign_bundle/README_for_AI.md §7），
所以狀態一律留在伺服器端，前端只是投影。

執行緒安全：pipeline 跑在 worker thread、HTTP 跑在各自的 handler thread，
所有讀寫都走同一把 RLock。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# stage 代號 → 前端 status-badge 的 CSS class。
# 前端 7 種 badge 樣式是設計稿既有的，這裡負責把後端階段對到正確的樣式。
STAGE_BADGE_CLASS: dict[str, str] = {
    "queued": "status-queued",
    "preparing": "status-preparing",
    "prefilter": "status-prefilter",
    "codetr": "status-codetr",
    "vlm": "status-vlm",
    "merging": "status-merging",
    "inference": "status-vlm",
    "completed": "status-completed",
    "error": "status-error",
}

STAGE_TEXT: dict[str, str] = {
    "queued": "排隊中",
    "preparing": "準備中",
    "prefilter": "YOLO 偵測中",
    "codetr": "Co-DETR 垃圾偵測中",
    "vlm": "VLM 判定中",
    "merging": "彙整結果中",
    "inference": "推論中",
    "completed": "完成",
    "error": "失敗",
}

# 距上次 VLM 呼叫超過這個秒數 → 該列標紅（對應 ENFORCE_VLM_IDLE_ALERT_SEC）
IDLE_ALERT_SEC = 120

_TERMINAL = {"completed", "error"}


@dataclass
class JobRecord:
    """一部影片的完整狀態。欄位是 models.VideoJob + ProgressUpdate 的聯集。

    多出來的欄位只有 `job_note`（稽查人員註記）。那是設計稿新增的資料維度，
    後端原本沒有；存法見 JobStore._note_path。
    """

    job_id: str
    video_path: str
    video_name: str
    station_id: str = "unknown"

    status: str = "queued"
    progress: float = 0.0
    frame: int = 0
    total_frames: int = 0
    violations: int = 0
    error: str = ""

    vlm_calls_done: int = 0
    vlm_latency_sum_ms: int = 0
    vlm_timeouts: int = 0
    last_vlm_at: float = 0.0

    output_dir: str = ""
    csv_path: str = ""
    snap_dir: str = ""
    job_note: str = ""

    events: list[dict[str, Any]] = field(default_factory=list)

    # 計時用（monotonic）。序列化時才換算成 elapsed_sec。
    started_at: float = 0.0
    finished_elapsed: float = 0.0

    def elapsed_sec(self) -> float:
        if self.status in _TERMINAL:
            return self.finished_elapsed
        if self.started_at <= 0.0:
            return 0.0
        return time.monotonic() - self.started_at

    def seconds_since_last_vlm(self) -> int:
        if self.last_vlm_at <= 0.0:
            return -1
        return int(time.monotonic() - self.last_vlm_at)

    def vlm_avg_latency(self) -> float:
        if self.vlm_calls_done <= 0:
            return 0.0
        return self.vlm_latency_sum_ms / self.vlm_calls_done / 1000.0

    def is_idle_alert(self) -> bool:
        """§5「閒置警示」：跑到一半但距上次 AI 呼叫超過 2 分鐘。

        只對執行中的 job 判定 —— 排隊中的 job 本來就沒呼叫過，不該標紅。
        """
        if self.status in _TERMINAL or self.status in ("queued", "preparing"):
            return False
        idle = self.seconds_since_last_vlm()
        return idle >= IDLE_ALERT_SEC

    def to_dict(self) -> dict[str, Any]:
        idle = self.seconds_since_last_vlm()
        return {
            "job_id": self.job_id,
            "video_name": self.video_name,
            "video_path": self.video_path,
            "station_id": self.station_id,
            "status": self.status,
            "badge_class": STAGE_BADGE_CLASS.get(self.status, "status-queued"),
            "stage_display": STAGE_TEXT.get(self.status, self.status),
            "progress": round(self.progress, 4),
            "frame": self.frame,
            "total_frames": self.total_frames,
            "elapsed_sec": round(self.elapsed_sec(), 1),
            "vlm_calls_done": self.vlm_calls_done,
            "vlm_avg_latency": round(self.vlm_avg_latency(), 2),
            "vlm_timeouts": self.vlm_timeouts,
            "seconds_since_last_vlm": idle,
            "idle_alert": self.is_idle_alert(),
            # 還沒完成過任何一次 VLM 呼叫 → 介面要說「等待首次判定」而不是「上次呼叫 N 秒前」
            "awaiting_first_vlm": self.vlm_calls_done == 0,
            "violations": self.violations,
            "output_dir": self.output_dir,
            "csv_path": self.csv_path,
            "snap_dir": self.snap_dir,
            "job_note": self.job_note,
            "error": self.error,
            "event_count": len(self.events),
        }


class JobStore:
    """所有 job 的狀態 + log + SSE 訂閱者。"""

    MAX_LOGS = 400

    def __init__(self, output_root: Path) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._logs: list[dict[str, Any]] = []
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._seq = 0
        self.output_root = output_root
        self._notes = self._load_notes()

    # ── 註記持久化 ────────────────────────────────────────────────
    # 存兩份：
    #   1. output_root/officer_notes.json — 索引，用影片路徑當 key，隨時查得到
    #   2. {job 輸出目錄}/note.json      — 跟舉證資料放在一起，資料夾整包帶走時註記不遺失
    # 不開資料庫是刻意的：舉證資料本來就是「一個案子一個資料夾」，註記跟著走最自然。

    def _note_index_path(self) -> Path:
        return self.output_root / "officer_notes.json"

    def _load_notes(self) -> dict[str, str]:
        path = self._note_index_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _persist_notes(self) -> None:
        path = self._note_index_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._notes, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def set_note(self, job_id: str, note: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.job_note = note
            self._notes[job.video_path] = note
            self._persist_notes()
            if job.output_dir:
                try:
                    out = Path(job.output_dir)
                    if out.is_dir():
                        (out / "note.json").write_text(
                            json.dumps(
                                {
                                    "video": job.video_name,
                                    "note": note,
                                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                except OSError:
                    pass
        self._broadcast()
        return True

    # ── job 管理 ──────────────────────────────────────────────────

    def add_video(self, video_path: Path, station_id: str) -> JobRecord | None:
        """加入一部影片。已存在（同路徑）則不重複加入，回傳 None。"""
        with self._lock:
            resolved = str(video_path)
            for job in self._jobs.values():
                if job.video_path == resolved:
                    return None
            self._seq += 1
            job_id = f"job_{self._seq:03d}"
            record = JobRecord(
                job_id=job_id,
                video_path=resolved,
                video_name=video_path.name,
                station_id=station_id,
                job_note=self._notes.get(resolved, ""),
            )
            self._jobs[job_id] = record
            self._order.append(job_id)
            return record

    def existing_video_paths(self) -> set[str]:
        """目前佇列裡所有影片路徑。掃描前先取一次，就不必對每個檔案都線性比對。"""
        with self._lock:
            return {job.video_path for job in self._jobs.values()}

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobRecord]:
        with self._lock:
            return [self._jobs[j] for j in self._order if j in self._jobs]

    def pending(self) -> list[JobRecord]:
        with self._lock:
            return [j for j in self.all() if j.status == "queued"]

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            # 進度單調遞增保護：pipeline 的 prefilter 回呼來自預取執行緒（跑在前面），
            # vlm 回呼來自判定幀（落在後面），兩者交錯會讓進度條前後跳動。
            # 只在執行中的階段夾住；retry 會把 status 設回 queued，那時允許歸零。
            if (
                "progress" in fields
                and job.status not in _TERMINAL
                and fields.get("status", job.status) not in ("queued", "preparing")
            ):
                try:
                    fields["progress"] = max(float(fields["progress"]), job.progress)
                except (TypeError, ValueError):
                    fields.pop("progress")
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
        # events 變動會改到「違規事件」分頁與統計，前端要重抓；
        # 其他欄位（進度、階段、遙測）只影響這一列，推單筆就好。
        if "events" in fields:
            self._broadcast()
        else:
            self._push_job(job_id)

    def mark_vlm_call(self, job_id: str, latency_ms: int, calls_done: int,
                      timeouts: int) -> None:
        """VLM 遙測。calls_done 由 pipeline 給的絕對值為準（它才知道真實次數）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if calls_done > job.vlm_calls_done:
                job.vlm_latency_sum_ms += max(0, latency_ms)
                job.vlm_calls_done = calls_done
                job.last_vlm_at = time.monotonic()
            job.vlm_timeouts = timeouts

    def clear_finished(self) -> int:
        with self._lock:
            drop = [j for j in self._order
                    if j in self._jobs and self._jobs[j].status in _TERMINAL]
            for job_id in drop:
                del self._jobs[job_id]
            self._order = [j for j in self._order if j in self._jobs]
        self._broadcast()
        return len(drop)

    # ── 事件（違規紀錄）────────────────────────────────────────────

    def all_events(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for job in self.all():
            for idx, ev in enumerate(job.events):
                item = dict(ev)
                item["job_id"] = job.job_id
                item["event_index"] = idx
                item["source_video"] = job.video_name
                item["job_note"] = job.job_note
                out.append(item)
        return out

    # ── log ──────────────────────────────────────────────────────

    def log(self, message: str, level: str = "info") -> None:
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > self.MAX_LOGS:
                del self._logs[: len(self._logs) - self.MAX_LOGS]
        self._push({"type": "log", "entry": entry})

    def logs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._logs)

    # ── 快照與推播 ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """統計。刻意不呼叫 to_dict() —— 那會為每個 job 建一個字典，
        佇列上千部時光是算統計就要序列化幾百 KB。這裡直接讀欄位。"""
        _RUNNING = ("preparing", "prefilter", "codetr", "vlm", "merging")
        with self._lock:
            jobs = list(self._jobs.values())
            return {
                "total_jobs": len(jobs),
                "running_jobs": sum(1 for j in jobs if j.status in _RUNNING),
                "queued_jobs": sum(1 for j in jobs if j.status == "queued"),
                "completed_jobs": sum(1 for j in jobs if j.status == "completed"),
                "failed_jobs": sum(1 for j in jobs if j.status == "error"),
                "idle_alert_count": sum(1 for j in jobs if j.is_idle_alert()),
                "total_violations": sum(j.violations for j in jobs),
            }

    def snapshot(self, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
        """一頁 job + 全部事件 + 統計。

        **jobs 一定要分頁。** 曾經整包回傳，掃到 1526 部影片時單次回應 1MB，
        而 SSE 每次狀態變動都推一份 —— 瀏覽器光是 JSON.parse 就卡死。
        現在 SSE 只推變動的那一筆（`_push_job`），整份快照只在載入與重連時抓。
        """
        with self._lock:
            ordered = [self._jobs[j] for j in self._order if j in self._jobs]
        total = len(ordered)
        start = max(0, offset)
        page = ordered[start:] if limit is None else ordered[start:start + max(0, limit)]
        return {
            "type": "state",
            "jobs": [j.to_dict() for j in page],
            "jobs_total": total,
            "jobs_offset": start,
            "jobs_limit": limit,
            "events": self.all_events(),
            "stats": self.stats(),
        }

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _push(self, payload: dict[str, Any]) -> None:
        with self._lock:
            targets: Iterable[Callable[[dict[str, Any]], None]] = list(self._subscribers)
        for callback in targets:
            try:
                callback(payload)
            except Exception:  # 單一訂閱者壞掉不能影響其他人
                pass

    def _push_job(self, job_id: str) -> None:
        """單筆 job 變動。這是進度回呼會打的路徑，一秒可能好幾次，所以必須小。"""
        with self._lock:
            job = self._jobs.get(job_id)
            payload = job.to_dict() if job is not None else None
        if payload is None:
            return
        self._push({"type": "job", "job": payload, "stats": self.stats()})

    def _broadcast(self) -> None:
        """整批變動（掃描新增、清除、事件產生）→ 叫前端重抓自己那一頁。

        不直接推 jobs：訂閱者各自停在不同頁，伺服器不知道誰要看哪一段。
        """
        self._push({"type": "reset", "stats": self.stats()})
