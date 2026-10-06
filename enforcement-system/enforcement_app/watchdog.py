"""Background watchdog that periodically dumps active-job state to a log file.

使用者 4/20 晚上送 15 支影片，4/21 早上發現系統跑完中斷掉。長批次 36+ 小時
若中斷，沒有狀態快照可判斷卡在哪支影片、哪個階段。本模組提供：

- 每 `interval_sec` 秒把 active jobs 的 stage/progress/idle_sec 寫到 log 檔
- 超過 `stale_timeout_sec` 沒心跳的 job 會被標記 STALE（提示卡住了）
- 不會中斷 Python thread（那樣容易毀資料），只做記錄 + UI 顯示

使用方式（概念）：
    wd = WatchdogLogger(log_path=cfg.watchdog_log_path)
    wd.start()
    ...
    wd.register(job_id, video_name)
    wd.heartbeat(job_id, stage, progress)  # 每次 progress callback
    wd.unregister(job_id)
    ...
    wd.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _JobHeartbeat:
    job_id: str
    video_name: str
    stage: str = "queued"
    progress: float = 0.0
    registered_at: float = field(default_factory=time.monotonic)
    last_tick: float = field(default_factory=time.monotonic)


class WatchdogLogger:
    def __init__(
        self,
        log_path: Path,
        interval_sec: float = 60.0,
        stale_timeout_sec: float = 1200.0,
    ) -> None:
        self._log_path = Path(log_path)
        # 下限 0.05s 足以防誤設（production 預設 60s），又能讓測試跑快。
        self._interval = max(0.05, float(interval_sec))
        # Stale timeout 下限 30s：production 不希望短暫停頓就誤報 STALE。
        self._stale_timeout = max(30.0, float(stale_timeout_sec))
        self._jobs: dict[str, _JobHeartbeat] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("watchdog log directory mkdir failed: %s", exc)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="enforce-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def register(self, job_id: str, video_name: str) -> None:
        with self._lock:
            self._jobs[job_id] = _JobHeartbeat(
                job_id=job_id, video_name=video_name
            )

    def heartbeat(self, job_id: str, stage: str, progress: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = stage
            job.progress = max(0.0, min(1.0, float(progress)))
            job.last_tick = time.monotonic()

    def unregister(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def snapshot(self) -> list[dict[str, Any]]:
        """Thread-safe copy of current job states, including staleness evaluation."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "job_id": j.job_id,
                    "video_name": j.video_name,
                    "stage": j.stage,
                    "progress": j.progress,
                    "idle_sec": max(0.0, now - j.last_tick),
                    "run_sec": max(0.0, now - j.registered_at),
                    "stale": (now - j.last_tick) > self._stale_timeout,
                }
                for j in self._jobs.values()
            ]

    def stale_job_ids(self) -> list[str]:
        return [j["job_id"] for j in self.snapshot() if j["stale"]]

    def _loop(self) -> None:
        # 先等一次 tick 再寫，避免啟動當下空空如也卻寫 log
        while not self._stop_event.wait(self._interval):
            try:
                self._write_snapshot()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("watchdog snapshot write failed: %s", exc)

    def _write_snapshot(self) -> None:
        snap = self.snapshot()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"--- watchdog {ts} ({len(snap)} active) ---"
        lines = [header]
        for j in snap:
            flag = "STALE" if j["stale"] else "OK   "
            lines.append(
                f"[{flag}] {j['job_id']} {j['video_name']} "
                f"stage={j['stage']} progress={int(j['progress'] * 100)}% "
                f"idle={int(j['idle_sec'])}s run={int(j['run_sec'])}s"
            )
        text = "\n".join(lines) + "\n"
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            logger.debug("watchdog log write skipped: %s", exc)
