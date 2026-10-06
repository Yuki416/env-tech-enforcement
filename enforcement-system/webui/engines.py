"""執行引擎 — 兩種實作共用同一個介面。

MockEngine : 不載入任何模型，用時間軸模擬整條 pipeline。用來先確認介面互動。
RealEngine : 包住 enforcement_app.pipeline_adapter.PipelineEngine，跑真影片。

兩者都把狀態寫回 JobStore，前端完全分不出差別 —— 這正是重點：
先用 mock 把介面跑順，再換引擎，前端一行都不用改。
"""
from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .store import JobRecord, JobStore

# ── mock 用的假資料池 ────────────────────────────────────────────
_MOCK_BEHAVIORS = ["亂丟垃圾", "亂丟菸蒂", "隨地便溺"]
_MOCK_REASONS = {
    "亂丟垃圾": "機車駕駛右手向下甩出黑色垃圾袋，垃圾袋離手並落地於路邊草叢。",
    "亂丟菸蒂": "汽車駕駛降下左側車窗，將點燃之菸蒂彈出車外落於柏油路面。",
    "隨地便溺": "行人於牆角處背向鏡頭停留約 45 秒，姿勢符合隨地便溺特徵。",
}
_MOCK_PLATES = ["AAA-0001", "AAA-0002", "AAA-0003", "AAA-0004", "", "000-A1"]


class BaseEngine:
    """共用的排程骨架：把 queued 的 job 丟進 thread pool，一次跑 N 部。"""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._running = False
        self.options: dict[str, Any] = {}

    @property
    def busy(self) -> bool:
        return self._running

    def start(self, options: dict[str, Any]) -> int:
        """把目前所有 queued 的 job 送進執行。回傳送出的數量。"""
        with self._lock:
            jobs = self.store.pending()
            if not jobs:
                return 0
            self.options = options
            parallel = max(1, min(8, int(options.get("parallel_jobs_count", 1))))
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=parallel, thread_name_prefix="webui-job"
                )
            self._running = True
            self._on_batch_start(parallel)
            for job in jobs:
                self.store.update(job.job_id, status="queued")
                self._executor.submit(self._guarded_run, job.job_id)
            return len(jobs)

    def _on_batch_start(self, parallel: int) -> None:
        """子類別可在此同步平行度設定（§5「平行影片數要真的生效」）。"""

    def _guarded_run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        try:
            # last_vlm_at 從開工就起算，而不是等第一次 VLM 呼叫完成才有值。
            # 原因：pipeline 的遙測只在「呼叫回來之後」才發，第一次呼叫如果卡住
            # （ollama 冷啟動要 60 秒以上、或退回 CPU 要好幾分鐘），介面會一直是
            # 「尚未呼叫」而不會觸發閒置警示 —— 那正是廠商回報「像當機」的情境。
            now = time.monotonic()
            self.store.update(
                job_id, started_at=now, last_vlm_at=now, status="preparing"
            )
            self.store.log(f"[ENGINE] 開始處理 {job.video_name}（站別 {job.station_id}）")
            self._run_one(job_id)
        except Exception as exc:  # noqa: BLE001 — 單一影片失敗不能拖垮整批
            record = self.store.get(job_id)
            elapsed = record.elapsed_sec() if record else 0.0
            self.store.update(
                job_id, status="error", error=str(exc), finished_elapsed=elapsed
            )
            self.store.log(
                f"[ERROR] {job.video_name} 處理失敗：{exc}", level="error"
            )
        finally:
            if not any(j.status in ("queued", "preparing", "prefilter", "codetr",
                                    "vlm", "merging")
                       for j in self.store.all()):
                self._running = False

    def _run_one(self, job_id: str) -> None:
        raise NotImplementedError


class MockEngine(BaseEngine):
    """時間軸模擬。不碰 GPU、不載模型，秒開。

    速度刻意壓縮（真實 VLM 一次 2.4 秒、一部片跑數小時），
    這裡一次「呼叫」約 0.5 秒，讓整條流程在一分鐘內走完，方便看互動。
    """

    def __init__(self, store: JobStore, demo_anomalies: bool = True) -> None:
        super().__init__(store)
        self.demo_anomalies = demo_anomalies
        self._counter = 0

    def _run_one(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        with self._lock:
            self._counter += 1
            index = self._counter

        rng = random.Random(job.video_name)
        total_frames = rng.choice([1800, 2400, 3600, 5400])
        self.store.update(job_id, total_frames=total_frames)

        # 模擬「毀損檔案」：檔名含 corrupt / broken 就直接失敗
        lowered = job.video_name.lower()
        if "corrupt" in lowered or "broken" in lowered:
            time.sleep(1.5)
            self.store.update(job_id, frame=int(total_frames * 0.12), progress=0.12)
            raise RuntimeError("FFmpeg 解碼失敗 (Corrupted header)")

        # ── preparing ──
        time.sleep(1.2)

        # ── prefilter：YOLO 掃過抽樣幀，進度 0 → 0.30 ──
        self.store.update(job_id, status="prefilter")
        self.store.log(f"[YOLO] {job.video_name} 開始人車預過濾")
        steps = 12
        for i in range(1, steps + 1):
            if self._stopped():
                return
            progress = 0.30 * i / steps
            self.store.update(
                job_id, progress=progress, frame=int(total_frames * progress)
            )
            time.sleep(0.35)

        # ── vlm：主要耗時階段，進度 0.30 → 0.95 ──
        self.store.update(job_id, status="vlm")
        self.store.log(f"[VLM] {job.video_name} 進入行為判定階段")
        calls = rng.randint(18, 34)
        # demo：第 5 部（index % 5 == 0）在中途卡住，展示「閒置警示」紅標
        stall_at = calls // 2 if (self.demo_anomalies and index % 5 == 0) else -1

        events: list[dict[str, Any]] = []
        for call in range(1, calls + 1):
            if self._stopped():
                return
            if call == stall_at:
                self.store.log(
                    f"[WARNING] {job.video_name} 推論服務無回應，觸發閒置警示"
                    "（此為 mock 模式刻意模擬）",
                    level="warn",
                )
                # 把 last_vlm_at 往回撥，讓前端立刻進入 idle_alert 狀態
                record = self.store.get(job_id)
                if record is not None:
                    self.store.update(
                        job_id, last_vlm_at=time.monotonic() - 185
                    )
                time.sleep(8)

            latency_ms = rng.randint(1900, 3100)
            time.sleep(0.5)
            self.store.mark_vlm_call(
                job_id, latency_ms=latency_ms, calls_done=call, timeouts=0
            )
            progress = 0.30 + 0.65 * call / calls
            frame = int(total_frames * progress)
            self.store.update(job_id, progress=progress, frame=frame)

            # 約 12% 的呼叫判定為違規
            if rng.random() < 0.12:
                behavior = rng.choice(_MOCK_BEHAVIORS)
                sec = int(frame / 30)
                events.append(
                    {
                        "frame_index": frame,
                        "frame_time": f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}",
                        "overlay_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "plate": rng.choice(_MOCK_PLATES),
                        "behavior": behavior,
                        "confidence": round(rng.uniform(0.90, 0.98), 2),
                        "reason": _MOCK_REASONS[behavior],
                        "snapshot": "",
                        "clip": "",
                        "candidate_box": None,
                    }
                )
                self.store.update(job_id, events=list(events), violations=len(events))
                self.store.log(
                    f"[VLM] {job.video_name} 判定違規「{behavior}」"
                    f"（信心 {events[-1]['confidence']}）",
                    level="warn",
                )

        # ── merging ──
        self.store.update(job_id, status="merging", progress=0.97)
        self.store.log(f"[MERGE] {job.video_name} 彙整事件與去重")
        time.sleep(1.5)

        record = self.store.get(job_id)
        elapsed = record.elapsed_sec() if record else 0.0
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"{job.station_id}/{stamp}_{Path(job.video_name).stem}"
        self.store.update(
            job_id,
            status="completed",
            progress=1.0,
            frame=total_frames,
            finished_elapsed=elapsed,
            output_dir=out_dir,
            csv_path=f"{out_dir}/違規紀錄.csv",
            snap_dir=f"{out_dir}/違規截圖",
        )
        self.store.log(
            f"[OUTPUT] {job.video_name} 完成，共 {len(events)} 件違規，"
            f"已寫入 {out_dir}/違規紀錄.csv",
            level="success",
        )

    def _stopped(self) -> bool:
        return False


class RealEngine(BaseEngine):
    """接上真的 VLM pipeline。

    模型載入很慢（YOLO + ollama 連線檢查），所以 warmup 放在背景執行緒，
    第一部影片會等它完成 —— 這跟 Tkinter 版行為一致。
    """

    def __init__(self, store: JobStore) -> None:
        super().__init__(store)
        self._engine: Any = None
        self._config: Any = None
        self._warm = threading.Event()
        self._warm_error = ""
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        try:
            from enforcement_app.config import load_config
            from enforcement_app.pipeline_adapter import PipelineEngine

            self.store.log("[SYSTEM] 正在載入 AI 推論引擎…")
            self._config = load_config()
            self._engine = PipelineEngine(self._config)
            self._engine.warmup()
            self.store.log("[SYSTEM] AI 推論引擎就緒", level="success")
            # 「Co-DETR 垃圾偵測中」這個階段只在真的跑到時才出現，
            # 所以開工時要明講現在用的是哪個 litter 偵測器 —— 否則看不到那個階段時，
            # 分不出是「沒啟用」還是「還沒跑到那一步」。
            codetr_url = os.getenv("ENFORCE_CODETR_URL", "").strip()
            if codetr_url:
                self.store.log(
                    f"[SYSTEM] 垃圾偵測：第 18 次訓練的 Co-DETR（{codetr_url}）",
                    level="success",
                )
            else:
                self.store.log(
                    "[SYSTEM] 垃圾偵測：YOLO prefilter"
                    "（注意：目前那顆是 stock COCO yolov8n，沒有 litter 類別）",
                    level="warn",
                )
        except Exception as exc:  # noqa: BLE001
            self._warm_error = str(exc)
            self.store.log(f"[SYSTEM] 引擎載入失敗：{exc}", level="error")
        finally:
            self._warm.set()

    def _on_batch_start(self, parallel: int) -> None:
        if self._mode() == "test":
            # 測試模式不經過 VLM，不必等引擎暖機 —— 這樣 ollama 掛了也能展示 Co-DETR。
            self.store.log("[ENGINE] 模式：測試（純 Co-DETR，不呼叫 VLM）")
            return
        # §5「平行影片數要真的生效」：VLM 槽位數必須同步，否則設 8 部只會跑 2 部。
        self._warm.wait(timeout=600)
        if self._engine is not None:
            try:
                self._engine.set_max_vlm_inflight(parallel)
                self.store.log(f"[ENGINE] VLM 並行槽位已同步為 {parallel}")
            except Exception:  # noqa: BLE001
                pass

    def _mode(self) -> str:
        """`full`（完整）或 `test`（測試）。未指定一律當完整，行為不變。"""
        return "test" if str(self.options.get("mode", "full")) == "test" else "full"

    def _run_one(self, job_id: str) -> None:
        if self._mode() == "test":
            self._run_codetr_test(job_id)
        else:
            self._run_full(job_id)

    # ── 測試模式：純 Co-DETR ──────────────────────────────────────
    def _run_codetr_test(self, job_id: str) -> None:
        """整支影片餵 Co-DETR，偵測到 litter/garbage 的幀就是違規片段。

        完全不碰 `PipelineEngine`，所以「完整」流程的行為一行都不受影響。
        """
        from datetime import datetime

        from enforcement_app.station import UNKNOWN_STATION, sanitize_station_id

        from .codetr_test_mode import CoDetrTestPipeline

        record = self.store.get(job_id)
        if record is None:
            return
        opts = self.options
        url = str(opts.get("codetr_url") or os.getenv("ENFORCE_CODETR_URL", "")).strip()
        if not url:
            raise RuntimeError(
                "測試模式需要 Co-DETR sidecar。請設定 ENFORCE_CODETR_URL "
                "（例如 http://codetr:8091）並確認 `docker compose up -d codetr`"
            )

        pipeline = CoDetrTestPipeline(
            base_url=url,
            score_thr=float(opts.get("codetr_score", 0.30)),
            stride=int(opts.get("heavy_frame_stride", 1)),
        )
        health = pipeline.health()
        if health is None:
            raise RuntimeError(f"連不上 Co-DETR sidecar：{url}")
        self.store.update(job_id, status="codetr")

        # 輸出路徑與完整模式同一套分層：{root}/{station}/{timestamp}_{stem}/
        output_root = Path(
            opts.get("output_root")
            or (self._config.output_root if self._config else self.store.output_root)
        )
        station_dir = sanitize_station_id(record.station_id or UNKNOWN_STATION)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = output_root / station_dir / f"{stamp}_{Path(record.video_path).stem}"

        def on_progress(update: dict[str, Any]) -> None:
            # 測試模式完全不呼叫 VLM，閒置時鐘（last_vlm_at）沒人更新，
            # 跑超過 ENFORCE_VLM_IDLE_ALERT_SEC（預設 120 秒）整列就會被誤標紅。
            # 閒置警示的用意是「推論後端還活著嗎」—— 這個模式的後端是 Co-DETR sidecar，
            # 每收到一批偵測結果就是它的存活證明，拿來當心跳語意才對得上。
            self.store.update(
                job_id,
                status=str(update.get("stage", "codetr")),
                progress=float(update.get("progress", 0.0)),
                frame=int(update.get("frame", 0)),
                total_frames=int(update.get("total_frames", 0)),
                last_vlm_at=time.monotonic(),
            )

        result = pipeline.run(
            video_path=Path(record.video_path),
            out_dir=out_dir,
            progress=on_progress,
            log=lambda msg, level="info": self.store.log(msg, level=level),
        )

        events = result["events"]
        cur = self.store.get(job_id)
        self.store.update(
            job_id,
            status="completed",
            progress=1.0,
            finished_elapsed=cur.elapsed_sec() if cur else 0.0,
            violations=len(events),
            events=list(events),
            output_dir=result["output_dir"],
            csv_path=result["csv"],
            snap_dir=result["snap_dir"],
        )
        if cur is not None and cur.job_note:
            self.store.set_note(job_id, cur.job_note)
        self.store.log(
            f"[OUTPUT] {record.video_name} 完成（測試模式）："
            f"掃 {result['scanned_frames']} 幀、{result['hit_frames']} 幀命中、"
            f"{len(events)} 個違規片段",
            level="success",
        )

    # ── 完整模式：原本的 VLM pipeline（邏輯未改）──────────────────
    def _run_full(self, job_id: str) -> None:
        from enforcement_app.models import ProgressUpdate, VideoJob
        from enforcement_app.pipeline_adapter import build_job_options

        self._warm.wait(timeout=600)
        if self._engine is None:
            raise RuntimeError(self._warm_error or "推論引擎尚未就緒")

        record = self.store.get(job_id)
        if record is None:
            return
        opts = self.options
        options = build_job_options(
            enable_litter=bool(opts.get("enable_litter", True)),
            enable_urination=bool(opts.get("enable_urination", True)),
            require_plate_for_csv=bool(opts.get("require_plate_for_csv", False)),
            heavy_frame_stride=int(opts.get("heavy_frame_stride", 1)),
            max_vlm_calls_per_video=int(opts.get("max_vlm_calls_per_video", 2400)),
        )
        job = VideoJob(
            job_id=job_id,
            video_path=Path(record.video_path),
            options=options,
            station_id=record.station_id,
        )

        # pipeline 的三個回呼來源回報的 frame 語意不同（prefilter 與心跳是「讀取前緣」、
        # vlm 是「判定前緣」），交錯時進度會前後跳。這裡照收，由 JobStore.update()
        # 的單調遞增保護統一處理 —— 讀取前緣領先的幅度被預取佇列上限
        # （_VLM_PREFETCH_BATCH * 3 = 24 個候選幀）夾住，不會脫韁。
        def on_progress(update: ProgressUpdate) -> None:
            self.store.mark_vlm_call(
                job_id,
                latency_ms=update.last_call_latency_ms,
                calls_done=update.vlm_calls_done,
                timeouts=update.vlm_timeouts,
            )
            self.store.update(
                job_id,
                status=update.stage,
                progress=update.progress,
                frame=update.frame,
                total_frames=update.total_frames,
            )

        output_root = Path(opts.get("output_root") or self._config.output_root)
        result = self._engine.process_video(job, on_progress, output_root)

        cur = self.store.get(job_id)
        elapsed = cur.elapsed_sec() if cur else result.elapsed_sec
        self.store.update(
            job_id,
            status="completed",
            progress=1.0,
            finished_elapsed=elapsed,
            violations=result.violations,
            events=list(result.events),
            output_dir=str(result.output_dir or ""),
            csv_path=str(result.csv_path or ""),
            snap_dir=str(result.snap_dir or ""),
        )
        # 完成後把註記寫進輸出目錄，讓舉證資料夾自帶稽查備註
        if cur is not None and cur.job_note:
            self.store.set_note(job_id, cur.job_note)
        self.store.log(
            f"[OUTPUT] {record.video_name} 完成，共 {result.violations} 件違規",
            level="success",
        )
