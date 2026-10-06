from __future__ import annotations

import csv
import threading
import time as _time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import AppConfig
from .models import ProgressUpdate, VideoJob
from .pipeline_adapter import PipelineEngine, build_job_options
from .scanner import scan_video_files
from .stage_display import display_stage
from .station import infer_station_id
from .watchdog import WatchdogLogger


@dataclass
class VLMStats:
    """Per-row running totals for VLM telemetry shown in the Treeview.

    Updated by the GUI on_progress callback (one record_call per VLM telemetry
    payload), read by the 1Hz tick to render the 'VLM' column.
    """
    calls: int = 0
    timeouts: int = 0
    total_latency_ms: int = 0
    last_call_at: float = 0.0  # monotonic seconds; 0 = never called
    last_call_latency_ms: int = 0

    def avg_latency_sec(self) -> float:
        if self.calls <= 0:
            return 0.0
        return self.total_latency_ms / self.calls / 1000.0

    def record_call(self, latency_ms: int, timed_out: bool) -> None:
        self.calls += 1
        self.total_latency_ms += int(latency_ms)
        if timed_out:
            self.timeouts += 1
        self.last_call_latency_ms = int(latency_ms)
        self.last_call_at = _time.monotonic()


def _format_vlm_stats(
    calls: int,
    avg_latency_sec: float,
    timeouts: int,
    last_call_ago_sec: int,
) -> str:
    if calls <= 0:
        return "尚未呼叫"
    return (
        f"已呼叫 {calls} / 平均 {avg_latency_sec:.1f}s / "
        f"逾時 {timeouts} / 上次 {int(last_call_ago_sec)}秒前"
    )


def _should_flag_idle(
    *,
    now: float,
    last_call_at: float,
    started_at: float,
    idle_alert_sec: float,
) -> bool:
    """Return True when this row should be highlighted as idle (probably stuck).

    Two cases:
    - Has been running > 60s and last VLM call was > idle_alert_sec ago
    - Has been running > idle_alert_sec but no VLM call yet (worker stuck in
      prefilter / preparing)
    """
    elapsed = now - started_at
    if elapsed < 60.0:
        return False
    if last_call_at <= 0.0:
        # No VLM call yet; only flag if we've been running long enough
        return elapsed >= idle_alert_sec
    return (now - last_call_at) >= idle_alert_sec


class EnforcementGUI:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.engine = PipelineEngine(config)
        self.executor = ThreadPoolExecutor(max_workers=config.max_parallel_videos)
        self.jobs: dict[str, VideoJob] = {}
        self.row_to_job: dict[str, str] = {}
        # Per-row 起跑時間 + VLM 統計，1Hz tick 用來即時更新「耗時」與「VLM」欄。
        self._row_started_at: dict[str, float] = {}
        self._row_vlm_stats: dict[str, VLMStats] = {}
        # 整批次 VLM 計數器（用於 status bar 全域統計顯示）。
        self._global_vlm_calls: int = 0
        self._global_vlm_timeouts: int = 0
        self._global_vlm_total_latency_ms: int = 0
        # 長批次 debug 用：每 N 秒把活躍 jobs 狀態寫到 log，超時標 STALE。
        self.watchdog = WatchdogLogger(
            log_path=config.watchdog_log_path,
            interval_sec=config.watchdog_interval_sec,
            stale_timeout_sec=config.stale_timeout_sec,
        )
        self.watchdog.start()
        self.root = tk.Tk()
        self.root.title("環保科技執法系統")
        self.root.geometry("1280x820")
        self._build_ui()
        self._apply_style()
        # 1Hz tick：每秒刷新活躍列的「耗時」「VLM」欄、idle 紅標、status bar。
        self.root.after(self.config.gui_tick_ms, self._tick_active_rows)
        threading.Thread(target=self._warmup, daemon=True).start()

    def _apply_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#EEF2F7")
        style.configure("TLabelframe", background="#FFFFFF", foreground="#0F3460")
        style.configure(
            "TLabelframe.Label",
            background="#FFFFFF",
            foreground="#0F3460",
            font=("Microsoft JhengHei", 11, "bold"),
        )
        style.configure(
            "TLabel",
            background="#EEF2F7",
            foreground="#1A1F36",
            font=("Microsoft JhengHei", 11),
        )
        style.configure("TButton", font=("Microsoft JhengHei", 11), padding=8)
        style.configure("Treeview", font=("Microsoft JhengHei", 10), rowheight=26)
        style.configure("Treeview.Heading", font=("Microsoft JhengHei", 10, "bold"))

    def _build_ui(self) -> None:
        self.root.configure(background="#EEF2F7")

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=16, pady=(14, 8))

        source_box = ttk.LabelFrame(top, text="資料來源")
        source_box.pack(fill=tk.X)
        self.folder_var = tk.StringVar()
        ttk.Entry(source_box, textvariable=self.folder_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10
        )
        ttk.Button(source_box, text="選擇資料夾", command=self._pick_folder).pack(
            side=tk.LEFT, padx=6, pady=10
        )
        ttk.Button(source_box, text="掃描影片", command=self._scan_folder).pack(
            side=tk.LEFT, padx=(0, 10), pady=10
        )

        out_box = ttk.LabelFrame(top, text="輸出資料夾")
        out_box.pack(fill=tk.X, pady=(8, 0))
        self.output_var = tk.StringVar(value=str(self.config.output_root))
        ttk.Entry(out_box, textvariable=self.output_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10
        )
        ttk.Button(out_box, text="選擇資料夾", command=self._pick_output_folder).pack(
            side=tk.LEFT, padx=(0, 10), pady=10
        )

        options = ttk.Frame(self.root)
        options.pack(fill=tk.X, padx=16, pady=8)

        opt_box = ttk.LabelFrame(options, text="偵測設定")
        opt_box.pack(fill=tk.X)
        self.enable_litter = tk.BooleanVar(value=True)
        self.enable_urine = tk.BooleanVar(value=True)
        self.require_plate = tk.BooleanVar(value=True)
        self.stride_var = tk.IntVar(value=self.config.heavy_frame_stride)
        self.workers_var = tk.IntVar(value=self.config.max_parallel_videos)
        self.vlm_budget_var = tk.IntVar(value=self.config.max_vlm_calls_per_video)

        tk.Checkbutton(
            opt_box,
            text="亂丟垃圾/菸蒂偵測",
            variable=self.enable_litter,
            bg="#FFFFFF",
            fg="#1A1F36",
            activebackground="#FFFFFF",
            activeforeground="#1A1F36",
            font=("Microsoft JhengHei", 11),
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Checkbutton(
            opt_box,
            text="隨地便溺偵測",
            variable=self.enable_urine,
            bg="#FFFFFF",
            fg="#1A1F36",
            activebackground="#FFFFFF",
            activeforeground="#1A1F36",
            font=("Microsoft JhengHei", 11),
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Label(opt_box, text="平行影片數").pack(side=tk.LEFT, padx=(16, 6), pady=10)
        ttk.Spinbox(
            opt_box, from_=1, to=8, textvariable=self.workers_var, width=5
        ).pack(side=tk.LEFT, pady=10)

        action = ttk.Frame(self.root)
        action.pack(fill=tk.X, padx=16, pady=8)
        ttk.Button(action, text="開始取締", command=self._start_processing).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(action, text="匯出整批 CSV", command=self._export_combined_csv).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(action, text="清除清單", command=self._clear_table).pack(
            side=tk.LEFT
        )

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 14))

        left = ttk.LabelFrame(body, text="執法工作佇列")

        cols = (
            "name",
            "station",
            "status",
            "progress",
            "vlm",
            "elapsed",
            "violations",
            "output",
            "error",
        )
        self.tree = ttk.Treeview(left, columns=cols, show="headings")
        headers = {
            "name": "檔名",
            "station": "站別",
            "status": "狀態",
            "progress": "進度",
            "vlm": "VLM",
            "elapsed": "耗時",
            "violations": "違規數",
            "output": "輸出目錄",
            "error": "錯誤訊息",
        }
        widths = {
            "name": 160,
            "station": 60,
            "status": 90,
            "progress": 70,
            "vlm": 220,
            "elapsed": 100,
            "violations": 70,
            "output": 280,
            "error": 220,
        }
        for key in cols:
            self.tree.heading(key, text=headers[key])
            self.tree.column(key, width=widths[key], anchor=tk.W)
        # idle_warn 紅底色 — 距上次 VLM call 超過 idle_alert 秒就掛 tag。
        self.tree.tag_configure("idle_warn", background="#FFEAEA")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_paned = ttk.PanedWindow(body, orient=tk.VERTICAL)

        log_box = ttk.LabelFrame(right_paned, text="執法紀錄")
        self.log_text = tk.Text(
            log_box,
            width=44,
            height=16,
            bg="#0B132B",
            fg="#E6EDF7",
            font=("Consolas", 10),
            relief=tk.FLAT,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        event_box = ttk.LabelFrame(right_paned, text="違規事件")
        event_cols = ("video", "time", "plate", "behavior", "confidence")
        self.events_tree = ttk.Treeview(
            event_box, columns=event_cols, show="headings", height=12
        )
        event_headers = {
            "video": "影片",
            "time": "右上角時間",
            "plate": "車牌",
            "behavior": "行為",
            "confidence": "信心",
        }
        self._event_cols = event_cols
        self._event_col_weights = {
            "video": 24,
            "time": 24,
            "plate": 22,
            "behavior": 18,
            "confidence": 12,
        }
        self._event_col_min_widths = {
            "video": 80,
            "time": 80,
            "plate": 70,
            "behavior": 56,
            "confidence": 48,
        }
        for key in event_cols:
            self.events_tree.heading(key, text=event_headers[key])
            self.events_tree.column(
                key, width=self._event_col_min_widths[key], anchor=tk.W
            )
        self.events_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        event_box.bind("<Configure>", self._on_event_box_configure)

        right_paned.add(log_box, weight=1)
        right_paned.add(event_box, weight=1)

        body.add(left, weight=65)
        body.add(right_paned, weight=35)

        self.status_var = tk.StringVar(value="系統初始化中...")
        ttk.Label(self.root, textvariable=self.status_var).pack(
            fill=tk.X, padx=16, pady=(0, 10)
        )

    def _warmup(self) -> None:
        try:
            self.engine.warmup()
            self.root.after(0, lambda: self._log("模型預熱完成，系統可開始執法。"))
            self.root.after(0, lambda: self.status_var.set("模型已就緒"))
        except Exception as exc:
            self.root.after(0, lambda: self._log(f"模型預熱失敗: {exc}"))
            self.root.after(0, lambda: self.status_var.set("模型預熱失敗"))

    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇影片資料夾")
        if folder:
            self.folder_var.set(folder)

    def _pick_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_var.set(folder)

    def _scan_folder(self) -> None:
        folder = Path(self.folder_var.get().strip())
        files = scan_video_files(folder, self.config.video_exts)
        self._clear_table(clear_logs=False)
        for path in files:
            job_id = uuid.uuid4().hex
            station_id = infer_station_id(path)
            job = VideoJob(
                job_id=job_id,
                video_path=path,
                options=build_job_options(
                    enable_litter=self.enable_litter.get(),
                    enable_urination=self.enable_urine.get(),
                    require_plate_for_csv=self.require_plate.get(),
                    heavy_frame_stride=self.stride_var.get(),
                    max_vlm_calls_per_video=self.vlm_budget_var.get(),
                ),
                station_id=station_id,
            )
            row_id = self.tree.insert(
                "",
                tk.END,
                values=(
                    path.name,
                    station_id,
                    "待處理",
                    "0%",
                    "尚未呼叫",
                    self._format_elapsed(0.0),
                    "0",
                    "",
                    "",
                ),
            )
            self.jobs[job_id] = job
            self.row_to_job[row_id] = job_id
        self.status_var.set(f"掃描完成，共 {len(files)} 支影片")
        station_counts: dict[str, int] = {}
        for job in self.jobs.values():
            station_counts[job.station_id] = station_counts.get(job.station_id, 0) + 1
        breakdown = "、".join(
            f"{s}:{n}" for s, n in sorted(station_counts.items())
        )
        self._log(f"已載入 {len(files)} 支影片（站別：{breakdown}）")

    def _start_processing(self) -> None:
        if not self.jobs:
            messagebox.showwarning("提醒", "請先選擇資料夾並掃描影片")
            return
        if self.config.vlm_provider != "ollama" and not self.config.glm_api_key.strip():
            messagebox.showwarning("提醒", "請先設定 GLM_API_KEY，再開始取締")
            return
        workers = max(1, int(self.workers_var.get()))
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.executor = ThreadPoolExecutor(max_workers=workers)
        # VLM semaphore 容量必須同步為 workers，否則 8 部並行只會有
        # ENFORCE_MAX_VLM_INFLIGHT（預設對齊 config.max_parallel_videos）支進入 VLM 階段，
        # 其餘卡在 semaphore.acquire()，導致看起來「多部 0%」的假卡住現象。
        self.engine.set_max_vlm_inflight(workers)
        self._log(f"開始取締。平行影片數={workers}，VLM 槽位已同步")
        self.status_var.set("執法中...")
        for row_id, job_id in list(self.row_to_job.items()):
            base_job = self.jobs[job_id]
            updated = VideoJob(
                job_id=base_job.job_id,
                video_path=base_job.video_path,
                options=build_job_options(
                    enable_litter=self.enable_litter.get(),
                    enable_urination=self.enable_urine.get(),
                    require_plate_for_csv=self.require_plate.get(),
                    heavy_frame_stride=self.stride_var.get(),
                    max_vlm_calls_per_video=self.vlm_budget_var.get(),
                ),
                station_id=base_job.station_id,
            )
            self.jobs[job_id] = updated
            # 提交到 thread pool 但尚未實際執行 → 顯示「排隊中」。
            # pipeline_adapter.process_video 開工時會立刻推一個 'preparing'
            # 進度回呼，UI 看到的就是「排隊中」→「準備中」→「YOLO 偵測中」...
            self.tree.set(row_id, "status", display_stage("queued"))
            self.tree.set(row_id, "progress", "0%")
            self.tree.set(row_id, "vlm", "尚未呼叫")
            self._row_started_at[row_id] = _time.monotonic()
            self._row_vlm_stats[row_id] = VLMStats()
            self.executor.submit(self._run_job, row_id, updated)

    def _run_job(self, row_id: str, job: VideoJob) -> None:
        self.watchdog.register(job.job_id, job.video_path.name)
        try:

            def on_progress(p: ProgressUpdate) -> None:
                self.watchdog.heartbeat(p.job_id, p.stage, p.progress)

                def apply() -> None:
                    stats = self._row_vlm_stats.get(row_id)
                    if stats is not None and p.last_call_latency_ms > 0:
                        # vlm_timeouts is monotonically increasing in the
                        # emitter; detect this delta as a timeout event.
                        is_timeout = p.vlm_timeouts > stats.timeouts
                        stats.record_call(
                            latency_ms=p.last_call_latency_ms,
                            timed_out=is_timeout,
                        )
                        self._global_vlm_calls += 1
                        self._global_vlm_total_latency_ms += p.last_call_latency_ms
                        if is_timeout:
                            self._global_vlm_timeouts += 1
                    self._update_progress(row_id, p.progress, p.stage)

                self.root.after(0, apply)

            output_root = Path(self.output_var.get().strip())
            done = self.engine.process_video(job, on_progress, output_root=output_root)
            self.root.after(0, lambda: self._finish_job(row_id, done))
        except Exception as exc:
            msg = f"{exc}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._fail_job(row_id, msg))
        finally:
            self.watchdog.unregister(job.job_id)

    def _update_progress(self, row_id: str, progress: float, stage: str) -> None:
        # 把內部代號（prefilter/vlm/preparing/...）翻成中文顯示名稱。
        self.tree.set(row_id, "status", display_stage(stage))
        self.tree.set(row_id, "progress", f"{int(progress * 100)}%")

    def _tick_active_rows(self) -> None:
        """每 gui_tick_ms 毫秒刷新活躍列的「耗時」「VLM」欄、idle 紅標。

        進度欄與 VLM 計數由 on_progress 寫入字典；本 tick 只負責把字典內容
        推到 Treeview，避免高頻 VLM 回呼直接打 Treeview redraw。
        """
        try:
            now = _time.monotonic()
            for row_id, started_at in list(self._row_started_at.items()):
                if row_id not in self.row_to_job:
                    continue
                # Stop ticking once row is in terminal state. Keep last value.
                status = self.tree.set(row_id, "status")
                if status in {"完成", "錯誤"}:
                    self._row_started_at.pop(row_id, None)
                    continue
                elapsed = now - started_at
                self.tree.set(row_id, "elapsed", self._format_elapsed(elapsed))
                stats = self._row_vlm_stats.get(row_id)
                if stats is not None:
                    last_ago = (
                        int(now - stats.last_call_at)
                        if stats.last_call_at > 0 else 0
                    )
                    self.tree.set(
                        row_id, "vlm",
                        _format_vlm_stats(
                            calls=stats.calls,
                            avg_latency_sec=stats.avg_latency_sec(),
                            timeouts=stats.timeouts,
                            last_call_ago_sec=last_ago,
                        ),
                    )
                    if _should_flag_idle(
                        now=now,
                        last_call_at=stats.last_call_at,
                        started_at=started_at,
                        idle_alert_sec=self.config.vlm_idle_alert_sec,
                    ):
                        self.tree.item(row_id, tags=("idle_warn",))
                    else:
                        self.tree.item(row_id, tags=())
            self._refresh_global_status()
        finally:
            self.root.after(self.config.gui_tick_ms, self._tick_active_rows)

    def _refresh_global_status(self) -> None:
        """Append global VLM totals to status_var (e.g. for end-of-batch report)."""
        if self._global_vlm_calls <= 0:
            return
        avg = self._global_vlm_total_latency_ms / self._global_vlm_calls / 1000.0
        base = self.status_var.get()
        # 拿掉先前 tick 添加的「| 全域 VLM ...」段落，重新加上最新值。
        prefix = base.split(" | 全域 VLM")[0]
        self.status_var.set(
            f"{prefix} | 全域 VLM 已呼叫 {self._global_vlm_calls} / "
            f"平均 {avg:.1f}s / 逾時 {self._global_vlm_timeouts}"
        )

    def _finish_job(self, row_id: str, job: VideoJob) -> None:
        self.jobs[job.job_id] = job
        self.tree.set(row_id, "status", "完成")
        self.tree.set(row_id, "progress", "100%")
        self.tree.set(row_id, "elapsed", self._format_elapsed(job.elapsed_sec))
        self.tree.set(row_id, "violations", str(job.violations))
        self.tree.set(row_id, "output", str(job.output_dir or ""))
        self.tree.set(row_id, "error", "")
        # Stop ticking this row, but keep VLM stats so final value stays visible.
        self._row_started_at.pop(row_id, None)
        self.tree.item(row_id, tags=())
        self._log(
            f"{job.video_path.name} 完成，違規 {job.violations} 件，耗時 {self._format_elapsed(job.elapsed_sec)}"
        )
        for event in job.events:
            overlay_time = str(event.get("overlay_time", "")).strip()
            plate = str(event.get("plate", "")).strip()
            behavior = str(event.get("behavior", "")).strip()
            confidence_obj = event.get("confidence", 0.0)
            confidence = (
                f"{float(confidence_obj):.2f}"
                if isinstance(confidence_obj, (int, float))
                else "0.00"
            )
            self.events_tree.insert(
                "",
                tk.END,
                values=(job.video_path.name, overlay_time, plate, behavior, confidence),
            )
        self._check_all_done()

    def _fail_job(self, row_id: str, msg: str) -> None:
        job_id = self.row_to_job[row_id]
        job = self.jobs[job_id]
        job.status = "error"
        job.error = msg
        self.tree.set(row_id, "status", "錯誤")
        self.tree.set(row_id, "error", msg.splitlines()[0][:200])
        self._log(f"{job.video_path.name} 發生錯誤: {msg.splitlines()[0]}")
        self._row_started_at.pop(row_id, None)
        self.tree.item(row_id, tags=())
        self._check_all_done()

    def _check_all_done(self) -> None:
        states = [self.tree.set(row, "status") for row in self.tree.get_children()]
        if states and all(s in {"完成", "錯誤"} for s in states):
            self.status_var.set("本批次執法完成")

    def _export_combined_csv(self) -> None:
        header: list[str] | None = None
        rows: list[list[str]] = []
        snapshot_col_index: int | None = None

        def to_snapshot_hyperlink(raw_value: str, base_dir: Path) -> str:
            value = raw_value.strip()
            if not value:
                return ""
            try:
                snapshot_path = Path(value)
                if not snapshot_path.is_absolute():
                    snapshot_path = (base_dir / snapshot_path).resolve()
                uri = snapshot_path.resolve().as_uri()
            except (OSError, ValueError):
                return raw_value
            return f'=HYPERLINK("{uri}","開啟截圖")'

        for job in self.jobs.values():
            if not job.csv_path or not job.csv_path.exists():
                continue
            with job.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                parsed_rows = list(reader)
            if not parsed_rows:
                continue

            source_header = parsed_rows[0]
            if header is None:
                header = ["source_video"] + source_header
                snapshot_col_index = next(
                    (
                        idx
                        for idx, col in enumerate(source_header)
                        if col.strip() in {"截圖路徑", "snapshot", "snapshot_path"}
                    ),
                    None,
                )

            for raw_row in parsed_rows[1:]:
                row = [job.video_path.name] + raw_row
                if snapshot_col_index is not None:
                    row_snapshot_index = snapshot_col_index + 1
                    if row_snapshot_index < len(row):
                        row[row_snapshot_index] = to_snapshot_hyperlink(
                            row[row_snapshot_index], job.csv_path.parent
                        )
                rows.append(row)
        if header is None:
            messagebox.showinfo("提示", "目前沒有可匯出的結果")
            return
        out = filedialog.asksaveasfilename(
            title="儲存整批彙整 CSV",
            defaultextension=".csv",
            initialfile="combined_violations.csv",
        )
        if not out:
            return
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        self._log(f"已匯出整批結果: {out}")
        messagebox.showinfo("完成", f"已匯出: {out}")

    def _clear_table(self, clear_logs: bool = True) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        for row in self.events_tree.get_children():
            self.events_tree.delete(row)
        self.jobs.clear()
        self.row_to_job.clear()
        if clear_logs:
            self.log_text.delete("1.0", tk.END)

    def _log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _on_event_box_configure(self, _: tk.Event[tk.Widget]) -> None:
        self._resize_event_columns()

    def _resize_event_columns(self) -> None:
        table_width = max(0, self.events_tree.winfo_width() - 4)
        if table_width <= 0:
            return

        cols = self._event_cols
        weights = self._event_col_weights
        min_widths = self._event_col_min_widths
        total_weight = sum(weights[c] for c in cols)

        widths: dict[str, int] = {}
        used = 0
        for idx, col in enumerate(cols):
            if idx == len(cols) - 1:
                width = table_width - used
            else:
                width = int(table_width * weights[col] / total_weight)
                used += width
            widths[col] = max(min_widths[col], width)

        total_width = sum(widths.values())
        if total_width > table_width:
            overflow = total_width - table_width
            shrinkable_cols = [c for c in cols if widths[c] > min_widths[c]]
            while overflow > 0 and shrinkable_cols:
                for col in list(shrinkable_cols):
                    if overflow <= 0:
                        break
                    if widths[col] > min_widths[col]:
                        widths[col] -= 1
                        overflow -= 1
                    else:
                        shrinkable_cols.remove(col)

        for col in cols:
            self.events_tree.column(col, width=widths[col], anchor=tk.W)

    def _format_elapsed(self, elapsed_sec: float) -> str:
        total_seconds = max(0, int(round(elapsed_sec)))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}小時{minutes}分{seconds}秒"

    def run(self) -> int:
        try:
            self.root.mainloop()
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.watchdog.stop()
        return 0


def launch_gui(config: AppConfig) -> int:
    app = EnforcementGUI(config)
    return app.run()
