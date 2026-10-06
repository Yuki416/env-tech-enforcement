from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _clean_env_str(value: str, default: str = "") -> str:
    text = (value or default).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _get_app_root() -> Path:
    """
    frozen (PyInstaller) 模式：
      從 exe 所在目錄開始，往上最多找 2 層，找到含 models/ 的目錄即回傳。
      找不到就退回 exe 目錄（讓後續路徑檢查給出清楚的錯誤訊息）。
    一般 Python 模式：
      回傳 repo 根目錄（config.py 的兩層 parent）。
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        for candidate in (base, base.parent, base.parent.parent):
            if (candidate / "models").is_dir():
                return candidate
        return base  # fallback：讓 FileNotFoundError 顯示正確路徑
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    output_root: Path
    model_litter: Path
    model_plate: Path
    ts_litter_dir: Path
    ts_urine_dir: Path
    pifpaf_ckpt: str
    max_parallel_videos: int
    heavy_frame_stride: int
    max_vlm_calls_per_video: int
    max_vlm_calls_per_minute: int
    max_vlm_inflight: int
    conf: float
    litter_conf: float
    nms_iou: float
    prefilter_conf: float
    glm_api_key: str
    vlm_provider: str
    glm_model: str
    glm_api_base: str
    glm_timeout_sec: float
    vlm_retry_count: int
    vlm_retry_backoff_sec: float
    vlm_cooldown_sec: float
    event_merge_sec: float
    min_violation_confidence: float
    confirm_violation_confidence: float
    max_candidates_per_frame: int
    roi_expand: float
    litter_clip_before_sec: float
    litter_clip_after_sec: float
    require_plate_for_csv: bool
    video_exts: tuple[str, ...]
    watchdog_interval_sec: float
    stale_timeout_sec: float
    watchdog_log_path: Path
    glm_connect_timeout_sec: float
    vlm_hard_timeout_sec: float
    vlm_idle_alert_sec: float
    gui_tick_ms: int
    log_level: str
    log_path: Path
    # 2026-04-30 anti-hallucination guard 改成 opt-in。
    # 預設 False：信任 VLM 判讀，即使 YOLO 沒在當幀看到 litter 也保留亂丟
    # 標籤。理由：case5 reproduce 顯示真正的亂丟動作（如丟出垃圾袋）也常
    # frame-miss litter class，舊版硬 relabel 把對的 lit 標成錯的便溺。
    # 設 True 還原舊行為（vendor 4/15 第3站：5 件真便溺被 VLM 誤判亂丟，
    # relabel 後反而變對）。詳見 enforcement_app/vlm_enforcement.py 註解。
    relabel_lit_to_urin: bool


def load_config() -> AppConfig:
    repo_root = _get_app_root()
    # frozen 模式下輸出放在 exe 旁的 output/；一般模式保持原路徑
    if getattr(sys, "frozen", False):
        output_root = repo_root / "output"
    else:
        output_root = repo_root / "enforcement_app" / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    max_parallel_videos = max(1, int(os.getenv("ENFORCE_MAX_PARALLEL", "5")))
    # max_vlm_inflight 預設對齊 max_parallel_videos；僅當使用者明確設定
    # ENFORCE_MAX_VLM_INFLIGHT 為非空字串時，才以環境變數為準。
    # 這是 4/20-4/23 批次「設 8 部並行卻只跑 2 部」的根因修正。
    _inflight_env = os.getenv("ENFORCE_MAX_VLM_INFLIGHT", "").strip()
    max_vlm_inflight = (
        max(1, int(_inflight_env)) if _inflight_env else max_parallel_videos
    )

    return AppConfig(
        repo_root=repo_root,
        output_root=output_root,
        model_litter=repo_root
        / "models"
        / "yolo"
        / "runs"
        / "detect"
        / "train_person_vehicle_litter"
        / "weights"
        / "best.pt",
        model_plate=repo_root / "models" / "license" / "best_license.pt",
        ts_litter_dir=repo_root / "models" / "timesformer" / "litter",
        ts_urine_dir=repo_root / "models" / "timesformer" / "urine",
        pifpaf_ckpt=_clean_env_str(os.getenv("PIFPAF_CKPT", "resnet50"), "resnet50"),
        max_parallel_videos=max_parallel_videos,
        heavy_frame_stride=max(1, int(os.getenv("ENFORCE_HEAVY_STRIDE", "1"))),
        max_vlm_calls_per_video=max(
            0, int(os.getenv("ENFORCE_MAX_VLM_CALLS_PER_VIDEO", "2400"))
        ),
        # 2026-04-24 新增：每分鐘 N 次 VLM 呼叫，與 per_video 共存時取 min。
        # 預設 30 ≈ 每 2 秒一次，對 2h18m 影片 = 138×30 = 4140 呼叫，
        # 若 per_video=2400 仍為上限，實際取 min(2400, 4140) = 2400（向後相容）。
        # 使用者若把 per_video 設為 0 並保留 per_minute=30，即進入純時長計費模式。
        max_vlm_calls_per_minute=max(
            0, int(os.getenv("ENFORCE_MAX_VLM_CALLS_PER_MINUTE", "30"))
        ),
        max_vlm_inflight=max_vlm_inflight,
        conf=float(os.getenv("ENFORCE_CONF", "0.10")),
        litter_conf=float(os.getenv("ENFORCE_LITTER_CONF", "0.14")),
        nms_iou=float(os.getenv("ENFORCE_NMS_IOU", "0.65")),
        prefilter_conf=float(os.getenv("ENFORCE_PREFILTER_CONF", "0.25")),
        glm_api_key=_clean_env_str(os.getenv("GLM_API_KEY", ""), ""),
        vlm_provider=_clean_env_str(
            os.getenv("VLM_PROVIDER", "ollama"), "ollama"
        ).lower(),
        glm_model=_clean_env_str(
            os.getenv("GLM_MODEL", "qwen3-enforcement"),
            "qwen3-enforcement",
        ),
        glm_api_base=_clean_env_str(
            os.getenv("GLM_API_BASE", "http://127.0.0.1:11434"),
            "http://127.0.0.1:11434",
        ),
        glm_timeout_sec=float(os.getenv("GLM_TIMEOUT_SEC", "60")),
        vlm_retry_count=max(0, int(os.getenv("VLM_RETRY_COUNT", "3"))),
        vlm_retry_backoff_sec=max(
            0.0, float(os.getenv("VLM_RETRY_BACKOFF_SEC", "5.0"))
        ),
        vlm_cooldown_sec=float(os.getenv("ENFORCE_VLM_COOLDOWN_SEC", "1.0")),
        event_merge_sec=float(os.getenv("ENFORCE_EVENT_MERGE_SEC", "3.0")),
        # 2026-04-30 default raised from 0.80 → 0.90 to absorb the precision
        # cost of B4 (normalize_behavior unblocked many simplified-Chinese
        # candidates, including some VLM hallucinations that arrive at high
        # but not-quite-perfect confidence). Set ENFORCE_MIN_VIOLATION_CONF
        # to roll back if recall drops too far for a given vendor scene.
        min_violation_confidence=max(
            0.0, min(1.0, float(os.getenv("ENFORCE_MIN_VIOLATION_CONF", "0.90")))
        ),
        confirm_violation_confidence=max(
            0.0,
            min(1.0, float(os.getenv("ENFORCE_CONFIRM_MIN_VIOLATION_CONF", "0.75"))),
        ),
        max_candidates_per_frame=max(
            1, int(os.getenv("ENFORCE_MAX_CANDIDATES_PER_FRAME", "8"))
        ),
        roi_expand=max(0.0, float(os.getenv("ENFORCE_ROI_EXPAND", "0.20"))),
        litter_clip_before_sec=max(
            0.0, float(os.getenv("ENFORCE_LITTER_CLIP_BEFORE_SEC", "2.0"))
        ),
        litter_clip_after_sec=max(
            0.0, float(os.getenv("ENFORCE_LITTER_CLIP_AFTER_SEC", "2.0"))
        ),
        require_plate_for_csv=os.getenv("ENFORCE_REQUIRE_PLATE", "0") == "1",
        video_exts=(".mp4", ".mov", ".avi", ".mkv", ".m4v"),
        watchdog_interval_sec=max(
            1.0, float(os.getenv("ENFORCE_WATCHDOG_INTERVAL_SEC", "60"))
        ),
        stale_timeout_sec=max(
            30.0, float(os.getenv("ENFORCE_STALE_TIMEOUT_SEC", "1200"))
        ),
        watchdog_log_path=Path(
            os.getenv("ENFORCE_WATCHDOG_LOG_PATH", "") or (output_root / "enforce_watchdog.log")
        ),
        # 2026-04-26 觀測能力：兩段式 timeout + GUI 即時欄位。
        # 預設值依 RTX 5090 校準（單張 ollama call 1-3s，60s 是 20× margin）；
        # 客戶 .env 已有 GLM_TIMEOUT_SEC=30 仍生效。
        glm_connect_timeout_sec=max(
            1.0, float(os.getenv("GLM_CONNECT_TIMEOUT_SEC", "10"))
        ),
        vlm_hard_timeout_sec=max(
            10.0, float(os.getenv("VLM_HARD_TIMEOUT_SEC", "90"))
        ),
        vlm_idle_alert_sec=max(
            30.0, float(os.getenv("ENFORCE_VLM_IDLE_ALERT_SEC", "120"))
        ),
        gui_tick_ms=max(
            100, int(os.getenv("ENFORCE_GUI_TICK_MS", "1000"))
        ),
        # 2026-04-26 ops 級別 log：frozen exe 客戶端看不到 stdout/stderr，
        # 所有 logger.info/.warning/.error 都會被 applog 串到檔案。
        log_level=_clean_env_str(os.getenv("ENFORCE_LOG_LEVEL", "INFO"), "INFO").upper(),
        log_path=Path(
            os.getenv("ENFORCE_LOG_PATH", "")
            or (output_root / "enforce_app.log")
        ),
        relabel_lit_to_urin=(
            os.getenv("ENFORCE_RELABEL_LIT_TO_URIN", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    )
