"""Pipeline stage code → 使用者可讀的中文名稱。

集中在此避免 GUI 直接把 'prefilter' / 'vlm' 等內部字串塞到畫面，
讓使用者誤以為「沒在跑」或看不懂狀態。也避免多處硬編碼 Chinese
字串導致翻譯不一致。

使用時機：
- `gui._update_progress` 把收到的 `stage` 翻成中文再寫入 Treeview。
- `gui._start_processing` 提交任務後顯示 'queued' → 「排隊中」。
- `pipeline_adapter.process_video` 在進入 pipeline 前回呼 'preparing'。
"""
from __future__ import annotations

STAGES: dict[str, str] = {
    # GUI / 排程階段
    "queued": "排隊中",
    "preparing": "準備中",
    # Pipeline 內部階段
    "prefilter": "YOLO 偵測中",
    "codetr": "Co-DETR 垃圾偵測中",  # 只有設了 ENFORCE_CODETR_URL 才會出現
    "vlm": "VLM 判定中",
    "merging": "彙整結果中",
    "inference": "推論中",  # pipeline_adapter 預設 fallback
}


def display_stage(stage: str) -> str:
    """把內部階段代號翻成中文；未知值原樣回傳（避免將 typo/未來新增階段藏起來）。"""
    return STAGES.get(stage, stage)
