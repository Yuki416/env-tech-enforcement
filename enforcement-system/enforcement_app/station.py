"""Utilities for inferring the monitoring station ID (CHx) from a video path.

使用者回報「跑完的資料夾除了標示時間點，也需標示該資料夾是站別幾」。
IPC 攝影機的檔名/資料夾多半帶有 'IPC{N}' (例如 `19_8_loc19_IPC3.mp4`)，
某些既有資料則直接用 `CH{N}`。本模組提供統一解析介面。
"""
from __future__ import annotations

import re
from pathlib import Path

# 獨立 token 的 IPCN / CHN （token 邊界：字串首尾或非字母數字字元）
_STATION_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:IPC|CH)0*(\d+)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)

UNKNOWN_STATION = "unknown"


def infer_station_id(video_path: Path) -> str:
    """
    從檔案路徑推斷站別編號 (e.g. 'CH3')。

    解析順序：檔名 → 父資料夾名。兩者都找不到則回傳 'unknown'。
    匹配樣式：`IPC{N}` 或 `CH{N}`，大小寫不敏感，支援前導零（IPC03 → CH3）。
    """
    for token in (video_path.name, video_path.parent.name):
        # re 需要 token 兩端有邊界，pad 首尾讓 pattern 在字串首尾也能比對
        match = _STATION_PATTERN.search(f"_{token}_")
        if match:
            return f"CH{int(match.group(1))}"
    return UNKNOWN_STATION


def sanitize_station_id(raw: str) -> str:
    """
    清除 station_id 中的路徑分隔字元與控制字元，避免寫入路徑時逃逸。

    僅保留字母、數字、底線、連字號；其餘全數移除。空結果回 'unknown'。
    """
    if not raw:
        return UNKNOWN_STATION
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", raw)
    return cleaned or UNKNOWN_STATION
