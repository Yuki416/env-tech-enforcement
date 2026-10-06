"""第 18 次訓練的 Co-DETR litter 來源 —— 接 `codetr_service` sidecar 的 HTTP client。

## 這個模組換掉的是什麼

`vlm_enforcement.PersonVehiclePrefilter` 的 YOLO 同時做兩件事：

1. **person/vehicle 候選** → 決定哪些幀要送 VLM、截圖怎麼框。**這一半不動。**
2. **litter/garbage 框** → 反幻覺 gate（`_analyze_candidates`：YOLO 當幀沒看到 litter，
   就強制 VLM 只能在「便溺/none」中選）。**這一半換成我們自己第 18 次訓練的 Co-DETR。**

只換第 2 個角色的理由：第 1 個角色實測沒問題（`vlm_decisions.csv` 裡 person/car/motorcycle
都抓得到），而且逐幀跑 Co-DETR 會慢一個量級；litter 才是我們模型贏最多的地方
（dynamic166 litter precision 0.840 vs 第 16 次 0.114、static594 FP 186→12）。

## 為什麼傳 frame index 而不是傳影像

sidecar 需要**循序**餵幀才建得出 KNN 背景模型（第 18 次的模型是 RGB+changemap 4 通道），
但這裡只拿得到每 stride 抽出來的稀疏幀。所以改由 sidecar 自己開同一支影片：
它先循序暖機，再依 index 取幀。兩個容器掛同一份影片路徑，座標一致。

## 失效時的行為

sidecar 掛掉、逾時、回錯 —— 一律回 `None`，呼叫端沿用 YOLO 原本的 litter 框。
**偵測器換新不該讓整條 pipeline 變得比較容易壞。**
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class CoDetrLitterSource:
    """向 sidecar 問「這些幀有沒有 litter/garbage」。執行緒安全由呼叫端的 lock 保證。"""

    def __init__(
        self,
        base_url: str,
        score_thr: float = 0.3,
        timeout_sec: float = 300.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._score_thr = float(score_thr)
        self._timeout = float(timeout_sec)
        self._video: str | None = None
        # 連續失敗就停用，避免每個批次都白等一次 timeout 拖垮整支影片。
        self._fail_count = 0
        self._disabled = False

    # ── 生命週期 ──────────────────────────────────────────────────
    def set_video(self, video_path: str) -> None:
        """換影片。sidecar 會在第一次 detect 時針對這支影片暖機建背景。"""
        if self._video and self._video != video_path:
            self._release(self._video)
        self._video = video_path
        self._fail_count = 0
        self._disabled = False

    def release(self) -> None:
        if self._video:
            self._release(self._video)
            self._video = None

    def _release(self, video_path: str) -> None:
        try:
            self._post("/release", {"video": video_path}, timeout=10.0)
        except Exception as exc:  # 釋放失敗不影響正確性，sidecar 有 LRU 上限
            logger.debug("codetr release failed: %s", exc)

    # ── 主要介面 ──────────────────────────────────────────────────
    def litter_boxes(self, frame_indices: list[int]) -> list[list[dict[str, Any]]] | None:
        """回傳每幀的 litter/garbage 框；任何失敗都回 None 讓呼叫端退回 YOLO。"""
        if self._disabled or not self._video or not frame_indices:
            return None
        try:
            resp = self._post(
                "/detect",
                {
                    "video": self._video,
                    "frames": [int(i) for i in frame_indices],
                    "score_thr": self._score_thr,
                },
                timeout=self._timeout,
            )
        except Exception as exc:
            self._fail_count += 1
            logger.warning(
                "Co-DETR litter 來源失敗（第 %d 次）：%s", self._fail_count, exc
            )
            if self._fail_count >= 3:
                self._disabled = True
                logger.error(
                    "Co-DETR 連續失敗 3 次，本支影片改用 YOLO 的 litter 框。"
                )
            return None

        boxes = resp.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != len(frame_indices):
            logger.warning(
                "Co-DETR 回傳長度不符（要 %d 幀、拿到 %s），退回 YOLO",
                len(frame_indices),
                type(boxes).__name__ if not isinstance(boxes, list) else len(boxes),
            )
            return None
        self._fail_count = 0
        return [[b for b in per_frame if isinstance(b, dict)] for per_frame in boxes]

    def health(self) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(f"{self._base}/health", timeout=10.0) as fh:
                return json.loads(fh.read())
        except Exception as exc:
            logger.warning("Co-DETR health check 失敗：%s", exc)
            return None

    # ── HTTP ────────────────────────────────────────────────────
    def _post(self, path: str, payload: dict, timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.loads(fh.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
