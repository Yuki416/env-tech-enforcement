from __future__ import annotations

import base64
import csv
import datetime
import importlib
import json
import math
import os
import re
import time
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
import logging

from .audit_log import AuditLogger

logger = logging.getLogger(__name__)


def _to_hms(seconds: float) -> str:
    whole = max(0, int(seconds))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_video_start_time(
    video_path: str | Path,
    fps: float,
    total_frames: int,
) -> datetime.datetime | None:
    """Return the wall-clock start time of a CCTV video, or None.

    Strategy 1 — filename: scan the stem for a date-time string.
    Common CCTV patterns handled::

        20240115_143225          → 2024-01-15 14:32:25
        CH01_20240115143225_000  → 2024-01-15 14:32:25
        2024-01-15_14-32-25      → 2024-01-15 14:32:25
        2024-01-15T14:32:25      → 2024-01-15 14:32:25

    Strategy 2 — file mtime: ``mtime − duration``.  Works when the OS
    preserves the original file-creation time (common on FAT32/NTFS drives
    from DVRs after direct copy).
    """
    stem = Path(video_path).stem
    # Single regex covers the common digit-separator variants
    _DT_PAT = re.compile(
        r"(\d{4})[^\d]?(\d{2})[^\d]?(\d{2})"   # date: YYYYMMDD with optional sep
        r"[^\d]?"                                  # optional separator
        r"(\d{2})[^\d]?(\d{2})[^\d]?(\d{2})"    # time: HHMMSS with optional sep
    )
    m = _DT_PAT.search(stem)
    if m:
        try:
            yr, mo, dy, hh, mm, ss = (int(g) for g in m.groups())
            if 2000 <= yr <= 2100 and 1 <= mo <= 12 and 1 <= dy <= 31:
                return datetime.datetime(yr, mo, dy, hh, mm, ss)
        except (ValueError, OverflowError):
            pass

    # Fallback: file mtime − video duration
    try:
        mtime = os.path.getmtime(str(video_path))
        duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
        return datetime.datetime.fromtimestamp(mtime - duration)
    except Exception:
        pass

    return None


_RE_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# GLM-4.5V/4.6 wraps its "final answer" in these sentinel tokens when the
# prompt asks for a plain-text reply.  They are not part of the real answer.
_RE_BOX_OPEN = re.compile(r"<\|begin_of_box\|>")
_RE_BOX_CLOSE = re.compile(r"<\|end_of_box\|>")


def _strip_think_tags(text: str) -> str:
    """Remove reasoning/answer sentinel tokens emitted by VLMs.

    Handles both ``<think>...</think>`` (reasoning-mode) and the GLM
    ``<|begin_of_box|>...<|end_of_box|>`` answer wrappers.  The wrappers are
    stripped but their inner content is kept.
    """
    text = _RE_THINK_BLOCK.sub("", text)
    text = _RE_BOX_OPEN.sub("", text)
    text = _RE_BOX_CLOSE.sub("", text)
    return text.strip()


def _extract_json_text(raw: str) -> str:
    text = _strip_think_tags(raw)
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def _clean_text(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


# Map of simplified-Chinese characters that GLM-4.6V Flash regularly emits
# despite the prompt explicitly requiring traditional Chinese. case12
# reproduce 2026-04-30: 105/319 candidates were rejected solely because the
# VLM returned "乱丢垃圾" instead of "亂丟垃圾". Other glyphs (垃 / 圾 / 便 /
# 溺 / 菸 / 蒂) already have identical trad/simp forms.
_SIMP_TO_TRAD: dict[str, str] = {
    "乱": "亂",
    "丢": "丟",
}

# Tokens VLM returned that mean "no violation" but in non-canonical forms.
_NO_VIOLATION_ALIASES: set[str] = {"none", "no", "n/a", "", "正常"}


def normalize_behavior(raw: str) -> str:
    """Map a VLM behavior string to one of {"亂丟垃圾", "便溺", "none"}.

    Handles three real failure modes seen in vendor reproduce on 2026-04-30:
      1. Simplified Chinese: "乱丢垃圾" → "亂丟垃圾"
      2. Template echo: "亂丟垃圾|便溺|none" or "乱丟垃圾/便溺" — VLM copied
         the schema choice line verbatim. We pick the first token that
         normalizes to a real action; if none, return "none".
      3. English / unrecognized: "litter", "urination", garbage strings
         → all collapse to "none".

    Returns "none" rather than raising for unknown input so the caller can
    treat "no usable label" the same as "VLM said no violation".
    """
    text = (raw or "").strip()
    # Strip enclosing quotes the VLM occasionally adds (incl. CJK paired
    # forms where opening and closing chars differ).
    _PAIRS = (("「", "」"), ("『", "』"), ('"', '"'), ("'", "'"))
    if text:
        for opener, closer in _PAIRS:
            if text[0] == opener and text[-1] == closer and len(text) >= 2:
                text = text[1:-1].strip()
                break
    # Translate any simplified-only chars to traditional.
    for simp, trad in _SIMP_TO_TRAD.items():
        text = text.replace(simp, trad)
    if not text:
        return "none"
    if text in {"亂丟垃圾", "便溺"}:
        return text
    # Template echoes use either "|" or "/" as separators.
    if "|" in text or "/" in text:
        for token in text.replace("/", "|").split("|"):
            t = token.strip()
            for simp, trad in _SIMP_TO_TRAD.items():
                t = t.replace(simp, trad)
            if t in {"亂丟垃圾", "便溺"}:
                return t
        return "none"
    if text.lower() in _NO_VIOLATION_ALIASES:
        return "none"
    return "none"


def compose_traditional_reason(
    behavior: str,
    confidence: float,
    has_violation: bool,
) -> str:
    conf = max(0.0, min(1.0, float(confidence)))
    if not has_violation:
        return "未觀察到明確違規證據。"

    if behavior == "亂丟垃圾":
        if conf >= 0.9:
            return "畫面清楚顯示疑似拋棄垃圾行為。"
        if conf >= 0.7:
            return "畫面顯示疑似拋棄垃圾行為，證據大致清楚。"
        return "畫面出現疑似拋棄垃圾行為，但證據有限。"

    if behavior == "便溺":
        if conf >= 0.9:
            return "畫面清楚顯示疑似便溺行為。"
        if conf >= 0.7:
            return "畫面顯示疑似便溺行為，證據大致清楚。"
        return "畫面出現疑似便溺行為，但證據有限。"

    return "畫面顯示疑似違規行為。"


@dataclass(frozen=True)
class ViolationEvent:
    frame_index: int
    frame_time: str
    overlay_time: str
    plate: str
    behavior: str
    confidence: float
    reason: str
    snapshot: str
    clip: str = ""
    candidate_box: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_time": self.frame_time,
            "overlay_time": self.overlay_time,
            "plate": self.plate,
            "behavior": self.behavior,
            "confidence": self.confidence,
            "reason": self.reason,
            "snapshot": self.snapshot,
            "clip": self.clip,
            "candidate_box": list(self.candidate_box)
            if self.candidate_box is not None
            else None,
        }


def _normalize_plate_for_merge(plate: str) -> str:
    token = (plate or "").strip().upper()
    if not token:
        return ""
    compact = "".join(ch for ch in token if ch.isalnum())
    if compact in {"", "UNKNOWN", "NONE", "NA", "N/A", "未辨識", "未知", "-"}:
        return ""
    return compact


def _box_iou_xyxy(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return (inter / denom) if denom > 0 else 0.0


_CJK_FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
]


def _put_chinese_text(
    img: Any,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> Any:
    """Draw text with CJK support via PIL; falls back to cv2.putText."""
    try:
        Image = importlib.import_module("PIL.Image")
        ImageDraw = importlib.import_module("PIL.ImageDraw")
        ImageFont = importlib.import_module("PIL.ImageFont")
        np = importlib.import_module("numpy")
        font_size = max(12, int(font_scale * 32))
        font = None
        for fp in _CJK_FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        img_rgb = img[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        r, g, b = int(color[2]), int(color[1]), int(color[0])
        stroke = max(0, thickness - 1)
        try:
            draw.text(
                pos,
                text,
                font=font,
                fill=(r, g, b),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0),
            )
        except TypeError:
            draw.text(pos, text, font=font, fill=(r, g, b))
        return np.array(pil_img)[:, :, ::-1].copy()
    except Exception:
        cv2_mod = importlib.import_module("cv2")
        cv2_mod.putText(
            img,
            text,
            pos,
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2_mod.LINE_AA,
        )
        return img


class PersonVehiclePrefilter:
    _NAME_KEYWORDS: set[str] = {
        "person",
        "people",
        "pedestrian",
        "car",
        "vehicle",
        "motorcycle",
        "motorbike",
        "bike",
        "bicycle",
        "bus",
        "truck",
        "van",
    }
    # Litter-class keywords for VLM hallucination check：當 YOLO 在當前幀沒看到
    # 任何 litter class，就算 VLM 自信地說「亂丟垃圾」也視為幻覺，把
    # detect_litter=False 強制讓 VLM 只能在「便溺/none」中選。
    # 對應 vendor 4/15 第3站 5 件全判亂丟（實際全是便溺）的情境。
    _LITTER_KEYWORDS: set[str] = {
        "litter",
        "trash",
        "rubbish",
        "garbage",
        "litter_bag",
        "trash_bag",
    }
    # Only used as a last-resort fallback when the loaded model does not expose
    # a class-name map (generic COCO yolov8n.pt style).  NEVER combine with
    # name-based filtering: custom models can reuse these IDs for unrelated
    # classes (e.g. our own model has id=2 → "litter", which collides with COCO
    # id=2 → "car" and leaked litter boxes into the "person/vehicle candidate"
    # list, which then caused VLM hallucinations and wrong snapshot framing).
    _COCO_IDS_FALLBACK: set[int] = {0, 1, 2, 3, 5, 7}

    def __init__(self, model_path: str, conf: float) -> None:
        ultralytics_mod = importlib.import_module("ultralytics")
        yolo_cls = getattr(ultralytics_mod, "YOLO")
        self._model: Any = yolo_cls(model_path)
        self._conf: float = conf
        # Pre-compute the allowed class IDs from the model's own name map.
        # Falls back to the COCO id set only when names are unavailable.
        names_obj = getattr(self._model, "names", None)
        allowed: set[int] = set()
        if isinstance(names_obj, dict) and names_obj:
            for cid, cname in names_obj.items():
                try:
                    if str(cname).strip().lower() in self._NAME_KEYWORDS:
                        allowed.add(int(cid))
                except (TypeError, ValueError):
                    continue
        if not allowed:
            allowed = set(self._COCO_IDS_FALLBACK)
        self._allowed_ids: set[int] = allowed
        logger.info(
            "PersonVehiclePrefilter: model classes=%s, allowed_ids=%s",
            names_obj,
            sorted(self._allowed_ids),
        )

    def warmup(self) -> None:
        return

    def has_person_or_vehicle(self, frame_bgr: Any) -> bool:
        return bool(self.detect_candidates(frame_bgr))

    def detect_candidates(self, frame_bgr: Any) -> list[dict[str, Any]]:
        results = self._model.predict(frame_bgr, conf=self._conf, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        cls_list = getattr(boxes, "cls", None)
        xyxy_list = getattr(boxes, "xyxy", None)
        conf_list = getattr(boxes, "conf", None)
        if cls_list is None or xyxy_list is None:
            return []

        names_obj = getattr(result, "names", {})
        names = names_obj if isinstance(names_obj, dict) else {}
        cls_vals = cls_list.tolist()
        xyxy_vals = xyxy_list.tolist()
        conf_vals = conf_list.tolist() if conf_list is not None else []

        out: list[dict[str, Any]] = []
        for idx, cls_tensor in enumerate(cls_vals):
            cls_id = int(cls_tensor)
            name = str(names.get(cls_id, "")).strip().lower()
            # Prefer name-based filtering (robust across custom models).
            # Only fall back to _allowed_ids (COCO fallback) if name map empty.
            if names:
                if name not in self._NAME_KEYWORDS:
                    continue
            else:
                if cls_id not in self._allowed_ids:
                    continue
            if idx >= len(xyxy_vals):
                continue
            box = xyxy_vals[idx]
            if not isinstance(box, list) or len(box) < 4:
                continue
            conf = 0.0
            if idx < len(conf_vals) and isinstance(conf_vals[idx], (int, float)):
                conf = float(conf_vals[idx])
            out.append(
                {
                    "x1": int(max(0, box[0])),
                    "y1": int(max(0, box[1])),
                    "x2": int(max(0, box[2])),
                    "y2": int(max(0, box[3])),
                    "label": name if name else f"cls_{cls_id}",
                    "score": conf,
                }
            )
        return out

    def detect_candidates_with_litter_batch(
        self, frames_bgr: list[Any]
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        """Batch YOLO inference returning (person/vehicle candidates, litter boxes) per frame.

        單次 YOLO 推論抽兩個 view：person/vehicle 給 VLM 當候選；litter 給後續
        consistency check（YOLO 沒看到 litter 就強制 VLM detect_litter=False）。
        """
        if not frames_bgr:
            return [], []
        results = self._model.predict(frames_bgr, conf=self._conf, verbose=False)
        if not results:
            return [[] for _ in frames_bgr], [[] for _ in frames_bgr]

        all_candidates: list[list[dict[str, Any]]] = []
        all_litter: list[list[dict[str, Any]]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                all_candidates.append([])
                all_litter.append([])
                continue

            cls_list = getattr(boxes, "cls", None)
            xyxy_list = getattr(boxes, "xyxy", None)
            conf_list = getattr(boxes, "conf", None)
            if cls_list is None or xyxy_list is None:
                all_candidates.append([])
                all_litter.append([])
                continue

            names_obj = getattr(result, "names", {})
            names = names_obj if isinstance(names_obj, dict) else {}
            cls_vals = cls_list.tolist()
            xyxy_vals = xyxy_list.tolist()
            conf_vals = conf_list.tolist() if conf_list is not None else []

            out: list[dict[str, Any]] = []
            litter_out: list[dict[str, Any]] = []
            for idx, cls_tensor in enumerate(cls_vals):
                cls_id = int(cls_tensor)
                name = str(names.get(cls_id, "")).strip().lower()
                if idx >= len(xyxy_vals):
                    continue
                box = xyxy_vals[idx]
                if not isinstance(box, list) or len(box) < 4:
                    continue
                conf = 0.0
                if idx < len(conf_vals) and isinstance(conf_vals[idx], (int, float)):
                    conf = float(conf_vals[idx])
                box_dict = {
                    "x1": int(max(0, box[0])),
                    "y1": int(max(0, box[1])),
                    "x2": int(max(0, box[2])),
                    "y2": int(max(0, box[3])),
                    "label": name if name else f"cls_{cls_id}",
                    "score": conf,
                }
                # 名稱優先（custom model robust）；無名稱才退 fallback id set。
                if names:
                    if name in self._NAME_KEYWORDS:
                        out.append(box_dict)
                    elif name in self._LITTER_KEYWORDS:
                        litter_out.append(box_dict)
                else:
                    if cls_id in self._allowed_ids:
                        out.append(box_dict)
            all_candidates.append(out)
            all_litter.append(litter_out)
        return all_candidates, all_litter

    def detect_candidates_batch(
        self, frames_bgr: list[Any]
    ) -> list[list[dict[str, Any]]]:
        """Batch YOLO inference: send multiple frames to GPU at once.

        Backward-compatible wrapper around `detect_candidates_with_litter_batch`.
        New callers should prefer the with_litter variant for hallucination check.
        """
        if not frames_bgr:
            return []
        results = self._model.predict(frames_bgr, conf=self._conf, verbose=False)
        if not results:
            return [[] for _ in frames_bgr]

        all_candidates: list[list[dict[str, Any]]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                all_candidates.append([])
                continue

            cls_list = getattr(boxes, "cls", None)
            xyxy_list = getattr(boxes, "xyxy", None)
            conf_list = getattr(boxes, "conf", None)
            if cls_list is None or xyxy_list is None:
                all_candidates.append([])
                continue

            names_obj = getattr(result, "names", {})
            names = names_obj if isinstance(names_obj, dict) else {}
            cls_vals = cls_list.tolist()
            xyxy_vals = xyxy_list.tolist()
            conf_vals = conf_list.tolist() if conf_list is not None else []

            out: list[dict[str, Any]] = []
            for idx, cls_tensor in enumerate(cls_vals):
                cls_id = int(cls_tensor)
                name = str(names.get(cls_id, "")).strip().lower()
                # Same policy as detect_candidates: name-based when available.
                if names:
                    if name not in self._NAME_KEYWORDS:
                        continue
                else:
                    if cls_id not in self._allowed_ids:
                        continue
                if idx >= len(xyxy_vals):
                    continue
                box = xyxy_vals[idx]
                if not isinstance(box, list) or len(box) < 4:
                    continue
                conf = 0.0
                if idx < len(conf_vals) and isinstance(conf_vals[idx], (int, float)):
                    conf = float(conf_vals[idx])
                out.append(
                    {
                        "x1": int(max(0, box[0])),
                        "y1": int(max(0, box[1])),
                        "x2": int(max(0, box[2])),
                        "y2": int(max(0, box[3])),
                        "label": name if name else f"cls_{cls_id}",
                        "score": conf,
                    }
                )
            all_candidates.append(out)
        return all_candidates


class VLMHardTimeout(RuntimeError):
    """Raised when a VLM call exceeds the configured hard wall-clock budget.

    The pipeline catches this in _analyze_candidates, logs a `rejected_hard_timeout`
    audit row, and continues with the next candidate so a single hung call can't
    stall the whole video.
    """


class HeartbeatEmitter:
    """Stateless wrapper around progress_callback that adds VLM telemetry.

    Why: _analyze_candidates can issue up to ~16 sequential VLM calls per frame
    (8 candidates × strict_confirm second pass). Without intermediate callbacks
    the GUI/watchdog see no progress for many minutes while the worker is fine.
    This emitter lets each VLM call broadcast counter + latency so the UI can
    display 'still working, just slow' rather than appearing frozen.
    """

    def __init__(
        self,
        callback: Callable[[dict[str, object]], None] | None,
        frame_index_ref: list[int],
        total_frames: int,
    ) -> None:
        self._cb = callback
        self._frame_idx = frame_index_ref  # mutable [int] from outer scope
        self._total = max(0, int(total_frames))
        self.vlm_calls_done = 0
        self.vlm_timeouts = 0

    def emit_vlm_call(self, latency_ms: int, *, timed_out: bool = False) -> None:
        self.vlm_calls_done += 1
        if timed_out:
            self.vlm_timeouts += 1
        if self._cb is None:
            return
        fi = self._frame_idx[0] if self._frame_idx else 0
        if self._total <= 0:
            progress = 0.0
        else:
            progress = min(1.0, max(0.0, fi / float(self._total)))
        self._cb({
            "progress": progress,
            "stage": "vlm",
            "frame": int(fi),
            "total_frames": int(self._total),
            "vlm_calls_done": int(self.vlm_calls_done),
            "last_call_latency_ms": int(latency_ms),
            "vlm_timeouts": int(self.vlm_timeouts),
        })


class GLMFlashClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        timeout_sec: float,
        provider: str,
        retry_count: int,
        retry_backoff_sec: float,
        connect_timeout_sec: float = 10.0,
        hard_timeout_sec: float = 90.0,
    ) -> None:
        self._provider: str = _clean_text(provider).lower()
        self._api_key: str = _clean_text(api_key)
        self._model: str = _clean_text(model)
        base = _clean_text(api_base).rstrip("/")
        if self._provider == "ollama":
            self._url = base + "/api/chat"
        else:
            self._url = base + "/chat/completions"
        # Two-stage timeout: connect=fast (localhost should be ~1s), read=full
        # ollama wall-clock allowance. requests.post(timeout=(connect, read))
        # gives finer control than a single value.
        self._connect_timeout: float = max(1.0, float(connect_timeout_sec))
        self._read_timeout: float = max(5.0, float(timeout_sec))
        # Hard wall-clock cap covers retry+backoff worst case. Floor is 0.1s
        # (just to reject zero/negative); production callers go through
        # AppConfig.vlm_hard_timeout_sec which has its own 10s floor.
        self._hard_timeout: float = max(0.1, float(hard_timeout_sec))
        # Kept for backward compatibility / debugging output:
        self._timeout_sec: float = self._read_timeout
        self._retry_count: int = max(0, retry_count)
        self._retry_backoff_sec: float = max(0.0, retry_backoff_sec)

    def _encode_jpeg_data_url(self, frame_bgr: Any) -> str:
        cv2_mod = importlib.import_module("cv2")
        ok, encoded = cv2_mod.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _encode_jpeg_base64(self, frame_bgr: Any) -> str:
        cv2_mod = importlib.import_module("cv2")
        ok, encoded = cv2_mod.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def analyze(
        self,
        frame_bgr: Any,
        detect_litter: bool,
        detect_urination: bool,
        target_hint: str = "",
        strict_confirm: bool = False,
        expected_behavior: str = "",
        is_mosaic: bool = False,
        context_frame_bgr: Any = None,
        hint_timestamp: str = "",
        hint_plate: str = "",
    ) -> dict[str, object]:
        if self._provider != "ollama" and not self._api_key:
            raise RuntimeError("Missing GLM_API_KEY")

        behavior_hint: list[str] = []
        if detect_litter:
            behavior_hint.append("亂丟垃圾")
        if detect_urination:
            behavior_hint.append("便溺")
        allowed = "/".join(behavior_hint) if behavior_hint else "none"

        verify_hint = ""
        if strict_confirm and expected_behavior in {"亂丟垃圾", "便溺"}:
            verify_hint = (
                f"【一致性驗證】請重新獨立分析此畫面，不要參考任何先前判斷。\n"
                f"若你看到「{expected_behavior}」的明確視覺證據，請維持判定。\n"
                "請具體描述你看到的：\n"
                "  1. 物體的顏色、大小、在畫面中的具體位置\n"
                "  2. 行為人的動作與姿態\n"
                "  3. 物體與行為人之間的空間關係變化\n"
                "若你無法從畫面中指出上述具體視覺細節，請回傳 has_violation=false、behavior=none。\n"
                "請如實給出你的 confidence，不必刻意提高或壓低。"
            )

        frame_desc = (
            "你收到的是【時序連續畫面】，由左至右代表時間推進（最右為當前幀）。"
            "請綜合所有畫格觀察人物動作軌跡，判斷是否有違規發生。"
        ) if is_mosaic else (
            "你收到的是單一監視器截圖（候選主體已用青色外框標示）。"
        )
        litter_criteria = (
            "【亂丟垃圾判斷準則—必須同時滿足，且 reason 必須逐項舉證】\n"
            "  ① 可見人物手臂有向外拋出、甩出、投放或棄置動作\n"
            "  ② 可見垃圾物體（袋子/瓶罐/紙張/食品等）脫離身體或落地\n"
            "  ③ 時序畫面中同一物體位置明顯變化（能指出「某一格」物體仍在手上，與「另一格」物體已離手/落地）\n"
            "  ※ 僅看到垃圾躺在地面、主體走路、車輛停放、或只看到「vehicle 在路上」皆不構成證據\n"
            "【嚴禁幻覺】若你無法在畫面中明確指出「垃圾離手前」與「垃圾離手後」兩個不同時刻的具體視覺差異，"
            "必須回傳 has_violation=false、behavior=none。\n"
            "禁止用通用敘述（例如「符合時序」「位置明顯變化」「在手→空中→落地」）代替具體觀察；"
            "若 reason 中出現這類通用句而無具體位置/顏色/動作描述，視為無效證據。\n"
        ) if detect_litter else ""
        urine_criteria = (
            "【便溺判斷準則—必須同時滿足】\n"
            "  ① 人物面向牆面/角落/路邊草叢且姿勢固定（非行走）\n"
            "  ② 可見液體排出、褲子部分脫下、或明顯便溺蹲姿\n"
            "  ③ 僅站立/停車/整理衣物/背對牆壁不構成便溺\n"
        ) if detect_urination else ""
        # Timestamp is now read from video metadata — VLM only needs to find plates.
        if context_frame_bgr is not None:
            plate_time_hint = (
                "你收到兩張圖：第一張為候選主體分析圖，第二張為原始監視器完整畫面。"
                "請從【第二張圖】中找尋候選主體附近的車輛車牌號碼（若清晰可見）。\n"
            )
        else:
            plate_time_hint = (
                "圖片為監視器完整場景（含頂端放大區），青色框標示候選違規主體。"
                "請辨識青色框附近車輛的車牌號碼（若清晰可見，否則填空字串）。\n"
            )

        # Plate hint if OCR pre-read succeeded
        pre_read_hint = ""
        if hint_plate:
            pre_read_hint = (
                f"【系統已辨識車牌】：「{hint_plate}」"
                "——請直接填入 plate（除非你在圖中看到明顯不同的車牌）。\n"
            )

        prompt = (
            "你是環保科技執法影像判讀助手，專責辨識亂丟垃圾與便溺的確切視覺證據。\n"
            "【輸出規定】僅回傳 JSON 物件，禁止輸出任何額外文字、markdown 或說明。\n"
            f"【畫面說明】{frame_desc}\n"
            f"【本次允許偵測的違規】{allowed}\n"
            "【禁止偵測以下行為】抽菸、違停、逆向、噪音等其他違規→has_violation=false，behavior=none\n"
            f"{litter_criteria}"
            f"{urine_criteria}"
            "【信心分數校準】\n"
            "  0.95以上：動作、物體、行為人三者均清晰可見，毫無疑問\n"
            "  0.85-0.94：主要動作清楚，局部遮擋但仍可確認\n"
            "  0.70-0.84：動作大致可辨，有遮擋或模糊但整體可信\n"
            "  0.50-0.69：可疑但證據不足→必須輸出 has_violation=false\n"
            "  0.50以下：不確定→has_violation=false\n"
            "【reason 欄位規定】必須繁體中文，條列 2-3 個**你實際從畫面中看到的**具體視覺觀察。\n"
            "每一條觀察必須包含：主體位置（畫面左/中/右、前景/背景）、物體顏色與大小、以及該觀察對應的畫面區域。\n"
            "禁止照抄本 prompt 中出現的描述句（例如「垃圾袋離手落地」「在手→空中→落地」「符合判斷準則」）。\n"
            "若你只能寫出通用句而無法指出具體位置與外觀，代表證據不足，請回傳 has_violation=false。\n"
            "禁止使用簡體中文或英文。\n"
            f"【車牌與時間】{plate_time_hint}"
            f"{pre_read_hint}"
            f"候選主體類型：{target_hint or 'unknown'}\n"
            f"{verify_hint}\n"
            "【JSON Schema】：\n"
            "{"
            '"has_violation":bool,'
            '"behavior":"亂丟垃圾|便溺|none",'
            '"plate":"車牌號碼，若無法辨識或看不清車牌請填入空字串",'
            '"confidence":0.0到1.0的浮點數,'
            '"reason":"條列具體視覺觀察依據（繁體中文）",'
            '"litter_box":[x1,y1,x2,y2]'
            "}\n"
            "亂丟垃圾：盡量提供 litter_box（垃圾物體在畫面中的相對座標 0~1）。"
            "若不確定請回傳空陣列[]。"
        )

        if self._provider == "ollama":
            images = [self._encode_jpeg_base64(frame_bgr)]
            if context_frame_bgr is not None:
                images.append(self._encode_jpeg_base64(context_frame_bgr))
            payload = {
                "model": self._model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images,
                    }
                ],
            }
        else:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_jpeg_data_url(frame_bgr)},
                },
            ]
            if context_frame_bgr is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_jpeg_data_url(context_frame_bgr)
                        },
                    }
                )
            payload = {
                "model": self._model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": content}],
            }

        requests_mod = importlib.import_module("requests")
        headers = {"Content-Type": "application/json"}
        if self._provider != "ollama":
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = None
        last_error: Exception | None = None
        total_attempts = self._retry_count + 1
        deadline = time.monotonic() + self._hard_timeout
        for attempt in range(total_attempts):  # noqa: SIM113
            if time.monotonic() >= deadline:
                raise VLMHardTimeout(
                    f"VLM hard timeout {self._hard_timeout}s exceeded before attempt {attempt}"
                ) from last_error
            try:
                response = requests_mod.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
                response.raise_for_status()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    break
                should_retry = False
                if isinstance(
                    exc,
                    (
                        requests_mod.exceptions.Timeout,
                        requests_mod.exceptions.ConnectionError,
                    ),
                ):
                    should_retry = True
                elif isinstance(exc, requests_mod.exceptions.HTTPError):
                    resp = getattr(exc, "response", None)
                    status = getattr(resp, "status_code", None)
                    should_retry = isinstance(status, int) and status >= 500
                if not should_retry:
                    break
                sleep_sec = self._retry_backoff_sec * (attempt + 1)
                if (time.monotonic() + sleep_sec) >= deadline:
                    raise VLMHardTimeout(
                        f"VLM hard timeout {self._hard_timeout}s exceeded during retry"
                    ) from exc
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        if response is None:
            if last_error is None:
                raise RuntimeError("VLM request failed without response")
            raise RuntimeError(f"VLM request failed: {last_error}") from last_error

        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("GLM response body is not object")

        if self._provider == "ollama":
            message = body.get("message")
            if not isinstance(message, dict):
                raise RuntimeError("Ollama response missing message")
            raw = message.get("content")
        else:
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("GLM response missing choices")
            choice0 = choices[0]
            if not isinstance(choice0, dict):
                raise RuntimeError("GLM response first choice invalid")
            message = choice0.get("message")
            if not isinstance(message, dict):
                raise RuntimeError("GLM response missing message")
            raw = message.get("content")
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

        if not raw_text.strip():
            raise RuntimeError("VLM returned empty content")

        parsed_obj = json.loads(_extract_json_text(raw_text))
        if not isinstance(parsed_obj, dict):
            raise RuntimeError("GLM response is not JSON object")
        return dict(parsed_obj)

    def read_text_from_image(self, frame_bgr: Any, prompt: str) -> str:
        """Simple plain-text VLM call — no JSON schema, no retry.

        Used for focused OCR tasks (timestamp / plate) where we only need the
        raw text answer rather than a structured JSON object.  Returns an empty
        string on any failure so callers never need to handle exceptions.
        """
        try:
            b64 = self._encode_jpeg_base64(frame_bgr)
            if self._provider == "ollama":
                payload: dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [b64],
                        }
                    ],
                    "stream": False,
                }
            else:
                payload = {
                    "model": self._model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                }
            requests_mod = importlib.import_module("requests")
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._provider != "ollama":
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = requests_mod.post(
                self._url,
                headers=headers,
                json=payload,
                timeout=(self._connect_timeout, self._read_timeout),
            )
            response.raise_for_status()
            body = response.json()
            if self._provider == "ollama":
                msg = body.get("message", {})
                return _strip_think_tags(str(msg.get("content", "")))
            choices = body.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return _strip_think_tags(str(msg.get("content", "")))
        except Exception:
            pass
        return ""


class VLMEnforcementPipeline:
    def __init__(
        self,
        model_path: str,
        prefilter_conf: float,
        glm_api_key: str,
        glm_model: str,
        glm_api_base: str,
        glm_timeout_sec: float,
        vlm_provider: str,
        vlm_retry_count: int,
        vlm_retry_backoff_sec: float,
        vlm_cooldown_sec: float,
        event_merge_sec: float,
        min_violation_confidence: float,
        confirm_violation_confidence: float,
        max_candidates_per_frame: int,
        roi_expand: float,
        max_vlm_calls_per_video: int,
        max_vlm_inflight: int,
        litter_clip_before_sec: float,
        litter_clip_after_sec: float,
        max_vlm_calls_per_minute: int = 0,
        glm_connect_timeout_sec: float = 10.0,
        vlm_hard_timeout_sec: float = 90.0,
        relabel_lit_to_urin: bool = False,
    ) -> None:
        self._prefilter: PersonVehiclePrefilter = PersonVehiclePrefilter(
            model_path=model_path,
            conf=prefilter_conf,
        )
        # 選用：把反幻覺 gate 用的 litter/garbage 框改由我們自己第 18 次訓練的 Co-DETR 提供。
        # 沒設 ENFORCE_CODETR_URL 就完全走原本的 YOLO，行為不變。
        # person/vehicle 候選一律仍由 YOLO 決定 —— 詳見 codetr_litter.py 開頭。
        self._codetr_litter: Any = None
        _codetr_url = os.getenv("ENFORCE_CODETR_URL", "").strip()
        if _codetr_url:
            from .codetr_litter import CoDetrLitterSource

            self._codetr_litter = CoDetrLitterSource(
                base_url=_codetr_url,
                score_thr=float(os.getenv("ENFORCE_CODETR_SCORE", "0.30")),
            )
            health = self._codetr_litter.health()
            if health is None:
                logger.error(
                    "ENFORCE_CODETR_URL=%s 連不上，litter 框退回 YOLO", _codetr_url
                )
                self._codetr_litter = None
            else:
                logger.info(
                    "Co-DETR litter 來源已啟用：%s (ckpt=%s)",
                    _codetr_url,
                    health.get("checkpoint"),
                )
        self._vlm: GLMFlashClient = GLMFlashClient(
            api_key=glm_api_key,
            model=glm_model,
            api_base=glm_api_base,
            timeout_sec=glm_timeout_sec,
            provider=vlm_provider,
            retry_count=vlm_retry_count,
            retry_backoff_sec=vlm_retry_backoff_sec,
            connect_timeout_sec=glm_connect_timeout_sec,
            hard_timeout_sec=vlm_hard_timeout_sec,
        )
        self._cooldown_sec: float = max(0.0, vlm_cooldown_sec)
        self._event_merge_sec: float = max(0.0, event_merge_sec)
        self._min_violation_confidence: float = max(
            0.0, min(1.0, min_violation_confidence)
        )
        self._confirm_violation_confidence: float = max(
            0.0, min(1.0, confirm_violation_confidence)
        )
        self._roi_expand: float = max(0.0, roi_expand)
        self._max_candidates_per_frame: int = max(1, int(max_candidates_per_frame))
        self._max_vlm_calls_per_video: int = max(0, int(max_vlm_calls_per_video))
        # 新預算機制：每分鐘 N 次 VLM 呼叫。與 per_video 共存時取 min
        # （per_video 作為「超長影片」的絕對硬上限）。0 = 該機制停用。
        self._max_vlm_calls_per_minute: int = max(0, int(max_vlm_calls_per_minute))
        self._max_vlm_inflight: int = max(1, int(max_vlm_inflight))
        self._vlm_slots: threading.BoundedSemaphore = threading.BoundedSemaphore(
            self._max_vlm_inflight
        )
        # 2026-04-30: anti-hallucination guard now opt-in. case5 reproduce
        # showed real "亂丟垃圾" frames where YOLO misses the in-flight bag,
        # then this guard silently relabels the event to "便溺" → wrong type
        # written to CSV. Default False means we trust VLM's verdict (no
        # relabel, no drop). See _analyze_candidates for the gate.
        self._relabel_lit_to_urin: bool = bool(relabel_lit_to_urin)
        self._prefilter_lock = threading.Lock()
        self._litter_clip_before_sec: float = max(0.0, float(litter_clip_before_sec))
        self._litter_clip_after_sec: float = max(0.0, float(litter_clip_after_sec))
        # Ring buffer for temporal mosaic (up to 4 stride-sampled frames)
        self._frame_ring: deque[tuple[int, Any]] = deque(maxlen=4)
        # Per-call telemetry state set inside _run_vlm_analyze; consumers in
        # _analyze_candidates read them when emitting audit rows.
        self._last_vlm_latency_ms: int = 0
        self._last_vlm_timed_out: bool = False

    def warmup(self) -> None:
        self._prefilter.warmup()

    def set_max_vlm_inflight(self, n: int) -> None:
        """
        動態調整 VLM 並行槽位容量。

        GUI 的「平行影片數」變更時由 `PipelineEngine.set_max_vlm_inflight` 轉呼叫，
        讓 BoundedSemaphore 容量與線程池 worker 數一致，避免 semaphore 成為瓶頸
        （2026-04-23 批次「開 8 部只跑 2 部」的根因）。

        注意：應在「無進行中推論」的時機點呼叫（例如 `_start_processing` 之前）。
        仍持有舊 semaphore 的 thread 會正常釋放舊物件；新 acquire 會走新物件。
        """
        n = max(1, int(n))
        self._vlm_slots = threading.BoundedSemaphore(n)
        self._max_vlm_inflight = n

    def infer_single_video(
        self,
        video_path: Path,
        out_csv_path: Path,
        snap_dir: Path,
        detect_litter: bool,
        detect_urination: bool,
        frame_stride: int,
        max_vlm_calls_per_video: int | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        audit_log_path: Path | None = None,
    ) -> dict[str, object]:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Optional per-video audit log. Caller (PipelineEngine.process_video)
        # supplies a path; tests / direct callers can omit and skip the file.
        audit_log: AuditLogger | None = None
        if audit_log_path is not None:
            audit_log = AuditLogger(Path(audit_log_path))

        cv2_mod = importlib.import_module("cv2")
        cap = cv2_mod.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Co-DETR sidecar 自己開同一支影片建 KNN 背景模型（4ch 的第 4 通道要用），
        # 所以要先告訴它現在處理哪一支。兩邊的 frame index 同基準（0-based）。
        if self._codetr_litter is not None:
            self._codetr_litter.set_video(str(video_path))

        fps = cap.get(cv2_mod.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0
        total_frames = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT) or 0)
        video_start_dt = _parse_video_start_time(video_path, fps, total_frames)
        effective_stride = max(1, int(frame_stride))
        # 預算計算：同時考慮 per_video (絕對上限) 與 per_minute (每分鐘 N 次)。
        # 4/20-4/23 回報同長度影片執行時間差 18 分鐘 vs 22 小時，根因是
        # per_video=2400 固定預算，違規密集影片一路跑到耗盡。改為以影片
        # 時長線性計算，讓「執行時間」主要隨「影片時長」而非「違規密度」變化。
        per_video_cap = (
            self._max_vlm_calls_per_video
            if max_vlm_calls_per_video is None
            else max(0, int(max_vlm_calls_per_video))
        )
        per_minute_cap = 0
        if self._max_vlm_calls_per_minute > 0 and fps > 0 and total_frames > 0:
            duration_min = total_frames / fps / 60.0
            per_minute_cap = max(1, int(round(self._max_vlm_calls_per_minute * duration_min)))

        if per_video_cap > 0 and per_minute_cap > 0:
            max_calls = min(per_video_cap, per_minute_cap)
        elif per_video_cap > 0:
            max_calls = per_video_cap
        elif per_minute_cap > 0:
            max_calls = per_minute_cap
        else:
            max_calls = 0  # 無限制
        if total_frames > 0 and max_calls > 0:
            budget_stride = int(math.ceil(total_frames / float(max_calls)))
            effective_stride = max(effective_stride, max(1, budget_stride))
        cooldown_frames = max(1, int(self._cooldown_sec * fps))
        last_vlm_frame = -cooldown_frames
        vlm_decision_frames = 0

        frame_index = -1
        events: list[ViolationEvent] = []
        self._frame_ring.clear()

        # frame_index_ref is mutable list so HeartbeatEmitter can read the
        # current frame_index without us having to thread updates through.
        frame_index_ref: list[int] = [0]
        heartbeat = HeartbeatEmitter(
            callback=progress_callback,
            frame_index_ref=frame_index_ref,
            total_frames=total_frames,
        )

        # State for prefilter_empty_range merging (see audit_log.log_prefilter_empty_range).
        empty_range_start: int | None = None

        def _flush_empty_range(end_frame: int) -> None:
            nonlocal empty_range_start
            if audit_log is None or empty_range_start is None:
                empty_range_start = None
                return
            audit_log.log_prefilter_empty_range(
                start_frame=empty_range_start,
                end_frame=end_frame,
                fps=fps,
            )
            empty_range_start = None

        import queue as _vlm_q

        _VLM_PREFETCH_BATCH = 8
        _vlm_pq: _vlm_q.Queue = _vlm_q.Queue(maxsize=_VLM_PREFETCH_BATCH * 3)
        _vlm_prefetch_stop = threading.Event()

        def _vlm_prefetch_worker():
            while not _vlm_prefetch_stop.is_set():
                ok, frm = cap.read()
                try:
                    _vlm_pq.put((ok, frm), timeout=5.0)
                except _vlm_q.Full:
                    pass
                if not ok:
                    break

        _vlm_prefetch_thread = threading.Thread(
            target=_vlm_prefetch_worker, daemon=True, name="vlm-prefetch"
        )
        _vlm_prefetch_thread.start()

        try:
            _vlm_eof = False
            while not _vlm_eof:
                batch_raw: list[tuple[int, Any]] = []
                while len(batch_raw) < _VLM_PREFETCH_BATCH and not _vlm_eof:
                    try:
                        ok, frm = _vlm_pq.get(timeout=30.0)
                    except _vlm_q.Empty:
                        _vlm_eof = True
                        break
                    if not ok:
                        _vlm_eof = True
                        break
                    frame_index += 1
                    frame_index_ref[0] = frame_index

                    if progress_callback is not None:
                        progress = (
                            0.0
                            if total_frames <= 0
                            else min(1.0, max(0.0, frame_index / max(total_frames, 1)))
                        )
                        progress_callback(
                            {
                                "progress": progress,
                                "stage": "prefilter",
                                "frame": frame_index,
                                "total_frames": total_frames,
                            }
                        )

                    if frame_index % effective_stride != 0:
                        continue

                    self._frame_ring.append((frame_index, frm))
                    batch_raw.append((frame_index, frm))

                if not batch_raw:
                    break

                if max_calls > 0 and vlm_decision_frames >= max_calls:
                    break

                # Batch YOLO prefilter — also fetch litter boxes per frame for
                # VLM hallucination consistency check (see _analyze_candidates).
                batch_frames = [frm for _, frm in batch_raw]
                with self._prefilter_lock:
                    batch_candidates, batch_litter = (
                        self._prefilter.detect_candidates_with_litter_batch(batch_frames)
                    )
                    # 只覆寫 litter 那一半；batch_candidates（person/vehicle）保持 YOLO 的。
                    # 回 None = sidecar 出問題 → 沿用 YOLO 的框，pipeline 不受影響。
                    if self._codetr_litter is not None:
                        # 這一步不快（R50 5-scale @1024²，實測 420ms/幀，一批 8 幀 ≈ 3.4 秒）。
                        # 不回報的話介面會看起來卡在「YOLO 偵測中」，所以自成一個階段。
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "progress": (
                                        0.0
                                        if total_frames <= 0
                                        else min(1.0, max(0.0, frame_index / max(total_frames, 1)))
                                    ),
                                    "stage": "codetr",
                                    "frame": frame_index,
                                    "total_frames": total_frames,
                                }
                            )
                        codetr_litter = self._codetr_litter.litter_boxes(
                            [fi for fi, _ in batch_raw]
                        )
                        if codetr_litter is not None:
                            batch_litter = codetr_litter

                for (fi, frm), candidates, litter_boxes in zip(
                    batch_raw, batch_candidates, batch_litter
                ):
                    if max_calls > 0 and vlm_decision_frames >= max_calls:
                        if audit_log is not None:
                            audit_log.log_skip(
                                frame_index=fi,
                                frame_time_sec=fi / fps if fps > 0 else 0.0,
                                overlay_time="",
                                event_type="budget_skip",
                                candidate_label="",
                                candidate_box=None,
                                decision_note=f"max_calls={max_calls}",
                            )
                        break

                    if not candidates:
                        # Track consecutive empty frames so we can collapse
                        # them into a single prefilter_empty_range row.
                        if empty_range_start is None:
                            empty_range_start = fi
                        continue
                    # Non-empty after prefilter — flush any pending empty range.
                    _flush_empty_range(end_frame=fi)

                    candidates = self._select_candidates(candidates, frm)
                    # Lightweight temporal pre-filter: skip candidates that
                    # are clearly just passing through the frame.
                    ring_snap = list(self._frame_ring)
                    pre_disp_count = len(candidates)
                    candidates = [
                        c for c in candidates
                        if self._is_candidate_worth_vlm(c, ring_snap, frm)
                    ]
                    if not candidates:
                        if audit_log is not None and pre_disp_count > 0:
                            audit_log.log_skip(
                                frame_index=fi,
                                frame_time_sec=fi / fps if fps > 0 else 0.0,
                                overlay_time="",
                                event_type="displacement_skip",
                                candidate_label="",
                                candidate_box=None,
                                decision_note=f"all_{pre_disp_count}_candidates_filtered",
                            )
                        continue

                    if fi - last_vlm_frame < cooldown_frames:
                        if audit_log is not None:
                            audit_log.log_skip(
                                frame_index=fi,
                                frame_time_sec=fi / fps if fps > 0 else 0.0,
                                overlay_time="",
                                event_type="cooldown_skip",
                                candidate_label=str(candidates[0].get("label", "") if candidates else ""),
                                candidate_box=None,
                                decision_note=f"cooldown_frames={cooldown_frames}",
                            )
                        continue
                    last_vlm_frame = fi
                    vlm_decision_frames += 1

                    if progress_callback is not None:
                        progress_callback(
                            {
                                "progress": min(
                                    1.0, max(0.0, fi / max(total_frames, 1))
                                ),
                                "stage": "vlm",
                                "frame": fi,
                                "total_frames": total_frames,
                            }
                        )

                    best_event = self._analyze_candidates(
                        frame=frm,
                        frame_index=fi,
                        fps=fps,
                        candidates=candidates,
                        detect_litter=detect_litter,
                        detect_urination=detect_urination,
                        snap_dir=snap_dir,
                        frame_ring=list(self._frame_ring),
                        video_start_dt=video_start_dt,
                        frame_litter_boxes=litter_boxes,
                        heartbeat=heartbeat,
                        audit_log=audit_log,
                    )
                    if best_event is not None:
                        events.append(best_event)
        finally:
            _vlm_prefetch_stop.set()
            cap.release()
            # Flush any tail empty range that ran to EOF.
            if empty_range_start is not None:
                _flush_empty_range(end_frame=frame_index)

        # Capture pre-merge state to log which originals get folded into winners.
        pre_merge_events = list(events)
        merged_events = self._merge_events(events=events, fps=fps)
        if audit_log is not None and len(pre_merge_events) > len(merged_events):
            survivor_frames = {e.frame_index for e in merged_events}
            for orig in pre_merge_events:
                if orig.frame_index in survivor_frames:
                    continue
                if merged_events:
                    winner = min(
                        merged_events,
                        key=lambda m: abs(m.frame_index - orig.frame_index),
                    )
                    winner_idx = winner.frame_index
                else:
                    winner_idx = orig.frame_index
                audit_log.log_event_merged(
                    frame_index=orig.frame_index,
                    frame_time_sec=orig.frame_index / fps if fps > 0 else 0.0,
                    overlay_time=orig.overlay_time,
                    winner_frame_index=winner_idx,
                )

        pre_dedup_events = list(merged_events)
        merged_events = self._deduplicate_by_plate(merged_events)
        if audit_log is not None and len(pre_dedup_events) > len(merged_events):
            survivor_frames = {e.frame_index for e in merged_events}
            for orig in pre_dedup_events:
                if orig.frame_index in survivor_frames:
                    continue
                plate_key = _normalize_plate_for_merge(orig.plate)
                audit_log.log_event_deduped(
                    frame_index=orig.frame_index,
                    frame_time_sec=orig.frame_index / fps if fps > 0 else 0.0,
                    overlay_time=orig.overlay_time,
                    winning_plate=plate_key,
                )

        merged_events = self._attach_litter_clips(
            video_path=video_path,
            events=merged_events,
            fps=fps,
            snap_dir=snap_dir,
        )

        self._write_csv(
            video_name=video_path.name,
            out_csv_path=out_csv_path,
            events=merged_events,
        )
        if audit_log is not None:
            audit_log.close()
        return {
            "events": [event.to_dict() for event in merged_events],
            "csv": str(out_csv_path),
            "snap_dir": str(snap_dir),
            "clip_dir": str(snap_dir.parent / "違規片段"),
        }

    def _merge_events(
        self, events: list[ViolationEvent], fps: float
    ) -> list[ViolationEvent]:
        if not events:
            return []
        if self._event_merge_sec <= 0:
            return events

        gap_frames = max(1, int(self._event_merge_sec * fps))
        sorted_events = sorted(events, key=lambda e: e.frame_index)
        merged: list[ViolationEvent] = []

        current = sorted_events[0]
        for event in sorted_events[1:]:
            same_behavior = event.behavior == current.behavior
            current_plate_key = _normalize_plate_for_merge(current.plate)
            event_plate_key = _normalize_plate_for_merge(event.plate)
            strict_plate_match = bool(current_plate_key) and (
                current_plate_key == event_plate_key
            )
            one_plate_missing = (not current_plate_key and bool(event_plate_key)) or (
                bool(current_plate_key) and not event_plate_key
            )
            both_plate_missing = (not current_plate_key) and (not event_plate_key)
            box_iou = _box_iou_xyxy(current.candidate_box, event.candidate_box)
            spatial_match = box_iou >= 0.10
            weak_spatial_match = box_iou >= 0.02
            has_both_boxes = (
                current.candidate_box is not None and event.candidate_box is not None
            )
            strict_subject_match = strict_plate_match and (
                (not has_both_boxes) or weak_spatial_match
            )
            same_subject = (
                strict_subject_match
                or (one_plate_missing and spatial_match)
                or (both_plate_missing and box_iou >= 0.20)
            )
            gap_allowance = gap_frames * (
                2 if (spatial_match and not both_plate_missing) else 1
            )
            close_enough = (event.frame_index - current.frame_index) <= gap_allowance

            if same_behavior and same_subject and close_enough:
                winner = event if event.confidence >= current.confidence else current
                loser = current if winner is event else event
                winner_plate_key = _normalize_plate_for_merge(winner.plate)
                loser_plate_key = _normalize_plate_for_merge(loser.plate)
                chosen_plate = winner.plate
                if loser_plate_key and not winner_plate_key:
                    chosen_plate = loser.plate
                current = ViolationEvent(
                    frame_index=winner.frame_index,
                    frame_time=winner.frame_time,
                    overlay_time=winner.overlay_time
                    if winner.overlay_time
                    else loser.overlay_time,
                    plate=chosen_plate,
                    behavior=winner.behavior,
                    confidence=winner.confidence,
                    reason=winner.reason if winner.reason else loser.reason,
                    snapshot=winner.snapshot if winner.snapshot else loser.snapshot,
                    clip=winner.clip if winner.clip else loser.clip,
                    candidate_box=winner.candidate_box
                    if winner.candidate_box is not None
                    else loser.candidate_box,
                )
                continue

            merged.append(current)
            current = event

        merged.append(current)
        return merged

    def _deduplicate_by_plate(
        self, events: list[ViolationEvent]
    ) -> list[ViolationEvent]:
        """Keep only the highest-confidence event per known plate in a video."""
        plate_best: dict[str, ViolationEvent] = {}
        no_plate: list[ViolationEvent] = []
        for event in events:
            plate_key = _normalize_plate_for_merge(event.plate)
            if not plate_key:
                no_plate.append(event)
            else:
                existing = plate_best.get(plate_key)
                if existing is None or event.confidence > existing.confidence:
                    plate_best[plate_key] = event
        result = list(plate_best.values()) + no_plate
        result.sort(key=lambda e: e.frame_index)
        return result

    def _is_candidate_worth_vlm(
        self,
        candidate: dict[str, Any],
        frame_ring: list[tuple[int, Any]],
        current_frame: Any,
    ) -> bool:
        """Lightweight temporal filter: skip candidates that are clearly just
        passing through (high displacement across frames).

        Littering / urination requires a person to *stay* roughly in the same
        area.  If the candidate's bounding-box centre has moved more than half
        the frame width across the ring buffer, it is very likely a moving
        vehicle or pedestrian walking by — not worth a VLM call.

        Returns True when the candidate should be sent to VLM.
        """
        if len(frame_ring) < 3:
            return True  # not enough history to judge; let VLM decide
        shape = getattr(current_frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return True
        h, w = int(shape[0]), int(shape[1])
        if w <= 0 or h <= 0:
            return True

        cx = (int(candidate.get("x1", 0)) + int(candidate.get("x2", 0))) / 2.0
        cy = (int(candidate.get("y1", 0)) + int(candidate.get("y2", 0))) / 2.0
        label = str(candidate.get("label", "")).lower()

        # Vehicles are expected to move; only filter persons with extreme
        # displacement.  For vehicles we still want VLM to check if something
        # was tossed from the window, so use a very generous threshold.
        max_disp_ratio = 0.30 if label in {"person", "people", "pedestrian"} else 0.50

        # Compare with the oldest frame in the ring — we don't have per-
        # candidate tracking across frames (that would need a full tracker),
        # so we use a heuristic: if this candidate's label appeared in the
        # oldest ring frame's YOLO results we could match by IoU.  Since we
        # don't persist per-frame detections in the ring, we fall back to a
        # simpler rule: the candidate centre must be within *max_disp_ratio*
        # of the frame diagonal from the frame centre band.  This is a very
        # loose filter that only blocks candidates at the extreme edges moving
        # fast.  More precise tracking-based filtering can be added later.
        #
        # For now, a useful fast check: skip if bounding box is tiny AND at
        # the very edge of frame (likely a car that just entered / exited).
        area = max(1, (int(candidate.get("x2", 0)) - int(candidate.get("x1", 0)))
                   * (int(candidate.get("y2", 0)) - int(candidate.get("y1", 0))))
        area_ratio = area / max(1, w * h)
        at_edge = (cx < w * 0.08 or cx > w * 0.92 or cy < h * 0.08 or cy > h * 0.92)
        if area_ratio < 0.005 and at_edge:
            return False  # tiny object at the edge — noise

        return True

    def _select_candidates(
        self,
        candidates: list[dict[str, Any]],
        frame: Any,
    ) -> list[dict[str, Any]]:
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return candidates[: self._max_candidates_per_frame]
        h = int(shape[0])
        w = int(shape[1])

        def _area(c: dict[str, Any]) -> int:
            x1 = int(c.get("x1", 0))
            y1 = int(c.get("y1", 0))
            x2 = int(c.get("x2", 0))
            y2 = int(c.get("y2", 0))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            return max(0, x2 - x1) * max(0, y2 - y1)

        frame_area = max(1, w * h)

        def _priority(c: dict[str, Any]) -> tuple[float, float, float]:
            area = float(_area(c))
            area_ratio = min(1.0, area / frame_area)
            score_raw = c.get("score", 0.0)
            score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
            label = str(c.get("label", "")).lower()
            person_bonus = 0.08 if label in {"person", "people", "pedestrian"} else 0.0
            priority = score * 0.60 + area_ratio * 0.32 + person_bonus
            return (priority, score, area)

        ranked = sorted(candidates, key=_priority, reverse=True)
        return ranked[: self._max_candidates_per_frame]

    def _candidate_roi_rect(
        self, frame: Any, candidate: dict[str, Any]
    ) -> tuple[int, int, int, int] | None:
        """候選框往外擴 roi_expand 後、夾在畫面內的矩形（全幀座標）。

        抽出來是為了讓 `_crop_candidate` 與 `_codetr_litter_box_for_roi` 用同一份
        幾何 —— 兩邊算得不一樣的話，Co-DETR 的框換算到 ROI 會偏掉。
        """
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None
        h = int(shape[0])
        w = int(shape[1])

        x1 = int(candidate.get("x1", 0))
        y1 = int(candidate.get("y1", 0))
        x2 = int(candidate.get("x2", w))
        y2 = int(candidate.get("y2", h))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * self._roi_expand)
        pad_y = int(bh * self._roi_expand)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return (cx1, cy1, cx2, cy2)

    def _crop_candidate(self, frame: Any, candidate: dict[str, Any]) -> Any:
        rect = self._candidate_roi_rect(frame, candidate)
        if rect is None:
            return frame
        cx1, cy1, cx2, cy2 = rect
        return frame[cy1:cy2, cx1:cx2]

    def _codetr_litter_box_for_roi(
        self,
        frame: Any,
        candidate: dict[str, Any],
        frame_litter_boxes: list[dict[str, Any]] | None,
    ) -> tuple[int, int, int, int] | None:
        """把 Co-DETR 的 litter/garbage 框（全幀座標）換算成這個候選 ROI 內的座標。

        用途：VLM 判「亂丟垃圾」卻沒指出垃圾在哪時，原本會因為
        `rejected_no_litter_box` 被更高的信心門檻擋掉。有偵測器實際看到垃圾的話，
        那個框就是比 VLM 自陳更可信的空間證據。

        只取**框心落在 ROI 內**、分數最高的一個；ROI 外的垃圾與這個候選無關。
        """
        if not frame_litter_boxes:
            return None
        rect = self._candidate_roi_rect(frame, candidate)
        if rect is None:
            return None
        cx1, cy1, cx2, cy2 = rect

        best: dict[str, Any] | None = None
        for box in frame_litter_boxes:
            try:
                bx1, by1 = int(box["x1"]), int(box["y1"])
                bx2, by2 = int(box["x2"]), int(box["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            mid_x = (bx1 + bx2) // 2
            mid_y = (by1 + by2) // 2
            if not (cx1 <= mid_x < cx2 and cy1 <= mid_y < cy2):
                continue
            score = float(box.get("score", 0.0))
            if best is None or score > float(best.get("score", 0.0)):
                best = box
        if best is None:
            return None

        rx1 = max(0, int(best["x1"]) - cx1)
        ry1 = max(0, int(best["y1"]) - cy1)
        rx2 = min(cx2 - cx1, int(best["x2"]) - cx1)
        ry2 = min(cy2 - cy1, int(best["y2"]) - cy1)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return (rx1, ry1, rx2, ry2)

    def _parse_litter_box(
        self, value: Any, roi_shape: Any
    ) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        if not (isinstance(roi_shape, tuple) and len(roi_shape) >= 2):
            return None
        h = int(roi_shape[0])
        w = int(roi_shape[1])
        if h <= 0 or w <= 0:
            return None

        try:
            raw = [float(v) for v in value]
        except (TypeError, ValueError):
            return None

        if all(0.0 <= v <= 1.0 for v in raw):
            x1 = int(raw[0] * w)
            y1 = int(raw[1] * h)
            x2 = int(raw[2] * w)
            y2 = int(raw[3] * h)
        else:
            x1, y1, x2, y2 = [int(v) for v in raw]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _annotate_candidate_on_roi(
        self,
        roi: Any,
        candidate: dict[str, Any],
        frame: Any,
    ) -> Any:
        """Draw the original candidate bounding box onto the cropped ROI image."""
        cv2_mod = importlib.import_module("cv2")
        shape = getattr(frame, "shape", None)
        roi_shape = getattr(roi, "shape", None)
        if not isinstance(shape, tuple) or not isinstance(roi_shape, tuple):
            return roi
        fh, fw = int(shape[0]), int(shape[1])
        rh, rw = int(roi_shape[0]), int(roi_shape[1])
        if fh <= 0 or fw <= 0 or rh <= 0 or rw <= 0:
            return roi

        cx1_orig = int(candidate.get("x1", 0))
        cy1_orig = int(candidate.get("y1", 0))
        cx2_orig = int(candidate.get("x2", fw))
        cy2_orig = int(candidate.get("y2", fh))
        bw = max(1, cx2_orig - cx1_orig)
        bh = max(1, cy2_orig - cy1_orig)
        pad_x = int(bw * self._roi_expand)
        pad_y = int(bh * self._roi_expand)
        roi_x1 = max(0, cx1_orig - pad_x)
        roi_y1 = max(0, cy1_orig - pad_y)
        # Translate candidate box into ROI coordinate space
        bx1 = cx1_orig - roi_x1
        by1 = cy1_orig - roi_y1
        bx2 = cx2_orig - roi_x1
        by2 = cy2_orig - roi_y1
        bx1 = max(0, min(rw - 1, bx1))
        by1 = max(0, min(rh - 1, by1))
        bx2 = max(1, min(rw, bx2))
        by2 = max(1, min(rh, by2))
        try:
            img = roi.copy() if hasattr(roi, "copy") else roi
            cv2_mod.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            return img
        except Exception:
            return roi

    def _ocr_timestamp(self, frame: Any) -> str:
        """Try to read the CCTV timestamp using pytesseract (if available).

        Returns the OCR'd time string, or an empty string on any error.
        Never returns a fallback value — callers must supply their own fallback
        so we don't accidentally inject a wrong relative-time string as a hint.
        """
        try:
            import pytesseract  # type: ignore[import]
            cv2_mod = importlib.import_module("cv2")
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                return ""
            h, w = int(shape[0]), int(shape[1])
            y2 = max(2, int(h * 0.12))
            x1 = max(0, int(w * 0.50))
            region = frame[0:y2, x1:w]
            rh = int(getattr(region, "shape", (0,))[0])
            rw = int(getattr(region, "shape", (0, 0))[1])
            if rh <= 0 or rw <= 0:
                return ""
            scale = max(1.0, 80.0 / rh)
            up = cv2_mod.resize(
                region,
                (max(1, int(rw * scale)), max(1, int(rh * scale))),
                interpolation=cv2_mod.INTER_LANCZOS4,
            )
            text = pytesseract.image_to_string(
                up, config="--psm 6 -c tessedit_char_whitelist=0123456789/:-. "
            ).strip()
            digits = sum(c.isdigit() for c in text)
            if digits >= 8:
                return text
        except Exception:
            pass
        return ""

    def _ocr_plate(self, frame: Any, candidate: dict[str, Any]) -> str:
        """Try to read a licence plate from the candidate bounding-box region.

        Uses pytesseract with a broad character set so it can handle both
        alphanumeric (international) and mixed-script (Taiwan) plates.
        Returns an empty string on any error or implausible result.
        """
        try:
            import pytesseract  # type: ignore[import]
            cv2_mod = importlib.import_module("cv2")
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                return ""
            fh, fw = int(shape[0]), int(shape[1])
            # Expand box by 20 px on each side for plate context
            x1 = max(0, int(candidate.get("x1", 0)) - 20)
            y1 = max(0, int(candidate.get("y1", 0)) - 20)
            x2 = min(fw, int(candidate.get("x2", fw)) + 20)
            y2 = min(fh, int(candidate.get("y2", fh)) + 20)
            if x2 <= x1 or y2 <= y1:
                return ""
            crop = frame[y1:y2, x1:x2]
            cw = int(getattr(crop, "shape", (0, 0))[1])
            if cw < 200:
                s = 200.0 / max(cw, 1)
                ch = int(getattr(crop, "shape", (0,))[0])
                crop = cv2_mod.resize(
                    crop,
                    (max(1, int(cw * s)), max(1, int(ch * s))),
                    interpolation=cv2_mod.INTER_LANCZOS4,
                )
            # PSM 7: treat image as single text line
            text = pytesseract.image_to_string(
                crop,
                lang="chi_tra+eng",
                config="--psm 7",
            ).strip()
            if not text:
                # fallback: eng only
                text = pytesseract.image_to_string(
                    crop,
                    config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                ).strip()
            # Must have at least 2 characters to be a plausible plate
            if len(text) >= 2:
                return text
        except Exception:
            pass
        return ""

    def _vlm_read_timestamp(self, frame: Any) -> str:
        """Dedicated focused VLM call to read the CCTV overlay timestamp.

        Crops the top-right timestamp region from the full frame, scales it to
        480 px tall (4–5× native size), and asks the VLM with a minimal prompt
        so it only has to read text — no violation detection.  Returns "" on
        failure.  This is called only when the primary VLM call failed to return
        an event_time, so it adds latency only in the failure path.
        """
        try:
            cv2_mod = importlib.import_module("cv2")
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                return ""
            h, w = int(shape[0]), int(shape[1])
            y2 = max(2, int(h * 0.15))
            x1 = max(0, int(w * 0.55))
            region = frame[0:y2, x1:w]
            rh = int(getattr(region, "shape", (0,))[0])
            rw = int(getattr(region, "shape", (0, 0))[1])
            if rh <= 0 or rw <= 0:
                return ""
            ts_scale = 480.0 / rh
            up = cv2_mod.resize(
                region,
                (max(1, int(rw * ts_scale)), 480),
                interpolation=cv2_mod.INTER_LANCZOS4,
            )
            text = self._vlm.read_text_from_image(
                up,
                (
                    "請讀取圖片中顯示的日期時間。"
                    "只回傳時間字串（格式如 2024-01-15 14:32:55），"
                    "不要回傳任何其他文字。若無法辨識請回傳空字串。"
                ),
            )
            digits = sum(c.isdigit() for c in text)
            if digits >= 6:
                return text.strip()
        except Exception:
            pass
        return ""

    def _vlm_read_plate(self, frame: Any, candidate: dict[str, Any]) -> str:
        """Dedicated focused VLM call to read a licence plate near the candidate.

        Uses the full-width horizontal band at the candidate's vertical position
        (so nearby vehicles' plates are visible) at ≤640 px wide.  Returns ""
        on failure or when no plate is visible.
        """
        try:
            cv2_mod = importlib.import_module("cv2")
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                return ""
            fh, fw = int(shape[0]), int(shape[1])
            y1 = max(0, int(candidate.get("y1", 0)) - 50)
            y2 = min(fh, int(candidate.get("y2", fh)) + 100)
            if y2 <= y1:
                return ""
            region = frame[y1:y2, 0:fw]
            rh = int(getattr(region, "shape", (0,))[0])
            rw = int(getattr(region, "shape", (0, 0))[1])
            if rh <= 0 or rw <= 0:
                return ""
            if rw > 640:
                s = 640.0 / rw
                region = cv2_mod.resize(
                    region,
                    (640, max(1, int(rh * s))),
                    interpolation=cv2_mod.INTER_AREA,
                )
            text = self._vlm.read_text_from_image(
                region,
                (
                    "圖片中是否有車牌號碼？"
                    "若有請只回傳車牌號碼文字，若沒有或看不清楚請回傳空字串。"
                    "不要回傳任何解釋或其他文字。"
                ),
            )
            # Defence in depth: strip_think already runs in read_text_from_image,
            # but the plate prompt is plain-text so sanitize once more and also
            # drop surrounding quotes (GLM sometimes emits '""' for "no plate").
            text = _clean_text(_strip_think_tags(text))
            _bad = {"無", "無法辨識", "不清楚", "沒有", "無車牌", "看不清楚", ""}
            if len(text) >= 2 and text not in _bad:
                return text
        except Exception:
            pass
        return ""

    def _make_fullframe_panel(
        self,
        frame: Any,
        candidate: dict[str, Any],
        panel_h: int,
    ) -> Any:
        """Return a full-frame thumbnail with the candidate box highlighted.

        Used so the VLM can read the top-right timestamp and nearby plates.
        Returns None if the frame cannot be processed.
        """
        cv2_mod = importlib.import_module("cv2")
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return None
        fh, fw = int(shape[0]), int(shape[1])
        if fh <= 0 or fw <= 0 or panel_h <= 0:
            return None
        scale = panel_h / max(fh, 1)
        thumb_w = max(1, int(fw * scale))
        thumb = cv2_mod.resize(frame, (thumb_w, panel_h), interpolation=cv2_mod.INTER_AREA)
        # Draw candidate box in green so VLM can locate the subject
        tx1 = max(0, int(candidate.get("x1", 0) * scale))
        ty1 = max(0, int(candidate.get("y1", 0) * scale))
        tx2 = min(thumb_w - 1, int(candidate.get("x2", fw) * scale))
        ty2 = min(panel_h - 1, int(candidate.get("y2", fh) * scale))
        cv2_mod.rectangle(thumb, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
        thumb = _put_chinese_text(thumb, "完整畫面", (4, 18), 0.5, (0, 255, 0), 1)
        return thumb

    def _resize_for_context(self, frame: Any, max_width: int = 1920) -> Any:
        """Return *frame* resized so its width ≤ max_width, preserving aspect ratio.

        The context frame is sent to the VLM at near-native resolution so the
        timestamp overlay and any vehicle plates remain clearly readable.
        """
        try:
            cv2_mod = importlib.import_module("cv2")
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                return frame
            h, w = int(shape[0]), int(shape[1])
            if w <= 0 or w <= max_width:
                return frame
            scale = max_width / w
            new_w = max_width
            new_h = max(1, int(h * scale))
            return cv2_mod.resize(frame, (new_w, new_h), interpolation=cv2_mod.INTER_AREA)
        except Exception:
            return frame

    def _build_vlm_composite(self, mosaic: Any, current_frame: Any) -> Any:
        """Stack a zoomed timestamp strip above *mosaic* for reliable time reading.

        Extracts the top-right corner of the full frame (where CCTV timestamps live),
        scales it UP so the text is clearly legible (≥40 px tall), then prepends it
        as a banner above the existing mosaic rows.  Falls back to the original mosaic
        on any error so it never breaks the main pipeline.
        """
        try:
            cv2_mod = importlib.import_module("cv2")
            np_mod = importlib.import_module("numpy")
            fshape = getattr(current_frame, "shape", None)
            mshape = getattr(mosaic, "shape", None)
            if not isinstance(fshape, tuple) or not isinstance(mshape, tuple):
                return mosaic
            fh, fw = int(fshape[0]), int(fshape[1])
            mosaic_h, mosaic_w = int(mshape[0]), int(mshape[1])
            if fh <= 0 or fw <= 0 or mosaic_h <= 0 or mosaic_w <= 0:
                return mosaic

            # Extract top 8 % of frame height, right 50 % of width —
            # the typical CCTV timestamp location.
            ts_y2 = max(2, int(fh * 0.08))
            ts_x1 = max(0, int(fw * 0.50))
            ts_region = current_frame[0:ts_y2, ts_x1:fw]
            ts_h = int(getattr(ts_region, "shape", (0,))[0])
            ts_w = int(getattr(ts_region, "shape", (0, 0))[1])
            if ts_h <= 0 or ts_w <= 0:
                return mosaic

            # Scale UP so the strip is exactly STAMP_H pixels tall.
            # For 1080 p source (ts_h ≈ 86 px) this gives a 1.16× upscale
            # → timestamp text grows from ~30 px to ~35 px (clearly readable).
            STAMP_H = 100
            scale = STAMP_H / max(ts_h, 1)
            new_w = max(1, int(ts_w * scale))
            ts_scaled = cv2_mod.resize(
                ts_region, (new_w, STAMP_H), interpolation=cv2_mod.INTER_LANCZOS4
            )
            cv2_mod.rectangle(ts_scaled, (0, 0), (new_w - 1, STAMP_H - 1), (0, 255, 255), 2)
            ts_scaled = _put_chinese_text(
                ts_scaled, "時間戳放大", (4, STAMP_H - 8), 0.5, (0, 255, 255), 1
            )

            # Pad (black) or crop to match mosaic width exactly.
            actual_w = int(getattr(ts_scaled, "shape", (0, 0))[1])
            if actual_w < mosaic_w:
                pad = np_mod.zeros(
                    (STAMP_H, mosaic_w - actual_w, 3), dtype=np_mod.uint8
                )
                ts_bar = np_mod.concatenate([ts_scaled, pad], axis=1)
            else:
                ts_bar = ts_scaled[:, :mosaic_w]

            return np_mod.concatenate([ts_bar, mosaic], axis=0)
        except Exception:
            return mosaic

    def _build_temporal_mosaic(
        self,
        current_frame: Any,
        current_idx: int,
        candidate: dict[str, Any],
        frame_ring: list[tuple[int, Any]],
    ) -> tuple[Any, bool]:
        """Return a composite: zoomed timestamp strip on top, full annotated frame below.

        VLMs process images at ~448 px internal resolution, so a 1920×1080 frame
        is downscaled ~4×, making a 20-30 px timestamp unreadable (~5-7 px).
        Zooming the timestamp region to 200 px tall before compositing keeps the
        text at ~50 px after VLM downscale — clearly legible.  Returns
        (composite, False); is_mosaic=False so the prompt uses single-frame wording.
        """
        cv2_mod = importlib.import_module("cv2")
        np_mod = importlib.import_module("numpy")
        shape = getattr(current_frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return current_frame, False
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return current_frame, False

        # Scale main frame to ≤1280 px wide (smaller total pixel budget for VLM)
        MAX_W = 1280
        if w > MAX_W:
            mscale = MAX_W / w
            mw = MAX_W
            mh = max(1, int(h * mscale))
            main = cv2_mod.resize(current_frame, (mw, mh), interpolation=cv2_mod.INTER_AREA)
            sx, sy = mscale, mscale
        else:
            main = current_frame.copy() if hasattr(current_frame, "copy") else current_frame
            sx, sy = 1.0, 1.0
            mh, mw = int(main.shape[0]), int(main.shape[1])

        mh, mw = int(main.shape[0]), int(main.shape[1])

        # Draw candidate bounding box on main frame
        bx1 = max(0, int(candidate.get("x1", 0) * sx))
        by1 = max(0, int(candidate.get("y1", 0) * sy))
        bx2 = min(mw - 1, int(candidate.get("x2", w) * sx))
        by2 = min(mh - 1, int(candidate.get("y2", h) * sy))
        cv2_mod.rectangle(main, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
        label = str(candidate.get("label", ""))
        if label:
            main = _put_chinese_text(main, label, (bx1, max(20, by1 - 4)), 0.6, (0, 255, 255), 2)

        # Build zoomed timestamp strip from the ORIGINAL (full-res) frame.
        # Crop top 12 %, right 45 % — the typical CCTV timestamp region.
        # Scale to STAMP_H=200 px tall so text survives VLM internal downscaling.
        STAMP_H = 200
        try:
            ts_y2 = max(2, int(h * 0.12))
            ts_x1 = max(0, int(w * 0.55))
            ts_region = current_frame[0:ts_y2, ts_x1:w]
            ts_h = int(getattr(ts_region, "shape", (0,))[0])
            ts_w = int(getattr(ts_region, "shape", (0, 0))[1])
            if ts_h > 0 and ts_w > 0:
                ts_scale = STAMP_H / ts_h
                new_ts_w = max(1, int(ts_w * ts_scale))
                ts_up = cv2_mod.resize(
                    ts_region, (new_ts_w, STAMP_H), interpolation=cv2_mod.INTER_LANCZOS4
                )
                cv2_mod.rectangle(ts_up, (0, 0), (new_ts_w - 1, STAMP_H - 1), (0, 255, 0), 3)
                ts_up = _put_chinese_text(
                    ts_up, "★時間戳放大區★", (4, STAMP_H - 12), 0.7, (0, 255, 0), 2
                )
                # Pad (black) or crop strip to match main frame width
                actual_ts_w = int(getattr(ts_up, "shape", (0, 0))[1])
                if actual_ts_w < mw:
                    pad = np_mod.zeros((STAMP_H, mw - actual_ts_w, 3), dtype=np_mod.uint8)
                    ts_bar = np_mod.concatenate([ts_up, pad], axis=1)
                else:
                    ts_bar = ts_up[:, :mw]
                composite = np_mod.concatenate([ts_bar, main], axis=0)
                return composite, False
        except Exception:
            pass

        return main, False

    def _upscale_if_small(self, roi: Any, min_side: int = 224) -> Any:
        """Upscale ROI if either dimension is below min_side."""
        cv2_mod = importlib.import_module("cv2")
        shape = getattr(roi, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return roi
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            return roi
        if h >= min_side and w >= min_side:
            return roi
        scale = max(min_side / max(h, 1), min_side / max(w, 1))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        return cv2_mod.resize(roi, (new_w, new_h), interpolation=cv2_mod.INTER_LANCZOS4)

    def _render_annotated_snapshot(
        self,
        roi: Any,
        behavior: str,
        confidence: float,
        reason: str,
        litter_box: tuple[int, int, int, int] | None,
    ) -> Any:
        cv2_mod = importlib.import_module("cv2")
        img = roi.copy() if hasattr(roi, "copy") else roi
        shape = getattr(img, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            return img
        h = int(shape[0])
        w = int(shape[1])
        if h <= 0 or w <= 0:
            return img

        try:
            cv2_mod.rectangle(
                img, (0, 0), (max(1, w - 1), max(1, h - 1)), (0, 255, 255), 2
            )
            if litter_box is not None:
                x1, y1, x2, y2 = litter_box
                cv2_mod.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                img = _put_chinese_text(
                    img,
                    "垃圾位置",
                    (max(2, x1), max(20, y1 - 6)),
                    0.6,
                    (0, 0, 255),
                    2,
                )
            title = f"{behavior}事件 {confidence:.2f}"
            img = _put_chinese_text(img, title, (10, 24), 0.7, (255, 255, 255), 2)
            if reason:
                img = _put_chinese_text(
                    img,
                    reason[:32],
                    (10, max(32, h - 12)),
                    0.55,
                    (255, 255, 255),
                    2,
                )
        except Exception:
            return roi
        return img

    def _analyze_candidates(
        self,
        frame: Any,
        frame_index: int,
        fps: float,
        candidates: list[dict[str, Any]],
        detect_litter: bool,
        detect_urination: bool,
        snap_dir: Path,
        frame_litter_boxes: list[dict[str, Any]] | None = None,
        frame_ring: list[tuple[int, Any]] | None = None,
        video_start_dt: datetime.datetime | None = None,
        heartbeat: "HeartbeatEmitter | None" = None,
        audit_log: "AuditLogger | None" = None,
    ) -> ViolationEvent | None:
        cv2_mod = importlib.import_module("cv2")
        frame_time_base = _to_hms(frame_index / fps)

        # Compute wall-clock timestamp directly from video metadata/filename.
        # This is far more reliable than asking the VLM to read the OSD overlay.
        # video_start_dt is None only if _parse_video_start_time failed entirely.
        if video_start_dt is not None:
            wall_dt = video_start_dt + datetime.timedelta(seconds=frame_index / fps)
            computed_wall_time: str = wall_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            computed_wall_time = ""

        # pytesseract OCR as bonus: only used if tesseract binary is installed
        # (not available in the compiled exe, always returns "").
        ocr_timestamp = self._ocr_timestamp(frame)

        # Cache for the dedicated VLM timestamp call (only triggered when both
        # file-metadata and pytesseract timestamp extraction fail).
        _vlm_ts: list[str] = []  # empty = not tried, one element = result (may be "")
        best: ViolationEvent | None = None
        # Build a downscaled context image (with timestamp bar) once per frame,
        # shared across all candidates.  The VLM receives two images:
        #   Image 1 – tightly cropped ROI (behaviour analysis)
        #   Image 2 – full scene with timestamp bar (plate / context)
        context_image, _ctx_mosaic = self._build_temporal_mosaic(
            current_frame=frame,
            current_idx=frame_index,
            candidate=candidates[0] if candidates else {},
            frame_ring=frame_ring or [],
        )
        # YOLO consistency check: detect VLM hallucinations on「亂丟垃圾」 by
        # checking if YOLO actually saw a litter-class box in this frame.
        # 經實測，在當前 prompt 下將 detect_litter=False 直接傳給 VLM 會讓便溺
        # confidence 跌破 0.80 threshold（v7: 0/1）；改用 post-hoc relabel：
        # VLM 高 conf 結果保留，只把「亂丟垃圾 + YOLO 沒看到 litter」的明顯
        # 幻覺 relabel 成「便溺」（仍保留原始 confidence）。
        yolo_saw_litter = bool(frame_litter_boxes)
        for cand_idx, candidate in enumerate(candidates):
            ocr_plate = self._ocr_plate(frame, candidate)
            # Tightly cropped ROI — used for both VLM input and snapshot rendering
            roi = self._crop_candidate(frame, candidate)
            vlm_roi = self._upscale_if_small(roi)
            target_hint = str(candidate.get("label", "unknown"))

            # Per-candidate audit metadata reused for every audit row this
            # candidate generates (vlm_call success / rejected_* / event_kept).
            cand_box = (
                int(candidate.get("x1", 0)),
                int(candidate.get("y1", 0)),
                int(candidate.get("x2", 0)),
                int(candidate.get("y2", 0)),
            )
            audit_meta = {
                "frame_index": frame_index,
                "frame_time_sec": frame_index / fps if fps > 0 else 0.0,
                "overlay_time": computed_wall_time or ocr_timestamp,
                "candidate_label": target_hint,
                "candidate_box": cand_box,
            }

            try:
                result = self._run_vlm_analyze(
                    heartbeat=heartbeat,
                    frame_bgr=vlm_roi,
                    detect_litter=detect_litter,
                    detect_urination=detect_urination,
                    target_hint=target_hint,
                    is_mosaic=False,
                    context_frame_bgr=context_image,
                    hint_timestamp=ocr_timestamp,
                    hint_plate=ocr_plate,
                )
            except VLMHardTimeout as exc:
                if audit_log is not None:
                    audit_log.log_vlm_call(
                        latency_ms=self._last_vlm_latency_ms,
                        timed_out=True,
                        vlm_response=None,
                        decision="rejected_hard_timeout",
                        decision_note=str(exc)[:120],
                        snapshot_path="",
                        **audit_meta,
                    )
                logger.warning(
                    "VLM hard timeout for frame %d cand %d: %s",
                    frame_index, cand_idx, exc,
                )
                continue
            except Exception as exc:
                if audit_log is not None:
                    audit_log.log_vlm_call(
                        latency_ms=self._last_vlm_latency_ms,
                        timed_out=False,
                        vlm_response=None,
                        decision="rejected_exception",
                        decision_note=str(exc)[:120],
                        snapshot_path="",
                        **audit_meta,
                    )
                logger.debug(
                    "VLM analyze failed for frame %d cand %d: %s",
                    frame_index, cand_idx, exc,
                )
                continue

            # Helper to log a decision against the *first* analyze call's result.
            def _log_first_call_decision(decision: str, note: str = "") -> None:
                if audit_log is not None:
                    audit_log.log_vlm_call(
                        latency_ms=self._last_vlm_latency_ms,
                        timed_out=self._last_vlm_timed_out,
                        vlm_response=result,
                        decision=decision,
                        decision_note=note,
                        snapshot_path="",
                        **audit_meta,
                    )

            has_violation = bool(result.get("has_violation", False))
            raw_behavior = str(result.get("behavior", "none"))
            behavior = normalize_behavior(raw_behavior)
            if not has_violation:
                _log_first_call_decision("rejected_no_violation")
                continue
            if behavior not in {"亂丟垃圾", "便溺"}:
                _log_first_call_decision(
                    "rejected_invalid_behavior",
                    f"raw={raw_behavior!r} normalized={behavior!r}",
                )
                continue
            # YOLO consistency override (anti-hallucination guard).
            #
            # Background: vendor 4/15 第3站 had 5 真便溺 systematically
            # mis-typed by VLM as 亂丟垃圾 with fabricated "yellow trash
            # bag" reasoning. The previous fix relabeled those events to
            # 便溺 whenever VLM said lit but YOLO frame-saw no litter.
            #
            # Why that fix backfires (case5 / vendor 4/27 reproduce
            # 2026-04-30): real lit events also frequently fail YOLO's
            # litter check — bags in flight are small, blurry, or occluded
            # for the exact frame the VLM analyzes. The relabel then
            # silently writes 便溺 to the CSV for genuine 亂丟 events
            # (case5: 3/3 events were mislabeled this way).
            #
            # Verdict: without 便溺-labelled training data the two
            # situations are indistinguishable from {VLM verdict, YOLO
            # boxes, confidence}. Both look like "VLM lit + YOLO no
            # litter". Recall is the priority per vendor PDFs (every
            # batch reports 0 violations), so the new default trusts
            # VLM's verdict. Type errors on 真便溺 → 亂丟 are recoverable
            # by human review; hard drops or silent relabels are not.
            #
            # 2026-04-30: relabel is now opt-in via ENFORCE_RELABEL_LIT_TO_URIN
            # (default False). Default = trust VLM. Set the env var to True
            # to restore the vendor-4/15 compensating behavior.
            relabeled = False
            if (
                behavior == "亂丟垃圾"
                and not yolo_saw_litter
                and self._relabel_lit_to_urin
                and detect_urination
            ):
                logger.info(
                    "frame %d cand %d: VLM said 亂丟垃圾 but YOLO saw 0 litter "
                    "→ relabel to 便溺 (ENFORCE_RELABEL_LIT_TO_URIN=on)",
                    frame_index, cand_idx,
                )
                behavior = "便溺"
                relabeled = True
            if behavior == "亂丟垃圾" and not detect_litter:
                _log_first_call_decision("rejected_disabled", "litter detection off")
                continue
            if behavior == "便溺" and not detect_urination:
                _log_first_call_decision("rejected_disabled", "urination detection off")
                continue

            confidence_raw = result.get("confidence", 0.0)
            confidence = (
                float(confidence_raw)
                if isinstance(confidence_raw, (int, float))
                else 0.0
            )
            if confidence < self._min_violation_confidence:
                _log_first_call_decision(
                    "rejected_low_confidence",
                    f"conf={confidence:.2f}<{self._min_violation_confidence:.2f}",
                )
                continue

            # 若 relabel 過（亂丟→便溺）：跳過 strict_confirm，把 confirm 設成
            # 原 result 讓下游 reason/litter_box fallback 仍有資料可取。理由：
            # (a) 原 VLM 已給高 conf，僅 type 是幻覺；YOLO 不見 litter 是更
            #     可信的 type 反證
            # (b) v7 實測：constrain VLM 到「便溺/none」會讓 conf 跌破 0.80
            #     threshold 而漏抓（recall 1/1 → 0/1）
            confirm: dict[str, object] = result
            if not relabeled:
                try:
                    confirm = self._run_vlm_analyze(
                        heartbeat=heartbeat,
                        frame_bgr=vlm_roi,
                        detect_litter=detect_litter,
                        detect_urination=detect_urination,
                        target_hint=target_hint,
                        strict_confirm=True,
                        expected_behavior=behavior,
                        is_mosaic=False,
                        context_frame_bgr=context_image,
                        hint_timestamp=ocr_timestamp,
                        hint_plate=ocr_plate,
                    )
                except VLMHardTimeout as exc:
                    if audit_log is not None:
                        audit_log.log_vlm_call(
                            latency_ms=self._last_vlm_latency_ms,
                            timed_out=True,
                            vlm_response=None,
                            decision="rejected_hard_timeout",
                            decision_note=f"strict_confirm: {str(exc)[:100]}",
                            snapshot_path="",
                            **audit_meta,
                        )
                    logger.warning(
                        "VLM confirm hard timeout for frame %d cand %d: %s",
                        frame_index, cand_idx, exc,
                    )
                    continue
                except Exception as exc:
                    if audit_log is not None:
                        audit_log.log_vlm_call(
                            latency_ms=self._last_vlm_latency_ms,
                            timed_out=False,
                            vlm_response=None,
                            decision="rejected_exception",
                            decision_note=f"strict_confirm: {str(exc)[:100]}",
                            snapshot_path="",
                            **audit_meta,
                        )
                    logger.debug(
                        "VLM confirm failed for frame %d cand %d: %s",
                        frame_index,
                        cand_idx,
                        exc,
                    )
                    continue
                c_has = bool(confirm.get("has_violation", False))
                c_raw_behavior = str(confirm.get("behavior", "none"))
                c_behavior = normalize_behavior(c_raw_behavior)
                c_conf_raw = confirm.get("confidence", 0.0)
                c_conf = float(c_conf_raw) if isinstance(c_conf_raw, (int, float)) else 0.0
                if (
                    not c_has
                    or c_behavior != behavior
                    or c_conf < self._confirm_violation_confidence
                ):
                    if audit_log is not None:
                        audit_log.log_vlm_call(
                            latency_ms=self._last_vlm_latency_ms,
                            timed_out=self._last_vlm_timed_out,
                            vlm_response=confirm,
                            decision="rejected_strict_confirm_disagree",
                            decision_note=f"first_conf={confidence:.2f},confirm_conf={c_conf:.2f},confirm_raw={c_raw_behavior!r},confirm_norm={c_behavior!r}",
                            snapshot_path="",
                            **audit_meta,
                        )
                    continue
                confidence = min(confidence, c_conf)

            if confidence < self._min_violation_confidence:
                if audit_log is not None:
                    audit_log.log_vlm_call(
                        latency_ms=self._last_vlm_latency_ms,
                        timed_out=self._last_vlm_timed_out,
                        vlm_response=confirm,
                        decision="rejected_low_confidence",
                        decision_note=f"post_confirm_conf={confidence:.2f}<{self._min_violation_confidence:.2f}",
                        snapshot_path="",
                        **audit_meta,
                    )
                continue
            if best is not None and confidence <= best.confidence:
                continue

            frame_time = frame_time_base
            # Priority for overlay_time (畫面時間):
            # 1. File-metadata wall time (filename parse or mtime - duration)
            # 2. pytesseract OCR result (only in dev env with tesseract installed)
            # 3. Dedicated focused VLM call on the zoomed timestamp region
            # 4. Relative frame offset (last resort)
            if computed_wall_time:
                overlay_time = computed_wall_time
            elif ocr_timestamp:
                overlay_time = ocr_timestamp
            else:
                vlm_event_time = str(result.get("event_time", "")).strip()
                if vlm_event_time:
                    overlay_time = vlm_event_time
                else:
                    if not _vlm_ts:
                        _vlm_ts.append(self._vlm_read_timestamp(frame))
                    overlay_time = _vlm_ts[0] or frame_time_base
            plate = _strip_think_tags(str(result.get("plate", ""))).strip()
            plate = _clean_text(plate)
            if plate in {"無法辨識", "無法辨識填空字串", "不詳", "unknown", "未知", "無"}:
                plate = ""
            if not plate:
                if ocr_plate:
                    plate = ocr_plate
                else:
                    # Dedicated VLM plate call with focused horizontal band
                    plate = self._vlm_read_plate(frame, candidate)
            # Use the model's actual evidence description; fall back to template
            reason_raw = _clean_text(str(result.get("reason", "") or ""))
            confirm_reason = _clean_text(str(confirm.get("reason", "") or ""))
            if reason_raw:
                reason = reason_raw
                if confirm_reason and confirm_reason != reason_raw:
                    reason = f"{reason_raw}；二次確認：{confirm_reason}"
            else:
                reason = compose_traditional_reason(
                    behavior=behavior,
                    confidence=confidence,
                    has_violation=True,
                )
            litter_box = self._parse_litter_box(
                result.get("litter_box"), getattr(roi, "shape", None)
            )
            if litter_box is None:
                litter_box = self._parse_litter_box(
                    confirm.get("litter_box"), getattr(roi, "shape", None)
                )
            # 第三順位：偵測器實際看到的垃圾框。只在 Co-DETR litter 來源啟用時才有值
            # （原本的 prefilter 是 stock COCO yolov8n，沒有 litter 類別，永遠是空的）。
            # 兩次 VLM 都沒指出垃圾位置時，偵測器的框是比 VLM 自陳更可信的空間證據。
            if litter_box is None and self._codetr_litter is not None:
                litter_box = self._codetr_litter_box_for_roi(
                    frame, candidate, frame_litter_boxes
                )
                if litter_box is not None:
                    logger.info(
                        "frame %d cand %d: VLM 沒給 litter_box，改用 Co-DETR 的框 %s",
                        frame_index, cand_idx, litter_box,
                    )
            # Lit candidates without a litter_box must clear an even higher
            # confidence bar (precision guard for VLMs that say "亂丟垃圾"
            # without identifying the trash spatially). The +0.12 buffer
            # is capped at 0.99 so the gate stays feasible after the
            # 2026-04-30 base bump to 0.90 (raw 0.90 + 0.12 = 1.02 would
            # never pass since VLM confidence is bounded at 1.0).
            if (
                behavior == "亂丟垃圾"
                and litter_box is None
                and confidence
                < min(
                    max(
                        self._min_violation_confidence + 0.12,
                        0.72,
                    ),
                    0.99,
                )
            ):
                if audit_log is not None:
                    audit_log.log_vlm_call(
                        latency_ms=self._last_vlm_latency_ms,
                        timed_out=self._last_vlm_timed_out,
                        vlm_response=confirm,
                        decision="rejected_no_litter_box",
                        decision_note=f"conf={confidence:.2f} below litter_box-required threshold",
                        snapshot_path="",
                        **audit_meta,
                    )
                continue
            snap_img = self._render_annotated_snapshot(
                roi=roi,
                behavior=behavior,
                confidence=confidence,
                reason=reason,
                litter_box=litter_box,
            )

            frame_time_safe = frame_time.replace(":", "-")
            snap_name = f"違規截圖_{frame_time_safe}_第{cand_idx + 1:02d}號.jpg"
            snap_path = snap_dir / snap_name
            snap_ok = bool(cv2_mod.imwrite(str(snap_path), snap_img))
            if not snap_ok:
                logger.warning("snapshot write failed: %s", snap_path)

            best = ViolationEvent(
                frame_index=frame_index,
                frame_time=frame_time,
                overlay_time=overlay_time,
                plate=plate,
                behavior=behavior,
                confidence=confidence,
                reason=reason,
                snapshot=str(snap_path) if snap_ok else "",
                candidate_box=(
                    int(candidate.get("x1", 0)),
                    int(candidate.get("y1", 0)),
                    int(candidate.get("x2", 0)),
                    int(candidate.get("y2", 0)),
                ),
            )
            if audit_log is not None:
                audit_log.log_event_kept(
                    frame_index=frame_index,
                    frame_time_sec=frame_index / fps if fps > 0 else 0.0,
                    overlay_time=overlay_time,
                    candidate_label=target_hint,
                    candidate_box=cand_box,
                    behavior=behavior,
                    confidence=confidence,
                    plate=plate,
                    snapshot_path=best.snapshot,
                )
        return best

    def _attach_litter_clips(
        self,
        video_path: Path,
        events: list[ViolationEvent],
        fps: float,
        snap_dir: Path,
    ) -> list[ViolationEvent]:
        if not events:
            return []
        clip_dir = snap_dir.parent / "違規片段"
        out: list[ViolationEvent] = []
        for event in events:
            if event.behavior != "亂丟垃圾":
                out.append(event)
                continue
            clip_path = self._export_event_clip(
                video_path=video_path,
                event=event,
                fps=fps,
                clip_dir=clip_dir,
            )
            out.append(replace(event, clip=clip_path))
        return out

    def _export_event_clip(
        self,
        video_path: Path,
        event: ViolationEvent,
        fps: float,
        clip_dir: Path,
    ) -> str:
        """匯出違規片段。所有 cv2 資源保證 release（外層 try/finally）。

        2026-04-23 回報：長批次跑 36+ 小時後系統中斷，疑似累積記憶體洩漏。
        本函式原有多個早期 return 點 + 內部 try/finally，新增 writer candidate
        選擇迴圈若在異常路徑下可能漏 release。改為單一 try/finally 包住 cap
        與 writer 生命週期，任何路徑（正常回傳 / 早期 return / raise）都會被
        清理。
        """
        cv2_mod = importlib.import_module("cv2")
        cap = cv2_mod.VideoCapture(str(video_path))
        writer: Any = None
        written = 0
        clip_path: Path | None = None
        try:
            if not cap.isOpened():
                logger.warning("clip export failed, cannot open video: %s", video_path)
                return ""

            src_fps_raw = cap.get(cv2_mod.CAP_PROP_FPS)
            src_fps = (
                float(src_fps_raw) if src_fps_raw and src_fps_raw > 0
                else max(1.0, fps)
            )
            total_frames = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT) or 0)
            before = int(max(0.0, self._litter_clip_before_sec) * src_fps)
            after = int(max(0.0, self._litter_clip_after_sec) * src_fps)
            start_f = max(0, event.frame_index - before)
            end_f = max(start_f, event.frame_index + after)
            if total_frames > 0:
                end_f = min(end_f, total_frames - 1)

            cap.set(cv2_mod.CAP_PROP_POS_FRAMES, start_f)
            ok, first = cap.read()
            if not ok:
                logger.warning(
                    "clip export failed, cannot read start frame %d from %s",
                    start_f,
                    video_path,
                )
                return ""
            shape = getattr(first, "shape", None)
            if not isinstance(shape, tuple) or len(shape) < 2:
                logger.warning(
                    "clip export failed, invalid frame shape for %s", video_path
                )
                return ""

            h = int(shape[0])
            w = int(shape[1])
            if h <= 0 or w <= 0:
                logger.warning(
                    "clip export failed, invalid frame size for %s", video_path
                )
                return ""

            clip_dir.mkdir(parents=True, exist_ok=True)
            clip_time_safe = event.frame_time.replace(":", "-")
            clip_path = clip_dir / f"違規片段_{clip_time_safe}.mp4"
            for codec in ("mp4v", "avc1", "MJPG", "XVID"):
                fourcc = cv2_mod.VideoWriter_fourcc(*codec)
                candidate = cv2_mod.VideoWriter(
                    str(clip_path), fourcc, src_fps, (w, h)
                )
                opened = True
                if hasattr(candidate, "isOpened"):
                    opened = bool(candidate.isOpened())
                if opened:
                    writer = candidate
                    break
                # 每個失敗的 candidate 都要 release，避免底層句柄累積。
                try:
                    candidate.release()
                except Exception:
                    pass
            if writer is None:
                logger.warning(
                    "clip export failed, cannot open writer for %s", clip_path
                )
                return ""

            max_frames_unknown = int(
                max(1.0, self._litter_clip_before_sec + self._litter_clip_after_sec + 1.0)
                * src_fps
            )
            frame = first
            cur_f = start_f
            while True:
                try:
                    frame = _put_chinese_text(
                        frame,
                        "亂丟垃圾事件",
                        (12, 28),
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                except Exception:
                    pass
                writer.write(frame)
                written += 1

                if total_frames > 0:
                    if cur_f >= end_f:
                        break
                elif written >= max_frames_unknown:
                    break

                ok, nxt = cap.read()
                if not ok:
                    break
                frame = nxt
                cur_f += 1
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            try:
                cap.release()
            except Exception:
                pass

        if written <= 0:
            logger.warning(
                "clip export produced empty output: %s",
                clip_path if clip_path is not None else "(no path)",
            )
            return ""
        return str(clip_path) if clip_path is not None else ""

    def _run_vlm_analyze(
        self,
        *,
        heartbeat: "HeartbeatEmitter | None" = None,
        **kwargs: Any,
    ) -> dict[str, Any]:  # type: ignore[override]
        """Wrap a single VLM analyze call with telemetry + slot acquisition.

        After the call (success or failure) we update self._last_vlm_latency_ms
        and self._last_vlm_timed_out so _analyze_candidates can read them when
        building audit log rows without changing the analyze return shape.
        Heartbeat fires either way so the GUI / watchdog see progress.
        """
        with self._vlm_slots:
            t0 = time.monotonic()
            timed_out = False
            try:
                return self._vlm.analyze(**kwargs)
            except VLMHardTimeout:
                timed_out = True
                raise
            finally:
                latency_ms = int((time.monotonic() - t0) * 1000)
                self._last_vlm_latency_ms = latency_ms
                self._last_vlm_timed_out = timed_out
                if heartbeat is not None:
                    heartbeat.emit_vlm_call(latency_ms, timed_out=timed_out)

    def _write_csv(
        self,
        video_name: str,
        out_csv_path: Path,
        events: list[ViolationEvent],
    ) -> None:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with out_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "來源影片",
                    "幀時間",
                    "畫面時間",
                    "車牌",
                    "違規行為",
                    "信心分數",
                    "判定理由",
                    "截圖路徑",
                    "影片片段路徑",
                ]
            )
            for event in events:
                writer.writerow(
                    [
                        video_name,
                        event.frame_time,
                        event.overlay_time,
                        event.plate,
                        event.behavior,
                        f"{event.confidence:.3f}",
                        event.reason,
                        event.snapshot,
                        event.clip,
                    ]
                )
