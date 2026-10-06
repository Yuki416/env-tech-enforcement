"""「測試」模式 —— 純 Co-DETR 的違規偵測流程。

## 跟「完整」模式的差別

| | 完整 | 測試 |
|---|---|---|
| 流程 | YOLO 找人車 → VLM 判定行為 → 事件合併 | **整支影片餵 Co-DETR** |
| 違規的定義 | VLM 說「亂丟垃圾／便溺」 | **Co-DETR 偵測到 litter/garbage 的幀** |
| 畫面上的框 | 只畫違規候選 | **全部 5 類都畫**（見下方說明） |
| 用到 VLM | 是 | **否** |
| 目的 | 正式執法流程 | **展示第 18 次訓練的 Co-DETR 成果** |

「完整」流程目前有多個環節還不能用（VLM 未微調、litter YOLO 是 stock COCO、
prompt 要求單幀給不出的時序證據），所以先做這個模式把偵測器的成果單獨秀出來。

**這個模組完全獨立，不動 `RealEngine` 的任何邏輯。**

## 「畫全部類別」與「違規判定」是兩件事

模型有 5 類：`car / litter / motorcycle / person / garbage`。
截圖與片段上**五類全畫**，但只有 `litter/garbage` 會構成違規事件
（`_TRIGGER_CLASSES`）—— 否則每個有人或車經過的幀都會被判成違規，事件表會被灌爆。

情境類別（人／車／機車）的作用是讓畫面有脈絡：看得到是誰、在哪台車旁邊丟的。
它們用冷色細框、且畫在違規框底下，不跟證據搶視覺重量。

## 輸出格式刻意與完整模式一致

事件 dict 的欄位照 `ViolationEvent.to_dict()`：
`frame_index / frame_time / overlay_time / plate / behavior / confidence / reason /
snapshot / clip / candidate_box`。
這樣網頁既有的事件表、片段牆、播放器、截圖 modal、稽查註記、CSV 匯出全部直接沿用，
不用為了新模式再改一次前端。

## 片段用 H.264，不用 cv2 的 mp4v

`vlm_enforcement._export_event_clip` 的 codec 順序是 `mp4v` 優先，那是 MPEG-4 Part 2，
**瀏覽器的 `<video>` 不播**。完整模式從來沒產出過事件，所以這個問題一直沒被發現。
這裡直接把幀 pipe 給 ffmpeg 用 libx264 編，並加 `+faststart` 讓串流可以邊下邊播。
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

logger = logging.getLogger(__name__)

# 一次向 sidecar 要幾幀。太小 → HTTP 往返成本高；太大 → 進度回報變粗。
_BATCH = 16
# 兩個命中幀相隔超過這個秒數就切成不同事件。
_EVENT_GAP_SEC = 2.0
# 片段在事件前後各延伸幾秒。
_CLIP_BEFORE_SEC = 3.0
_CLIP_AFTER_SEC = 3.0
# 片段最長幾秒，避免長事件產生巨大檔案。
_CLIP_MAX_SEC = 30.0

_LABEL_ZH = {"litter": "亂丟垃圾", "garbage": "垃圾堆置"}

# 第 18 次訓練的 Co-DETR 有 5 個類別（codetr_service/server.py:56 的 CLASSES）。
# 其中只有 litter/garbage 會「觸發」違規事件，car/motorcycle/person 純粹是情境資訊
# —— 否則每個有人或車經過的幀都會變成違規，事件表會被灌爆。
_TRIGGER_CLASSES = {"litter", "garbage"}

# BGR。觸發類別用暖色（紅/橘）搶眼，情境類別用冷色，一眼分得出哪個框是違規證據。
_BOX_COLOR = {
    "litter": (0, 0, 255),        # 紅
    "garbage": (0, 165, 255),     # 橘
    "person": (0, 200, 0),        # 綠
    "car": (255, 160, 0),         # 藍
    "motorcycle": (255, 0, 200),  # 紫
}
# 情境類別的框線細一點，才不會在舉證畫面上跟違規證據搶視覺重量。
_CONTEXT_THICKNESS_RATIO = 0.6

# 框線與標籤都隨影像寬度縮放（監視器影片 2592px，但播放器上只有約 1000px）。
# 原本固定 4px 太粗會蓋掉小垃圾的輪廓；改成寬度/1300 → 2592px 影片得到 2px。
# 覺得還是太細/太粗就調這兩個數字，其他都不用動。
_BOX_THICKNESS_DIV = 1300
_LABEL_SCALE_DIV = 2900


@dataclass
class _Hit:
    """一個命中幀。

    `boxes` 是該幀「全部類別」的偵測框（畫在影片上用）；
    `trigger_boxes` 只有 litter/garbage（決定違規事件、代表幀、信心分數用）。
    兩者刻意分開 —— 畫面要資訊完整，判定要維持原本的語意。
    """

    frame_index: int
    boxes: list[dict[str, Any]]

    @property
    def trigger_boxes(self) -> list[dict[str, Any]]:
        return [b for b in self.boxes if str(b.get("label", "")) in _TRIGGER_CLASSES]

    @property
    def score(self) -> float:
        """事件信心 = 觸發類別的最高分。情境類別（人車）不計入。"""
        return max(
            (float(b.get("score", 0.0)) for b in self.trigger_boxes), default=0.0
        )


@dataclass
class _Segment:
    """連續命中幀合併成的一個違規片段。"""

    hits: list[_Hit] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.hits[0].frame_index

    @property
    def end(self) -> int:
        return self.hits[-1].frame_index

    def peak(self) -> _Hit:
        """分數最高的那一幀 —— 拿來當代表幀與截圖。"""
        return max(self.hits, key=lambda h: h.score)


class CoDetrTestPipeline:
    """整支影片掃 Co-DETR，把命中幀合併成違規片段。"""

    def __init__(
        self,
        base_url: str,
        score_thr: float = 0.30,
        stride: int = 1,
        timeout_sec: float = 900.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._score_thr = float(score_thr)
        self._stride = max(1, int(stride))
        self._timeout = float(timeout_sec)

    # ── sidecar ──────────────────────────────────────────────────
    def _detect(self, video: str, frames: list[int]) -> list[list[dict[str, Any]]]:
        body = json.dumps(
            {
                "video": video,
                "frames": frames,
                "score_thr": self._score_thr,
                # 要回全部 5 類：畫面上要看到人車機車的框。
                # 違規判定仍只看 litter/garbage，在本模組這端過濾（見 _TRIGGER_CLASSES）。
                "all_classes": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/detect",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as fh:
            resp = json.loads(fh.read())
        boxes = resp.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != len(frames):
            raise RuntimeError(f"sidecar 回傳長度不符：要 {len(frames)} 幀")
        return boxes

    def health(self) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(f"{self._base}/health", timeout=10.0) as fh:
                return json.loads(fh.read())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Co-DETR health check 失敗：%s", exc)
            return None

    def release(self, video: str) -> None:
        try:
            req = urllib.request.Request(
                f"{self._base}/release",
                data=json.dumps({"video": video}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10.0).read()
        except Exception:  # noqa: BLE001
            pass

    # ── 主流程 ───────────────────────────────────────────────────
    def run(
        self,
        video_path: Path,
        out_dir: Path,
        progress: Callable[[dict[str, Any]], None] | None = None,
        log: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        video = str(video_path)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        def emit(msg: str, level: str = "info") -> None:
            if log is not None:
                log(msg, level)

        scan_frames = list(range(0, total, self._stride))
        emit(
            f"[CODETR] {video_path.name}：{total} 幀 / {fps:.0f} fps，"
            f"抽幀間隔 {self._stride} → 需掃 {len(scan_frames)} 幀"
            f"（約 {len(scan_frames) * 0.42 / 60:.1f} 分鐘）"
        )

        # ── 掃描 ──
        # all_boxes：每個掃過的幀的「全部類別」框，給影片逐幀畫框用。包含只有人車、
        # 沒有垃圾的幀 —— 事件前後緩衝那幾秒才不會整片空白，畫面才連貫。
        # hits：只收含 litter/garbage 的幀。違規事件的合併與計數一律以它為準。
        hits: list[_Hit] = []
        all_boxes: dict[int, list[dict[str, Any]]] = {}
        n_context_only = 0
        t0 = time.monotonic()
        for i in range(0, len(scan_frames), _BATCH):
            chunk = scan_frames[i : i + _BATCH]
            for per_frame, fi in zip(self._detect(video, chunk), chunk):
                if not per_frame:
                    continue
                all_boxes[fi] = per_frame
                hit = _Hit(frame_index=fi, boxes=per_frame)
                if hit.trigger_boxes:
                    hits.append(hit)
                else:
                    n_context_only += 1
            if progress is not None:
                done = min(i + _BATCH, len(scan_frames))
                progress(
                    {
                        "stage": "codetr",
                        "progress": done / max(1, len(scan_frames)) * 0.85,
                        "frame": chunk[-1],
                        "total_frames": total,
                    }
                )
        scan_sec = time.monotonic() - t0
        emit(
            f"[CODETR] 掃描完成 {scan_sec:.0f}s："
            f"{len(hits)}/{len(scan_frames)} 幀偵測到 litter/garbage（觸發違規）、"
            f"另有 {n_context_only} 幀只偵測到人／車／機車（僅畫框，不觸發違規）"
        )

        # ── 合併成片段 ──
        segments = self._group(hits, fps)
        emit(f"[CODETR] 合併為 {len(segments)} 個違規片段")

        # ── 產出證物 ──
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = out_dir / "違規截圖"
        clip_dir = out_dir / "違規片段"
        snap_dir.mkdir(exist_ok=True)
        clip_dir.mkdir(exist_ok=True)

        events: list[dict[str, Any]] = []
        for idx, seg in enumerate(segments, start=1):
            if progress is not None:
                progress(
                    {
                        "stage": "merging",
                        "progress": 0.85 + 0.15 * idx / max(1, len(segments)),
                        "frame": seg.end,
                        "total_frames": total,
                    }
                )
            ev = self._build_event(
                video_path, seg, idx, fps, snap_dir, clip_dir, all_boxes
            )
            if ev is not None:
                events.append(ev)

        self._write_csv(out_dir / "違規紀錄.csv", video_path.name, events)
        self._write_scan_log(out_dir / "codetr_scan.csv", all_boxes, fps)
        self.release(video)

        return {
            "events": events,
            "csv": str(out_dir / "違規紀錄.csv"),
            "snap_dir": str(snap_dir),
            "output_dir": str(out_dir),
            "scanned_frames": len(scan_frames),
            "hit_frames": len(hits),
            "context_only_frames": n_context_only,
            "scan_sec": scan_sec,
        }

    # ── 內部 ─────────────────────────────────────────────────────
    def _group(self, hits: list[_Hit], fps: float) -> list[_Segment]:
        """相鄰命中幀間隔在 _EVENT_GAP_SEC 內就算同一個事件。"""
        if not hits:
            return []
        gap = max(1, int(_EVENT_GAP_SEC * fps))
        segments: list[_Segment] = [_Segment(hits=[hits[0]])]
        for hit in hits[1:]:
            if hit.frame_index - segments[-1].end <= gap:
                segments[-1].hits.append(hit)
            else:
                segments.append(_Segment(hits=[hit]))
        return segments

    def _build_event(
        self,
        video_path: Path,
        seg: _Segment,
        idx: int,
        fps: float,
        snap_dir: Path,
        clip_dir: Path,
        all_boxes: dict[int, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        peak = seg.peak()
        t_sec = peak.frame_index / fps
        frame_time = _fmt_time(t_sec)
        # 代表框只從觸發類別挑 —— 否則旁邊一台信心 0.99 的車會蓋過信心 0.35 的垃圾，
        # 事件的 behavior 與 candidate_box 就會指到車上。
        best = max(peak.trigger_boxes, key=lambda b: float(b.get("score", 0.0)))
        label = str(best.get("label", "litter"))

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak.frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            logger.warning("讀不到代表幀 %d，跳過此事件", peak.frame_index)
            return None

        snap_name = f"違規截圖_{frame_time.replace(':', '-')}_第{idx:02d}號.jpg"
        snap_path = snap_dir / snap_name
        cv2.imwrite(str(snap_path), _draw_boxes(frame, peak.boxes), [cv2.IMWRITE_JPEG_QUALITY, 90])

        clip_name = f"違規片段_{frame_time.replace(':', '-')}_第{idx:02d}號.mp4"
        clip_path = clip_dir / clip_name
        # 片段逐幀畫上「該幀自己的」偵測框 —— 框會跟著目標移動，不是整段套同一組座標。
        # 用 all_boxes 而非 seg.hits：後者只有含垃圾的幀，前後緩衝那幾秒會完全空白。
        # 傳整份 dict 就好，超出片段範圍的鍵不會被查到。
        boxes_by_frame = all_boxes
        clip_ok, clip_first = _export_clip_h264(
            video_path, seg.start, seg.end, fps, clip_path, boxes_by_frame
        )

        n_frames = len(seg.hits)
        dur = (seg.end - seg.start) / fps
        reason = (
            f"Co-DETR（第 18 次訓練，v18 BGS 4ch）在第 {seg.start}~{seg.end} 幀"
            f"（共 {n_frames} 幀、持續 {dur:.1f} 秒）偵測到 {label}，"
            f"最高信心 {peak.score:.2f}。"
            "※ 測試模式：違規判定僅來自偵測器，未經 VLM 行為判讀。"
        )
        return {
            "frame_index": peak.frame_index,
            "frame_time": frame_time,
            "overlay_time": "",
            "plate": "",
            "behavior": _LABEL_ZH.get(label, label),
            "confidence": round(peak.score, 3),
            "reason": reason,
            "snapshot": str(snap_path),
            "clip": str(clip_path) if clip_ok else "",
            # 片段是從「事件前 _CLIP_BEFORE_SEC 秒」開始剪的，所以事件在片段裡的位置
            # 不等於它在原始影片裡的時間。前端要靠這兩個值把紅旗標到正確位置、
            # 以及把播放秒數換算成幀號。
            "clip_start_sec": round(clip_first / fps, 3) if clip_ok else 0.0,
            "clip_fps": round(fps, 3),
            # 違規區間本身的起訖（原始影片座標）。代表幀 frame_time 只是「分數最高」
            # 的那一幀，跟區間的頭尾是兩回事，介面上要分開顯示。
            "seg_start_sec": round(seg.start / fps, 3),
            "seg_end_sec": round(seg.end / fps, 3),
            "seg_start_time": _fmt_time(seg.start / fps),
            "seg_end_time": _fmt_time(seg.end / fps),
            "candidate_box": [
                int(best.get("x1", 0)),
                int(best.get("y1", 0)),
                int(best.get("x2", 0)),
                int(best.get("y2", 0)),
            ],
        }

    def _write_csv(self, path: Path, video_name: str, events: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["來源影片", "幀時間", "違規起始", "違規結束", "畫面時間", "車牌",
                 "違規行為", "信心分數", "判定理由", "截圖路徑", "影片片段路徑"]
            )
            for ev in events:
                w.writerow([
                    video_name, ev["frame_time"], ev["seg_start_time"], ev["seg_end_time"],
                    ev["overlay_time"], ev["plate"], ev["behavior"], ev["confidence"],
                    ev["reason"], ev["snapshot"], ev["clip"],
                ])

    def _write_scan_log(
        self, path: Path, all_boxes: dict[int, list[dict[str, Any]]], fps: float
    ) -> None:
        """逐幀流水帳，對應完整模式的 vlm_decisions.csv —— 漏抓時可回頭查。

        現在含全部 5 類。`triggers_event` 欄標出哪些框才會構成違規事件，
        免得日後看到滿滿的 person 列以為判定邏輯壞掉。
        """
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["frame_index", "frame_time_sec", "label", "score", "box",
                 "triggers_event"]
            )
            for fi in sorted(all_boxes):
                for b in all_boxes[fi]:
                    label = str(b.get("label", ""))
                    w.writerow([
                        fi, f"{fi / fps:.2f}",
                        label, round(float(b.get("score", 0.0)), 4),
                        f"{b.get('x1')},{b.get('y1')},{b.get('x2')},{b.get('y2')}",
                        "1" if label in _TRIGGER_CLASSES else "0",
                    ])


def _fmt_time(sec: float) -> str:
    total = int(sec)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _draw_boxes(frame: Any, boxes: list[dict[str, Any]], inplace: bool = False) -> Any:
    """畫 Co-DETR 的偵測框（全部 5 類）。截圖與片段共用同一套外觀。

    標籤用實心底色塊 + 白字：純彩色文字在淺色路面或深色陰影上都會糊掉，
    而這是舉證要用的畫面，類別和分數必須任何情況下都讀得出來。

    情境類別（person/car/motorcycle）線畫細一點，而且先畫、讓觸發類別
    （litter/garbage）疊在最上層 —— 路口畫面常常人車重疊，違規證據不能被蓋掉。
    標籤文字用英文類別名：cv2.putText 畫不出中文，硬塞會變成一排問號。
    """
    out = frame if inplace else frame.copy()
    h, w = out.shape[:2]
    th = max(1, round(w / _BOX_THICKNESS_DIV))
    fs = max(0.45, w / _LABEL_SCALE_DIV)
    ft = max(1, round(th * 0.8))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 情境框先畫、觸發框後畫（後畫的疊在上層）
    ordered = sorted(
        boxes, key=lambda b: str(b.get("label", "")) in _TRIGGER_CLASSES
    )
    for b in ordered:
        label = str(b.get("label", ""))
        is_trigger = label in _TRIGGER_CLASSES
        color = _BOX_COLOR.get(label, (0, 0, 255))
        bth = th if is_trigger else max(1, round(th * _CONTEXT_THICKNESS_RATIO))
        x1, y1 = int(b.get("x1", 0)), int(b.get("y1", 0))
        x2, y2 = int(b.get("x2", 0)), int(b.get("y2", 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, bth)

        text = f"{b.get('label', '')} {float(b.get('score', 0.0)):.2f}"
        (tw, tht), _ = cv2.getTextSize(text, font, fs, ft)
        pad = max(3, bth * 2)
        # 標籤預設放在框上方；貼近畫面頂端時改放框內，避免被裁掉
        ly2 = y1 - bth if y1 - tht - pad * 2 > 0 else y1 + tht + pad * 2
        ly1 = ly2 - tht - pad * 2
        lx2 = min(w, x1 + tw + pad * 2)
        cv2.rectangle(out, (x1, max(0, ly1)), (lx2, max(0, ly2)), color, -1)
        cv2.putText(out, text, (x1 + pad, max(tht, ly2 - pad)),
                    font, fs, (255, 255, 255), ft, cv2.LINE_AA)
    return out


def _export_clip_h264(
    video_path: Path,
    start: int,
    end: int,
    fps: float,
    out_path: Path,
    boxes_by_frame: dict[int, list[dict[str, Any]]] | None = None,
) -> tuple[bool, int]:
    """把 [start-before, end+after] 這段用 libx264 編出來。

    刻意不用 `cv2.VideoWriter` —— 它在這個 build 上會拿到 mp4v（MPEG-4 Part 2），
    瀏覽器的 `<video>` 不播。改成把 BGR 原始幀 pipe 給 ffmpeg。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        return False, 0

    first = max(0, start - int(_CLIP_BEFORE_SEC * fps))
    last = min(total - 1 if total > 0 else end, end + int(_CLIP_AFTER_SEC * fps))
    last = min(last, first + int(_CLIP_MAX_SEC * fps))

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.3f}", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        # yuv420p + faststart：前者是瀏覽器解碼的硬需求，後者讓 moov box 在檔頭，
        # 才能邊下載邊播（時間軸拖曳要靠這個）。
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for offset in range(last - first + 1):
            ok, frame = cap.read()
            if not ok:
                break
            boxes = (boxes_by_frame or {}).get(first + offset)
            if boxes:
                # inplace：cap.read() 每次都給新的 buffer，不必再複製一份
                _draw_boxes(frame, boxes, inplace=True)
            proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError) as exc:
        logger.warning("片段寫入中斷：%s", exc)
    finally:
        cap.release()
    # 不要用 communicate()：stdin 已經自己關過，communicate 會再 flush 一次
    # 已關閉的檔案而拋 ValueError("flush of closed file")。
    # stderr 用 -loglevel error，輸出量遠小於 pipe 容量，關 stdin 後再讀不會卡死。
    try:
        proc.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    err = proc.stderr.read() or b""
    proc.stderr.close()
    proc.wait()
    if proc.returncode != 0:
        logger.warning("ffmpeg 失敗 (%s)：%s", proc.returncode, err.decode("utf-8", "replace")[:300])
        return False, first
    ok = out_path.is_file() and out_path.stat().st_size > 0
    return ok, first
