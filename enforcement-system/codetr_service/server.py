"""第 18 次訓練的 Co-DETR（資料集 v18，BGS 4ch）litter 偵測 sidecar 服務。

**跑在 `co-detr:latest` image 裡，不是 enforcement image 裡。**
enforcement image 是 python:3.12-slim + torch + ultralytics，沒有 mmdet/mmcv；
把 mmcv 塞進去要對特定 torch 版本編譯，風險高。改成獨立服務，用 HTTP 溝通
（系統本來就已經用 HTTP 跟 ollama 說話，這個模式是現成的）。

## 為什麼是這個服務自己讀影片，而不是由 enforcement 傳幀過來

第 18 次的模型吃 4 通道：RGB + KNN background subtraction 的 changemap。
changemap 必須**循序**餵幀才建得出背景模型，但 enforcement 的 prefilter 只會
把「每 stride 幀」抽出來送批次 —— 稀疏幀餵 KNN 等於沒有背景。

所以這裡自己開同一支影片：先循序暖機建背景，再依 frame index 取幀推論。
兩個容器掛同一份 /videos、/testset，路徑一致，座標也就一致（都用原生解析度）。

## changemap 參數與訓練時完全一致

沿用 `co-detr/experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/gen_changemaps_v18.py`：
KNN(history=500, dist2Threshold=500, detectShadows=True)、half-res 算再 INTER_NEAREST 升回、
rm5 移除 <5px 碎塊、pass2 用 learningRate=0 凍結背景。

**唯一的偏離**：訓練時 pass1 暖機餵的是「事件前」的 pre-roll 幀（離線才知道事件在哪）；
線上不知道事件位置，改餵**影片開頭 WARM 幀**。這是因果的、可部署的，也是第 18 次 README
選這個方法的理由。代價是長影片（兩小時、有光照變化）凍結背景會漂 —— 見 README 的實測。

啟動：
    python /app/codetr_service/server.py --port 8091
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [codetr-svc] %(message)s",
)
logger = logging.getLogger("codetr_service")


# ── PyTorch >= 2.6 讀舊 checkpoint 的相容性處理 ──────────────────────────────
# 症狀：載入第 18 次訓練的 .pth 時噴
#   _pickle.UnpicklingError: Weights only load failed ...
#
# 原因：PyTorch 2.6 起 torch.load() 的 weights_only 預設由 False 改為 True，
# 會擋掉 checkpoint 裡序列化的非純 tensor 物件。我們的 .pth 是舊版 mmengine 存的，
# 內含 mmengine.logging.history_buffer.HistoryBuffer 與 numpy 的重建函式。
# 新環境 co-detr:blackwell 用 torch 2.7.1，因此會踩到。
#
# 為什麼不用 torch.serialization.add_safe_globals() 逐項白名單（已實測失敗）：
#   HistoryBuffer 可以順利加進白名單，但 numpy 那項不行 —— numpy 2.x 已把內部模組
#   由 numpy.core 改名為 numpy._core，而 checkpoint 裡序列化的字串仍是舊的
#   "numpy.core.multiarray._reconstruct"，與現在物件的實際 __module__ 對不起來，
#   註冊了也匹配不到。逐項白名單在這個組合下無解，且會變成打地鼠。
#
# 因此改為還原 torch < 2.6 的行為：weights_only 預設 False。
#
# ⚠️ 安全性取捨：weights_only=False 會在反序列化時允許執行 checkpoint 內嵌的程式碼，
# 對「來路不明的 .pth」是實際風險。這裡可以接受，因為本服務只載入我們自己訓練產出、
# 存放在自己伺服器上的 checkpoint（路徑見下方 RESULT_DIR）。
# **若日後要改成載入外部來源的 checkpoint，必須重新評估這個決定。**
#
# 同樣的問題會出現在任何用 torch>=2.6 讀舊 checkpoint 的地方（例如 tools/train.py
# 的 --resume、mmengine Runner 內部載 checkpoint），需比照處理。
def _restore_legacy_torch_load() -> None:
    import functools

    import torch

    # torch < 2.6 沒有這個預設值變更，不需處理
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
        return

    if getattr(torch.load, "_legacy_weights_only_patched", False):
        return  # 避免重複套用

    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._legacy_weights_only_patched = True
    torch.load = _load
    logger.info(
        "已將 torch.load 的 weights_only 預設改回 False（torch %s，相容舊 mmengine checkpoint）",
        torch.__version__,
    )


_restore_legacy_torch_load()

# ── 模型與資料集常數（照抄 infer_18th_v18_4ch_video_classification.py）─────────
CONFIG = "/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"
RESULT_DIR = "/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e"
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
LITTER_CLASSES = {"litter", "garbage"}
IMAGE_SIZE = (1024, 1024)
PAD4 = (114, 114, 114, 0)  # 第 4 通道 pad 0，不是 114

# ── changemap 參數（與 gen_changemaps_v18.py 逐項對齊，不要單獨改）──────────
CM_SCALE = 0.5  # S：half-res 算
CM_HISTORY = 500  # HIST
CM_DIST2 = 500.0  # DIST
CM_MIN_AREA = 5  # RM：移除 <5px 碎塊
CM_WARM_FRAMES = 150  # WARM

MAX_SESSIONS = 4  # 同時快取幾支影片的背景模型（webui 平行度通常 2-3）


def rm5(mask: np.ndarray) -> np.ndarray:
    """移除面積 <CM_MIN_AREA 的碎塊，並把 shadow(127) 一併丟掉 → 嚴格二值 {0,255}。

    只取 ==255 是關鍵：detectShadows=True 會把陰影標成 127，訓練時就是這樣濾掉的。

    向量化寫法：原本逐一走訪連通元件、每個都對整張圖做一次 `lab == i` 布林比較
    （訓練資料生成 gen_changemaps_v18.py:51 仍是那一版，一次性批次不值得改）。
    真實路口遮罩的元件數中位數 4871 個，等於對全圖掃近五千遍 —— 中位數 125.0 ms/張，
    最忙的一張要 3.6 秒。改成建一張「元件編號 → 保留與否」的查表再一次索引，
    降到 5.42 ms/張（中位數 23 倍、最忙那張 304 倍），成本不再隨元件數成長。
    輸出逐像素完全相同：537_8_loc537 全片 100 張遮罩 100/100 相等，
    另加邊界情況測試。
    量測與論證見 優化_rm5向量化.md，重跑用 bench_rm5_full.py。
    """
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= CM_MIN_AREA)   # 0 是背景，永遠不保留
    return (keep[lab] * 255).astype(np.uint8)


class VideoSession:
    """一支影片的背景模型 + 讀取狀態。

    背景模型在建構時就暖機完成（阻塞），之後 `changemap()` 一律 learningRate=0。
    """

    def __init__(self, video_path: str, warm_frames: int = CM_WARM_FRAMES) -> None:
        self.video_path = video_path
        self.lock = threading.Lock()
        self.last_used = time.monotonic()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._pos = -1  # cap 目前讀到的 frame index（-1 = 還沒讀）

        # ── pass1：暖機。用獨立的 capture，避免污染主 capture 的位置 ──
        t0 = time.monotonic()
        self.knn = cv2.createBackgroundSubtractorKNN(
            history=CM_HISTORY, dist2Threshold=CM_DIST2, detectShadows=True
        )
        warm_cap = cv2.VideoCapture(video_path)
        self.n_warm = 0
        try:
            for _ in range(warm_frames):
                ok, frame = warm_cap.read()
                if not ok:
                    break
                self.knn.apply(cv2.resize(frame, None, fx=CM_SCALE, fy=CM_SCALE))
                self.n_warm += 1
        finally:
            warm_cap.release()
        self.warm_sec = time.monotonic() - t0

        if self.n_warm == 0:
            # 沒暖機成功 → changemap 全零。訓練 config 明確警告第 4 通道補零會讓
            # 結果全錯，所以這裡拒絕服務而不是靜默降級。
            raise RuntimeError(f"changemap 暖機讀不到任何幀: {video_path}")
        logger.info(
            "session ready: %s (warm %d frames in %.1fs, total=%d)",
            Path(video_path).name,
            self.n_warm,
            self.warm_sec,
            self.total_frames,
        )

    def read_frame(self, index: int) -> np.ndarray | None:
        """取第 index 幀（0-based，與 enforcement 的 frame_index 同基準）。

        pipeline 是單調前進的，所以絕大多數情況只要連續 read；只有跳號時才 seek。
        """
        if index < self._pos:
            # 倒退 → 只能 seek。avi 的 seek 不一定精準，但這條路很少走到。
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._pos = index - 1
        while self._pos < index:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._pos += 1
            if self._pos == index:
                return frame
        return None

    def changemap(self, frame_bgr: np.ndarray) -> np.ndarray:
        """凍結背景取前景 → 全解析度二值 changemap。"""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, None, fx=CM_SCALE, fy=CM_SCALE)
        fg = self.knn.apply(small, learningRate=0)
        return cv2.resize(rm5(fg), (w, h), interpolation=cv2.INTER_NEAREST)

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class Detector:
    """第 18 次的 4ch Co-DETR。模型只載一次，推論用 lock 序列化（單卡、batch=1）。"""

    def __init__(self, checkpoint: str | None, device: str) -> None:
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.utils import import_modules_from_strings

        if not checkpoint:
            ckpts = sorted(
                glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth")
            )
            if not ckpts:
                raise SystemExit(f"找不到 best ckpt，看過 {RESULT_DIR}")
            checkpoint = ckpts[-1]
        logger.info("loading checkpoint: %s", checkpoint)

        init_default_scope("mmdet")
        cfg = Config.fromfile(CONFIG)
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg["custom_imports"])
        self.model = init_detector(cfg, checkpoint, device=device)
        # 沒有 LoadImageAndChangemap —— 4ch 影像由 detect() 直接組好放進 results['img']。
        self.pipeline = Compose(
            [
                dict(type="Resize", scale=IMAGE_SIZE, keep_ratio=True),
                dict(type="Pad", size=IMAGE_SIZE, pad_val=dict(img=PAD4)),
                dict(
                    type="PackDetInputs",
                    meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
                ),
            ]
        )
        self.checkpoint = checkpoint
        self.lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray, changemap: np.ndarray) -> list[dict]:
        import torch
        from mmengine.dataset import pseudo_collate

        # LoadImageAndChangemap 用 channel_order='rgb' 讀圖、config 是 bgr_to_rgb=False，
        # 所以這裡必須自己把 cv2 的 BGR 轉成 RGB，否則色彩通道對調、結果全錯。
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img4 = np.concatenate([rgb, changemap[..., None]], axis=2)
        results = {
            "img": img4,
            "img_id": 0,
            "img_path": "<memory>",
            "img_shape": img4.shape[:2],
            "ori_shape": img4.shape[:2],
        }
        data = pseudo_collate([self.pipeline(results)])
        with self.lock, torch.no_grad():
            out = self.model.test_step(data)
        pred = out[0].pred_instances
        boxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        return [
            {
                "x1": int(max(0, b[0])),
                "y1": int(max(0, b[1])),
                "x2": int(max(0, b[2])),
                "y2": int(max(0, b[3])),
                "label": CLASSES[int(li)] if int(li) < len(CLASSES) else f"cls_{int(li)}",
                "score": float(sc),
            }
            for b, sc, li in zip(boxes, scores, labels)
        ]


class SessionPool:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, VideoSession] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def get(self, video_path: str, warm_frames: int) -> VideoSession:
        with self._lock:
            sess = self._sessions.get(video_path)
            if sess is not None:
                self._sessions.move_to_end(video_path)
                sess.last_used = time.monotonic()
                return sess
        # 暖機要讀 150 幀，不佔著全域鎖做（否則會擋住其他影片的推論）。
        sess = VideoSession(video_path, warm_frames=warm_frames)
        with self._lock:
            existing = self._sessions.get(video_path)
            if existing is not None:  # 競賽：別人先建好了，用他的
                sess.close()
                return existing
            self._sessions[video_path] = sess
            while len(self._sessions) > self._max:
                _, victim = self._sessions.popitem(last=False)
                victim.close()
        return sess

    def drop(self, video_path: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(video_path, None)
        if sess is None:
            return False
        sess.close()
        return True


DETECTOR: Detector | None = None
POOL = SessionPool()
ALLOWED_ROOTS: list[Path] = []


def _is_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # 靜音預設的 stderr access log
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send(
                200,
                {
                    "ok": DETECTOR is not None,
                    "checkpoint": DETECTOR.checkpoint if DETECTOR else None,
                    "classes": CLASSES,
                    "sessions": list(POOL._sessions.keys()),
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        route = self.path.split("?")[0].rstrip("/")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"bad request body: {exc}"})
            return

        if route == "/release":
            video = str(body.get("video") or "")
            self._send(200, {"released": POOL.drop(video)})
            return

        if route != "/detect":
            self._send(404, {"error": "not found"})
            return

        video = str(body.get("video") or "")
        frames = body.get("frames") or []
        score_thr = float(body.get("score_thr", 0.3))
        warm_frames = int(body.get("warm_frames", CM_WARM_FRAMES))
        want_all = bool(body.get("all_classes", False))

        if not video or not isinstance(frames, list):
            self._send(400, {"error": "need video + frames[]"})
            return
        path = Path(video)
        if not _is_allowed(path):
            self._send(403, {"error": f"path not in allowed roots: {video}"})
            return
        if not path.is_file():
            self._send(404, {"error": f"video not found: {video}"})
            return
        if DETECTOR is None:
            self._send(503, {"error": "model not loaded"})
            return

        try:
            sess = POOL.get(str(path), warm_frames)
        except RuntimeError as exc:
            self._send(500, {"error": str(exc)})
            return

        out: list[list[dict]] = []
        t0 = time.monotonic()
        with sess.lock:
            for raw_index in frames:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    out.append([])
                    continue
                frame = sess.read_frame(index)
                if frame is None:
                    out.append([])
                    continue
                cm = sess.changemap(frame)
                boxes = DETECTOR.detect(frame, cm)
                out.append(
                    [
                        b
                        for b in boxes
                        if b["score"] >= score_thr
                        and (want_all or b["label"] in LITTER_CLASSES)
                    ]
                )
        elapsed = time.monotonic() - t0
        self._send(
            200,
            {
                "boxes": out,
                "elapsed_ms": int(elapsed * 1000),
                "ms_per_frame": int(elapsed * 1000 / max(1, len(frames))),
            },
        )


def main() -> None:
    global DETECTOR, ALLOWED_ROOTS

    parser = argparse.ArgumentParser(description="Co-DETR (18th, v18 4ch) litter sidecar")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--checkpoint", default=os.getenv("CODETR_CKPT") or None)
    parser.add_argument("--device", default=os.getenv("CODETR_DEVICE", "cuda:0"))
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/videos", "/testset", "/testset2"],
        help="允許讀取影片的根目錄白名單",
    )
    args = parser.parse_args()

    ALLOWED_ROOTS = [Path(r).resolve() for r in args.roots if Path(r).is_dir()]
    logger.info("allowed roots: %s", [str(r) for r in ALLOWED_ROOTS])

    DETECTOR = Detector(args.checkpoint, args.device)
    logger.info("model ready on %s", args.device)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
