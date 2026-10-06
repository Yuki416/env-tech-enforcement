#!/usr/bin/env python3
"""Stage 2 第一步：對每部 clip 的「事件前後窗口」逐幀跑 Co-DETR，把預測存成 JSON。

## 為什麼可以只跑一次就掃完所有抽幀間隔

線上設計是 **U=1**：KNN 背景模型每一幀都 apply（7.79 ms/幀，相對 Co-DETR 的數百毫秒
可以忽略），只有 Co-DETR 才依抽幀間隔挑幀。
在這個設計下，**第 f 幀的 changemap 與抽幀間隔無關**——不管間隔設 1 還是 35，
KNN 看到的幀序列都一樣，所以第 f 幀的第 4 通道完全相同。

因此：逐幀跑一次拿到「每一幀的預測」，之後任何抽幀間隔都只是對這份結果做子抽樣。
stage2_sweep.py 負責那一步，不需要 GPU、可以反覆重跑。

## 窗口切法

    [first_abs-300, first_abs-150)   暖機 150 幀（自動模式，等同訓練 pass1 的 pre-roll）
    [first_abs-150, first_abs)       誤報窗 150 幀 —— 事件還沒發生，任何 litter/garbage 預測都是誤報
    [first_abs,     first_abs+span)  事件窗 —— 有 GT，用來判斷這次丟棄有沒有被抓到

誤報窗與事件窗都逐幀 apply(learningRate=0) 並逐幀推論。

## 幀對應

dataset 第 fn 幀 → 原始影片第 `first_abs + round((fn - fn0) * slope)` 幀
（fn0 = 該 clip 的第一個 dataset 幀號；公式取自 build_alignment_v18.py:98）。

輸出：stage2_out/<split>__<ts>_<MMmSSs>.json
"""
import argparse, json, re, collections, time, os
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
V18 = Path("/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids")
ALIGN = HERE / "資料前處理" / "alignment_v18.json"
OUT = HERE / "stage2_out"

CONFIG = "/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"
RESULT_DIR = "/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e"
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
IMAGE_SIZE = (1024, 1024)
PAD4 = (114, 114, 114, 0)

CM_SCALE, CM_HISTORY, CM_DIST2, CM_MIN_AREA = 0.5, 500, 500.0, 5
WARM = 150          # 暖機幀數，與 gen_changemaps_v18.py / server.py 一致
FP_FRAMES = 150     # 誤報窗長度（幀）
SAVE_THR = 0.05     # 低於此分數的預測不存，省空間；分析時的門檻一定高於它

LITTER_CID, GARBAGE_CID = 2, 5      # COCO category_id
LITTER_LID, GARBAGE_LID = 1, 4      # 模型 label id（CLASSES 的索引）

# alignment_v18.json 存的是 host 路徑；容器內同一顆硬碟掛在別的位置。
HOST_PREFIX, CONTAINER_PREFIX = "/path/to/data/", "/workspace/hdd2_storage/"


def remap(path: str) -> str:
    """把 alignment 裡的 host 影片路徑換成容器內的掛載點；已經是容器路徑就原樣回傳。"""
    if path.startswith(HOST_PREFIX) and not os.path.exists(path):
        return CONTAINER_PREFIX + path[len(HOST_PREFIX):]
    return path


def rm5(mask):
    """移除面積 <CM_MIN_AREA 的碎塊、丟掉 shadow(127) → 嚴格二值 {0,255}。

    向量化版本（與逐元件走訪的舊寫法輸出逐像素相同，實測快 24 倍）。
    """
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= CM_MIN_AREA)
    return (keep[lab] * 255).astype(np.uint8)


class Detector:
    def __init__(self, checkpoint=None, device="cuda:0"):
        import glob, torch
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.utils import import_modules_from_strings
        if not checkpoint:
            c = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
            if not c:
                raise SystemExit(f"找不到 best ckpt，看過 {RESULT_DIR}")
            checkpoint = c[-1]
        print(f"[model] {checkpoint}", flush=True)
        init_default_scope("mmdet")
        cfg = Config.fromfile(CONFIG)
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg["custom_imports"])
        self.model = init_detector(cfg, checkpoint, device=device)
        self.pipeline = Compose([
            dict(type="Resize", scale=IMAGE_SIZE, keep_ratio=True),
            dict(type="Pad", size=IMAGE_SIZE, pad_val=dict(img=PAD4)),
            dict(type="PackDetInputs",
                 meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
        ])

    def detect(self, frame_bgr, changemap):
        import torch
        from mmengine.dataset import pseudo_collate
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # config 是 bgr_to_rgb=False，必須自己轉
        img4 = np.concatenate([rgb, changemap[..., None]], axis=2)
        res = {"img": img4, "img_id": 0, "img_path": "<memory>",
               "img_shape": img4.shape[:2], "ori_shape": img4.shape[:2]}
        data = pseudo_collate([self.pipeline(res)])
        with torch.no_grad():
            out = self.model.test_step(data)
        p = out[0].pred_instances
        lb = p.labels.cpu().numpy(); sc = p.scores.cpu().numpy(); bb = p.bboxes.cpu().numpy()
        m = sc >= SAVE_THR
        return [[int(a), round(float(b), 4), [round(float(v), 1) for v in c]]
                for a, b, c in zip(lb[m], sc[m], bb[m])]


def load_clip_gt():
    """回傳 {clip_key: {'gt': {dataset_fn: [(cat_id, xyxy)]}, 'fn0': int, 'split': str}}"""
    out = {}
    for sp in ("train", "valid", "test"):
        d = json.load(open(V18 / f"annotations/{sp}.json"))
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for an in d["annotations"]:
            if an["category_id"] not in (LITTER_CID, GARBAGE_CID):
                continue
            fn = id2fn[an["image_id"]]
            g = re.search(r"(\d{14})_(?:\d{2}h)?\d{2}m\d{2}s", fn)
            k = f"{sp}/{g.group(0)}"
            no = int(re.search(r"_frame_(\d+)", fn).group(1))
            x, y, w, h = an["bbox"]
            per[k][no].append((an["category_id"], [x, y, x + w, y + h]))
        # 也要記錄「這個 clip 有哪些幀」（含沒有 litter 標註的幀），才知道 fn0 與 span
        allfn = collections.defaultdict(set)
        for im in d["images"]:
            g = re.search(r"(\d{14})_(?:\d{2}h)?\d{2}m\d{2}s", im["file_name"])
            allfn[f"{sp}/{g.group(0)}"].add(int(re.search(r"_frame_(\d+)", im["file_name"]).group(1)))
        for k, fns in allfn.items():
            out[k] = dict(gt={str(n): per[k].get(n, []) for n in sorted(fns)},
                          fn0=min(fns), fnmax=max(fns), split=sp)
    return out


def run_clip(det, key, info, clip, outpath):
    fps, slope, first_abs = info["fps"], info["slope"], info["first_abs"]
    fn0, fnmax = clip["fn0"], clip["fnmax"]
    span = int(round((fnmax - fn0) * slope)) + 1
    warm_start = first_abs - FP_FRAMES - WARM
    fp_start = first_abs - FP_FRAMES
    end = first_abs + span

    avi = remap(info["avi"])
    cap = cv2.VideoCapture(avi)
    if not cap.isOpened():
        return {"error": f"無法開啟 {avi}"}
    knn = cv2.createBackgroundSubtractorKNN(history=CM_HISTORY, dist2Threshold=CM_DIST2,
                                            detectShadows=True)
    # ── 暖機：自動模式，與訓練 pass1 相同 ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
    n_warm = 0
    for _ in range(WARM):
        ok, f = cap.read()
        if not ok:
            break
        knn.apply(cv2.resize(f, None, fx=CM_SCALE, fy=CM_SCALE))
        n_warm += 1
    if n_warm < WARM:
        cap.release()
        return {"error": f"暖機只讀到 {n_warm}/{WARM} 幀"}

    # ── 誤報窗 + 事件窗：逐幀 apply(lr=0) 逐幀推論 ──
    frames, t0 = {}, time.monotonic()
    idx = fp_start
    while idx < end:
        ok, f = cap.read()
        if not ok:
            break
        cm_small = knn.apply(cv2.resize(f, None, fx=CM_SCALE, fy=CM_SCALE), learningRate=0)
        H, W = f.shape[:2]
        cm = cv2.resize(rm5(cm_small), (W, H), interpolation=cv2.INTER_NEAREST)
        frames[str(idx)] = det.detect(f, cm)
        idx += 1
    cap.release()
    dt = time.monotonic() - t0

    rec = dict(key=key, split=clip["split"], avi=avi, fps=fps, slope=slope,
               first_abs=first_abs, fn0=fn0, fnmax=fnmax, span=span,
               warm_start=warm_start, fp_start=fp_start, event_start=first_abs, end=idx,
               n_frames=len(frames), sec=round(dt, 1),
               ms_per_frame=round(dt / max(1, len(frames)) * 1000, 1),
               gt=clip["gt"], preds=frames, classes=CLASSES)
    json.dump(rec, open(outpath, "w"), ensure_ascii=False)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="逗號分隔的 clip key，或 @檔名 逐行讀")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    keys = (open(a.clips[1:]).read().split() if a.clips.startswith("@")
            else [k for k in a.clips.split(",") if k])

    OUT.mkdir(exist_ok=True)
    align = json.load(open(ALIGN))
    clips = load_clip_gt()
    det = Detector(device=a.device)

    for i, k in enumerate(keys, 1):
        outp = OUT / (k.replace("/", "__") + ".json")
        if outp.exists():
            print(f"[{i}/{len(keys)}] {k} 已存在，跳過", flush=True)
            continue
        if k not in align or k not in clips:
            print(f"[{i}/{len(keys)}] {k} 找不到對齊或標註，跳過", flush=True)
            continue
        print(f"[{i}/{len(keys)}] {k} …", flush=True)
        r = run_clip(det, k, align[k], clips[k], outp)
        if "error" in r:
            print(f"    ✗ {r['error']}", flush=True)
        else:
            print(f"    {r['n_frames']} 幀 / {r['sec']}s = {r['ms_per_frame']} ms/幀", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
