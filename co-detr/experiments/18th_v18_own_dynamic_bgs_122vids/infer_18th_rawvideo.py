#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_18th_rawvideo.py — 第 18 次 Co-DETR（v18 4ch）對**原始 .avi** 跑全片推論。

與 infer_18th_v18_4ch_video_classification.py 的差別（所以另開一支，不改那支）：
  那支讀 dataset JPG + 事先生好的 {split}_changemap/ 目錄；
  原始影片兩者都沒有 → changemap 必須**線上逐幀生成**。

★ 為什麼是「逐幀」apply 而不是 stride：
  實測 v18 dataset 的相鄰幀編號差＝1（917/917 全連續），
  也就是訓練時 changemap 是逐幀 apply 的，靜止物體壽命約 5 幀 ≈ 0.5 秒。
  逐幀 apply 才能複製訓練條件。
  ⚠️ 這**不等於**線上執法 pipeline（stride 35 → 壽命 17.5 秒），兩者要分開看。

★ 沒有 GT → 不做 TP/TN/FP/FN 分類，也不做 IoU≥0.5 校正（那是給有標註的 clip 用的）。

★ BGR→RGB 必須自己轉：config 是 bgr_to_rgb=False，訓練時的 LoadImageAndChangemap
  以 channel_order='rgb' 讀圖。不轉的話色彩通道對調、結果全錯。

用法（容器內）：
    PYTHONUTF8=1 python infer_18th_rawvideo.py --video <path> --out-dir <dir> [--limit N]
"""
import argparse, csv, glob, os, sys, time
from pathlib import Path

import cv2
import numpy as np

CONFIG = "/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"
RESULT_DIR = "/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e"

CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
PRED_COLORS = {"car": (0, 255, 255), "litter": (0, 0, 255), "motorcycle": (255, 255, 0),
               "person": (255, 0, 255), "garbage": (0, 165, 255)}

# ── 全部對齊訓練時的設定（gen_changemaps_v18.py）─────────────────────────
CM_SCALE = 0.5        # S：half-res 計算再 INTER_NEAREST 升回全解析度
CM_HISTORY = 500
CM_DIST2 = 500.0
CM_MIN_AREA = 5       # rm5：移除 <5px 碎塊
CM_WARM_FRAMES = 150

IMAGE_SIZE = (1024, 1024)
PAD4 = (114, 114, 114, 0)          # 第 4 通道 pad 0，不是 114
DRAW_THRESH = 0.30                 # 影片上畫框的門檻
CSV_THRESH = 0.05                  # CSV 記到這麼低，之後改門檻不用重跑
OUT_W, OUT_H = 1296, 972           # 半解析度輸出（沿用既有慣例）


def rm5(mask: np.ndarray) -> np.ndarray:
    """(mask==255) 順便濾掉陰影(127)；再用連通元件移除 <CM_MIN_AREA 的碎塊。"""
    fg = (mask == 255).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= CM_MIN_AREA:
            out[lab == i] = 255
    return out


class Detector:
    """4ch Co-DETR。4 通道影像在記憶體組好直接餵，不經 LoadImageAndChangemap。"""

    def __init__(self, checkpoint, device="cuda:0"):
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmengine.utils import import_modules_from_strings

        if not checkpoint:
            ck = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
            if not ck:
                sys.exit(f"❌ 找不到 best ckpt，看過 {RESULT_DIR}")
            checkpoint = ck[-1]
        print(f"載入 checkpoint: {checkpoint}", flush=True)

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
        self.checkpoint = checkpoint

    def detect(self, frame_bgr, changemap):
        import torch
        from mmengine.dataset import pseudo_collate
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)     # ★ 必須轉
        img4 = np.concatenate([rgb, changemap[..., None]], axis=2)
        res = {"img": img4, "img_id": 0, "img_path": "<memory>",
               "img_shape": img4.shape[:2], "ori_shape": img4.shape[:2]}
        data = pseudo_collate([self.pipeline(res)])
        with torch.no_grad():
            out = self.model.test_step(data)
        p = out[0].pred_instances
        return list(zip(p.labels.cpu().numpy().tolist(),
                        p.scores.cpu().numpy().tolist(),
                        p.bboxes.cpu().numpy().tolist()))


def draw(img, preds, sx, sy, header):
    for lid, score, box in preds:
        if score < DRAW_THRESH:
            continue
        name = CLASSES[lid] if lid < len(CLASSES) else str(lid)
        color = PRED_COLORS.get(name, (200, 200, 200))
        x1, y1 = int(box[0] * sx), int(box[1] * sy)
        x2, y2 = int(box[2] * sx), int(box[3] * sy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        t = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, t, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 幀（冒煙測試用）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"❌ 開不了影片：{args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W, H = int(cap.get(3)), int(cap.get(4))
    if args.limit:
        total = min(total, args.limit)
    print(f"影片 {args.video}", flush=True)
    print(f"  {total} 幀 / {fps:.1f} fps / {total/fps/60:.1f} 分 / {W}×{H}", flush=True)

    det = Detector(args.checkpoint, args.device)

    knn = cv2.createBackgroundSubtractorKNN(
        history=CM_HISTORY, dist2Threshold=CM_DIST2, detectShadows=True)
    sx, sy = OUT_W / W, OUT_H / H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_main = cv2.VideoWriter(str(out / "inference.mp4"), fourcc, fps, (OUT_W, OUT_H))
    vw_cmp = cv2.VideoWriter(str(out / "inference_changemap.mp4"), fourcc, fps, (OUT_W * 2, OUT_H))
    csv_f = open(out / "detections.csv", "w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow(["frame", "time_sec", "class", "score", "x1", "y1", "x2", "y2"])

    t0 = time.time()
    n_det = 0
    counts = {c: 0 for c in CLASSES}
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            print(f"  ! 第 {idx} 幀讀不到，提前結束", flush=True)
            total = idx
            break

        small = cv2.resize(frame, None, fx=CM_SCALE, fy=CM_SCALE)
        if idx < CM_WARM_FRAMES:                    # 暖機：只餵背景，不推論
            knn.apply(small)
            continue

        fg = knn.apply(small, learningRate=0)       # 同訓練：逐幀 apply
        cm = cv2.resize(rm5(fg), (W, H), interpolation=cv2.INTER_NEAREST)
        preds = det.detect(frame, cm)

        sec = idx / fps
        for lid, score, box in preds:
            if score < CSV_THRESH:
                continue
            name = CLASSES[lid] if lid < len(CLASSES) else str(lid)
            cw.writerow([idx, f"{sec:.2f}", name, f"{score:.4f}",
                         int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            if score >= DRAW_THRESH:
                n_det += 1; counts[name] = counts.get(name, 0) + 1

        vis = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        hdr = f"frame {idx}  |  {int(sec//60):02d}:{sec%60:05.2f}  |  18th Co-DETR 4ch  thr={DRAW_THRESH}"
        vis = draw(vis, preds, sx, sy, hdr)
        vw_main.write(vis)

        cmv = cv2.cvtColor(cv2.resize(cm, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST),
                           cv2.COLOR_GRAY2BGR)
        cv2.rectangle(cmv, (0, 0), (OUT_W, 26), (0, 0, 0), -1)
        cv2.putText(cmv, "changemap (S=0.5, rm5, learningRate=0, per-frame apply)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw_cmp.write(np.hstack([vis, cmv]))

        done = idx - CM_WARM_FRAMES + 1
        if done % 200 == 0:
            el = time.time() - t0
            per = el / done
            left = (total - CM_WARM_FRAMES - done) * per
            print(f"  {done}/{total-CM_WARM_FRAMES} 幀  {per*1000:.0f} ms/幀  "
                  f"已耗時 {el/60:.1f} 分  預估剩餘 {left/60:.1f} 分  "
                  f"累計框數 {n_det}", flush=True)

    cap.release(); vw_main.release(); vw_cmp.release(); csv_f.close()
    el = time.time() - t0
    print("\n" + "=" * 70, flush=True)
    print(f"完成。推論 {total-CM_WARM_FRAMES} 幀，耗時 {el/60:.1f} 分", flush=True)
    print(f"score>={DRAW_THRESH} 的框：{n_det} 個", flush=True)
    for c in CLASSES:
        print(f"    {c:<12} {counts.get(c,0)}", flush=True)
    print(f"輸出：{out}", flush=True)


if __name__ == "__main__":
    main()
