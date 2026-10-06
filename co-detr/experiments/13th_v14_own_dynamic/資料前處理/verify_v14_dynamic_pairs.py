#!/usr/bin/env python3
"""動態種子『前後幀配對』驗證圖（在容器內用 cv2 跑）。

把每個動態種子幀 f 和它最近鄰配到的下一幀並排：
  左=種子幀 f（紅框=來源框），右=下一幀（綠框=配到的框；黃框=來源框位置疊上去看位移）。
  中間畫位移向量。標題=類別/frame gap/px每幀。
判讀：黃框→綠框若是同一顆垃圾真的移動 = 真動態；若黃綠是畫面裡兩顆不同的靜止垃圾 = 配對假象。

litter、garbage 各出一張，各挑涵蓋低速→高速的 6 個種子。
"""
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import importlib.util, sys

_HERE = "/workspace/project/experiments/13th_v14_own_dynamic/資料前處理"  # 容器內路徑
spec = importlib.util.spec_from_file_location("b", f"{_HERE}/build_v14_dynamic_litter.py")
b = importlib.util.module_from_spec(spec); sys.modules["b"] = b; spec.loader.exec_module(b)

SPLIT = "train"
SRC = Path("/workspace/datasets/coco/chiayi_v11_garbage_litter_corrected")
b.SRC = SRC  # 容器內路徑覆寫（build 模組預設是主機 /mnt 路徑）
CATS_NAME = {2: "litter", 5: "garbage"}
OUT_DIR = Path(_HERE)


def matched_pairs(imgs, cat_by_img):
    """回傳每個種子幀的配對明細：(img_id, nxt_id, src_box, dst_box, dframe, speed)。"""
    clip_frames = defaultdict(list)
    for img_id in cat_by_img:
        im = imgs[img_id]
        if im["clip"] is not None:
            clip_frames[im["clip"]].append((im["frame"], img_id))
    out = []
    for clip, frames in clip_frames.items():
        frames.sort()
        for i, (f_idx, img_id) in enumerate(frames):
            im = imgs[img_id]
            gate = b.MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
            nxt = next((frames[j] for j in range(i + 1, len(frames))
                        if 0 < frames[j][0] - f_idx <= b.GAP_CAP), None)
            if nxt is None:
                continue
            dframe = nxt[0] - f_idx
            dst_boxes = cat_by_img[nxt[1]]
            dst_c = [b.center(x) for x in dst_boxes]
            best = None
            for src in cat_by_img[img_id]:
                cx, cy = b.center(src)
                dists = [(((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5, k) for k, (dx, dy) in enumerate(dst_c)]
                d, k = min(dists)
                if d <= gate and d / dframe >= b.SEED_THRESH:
                    if best is None or d / dframe > best[-1]:
                        best = (img_id, nxt[1], src, dst_boxes[k], dframe, d / dframe)
            if best:
                out.append(best)
    return out


def draw_box(img, box, color, thick=3):
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thick)


def render(cat):
    j, imgs, boxes_by_cat, seeds, selected, _ = b.select_frames(SPLIT)
    pairs = matched_pairs(imgs, boxes_by_cat[cat])
    pairs.sort(key=lambda p: p[-1])  # 依 speed
    if not pairs:
        print(f"{CATS_NAME[cat]}: 無配對"); return
    # 涵蓋低→高速取 6 個
    idxs = sorted({int(round(t * (len(pairs) - 1))) for t in [0, .2, .4, .6, .8, 1.0]})
    picks = [pairs[i] for i in idxs]

    H = 360
    rows = []
    for img_id, nxt_id, src, dst, dframe, speed in picks:
        imL = cv2.imread(str(SRC / SPLIT / imgs[img_id]["file_name"]))
        imR = cv2.imread(str(SRC / SPLIT / imgs[nxt_id]["file_name"]))
        s = H / imL.shape[0]
        imL = cv2.resize(imL, (int(imL.shape[1] * s), H))
        imR = cv2.resize(imR, (int(imR.shape[1] * s), H))
        sb = [v * s for v in src]; db = [v * s for v in dst]
        draw_box(imL, sb, (0, 0, 255))                       # 左：來源框 紅
        draw_box(imR, db, (0, 255, 0))                       # 右：配到框 綠
        draw_box(imR, sb, (0, 255, 255), 2)                  # 右：來源框位置 黃（看位移）
        c1 = (int(sb[0] + sb[2] / 2), int(sb[1] + sb[3] / 2))
        c2 = (int(db[0] + db[2] / 2), int(db[1] + db[3] / 2))
        cv2.arrowedLine(imR, c1, c2, (0, 255, 255), 2, tipLength=0.3)
        div = np.full((H, 6, 3), 40, np.uint8)
        row = np.hstack([imL, div, imR])
        bar = np.full((28, row.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f"{CATS_NAME[cat]} | dframe={dframe} | {speed:.0f}px/f | L=f{imgs[img_id]['frame']} R=f{imgs[nxt_id]['frame']}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        rows.append(np.vstack([bar, row]))
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 20, np.uint8)]) if r.shape[1] < W else r for r in rows]
    out = np.vstack([np.vstack([r, np.full((8, W, 3), 20, np.uint8)]) for r in rows])
    p = OUT_DIR / f"v14_verify_pairs_{CATS_NAME[cat]}.png"
    cv2.imwrite(str(p), out)
    print(f"saved {p}  ({len(picks)} 對, speed {picks[0][-1]:.0f}~{picks[-1][-1]:.0f}px/f)")


if __name__ == "__main__":
    for cat in b.TARGET_CATS:
        render(cat)
