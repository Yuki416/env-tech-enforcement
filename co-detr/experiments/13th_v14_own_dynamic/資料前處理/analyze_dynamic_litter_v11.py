#!/usr/bin/env python3
"""量化分析：v11 資料集裡「動態 litter」有多少 —— 只讀 JSON，不產資料集、不訓練。

目的（回答兩個 blocker）：
  1. 跨幀 litter 框的可配對率：RoboFlow 稀疏抽幀 + 無 track ID，能不能靠最近鄰把
     同一顆 litter 跨相鄰幀配起來算位移？配對率太低 → 位移法不可信。
  2. 動態 litter 數量：用 px/幀 門檻掃出動態 litter 框 / 幀，看 train/valid 過濾後
     還剩多少（判斷 Co-DETR 訓不訓得動）。

方法：
  - 同一 clip（檔名 _frame_ 前的字串為 clip key）內，把 litter 框按 frame index 排序。
  - 相鄰「有 litter 的幀」配對：對每顆 litter 框，在下一個有 litter 的幀裡找中心最近的 litter 框，
    位移 / Δframe = px/幀（正規化，抵銷抽幀不連續）。配對需通過閘門：
      * Δframe <= GAP_CAP（跨太多幀不可信）
      * 中心距 <= MATCH_GATE_FRAC * 影像對角線（太遠視為不同物件，不配）
  - litter 框「動態」定義：該框的 px/幀 >= 門檻。掃多個門檻對照。
"""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

ANN = Path("/path/to/data/coco/chiayi_v11_garbage_litter_corrected/annotations")
LITTER_CAT = 2  # v11: 2=litter
SPLITS = ["train", "valid", "test"]

GAP_CAP = 15                 # 相鄰有-litter幀 frame index 差超過這個就不配對
MATCH_GATE_FRAC = 0.20       # 中心距 > 對角線*此值 視為不同物件
THRESHOLDS = [10, 15, 20, 30]  # px/幀 動態門檻（與別組「相鄰幀20px」可比）

frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def parse(fn):
    m = frame_re.match(fn)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def center(b):  # coco bbox [x,y,w,h]
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def analyze(split):
    j = json.load(open(ANN / f"{split}.json"))
    imgs = {im["id"]: im for im in j["images"]}
    for im in j["images"]:
        im["clip"], im["frame"] = parse(im["file_name"])

    # clip -> [(frame_idx, img_id)]，只收有 litter 的幀
    litter_by_img = defaultdict(list)
    for a in j["annotations"]:
        if a["category_id"] == LITTER_CAT:
            litter_by_img[a["image_id"]].append(a["bbox"])

    clip_frames = defaultdict(list)
    for img_id, boxes in litter_by_img.items():
        im = imgs[img_id]
        if im["clip"] is None:
            continue
        clip_frames[im["clip"]].append((im["frame"], img_id))

    total_litter = sum(len(b) for b in litter_by_img.values())
    n_clips_with_litter = len(clip_frames)

    # 逐 clip 配對，算每顆「起點框」的 px/幀（配得起來的才有速度）
    matched = 0          # 有配到下一幀的 litter 框數
    unmatched_no_next = 0  # 沒有下一個有-litter幀可配（clip 尾/孤立）
    unmatched_gated = 0    # 有下一幀但被閘門擋掉
    speeds = []          # (img_id, px_per_frame)，每個成功配對的起點框

    for clip, frames in clip_frames.items():
        frames.sort()
        diag = None
        for i, (f_idx, img_id) in enumerate(frames):
            im = imgs[img_id]
            diag = (im["width"] ** 2 + im["height"] ** 2) ** 0.5
            gate = MATCH_GATE_FRAC * diag
            # 找下一個 frame index 在 GAP_CAP 內的有-litter幀
            nxt = None
            for j2 in range(i + 1, len(frames)):
                if frames[j2][0] - f_idx <= GAP_CAP and frames[j2][0] > f_idx:
                    nxt = frames[j2]
                    break
            src_boxes = litter_by_img[img_id]
            if nxt is None:
                unmatched_no_next += len(src_boxes)
                continue
            dframe = nxt[0] - f_idx
            dst_boxes = litter_by_img[nxt[1]]
            dst_centers = [center(b) for b in dst_boxes]
            for b in src_boxes:
                cx, cy = center(b)
                best_d, best = 1e18, None
                for (dx, dy) in dst_centers:
                    d = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best = d, (dx, dy)
                if best is None or best_d > gate:
                    unmatched_gated += 1
                    continue
                matched += 1
                speeds.append((img_id, best_d / dframe))

    # 動態統計（多門檻）
    print(f"\n===== [{split}] litter 動態分析 =====")
    print(f"  litter 框總數           : {total_litter}")
    print(f"  有 litter 的 clip 數     : {n_clips_with_litter}")
    print(f"  有 litter 的幀數         : {len(litter_by_img)}")
    print(f"  --- 可配對率 ---")
    total_src = matched + unmatched_no_next + unmatched_gated
    print(f"  成功配對（有速度）      : {matched}  ({matched/max(total_src,1)*100:.1f}%)")
    print(f"  無下一有-litter幀可配    : {unmatched_no_next}  ({unmatched_no_next/max(total_src,1)*100:.1f}%)")
    print(f"  有下一幀但被閘門擋       : {unmatched_gated}  ({unmatched_gated/max(total_src,1)*100:.1f}%)")
    if speeds:
        sp = sorted(s for _, s in speeds)
        med = sp[len(sp) // 2]
        print(f"  配對框位移中位數         : {med:.1f} px/幀   (max {sp[-1]:.0f})")
    print(f"  --- 動態 litter（在成功配對的 {matched} 框中）---")
    for th in THRESHOLDS:
        dyn_boxes = [(img_id, s) for img_id, s in speeds if s >= th]
        dyn_frames = {img_id for img_id, _ in dyn_boxes}
        pct = len(dyn_boxes) / max(matched, 1) * 100
        print(f"  >= {th:>2} px/幀 : 動態框 {len(dyn_boxes):>4} ({pct:4.1f}%) | 落在 {len(dyn_frames):>3} 幀")


if __name__ == "__main__":
    for sp in SPLITS:
        analyze(sp)
    print("\n注意：動態框/幀數是『成功配對子集』內的計數；無法配對的框(孤立幀/尾幀)無法判定動靜，")
    print("      若配對率低，代表位移法在 v11 稀疏抽幀上不可靠，需改用別的動態定義。")
