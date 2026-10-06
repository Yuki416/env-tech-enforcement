#!/usr/bin/env python3
"""建立 v14 = 從 v11 過濾出的『動態 litter』資料集（路線 A：動態種子 + 時間窗擴量）。

設計（依使用者決定）：
  - 只過濾 train/valid；test 直接沿用原始 v11（靜止版，不動）。
  - 保留 5 類（car/litter/motorcycle/person/garbage）：被選中的幀，其『全部 5 類標註』照留，
    只是「選哪些幀」由 litter 的動態與否決定。
  - 動態種子：litter 框跨相鄰有-litter幀的位移 / Δframe >= SEED_THRESH（px/幀）。
  - 時間窗擴量：同 clip 內，凡是落在任一種子幀 ±WINDOW native-frame 內、且含 litter 的幀都收進來
    （抓下落→落地整段弧，不是只留位移超標那一瞬）。

輸出：/path/to/data/coco/chiayi_v14_own_dynamic_litter/
  annotations/{train,valid,test}.json + {train,valid}/ 影像(複製) + test/ (symlink 到 v11)
"""
import json
import os
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v11_garbage_litter_corrected")
DST = Path("/path/to/data/coco/chiayi_v14_own_dynamic_litter")
# 動態目標類別：litter(2) + garbage(5)。各自類別內跨幀配對算位移，任一超標即為種子。
TARGET_CATS = [2, 5]
SEED_THRESH = 20     # px/幀，動態種子門檻
# 連續片段 padding（native frame index）：頭段從第一個 seed 往前、尾段從最後一個 seed 往後。
# 尾段刻意小 → 動態結束後很快收尾，避免片尾拖一長串靜止/無關幀（見 analyze_tail_gap.py）。
HEAD_WINDOW = 30
TAIL_WINDOW = 5

# 逐 clip 尾巴 padding 覆寫：預設用 TAIL_WINDOW(5)，這幾部使用者看過 tail5/tail10 比較後決定用 10。
# clip 前綴 -> tail 值。
TAIL_OVERRIDE = {
    "Day_Litter_12_8_loc12_bureau_20240603181958_04m49s_RoboFlow": 10,
    "Day_Litter_538_8_loc538_20240803182230_08m49s_RoboFlow": 10,
}

# 逐 clip 末幀『絕對上限』：使用者看過後指定這部只取到某 frame（覆寫 seed+tail 的結果）。
# 用於演算法把整段都當動態、但實際只有前段有意義的 clip。clip 前綴 -> 最大 frame index。
END_CAP = {
    "Day_Litter_535_8_loc535_20240704064730_00m12s_RoboFlow": 13,
}

# 人工補入的動態 clip：自動位移門檻(20px/幀)抓不到（動幅太小），但使用者肉眼確認是丟垃圾動作。
# clip 前綴(不含 _RoboFlow) -> (看到垃圾frame, 落地frame)＝起訖動態種子，套一樣的 HEAD/TAIL padding。
# 心裡明白：這 4 部是「較難偵測」的弱動態樣本。
MANUAL_DYNAMIC = {
    "Day_Litter_537_8_loc537_20240807184204_01m10s": (16, 18),
    "Night_Litter_544_8_loc544_20240911024010_02m09s": (7, 10),
    "Day_Litter_538_8_loc538_20240804060854_15m33s": (18, 21),
    "Day_Litter_536_8_loc536_20240716055530_05m10s": (16, 17),
}
GAP_CAP = 15         # 配對相鄰有-litter幀的最大 frame gap
MATCH_GATE_FRAC = 0.20

frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
CATS_NAME = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}


def parse(fn):
    m = frame_re.match(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def _seeds_for_cat(imgs, cat_by_img):
    """對單一類別做跨幀最近鄰配對，回傳(該類別種子 image_id 集合, clip->該類別幀清單)。"""
    clip_frames = defaultdict(list)  # clip -> [(frame_idx, img_id)] 含此類別的幀
    for img_id in cat_by_img:
        im = imgs[img_id]
        if im["clip"] is not None:
            clip_frames[im["clip"]].append((im["frame"], img_id))

    seeds = set()
    for clip, frames in clip_frames.items():
        frames.sort()
        for i, (f_idx, img_id) in enumerate(frames):
            im = imgs[img_id]
            gate = MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
            nxt = next((frames[j2] for j2 in range(i + 1, len(frames))
                        if 0 < frames[j2][0] - f_idx <= GAP_CAP), None)
            if nxt is None:
                continue
            dframe = nxt[0] - f_idx
            dst_c = [center(b) for b in cat_by_img[nxt[1]]]
            for b in cat_by_img[img_id]:
                cx, cy = center(b)
                best = min((((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5) for dx, dy in dst_c)
                if best <= gate and best / dframe >= SEED_THRESH:
                    seeds.add(img_id)
                    break
    return seeds, clip_frames


def select_frames(split):
    """回傳被選中的 image_id 集合 + 診斷資訊。litter(2)+garbage(5) 各自算動態。"""
    j = json.load(open(SRC / "annotations" / f"{split}.json"))
    imgs = {im["id"]: im for im in j["images"]}
    for im in j["images"]:
        im["clip"], im["frame"] = parse(im["file_name"])

    # 各目標類別的框
    boxes_by_cat = {c: defaultdict(list) for c in TARGET_CATS}
    for a in j["annotations"]:
        if a["category_id"] in TARGET_CATS:
            boxes_by_cat[a["category_id"]][a["image_id"]].append(a["bbox"])

    # 種子（各類別分開配對，取聯集）+ 每類別種子數
    seed_frames = set()
    seeds_per_cat = {}
    for c in TARGET_CATS:
        s, _ = _seeds_for_cat(imgs, boxes_by_cat[c])
        seeds_per_cat[c] = s
        seed_frames |= s

    # 連續片段擷取：pool = 該 clip『全部』的幀（不再只收有標註的幀），
    # 每個含種子的 clip 取 [第一個種子 - WINDOW, 最後一個種子 + WINDOW] 之間的所有幀，
    # 中間沒有目標標註的幀也一併納入 → 確保是連續片段。
    pool = defaultdict(list)  # clip -> [(frame_idx, img_id)]  該 clip 全部幀
    for im in imgs.values():
        if im["clip"] is not None:
            pool[im["clip"]].append((im["frame"], im["id"]))

    seed_by_clip = defaultdict(list)
    for img_id in seed_frames:
        seed_by_clip[imgs[img_id]["clip"]].append(imgs[img_id]["frame"])

    # 人工補入的弱動態 clip：把使用者給的起訖 frame 當種子加進來（若該 clip 在此 split）
    for clip in pool:
        for key, (f0, f1) in MANUAL_DYNAMIC.items():
            if clip.startswith(key):
                seed_by_clip[clip].extend([f0, f1])

    selected = set()
    for clip, frames in pool.items():
        seeds = seed_by_clip.get(clip)
        if not seeds:
            continue
        tail = next((v for k, v in TAIL_OVERRIDE.items() if clip.startswith(k)), TAIL_WINDOW)
        lo, hi = min(seeds) - HEAD_WINDOW, max(seeds) + tail
        cap = next((v for k, v in END_CAP.items() if clip.startswith(k)), None)
        if cap is not None:
            hi = min(hi, cap)
        for f_idx, img_id in frames:
            if lo <= f_idx <= hi:
                selected.add(img_id)

    return j, imgs, boxes_by_cat, seed_frames, selected, seeds_per_cat


def build_split(split):
    j, imgs, boxes_by_cat, seed_frames, selected, seeds_per_cat = select_frames(split)
    ann_by_img = defaultdict(list)
    for a in j["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    out_imgs, out_anns = [], []
    new_img_id, new_ann_id = 0, 0
    cls_counter = Counter()
    n_no_target = 0   # 無 litter/garbage 標註的幀（連續片段中間的填充幀 = negative）
    dst_dir = DST / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    for img_id in sorted(selected):
        im = imgs[img_id]
        shutil.copy2(SRC / split / im["file_name"], dst_dir / im["file_name"])
        out_imgs.append({"id": new_img_id, "file_name": im["file_name"],
                         "width": im["width"], "height": im["height"]})
        cats_here = {a["category_id"] for a in ann_by_img[img_id]}
        if not (cats_here & set(TARGET_CATS)):
            n_no_target += 1
        for a in ann_by_img[img_id]:
            out_anns.append({**a, "id": new_ann_id, "image_id": new_img_id})
            cls_counter[a["category_id"]] += 1
            new_ann_id += 1
        new_img_id += 1

    coco = {"images": out_imgs, "annotations": out_anns,
            "categories": j["categories"]}
    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(DST / "annotations" / f"{split}.json", "w"), ensure_ascii=False)

    clips = {imgs[i]["clip"] for i in selected}
    print(f"\n===== [{split}] v14 動態版 =====")
    print(f"  原始 v11         : {len(j['images'])} 幀")
    print(f"  動態種子幀(聯集)  : {len(seed_frames)}   ("
          + ", ".join(f"{CATS_NAME[c]}={len(seeds_per_cat[c])}" for c in TARGET_CATS) + ")")
    print(f"  連續片段選中      : {len(selected)} 幀  (來自 {len(clips)} 個 clip)")
    print(f"    其中無litter/garbage(negative填充幀): {n_no_target} ({n_no_target/max(len(selected),1)*100:.0f}%)")
    print(f"  → 5 類分布        : " + ", ".join(f"{CATS_NAME[k]}={cls_counter[k]}"
                                               for k in sorted(CATS_NAME)))


def copy_test():
    """test 沿用原始 v11（靜止版），直接『複製』影像進來（不用 symlink，避免容器內路徑解不開）。"""
    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "annotations" / "test.json", DST / "annotations" / "test.json")
    dst_test = DST / "test"
    if dst_test.is_symlink():          # 清掉舊的 symlink
        dst_test.unlink()
    dst_test.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in (SRC / "test").iterdir():
        if not (dst_test / f.name).exists():
            shutil.copy2(f, dst_test / f.name)
        n += 1
    print(f"\n===== [test] 沿用原始 v11（複製 {n} 張影像） =====\n  {dst_test}")


if __name__ == "__main__":
    print(f"參數: SEED_THRESH={SEED_THRESH}px/幀, HEAD_WINDOW={HEAD_WINDOW} TAIL_WINDOW={TAIL_WINDOW}幀, 目標類別={[CATS_NAME[c] for c in TARGET_CATS]}")
    # 清掉舊的 train/valid（避免上一版只含 litter 的殘檔）
    for sp in ["train", "valid"]:
        if (DST / sp).exists():
            shutil.rmtree(DST / sp)
    for sp in ["train", "valid"]:
        build_split(sp)
    copy_test()
    print(f"\n完成 → {DST}")
