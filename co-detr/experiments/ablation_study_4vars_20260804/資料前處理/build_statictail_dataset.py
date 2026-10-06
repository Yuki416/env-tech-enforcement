#!/usr/bin/env python3
"""變因① 處理組：在 16th 動態子集之上，加入「垃圾落地後」的靜止正樣本影格。

單變因設計（使用者 2026-08-04 定案）：
  - 自變因 = 訓練集是否納入「垃圾落地後」的靜止正樣本影格。
  - 對照組 = 16th `chiayi_v17_own_dynamic_litter_head10_manual`（HEAD10/TAIL5 動態子集）。
  - 處理組（本腳本）= 16th 的**嚴格超集**：每個 clip 從「16th 選幀的終點 hi」往後，
    延伸到該 clip **最後一個 litter/garbage(cat 2/5) 有標幀**為止，延伸段**每 3 幀取 1 張**抽稀。
  - HEAD、動態密取區間、8 部人工 override 的「起點」全部沿用 16th（因為對照組直接取自 16th json）。
  - 多段 burst：延到全 clip 最後有標幀（可能含第二次丟棄）→ 生成影片後再人工修（使用者定案）。
  - 8 部人工 override clip：其終點也一併往後延伸＋抽稀（使用者定案）。

為何是嚴格超集：對照組直接讀 16th 的 train.json，處理組 = 對照組 ∪ 抽稀尾巴。
→ 兩組差異只有「有沒有落地靜止尾」這一項，事件集合、HEAD、超參數全同。

annotations 一律取自母集（full 5 類/幀），與 16th 建法一致。
valid / test / test_static594 直接複製 16th，維持完全相同（ckpt 選取機制不變）。

用法：
  PYTHONUTF8=1 python3 build_statictail_dataset.py --dry   # 只統計、不寫檔
  PYTHONUTF8=1 python3 build_statictail_dataset.py         # 實際建置
"""
import argparse
import json
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

SRC  = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")  # 全幀母集
CTRL = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual")  # 16th=對照組
DST  = Path("/path/to/data/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation")

TARGET_CATS = [2, 5]   # litter, garbage
THIN = 3               # 延伸段每 THIN 幀取 1 張
CATS_NAME = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def parse(fn):
    m = frame_re.match(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load(split, root):
    j = json.load(open(root / "annotations" / f"{split}.json", encoding="utf-8"))
    return j


def build_train(dry=False):
    src = load("train", SRC)
    ctrl = load("train", CTRL)

    src_imgs = {im["id"]: im for im in src["images"]}
    for im in src["images"]:
        im["clip"], im["frame"] = parse(im["file_name"])
    src_ann_by_img = defaultdict(list)
    for a in src["annotations"]:
        src_ann_by_img[a["image_id"]].append(a)

    # 母集：每 clip 的全幀 + litter/garbage 有標幀
    clip_frames = defaultdict(list)          # clip -> [(frame, file_name)]
    clip_lg_frames = defaultdict(list)       # clip -> [frame with cat2/5]
    fname_by_clipframe = {}                   # (clip,frame) -> file_name
    for im in src["images"]:
        if im["clip"] is None:
            continue
        clip_frames[im["clip"]].append((im["frame"], im["file_name"]))
        fname_by_clipframe[(im["clip"], im["frame"])] = im["file_name"]
    for a in src["annotations"]:
        if a["category_id"] in TARGET_CATS:
            im = src_imgs[a["image_id"]]
            if im["clip"] is not None:
                clip_lg_frames[im["clip"]].append(im["frame"])

    # 對照組（16th）選中的檔名 + 每 clip 的終點 hi
    ctrl_fnames = {im["file_name"] for im in ctrl["images"]}
    ctrl_hi = {}
    for im in ctrl["images"]:
        c, f = parse(im["file_name"])
        if c is None:
            continue
        ctrl_hi[c] = max(ctrl_hi.get(c, f), f)

    # 逐 clip 算抽稀尾巴
    tail_fnames = set()
    per_clip = []  # (clip, hi, endpoint, n_tail_candidate, n_tail_kept)
    for clip, hi in ctrl_hi.items():
        lg = clip_lg_frames.get(clip, [])
        endpoint = max(lg) if lg else hi
        cand = sorted(f for f, _ in clip_frames[clip] if hi < f <= endpoint)
        kept = [f for r, f in enumerate(cand) if r % THIN == 0]   # 每 THIN 幀取第 1 張
        for f in kept:
            tail_fnames.add(fname_by_clipframe[(clip, f)])
        if cand:
            per_clip.append((clip, hi, endpoint, len(cand), len(kept)))

    selected_fnames = ctrl_fnames | tail_fnames

    # 由母集重建 train.json（對照組幀的標註與 16th 一致；尾巴幀補上母集標註）
    out_imgs, out_anns = [], []
    nid = aid = 0
    cls_counter = Counter()
    n_neg = 0
    if not dry:
        (DST / "train").mkdir(parents=True, exist_ok=True)
    for im in sorted(src["images"], key=lambda x: x["file_name"]):
        if im["file_name"] not in selected_fnames:
            continue
        if not dry:
            shutil.copy2(SRC / "train" / im["file_name"], DST / "train" / im["file_name"])
        out_imgs.append({"id": nid, "file_name": im["file_name"],
                         "width": im["width"], "height": im["height"]})
        anns = src_ann_by_img[im["id"]]
        if not ({a["category_id"] for a in anns} & set(TARGET_CATS)):
            n_neg += 1
        for a in anns:
            out_anns.append({**a, "id": aid, "image_id": nid})
            cls_counter[a["category_id"]] += 1
            aid += 1
        nid += 1

    if not dry:
        (DST / "annotations").mkdir(parents=True, exist_ok=True)
        json.dump({"images": out_imgs, "annotations": out_anns, "categories": src["categories"]},
                  open(DST / "annotations" / "train.json", "w", encoding="utf-8"),
                  ensure_ascii=False)

    # ---- 報表 ----
    print("=" * 80)
    print(f"[train] 變因① 處理組（含落地靜止尾，THIN={THIN}）" + ("   [DRY]" if dry else ""))
    print("=" * 80)
    print(f"  對照組 16th 幀數        : {len(ctrl_fnames)}")
    print(f"  抽稀尾巴新增幀數        : {len(tail_fnames)}")
    print(f"  處理組總幀數            : {len(selected_fnames)}   "
          f"(Δ +{len(tail_fnames)}, +{len(tail_fnames)/len(ctrl_fnames)*100:.1f}%)")
    print(f"  其中無 litter/garbage(negative): {n_neg} ({n_neg/max(len(selected_fnames),1)*100:.0f}%)")
    print(f"  5 類框分布              : "
          + ", ".join(f"{CATS_NAME[k]}={cls_counter[k]}" for k in sorted(CATS_NAME)))
    print(f"\n  有新增尾巴的 clip = {len(per_clip)} 部（hi=16th終點, endpoint=最後litter/garbage幀）:")
    for clip, hi, ep, ncand, nkept in sorted(per_clip, key=lambda x: -x[4]):
        print(f"    +{nkept:2d}張 (候選{ncand:2d}, {hi}->{ep})  {clip[:58]}")
    return len(ctrl_fnames), len(tail_fnames), len(selected_fnames)


def copy_asis(dry=False):
    """valid / test / test_static594 直接複製 16th，維持完全相同。"""
    print("\n" + "=" * 80)
    print("[valid/test] 直接複製 16th（完全相同）" + ("   [DRY]" if dry else ""))
    print("=" * 80)
    for split in ["valid", "test"]:
        j = load(split, CTRL)
        print(f"  {split}: {len(j['images'])} 幀 / {len(j['annotations'])} 框")
        if not dry:
            (DST / split).mkdir(parents=True, exist_ok=True)
            for im in j["images"]:
                s = CTRL / split / im["file_name"]
                d = DST / split / im["file_name"]
                if not d.exists():
                    shutil.copy2(s, d)
            shutil.copy2(CTRL / "annotations" / f"{split}.json",
                         DST / "annotations" / f"{split}.json")
    # test_static594
    sub = CTRL / "annotations" / "test_static594.json"
    if sub.exists():
        js = json.load(open(sub, encoding="utf-8"))
        print(f"  test_static594: {len(js['images'])} 幀 / {len(js['annotations'])} 框")
        if not dry:
            shutil.copy2(sub, DST / "annotations" / "test_static594.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只統計不寫檔")
    args = ap.parse_args()

    for p in [SRC, CTRL]:
        assert p.exists(), f"找不到 {p}"
    if not args.dry:
        (DST / "annotations").mkdir(parents=True, exist_ok=True)
        for sp in ["train"]:
            if (DST / sp).exists():
                shutil.rmtree(DST / sp)

    print(f"母集 : {SRC.name}")
    print(f"對照 : {CTRL.name}  (16th)")
    print(f"輸出 : {DST}\n")
    build_train(dry=args.dry)
    copy_asis(dry=args.dry)
    print(f"\n{'（DRY，未寫檔）' if args.dry else '完成 →'} {DST}")
