#!/usr/bin/env python3
"""建立 v16 動態子集 = 從 chiayi_v16(121部) 過濾出「動態 litter/garbage」訓練集（第 15 次訓練）。

配方完全沿用第 13 次 `13th_v14_own_dynamic/資料前處理/build_v14_dynamic_litter.py`，
唯一差別是資料來源 v11(65部) → v16(121部)。**這是本次的主變因，其餘一律不動。**

設計：
  - 只過濾 train/valid；test 沿用 v16 完整版（8 部 / 655 張，不過濾）。
  - 保留 5 類：被選中的幀，其「全部 5 類標註」照留，只是「選哪些幀」由 litter/garbage 動態與否決定。
  - 動態種子：litter(2)/garbage(5) 各自跨相鄰含該類幀最近鄰配對，位移/Δframe >= SEED_THRESH。
  - 連續片段擷取：含種子的 clip 取 [首個種子 - HEAD_WINDOW, 末個種子 + TAIL_WINDOW] 之間所有幀
    （含中間無標註幀 → negative，配合 filter_empty_gt=False）。

人工覆寫表：直接沿用第 13 次。v11 的 65 部在 v16 中影像與標註 **完全未變動**（已實測驗證），
故使用者當初對這些 clip 的判斷仍然有效。新增的 56 部尚無覆寫，待人工 review 後再填。

輸出：/path/to/data/coco/chiayi_v16_own_dynamic_litter/
  annotations/{train,valid,test}.json + {train,valid,test}/ 影像
  annotations/test_static594.json  ← 主判定用：v16 test 中屬於 v11 原 594 張的子集
"""
import json
import re
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v16_garbage_litter_121vids")
DST = Path("/path/to/data/coco/chiayi_v16_own_dynamic_litter")
V11_TEST = Path("/path/to/data/coco/chiayi_v11_garbage_litter_corrected/annotations/test.json")

# ---- 配方常數（與第 13 次逐字相同，勿改；改了就不再是單變因對照）----
TARGET_CATS = [2, 5]
SEED_THRESH = 20     # px/幀
HEAD_WINDOW = 30
TAIL_WINDOW = 5
GAP_CAP = 15
MATCH_GATE_FRAC = 0.20

# ---- 人工覆寫表：沿用第 13 次（僅對 v11 那 65 部有效）----
# 逐 clip 尾巴 padding 覆寫（使用者比較 tail5/tail10 後決定）
TAIL_OVERRIDE = {
    "Day_Litter_12_8_loc12_bureau_20240603181958_04m49s_RoboFlow": 10,
    "Day_Litter_538_8_loc538_20240803182230_08m49s_RoboFlow": 10,
}
# 逐 clip 末幀絕對上限（演算法把整段當動態，但實際只有前段有意義）
END_CAP = {
    "Day_Litter_535_8_loc535_20240704064730_00m12s_RoboFlow": 13,
}
# 自動門檻抓不到、但使用者肉眼確認是丟垃圾動作的弱動態 clip → (起, 訖) 當種子
MANUAL_DYNAMIC = {
    "Day_Litter_537_8_loc537_20240807184204_01m10s": (16, 18),
    "Night_Litter_544_8_loc544_20240911024010_02m09s": (7, 10),
    "Day_Litter_538_8_loc538_20240804060854_15m33s": (18, 21),
    "Day_Litter_536_8_loc536_20240716055530_05m10s": (16, 17),
}
# 新增 56 部的覆寫：使用者看過「需重點觀察_無動態種子_4部_完整clip/」的診斷影片後指定（2026-07-17）。
# 語意與 MANUAL_DYNAMIC 相同 = **動態種子**，仍會套 HEAD_WINDOW/TAIL_WINDOW padding（非最終範圍）。
# 這 4 部自動門檻(20px/幀)抓不到（peak 位移 0.0~16.3px/幀），但使用者肉眼確認有丟棄動作。
MANUAL_DYNAMIC_V16_NEW = {
    # peak 0.0px/幀；種子 23→片尾(43)　→ padding 後 [1,43]（全片 43 張）
    "Day_Garbage_538_8_loc538_20240801172230_01m46s": (23, 43),
    # peak 5.7px/幀　→ padding 後 [1,53]（53 張）
    "Day_Litter_12_8_loc12_20240802173010_00m54s": (22, 48),
    # peak 16.3px/幀（最接近門檻）　→ padding 後 [1,32]（32 張）
    "Day_Litter_535_8_loc535_20240727161815_01m35s": (17, 27),
    # peak 0.0px/幀；單幀種子　→ padding 後 [6,41]（36 張）
    "Day_Litter_537_8_loc537_20240911175753_10m12s": (36, 36),
}

frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
CATS_NAME = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}


def parse(fn):
    m = frame_re.match(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def is_new_clip(clip):
    """v16 新增的 56 部不帶 _RoboFlow 後綴；v11 原有 65 部帶。"""
    return not clip.endswith("_RoboFlow")


def key_matches(clip, key):
    """精確比對：clip 等於 key，或等於 key + '_RoboFlow'。
    （第 13 次用 startswith，在 v16 新舊命名並存下改用精確比對以免誤中。）"""
    return clip == key or clip == f"{key}_RoboFlow"


def lookup(clip, table, default=None):
    for k, v in table.items():
        if key_matches(clip, k):
            return v
    return default


def validate_overrides(all_clips):
    """每個覆寫 key 必須且只能命中 1 個 clip，否則中止。
    （第 13 次的教訓：硬編碼路徑/名稱失效會靜默產生錯誤結果。）"""
    print("=" * 78)
    print("覆寫表驗證")
    print("=" * 78)
    ok = True
    for name, table in [("TAIL_OVERRIDE", TAIL_OVERRIDE), ("END_CAP", END_CAP),
                        ("MANUAL_DYNAMIC", MANUAL_DYNAMIC),
                        ("MANUAL_DYNAMIC_V16_NEW", MANUAL_DYNAMIC_V16_NEW)]:
        for k in table:
            hits = [c for c in all_clips if key_matches(c, k)]
            if len(hits) != 1:
                print(f"  ❌ {name}['{k[:50]}...'] 命中 {len(hits)} 個 clip（應為 1）")
                ok = False
            else:
                print(f"  ✅ {name}: {hits[0][:60]}")
    if not ok:
        sys.exit("覆寫表有 key 對不到 clip，中止（勿讓錯誤靜默通過）")
    print()


def _seeds_for_cat(imgs, cat_by_img):
    clip_frames = defaultdict(list)
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
    return seeds


def select_frames(split):
    j = json.load(open(SRC / "annotations" / f"{split}.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in j["images"]}
    for im in j["images"]:
        im["clip"], im["frame"] = parse(im["file_name"])

    boxes_by_cat = {c: defaultdict(list) for c in TARGET_CATS}
    for a in j["annotations"]:
        if a["category_id"] in TARGET_CATS:
            boxes_by_cat[a["category_id"]][a["image_id"]].append(a["bbox"])

    seed_frames, seeds_per_cat = set(), {}
    for c in TARGET_CATS:
        s = _seeds_for_cat(imgs, boxes_by_cat[c])
        seeds_per_cat[c] = s
        seed_frames |= s

    pool = defaultdict(list)
    for im in imgs.values():
        if im["clip"] is not None:
            pool[im["clip"]].append((im["frame"], im["id"]))

    seed_by_clip = defaultdict(list)
    for img_id in seed_frames:
        seed_by_clip[imgs[img_id]["clip"]].append(imgs[img_id]["frame"])

    auto_seeded = set(seed_by_clip)  # 自動門檻抓到的（人工補入前）

    for clip in pool:
        for table in (MANUAL_DYNAMIC, MANUAL_DYNAMIC_V16_NEW):
            hit = lookup(clip, table)
            if hit:
                seed_by_clip[clip].extend([hit[0], hit[1]])

    selected = set()
    for clip, frames in pool.items():
        seeds = seed_by_clip.get(clip)
        if not seeds:
            continue
        tail = lookup(clip, TAIL_OVERRIDE, TAIL_WINDOW)
        lo, hi = min(seeds) - HEAD_WINDOW, max(seeds) + tail
        cap = lookup(clip, END_CAP)
        if cap is not None:
            hi = min(hi, cap)
        for f_idx, img_id in frames:
            if lo <= f_idx <= hi:
                selected.add(img_id)

    return j, imgs, seed_frames, selected, seeds_per_cat, set(pool), auto_seeded


def build_split(split):
    j, imgs, seed_frames, selected, seeds_per_cat, all_clips, auto_seeded = select_frames(split)
    ann_by_img = defaultdict(list)
    for a in j["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    out_imgs, out_anns = [], []
    new_img_id = new_ann_id = 0
    cls_counter, n_no_target = Counter(), 0
    dst_dir = DST / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    for img_id in sorted(selected):
        im = imgs[img_id]
        shutil.copy2(SRC / split / im["file_name"], dst_dir / im["file_name"])
        out_imgs.append({"id": new_img_id, "file_name": im["file_name"],
                         "width": im["width"], "height": im["height"]})
        if not ({a["category_id"] for a in ann_by_img[img_id]} & set(TARGET_CATS)):
            n_no_target += 1
        for a in ann_by_img[img_id]:
            out_anns.append({**a, "id": new_ann_id, "image_id": new_img_id})
            cls_counter[a["category_id"]] += 1
            new_ann_id += 1
        new_img_id += 1

    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    json.dump({"images": out_imgs, "annotations": out_anns, "categories": j["categories"]},
              open(DST / "annotations" / f"{split}.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    sel_clips = {imgs[i]["clip"] for i in selected}
    new_sel = {c for c in sel_clips if is_new_clip(c)}
    old_sel = sel_clips - new_sel
    new_all = {c for c in all_clips if is_new_clip(c)}
    old_all = all_clips - new_all
    no_seed_new = sorted(new_all - auto_seeded)

    print("=" * 78)
    print(f"[{split}] v16 動態子集")
    print("=" * 78)
    print(f"  來源 v16          : {len(j['images'])} 幀 / {len(all_clips)} clips"
          f"  (v11舊 {len(old_all)} + 新增 {len(new_all)})")
    print(f"  動態種子幀(聯集)   : {len(seed_frames)}   ("
          + ", ".join(f"{CATS_NAME[c]}={len(seeds_per_cat[c])}" for c in TARGET_CATS) + ")")
    print(f"  連續片段選中       : {len(selected)} 幀 / {len(sel_clips)} clips"
          f"  (v11舊 {len(old_sel)} + 新增 {len(new_sel)})")
    print(f"    其中無 litter/garbage（negative 填充幀）: {n_no_target}"
          f" ({n_no_target/max(len(selected),1)*100:.0f}%)")
    print(f"  5 類分布          : " + ", ".join(f"{CATS_NAME[k]}={cls_counter[k]}"
                                                for k in sorted(CATS_NAME)))
    # 無自動種子的新 clip：區分「已人工補入」與「仍待 review」
    manual_added = [c for c in no_seed_new
                    if lookup(c, MANUAL_DYNAMIC_V16_NEW) or lookup(c, MANUAL_DYNAMIC)]
    pending = [c for c in no_seed_new if c not in manual_added]
    if manual_added:
        print(f"\n  ℹ️ 新增 clip 無自動種子但『已人工補入』= {len(manual_added)} 部:")
        for c in manual_added:
            f0, f1 = lookup(c, MANUAL_DYNAMIC_V16_NEW) or lookup(c, MANUAL_DYNAMIC)
            print(f"      - {c}  種子=({f0},{f1}) {'✅ 已選中' if c in sel_clips else '❌ 未選中'}")
    if pending:
        print(f"\n  ⚠️ 新增 clip 無動態種子且『尚未人工處理』= {len(pending)} 部（需 review）:")
        for c in pending:
            print(f"      - {c}")
    return pending


def copy_test():
    """test 沿用 v16 完整版（8 部 / 655 張），不過濾。另產出 v11 原 594 張子集供主判定。"""
    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    j = json.load(open(SRC / "annotations" / "test.json", encoding="utf-8"))
    json.dump(j, open(DST / "annotations" / "test.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    dst_test = DST / "test"
    dst_test.mkdir(parents=True, exist_ok=True)
    for f in (SRC / "test").iterdir():
        if not (dst_test / f.name).exists():
            shutil.copy2(f, dst_test / f.name)

    # ---- 主判定子集：v16 test ∩ v11 原 594 張 ----
    v11_names = {i["file_name"] for i in
                 json.load(open(V11_TEST, encoding="utf-8"))["images"]}
    keep_ids = {i["id"] for i in j["images"] if i["file_name"] in v11_names}
    sub_imgs = [i for i in j["images"] if i["id"] in keep_ids]
    sub_anns = [a for a in j["annotations"] if a["image_id"] in keep_ids]
    json.dump({"images": sub_imgs, "annotations": sub_anns, "categories": j["categories"]},
              open(DST / "annotations" / "test_static594.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    print("=" * 78)
    print("[test] 沿用 v16 完整版（不過濾）")
    print("=" * 78)
    print(f"  test.json            : {len(j['images'])} 張 / {len(j['annotations'])} 框（8 部）")
    print(f"  test_static594.json  : {len(sub_imgs)} 張 / {len(sub_anns)} 框"
          f"  ← 主判定，與第 13 次同一把尺")
    assert len(sub_imgs) == 594, f"❌ 靜止子集應為 594 張，實得 {len(sub_imgs)}"
    assert len(sub_anns) == 1823, f"❌ 靜止子集應為 1823 框，實得 {len(sub_anns)}"
    print("  ✅ 594 張 / 1823 框，與 v11 test 完全一致")


if __name__ == "__main__":
    print(f"配方: SEED_THRESH={SEED_THRESH}px/幀, HEAD={HEAD_WINDOW} TAIL={TAIL_WINDOW}, "
          f"目標類別={[CATS_NAME[c] for c in TARGET_CATS]}\n")

    all_clips = set()
    for sp in ["train", "valid"]:
        j = json.load(open(SRC / "annotations" / f"{sp}.json", encoding="utf-8"))
        all_clips |= {parse(i["file_name"])[0] for i in j["images"]}
    validate_overrides(all_clips)

    for sp in ["train", "valid"]:
        if (DST / sp).exists():
            shutil.rmtree(DST / sp)

    no_seed = {}
    for sp in ["train", "valid"]:
        no_seed[sp] = build_split(sp)
        print()
    copy_test()

    print(f"\n完成 → {DST}")
    total_pending = sum(len(v) for v in no_seed.values())
    if total_pending:
        print(f"\n⚠️ 尚有 {total_pending} 部新 clip 無動態種子且未人工處理，已列於上方。")
        print("   下一步：跑 gen_review_videos_15th.py 產出完整 clip 影片供人工判斷，")
        print("   確認有丟垃圾動作者，把 (起,訖) frame 填入 MANUAL_DYNAMIC_V16_NEW 後重跑本腳本。")
    else:
        print("\n✅ 所有無自動種子的新 clip 皆已人工判定完畢，資料集為定案版。")
