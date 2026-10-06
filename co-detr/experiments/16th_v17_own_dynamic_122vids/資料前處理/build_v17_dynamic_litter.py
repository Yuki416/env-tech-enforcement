#!/usr/bin/env python3
"""建立 v17 動態子集 = 從 chiayi_v17(122部) 過濾出「動態 litter/garbage」訓練集（第 16 次訓練）。

配方沿用第 15 次 `15th_v16_own_dynamic_121vids/資料前處理/build_v16_dynamic_litter.py`，
**唯一刻意變動的旋鈕 = HEAD_WINDOW（收緊動態範圍）**，其餘（SEED_THRESH/TAIL_WINDOW/
純RGB取幀邏輯/5類保留/negative填充/filter_empty_gt=False 語意）一律不動。使用者決定：
只縮 HEAD_WINDOW，數值用 sweep 挑（見 --head / --dry）。

相對第 15 次，因 v17 換了資料集而重新推導的點（詳見實驗夾 README）：
  1. 資料來源 v16(121部) → v17(122部)：7 部原地重標（train，改名去 _RoboFlow）+ 1 部新增(valid)。
  2. ⚠️ 04m49s 這部被重標 → 第 15 次的 TAIL_OVERRIDE tail=10 是基於舊標註，**本次移除該覆寫**
     （使用者決定「重新推導、丟掉覆寫」），改吃預設 TAIL_WINDOW。
  3. 新增的 valid clip（Day_Garbage_16_8_loc16，garbage=4/litter=0/42張）**照動態過濾**，
     不特別強制納入（使用者決定）；garbage 框極少，極可能落選，build 後會回報是否入選。
  4. 7 部重標 clip 的框變了 → 自動種子會重算；若有 clip 因此完全沒種子，會列在「無種子」清單供 review。

其餘人工覆寫表（END_CAP / MANUAL_DYNAMIC / MANUAL_DYNAMIC_V16_NEW、及 TAIL_OVERRIDE 剩下那筆）
對應的 clip 在 v17 中影像與標註均未變動（已逐一驗證命中 1 個），故沿用有效。

輸出：/path/to/data/coco/chiayi_v17_own_dynamic_litter/
  annotations/{train,valid,test}.json + {train,valid,test}/ 影像
  annotations/test_static594.json  ← 主判定用：v17 test 中屬於 v11 原 594 張的子集
                                     （v17 test == v16 test，不過濾，assert 594/1823）

用法：
  python build_v17_dynamic_litter.py --head 15          # 建置到磁碟（HEAD_WINDOW=15）
  python build_v17_dynamic_litter.py --head 10 --dry     # 只計數、不寫檔（sweep 用）
"""
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")
DST = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter")
V11_TEST = Path("/path/to/data/coco/chiayi_v11_garbage_litter_corrected/annotations/test.json")

# ---- 配方常數（除 HEAD_WINDOW 外與第 15 次逐字相同，勿改）----
TARGET_CATS = [2, 5]
SEED_THRESH = 20     # px/幀
HEAD_WINDOW = 15     # ← 本次唯一變動旋鈕，預設 15，實際值由 --head 覆寫
TAIL_WINDOW = 5
GAP_CAP = 15
MATCH_GATE_FRAC = 0.20

# ---- 人工覆寫表：沿用第 15 次，但移除被重標的 04m49s 那筆 TAIL_OVERRIDE（使用者決定重新推導）----
TAIL_OVERRIDE = {
    # 04m49s 已因重標移除（原為 10）。
    "Day_Litter_538_8_loc538_20240803182230_08m49s_RoboFlow": 10,
}
# 逐 clip 末幀絕對上限
END_CAP = {
    "Day_Litter_535_8_loc535_20240704064730_00m12s_RoboFlow": 13,
}
# 逐 clip 首幀絕對下限（對稱於 END_CAP：lo = max(lo, N)）
START_CAP = {}
# 逐 clip 直接指定最終擷取範圍 (起, 訖)：完全跳過種子±HEAD/TAIL 與 END/START_CAP，所見即所得。
# 絕對幀號 → 該 clip 與 HEAD 無關。值由手動腳本（如 build_v17_manual_head10.py）填入，
# 共用 build 腳本此表保持空，純版不受影響。
RANGE_OVERRIDE = {}
# 自動門檻抓不到、但使用者肉眼確認是丟垃圾動作的弱動態 clip → (起, 訖) 當種子（沿用第 15 次）
MANUAL_DYNAMIC = {
    "Day_Litter_537_8_loc537_20240807184204_01m10s": (16, 18),
    "Night_Litter_544_8_loc544_20240911024010_02m09s": (7, 10),
    "Day_Litter_538_8_loc538_20240804060854_15m33s": (18, 21),
    "Day_Litter_536_8_loc536_20240716055530_05m10s": (16, 17),
}
# v16 新增 56 部的人工種子（沿用第 15 次；語意=動態種子，仍套 HEAD/TAIL padding）
MANUAL_DYNAMIC_V16_NEW = {
    "Day_Garbage_538_8_loc538_20240801172230_01m46s": (23, 43),
    "Day_Litter_12_8_loc12_20240802173010_00m54s": (22, 48),
    "Day_Litter_535_8_loc535_20240727161815_01m35s": (17, 27),
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
    """不帶 _RoboFlow 後綴者 = v16 新增 56 部 + 本次重標 7 部（均重新匯出，不帶後綴）。
    v11 原有者帶 _RoboFlow。用於報表分組與『無種子需 review』判斷。"""
    return not clip.endswith("_RoboFlow")


def key_matches(clip, key):
    return clip == key or clip == f"{key}_RoboFlow"


def lookup(clip, table, default=None):
    for k, v in table.items():
        if key_matches(clip, k):
            return v
    return default


def validate_overrides(all_clips):
    """每個覆寫 key 必須且只能命中 1 個 clip，否則中止（防硬編碼名稱失效靜默出錯）。"""
    print("=" * 78)
    print("覆寫表驗證")
    print("=" * 78)
    ok = True
    for name, table in [("TAIL_OVERRIDE", TAIL_OVERRIDE), ("END_CAP", END_CAP),
                        ("START_CAP", START_CAP), ("RANGE_OVERRIDE", RANGE_OVERRIDE),
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
        rng = lookup(clip, RANGE_OVERRIDE)
        if rng is not None:
            # 手動直接指定最終擷取範圍，跳過種子±HEAD/TAIL 與 END/START_CAP（所見即所得）。
            lo, hi = rng
        else:
            seeds = seed_by_clip.get(clip)
            if not seeds:
                continue
            tail = lookup(clip, TAIL_OVERRIDE, TAIL_WINDOW)
            lo, hi = min(seeds) - HEAD_WINDOW, max(seeds) + tail
            cap = lookup(clip, END_CAP)
            if cap is not None:
                hi = min(hi, cap)
            scap = lookup(clip, START_CAP)
            if scap is not None:
                lo = max(lo, scap)
        for f_idx, img_id in frames:
            if lo <= f_idx <= hi:
                selected.add(img_id)

    return j, imgs, seed_frames, selected, seeds_per_cat, set(pool), auto_seeded


def build_split(split, dry=False):
    j, imgs, seed_frames, selected, seeds_per_cat, all_clips, auto_seeded = select_frames(split)
    ann_by_img = defaultdict(list)
    for a in j["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    out_imgs, out_anns = [], []
    new_img_id = new_ann_id = 0
    cls_counter, n_no_target = Counter(), 0
    if not dry:
        dst_dir = DST / split
        dst_dir.mkdir(parents=True, exist_ok=True)

    for img_id in sorted(selected):
        im = imgs[img_id]
        if not dry:
            shutil.copy2(SRC / split / im["file_name"], DST / split / im["file_name"])
        out_imgs.append({"id": new_img_id, "file_name": im["file_name"],
                         "width": im["width"], "height": im["height"]})
        if not ({a["category_id"] for a in ann_by_img[img_id]} & set(TARGET_CATS)):
            n_no_target += 1
        for a in ann_by_img[img_id]:
            out_anns.append({**a, "id": new_ann_id, "image_id": new_img_id})
            cls_counter[a["category_id"]] += 1
            new_ann_id += 1
        new_img_id += 1

    if not dry:
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
    print(f"[{split}] v17 動態子集   (HEAD={HEAD_WINDOW} TAIL={TAIL_WINDOW} SEED={SEED_THRESH}px)"
          + ("   [DRY]" if dry else ""))
    print("=" * 78)
    print(f"  來源 v17          : {len(j['images'])} 幀 / {len(all_clips)} clips"
          f"  (帶_RoboFlow {len(old_all)} + 無後綴 {len(new_all)})")
    print(f"  動態種子幀(聯集)   : {len(seed_frames)}   ("
          + ", ".join(f"{CATS_NAME[c]}={len(seeds_per_cat[c])}" for c in TARGET_CATS) + ")")
    print(f"  連續片段選中       : {len(selected)} 幀 / {len(sel_clips)} clips"
          f"  (帶_RoboFlow {len(old_sel)} + 無後綴 {len(new_sel)})")
    print(f"    其中無 litter/garbage（negative 填充幀）: {n_no_target}"
          f" ({n_no_target/max(len(selected),1)*100:.0f}%)")
    print(f"  5 類分布          : " + ", ".join(f"{CATS_NAME[k]}={cls_counter[k]}"
                                                for k in sorted(CATS_NAME)))
    # 無自動種子的「無後綴」clip（含 v16 新增與本次重標）：區分已人工補入 vs 仍待 review
    manual_added = [c for c in no_seed_new
                    if lookup(c, MANUAL_DYNAMIC_V16_NEW) or lookup(c, MANUAL_DYNAMIC)]
    pending = [c for c in no_seed_new if c not in manual_added]
    if manual_added:
        print(f"\n  ℹ️ 無後綴 clip 無自動種子但『已人工補入』= {len(manual_added)} 部:")
        for c in manual_added:
            f0, f1 = lookup(c, MANUAL_DYNAMIC_V16_NEW) or lookup(c, MANUAL_DYNAMIC)
            print(f"      - {c}  種子=({f0},{f1}) {'✅ 已選中' if c in sel_clips else '❌ 未選中'}")
    if pending:
        print(f"\n  ⚠️ 無後綴 clip 無動態種子且『尚未人工處理』= {len(pending)} 部（需 review）:")
        for c in pending:
            print(f"      - {c}")
    return pending, {"selected": len(selected), "clips": len(sel_clips),
                     "seed": len(seed_frames), "litter_box": cls_counter[2],
                     "garbage_box": cls_counter[5], "neg": n_no_target}


def copy_test(dry=False):
    """test 沿用 v17 完整版（8 部 / 655 張，== v16 test），不過濾。另產出 v11 原 594 張子集供主判定。"""
    j = json.load(open(SRC / "annotations" / "test.json", encoding="utf-8"))
    if not dry:
        (DST / "annotations").mkdir(parents=True, exist_ok=True)
        json.dump(j, open(DST / "annotations" / "test.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        dst_test = DST / "test"
        dst_test.mkdir(parents=True, exist_ok=True)
        for f in (SRC / "test").iterdir():
            if not (dst_test / f.name).exists():
                shutil.copy2(f, dst_test / f.name)

    v11_names = {i["file_name"] for i in
                 json.load(open(V11_TEST, encoding="utf-8"))["images"]}
    keep_ids = {i["id"] for i in j["images"] if i["file_name"] in v11_names}
    sub_imgs = [i for i in j["images"] if i["id"] in keep_ids]
    sub_anns = [a for a in j["annotations"] if a["image_id"] in keep_ids]
    if not dry:
        json.dump({"images": sub_imgs, "annotations": sub_anns, "categories": j["categories"]},
                  open(DST / "annotations" / "test_static594.json", "w", encoding="utf-8"),
                  ensure_ascii=False)

    print("=" * 78)
    print("[test] 沿用 v17 完整版（不過濾，== v16 test）" + ("   [DRY]" if dry else ""))
    print("=" * 78)
    print(f"  test.json            : {len(j['images'])} 張 / {len(j['annotations'])} 框（8 部）")
    print(f"  test_static594.json  : {len(sub_imgs)} 張 / {len(sub_anns)} 框  ← 主判定，與第 13/15 次同尺")
    assert len(sub_imgs) == 594, f"❌ 靜止子集應為 594 張，實得 {len(sub_imgs)}"
    assert len(sub_anns) == 1823, f"❌ 靜止子集應為 1823 框，實得 {len(sub_anns)}"
    print("  ✅ 594 張 / 1823 框，與 v11 test 完全一致")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=int, default=HEAD_WINDOW, help="HEAD_WINDOW（收緊動態範圍的旋鈕）")
    ap.add_argument("--dry", action="store_true", help="只計數不寫檔（sweep 用）")
    ap.add_argument("--out", type=str, default=None,
                    help="輸出資料集目錄（預設 chiayi_v17_own_dynamic_litter；HEAD=10 等其他版本用此指定獨立目錄）")
    args = ap.parse_args()
    HEAD_WINDOW = args.head
    if args.out:
        DST = Path(args.out)
        print(f"輸出目錄改為: {DST}")

    print(f"配方: SEED_THRESH={SEED_THRESH}px/幀, HEAD={HEAD_WINDOW} TAIL={TAIL_WINDOW}, "
          f"目標類別={[CATS_NAME[c] for c in TARGET_CATS]}"
          + ("   [DRY RUN]" if args.dry else "") + "\n")

    all_clips = set()
    for sp in ["train", "valid"]:
        j = json.load(open(SRC / "annotations" / f"{sp}.json", encoding="utf-8"))
        all_clips |= {parse(i["file_name"])[0] for i in j["images"]}
    validate_overrides(all_clips)

    if not args.dry:
        for sp in ["train", "valid"]:
            if (DST / sp).exists():
                shutil.rmtree(DST / sp)

    no_seed, stats = {}, {}
    for sp in ["train", "valid"]:
        no_seed[sp], stats[sp] = build_split(sp, dry=args.dry)
        print()
    copy_test(dry=args.dry)

    print(f"\n{'（DRY，未寫檔）' if args.dry else '完成 →'} {DST}")
    total_pending = sum(len(v) for v in no_seed.values())
    if total_pending:
        print(f"\n⚠️ 尚有 {total_pending} 部無後綴 clip 無動態種子且未人工處理，已列於上方（可能是重標後掉種子）。")
    else:
        print("\n✅ 所有無後綴且無自動種子的 clip 皆已人工判定，無待 review。")
