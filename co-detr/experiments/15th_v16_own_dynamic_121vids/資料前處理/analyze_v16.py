#!/usr/bin/env python3
"""v16 探查：(1) v11 是否原封不動 (2) 解析度分布 (3) 新舊資料的 litter/garbage 動態語意是否一致"""
import json, re
from collections import defaultdict, Counter
from pathlib import Path

V16 = Path("/tmp/v16_ann")
V11 = Path("/path/to/data/coco/chiayi_v11_garbage_litter_corrected/annotations")
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.I)
CATS = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}
GAP_CAP, MATCH_GATE_FRAC = 15, 0.20


def load(p):
    j = json.load(open(p, encoding="utf-8"))
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        im["clip"], im["frame"] = (m.group(1), int(m.group(2))) if m else (None, None)
    return j


def is_new(clip):
    """新增的 56 部不帶 _RoboFlow 後綴"""
    return not clip.endswith("_RoboFlow")


print("=" * 78)
print("(1) v11 的 65 部是否原封不動？（比對 file_name 集合）")
print("=" * 78)
for sp in ["train", "valid", "test"]:
    a, b = load(V16 / f"{sp}.json"), load(V11 / f"{sp}.json")
    fa = {i["file_name"] for i in a["images"]}
    fb = {i["file_name"] for i in b["images"]}
    kept, lost, added = fa & fb, fb - fa, fa - fb
    print(f"\n[{sp}] v11 {len(fb)} 張 → v16 {len(fa)} 張")
    print(f"  v11 影像保留: {len(kept)}/{len(fb)}   遺失: {len(lost)}   新增: {len(added)}")
    if lost:
        print(f"  ⚠️ 遺失範例: {list(lost)[:3]}")
    # 標註數比對（僅保留的影像）
    ann_a, ann_b = defaultdict(list), defaultdict(list)
    ida = {i["id"]: i["file_name"] for i in a["images"]}
    idb = {i["id"]: i["file_name"] for i in b["images"]}
    for x in a["annotations"]:
        ann_a[ida[x["image_id"]]].append((x["category_id"], tuple(round(v, 2) for v in x["bbox"])))
    for x in b["annotations"]:
        ann_b[idb[x["image_id"]]].append((x["category_id"], tuple(round(v, 2) for v in x["bbox"])))
    diff = [f for f in kept if sorted(ann_a[f]) != sorted(ann_b[f])]
    print(f"  保留影像中標註有變動: {len(diff)} 張", f"例: {diff[:2]}" if diff else "✅ 完全一致")

print("\n" + "=" * 78)
print("(2) 解析度分布：v11 舊資料 vs 新增 56 部（SEED_THRESH=20px 是絕對值！）")
print("=" * 78)
for sp in ["train", "valid", "test"]:
    j = load(V16 / f"{sp}.json")
    res = defaultdict(Counter)
    clips_by_grp = defaultdict(set)
    for im in j["images"]:
        grp = "新增56部" if is_new(im["clip"]) else "v11舊"
        res[grp][f'{im["width"]}x{im["height"]}'] += 1
        clips_by_grp[grp].add(im["clip"])
    print(f"\n[{sp}]")
    for grp in ["v11舊", "新增56部"]:
        if grp in res:
            print(f"  {grp:9s} ({len(clips_by_grp[grp])} clips): " +
                  ", ".join(f"{k}={v}張" for k, v in res[grp].most_common()))

print("\n" + "=" * 78)
print("(3) 標註語意：litter/garbage 的動態比例（新資料 vs v11 舊資料）")
print("    位移 = 跨相鄰含該類幀的最近鄰配對 / Δframe，單位 px/幀")
print("=" * 78)


def displacements(j, cat, want_new):
    imgs = {i["id"]: i for i in j["images"]}
    by_img = defaultdict(list)
    for a in j["annotations"]:
        if a["category_id"] == cat:
            by_img[a["image_id"]].append(a["bbox"])
    clip_frames = defaultdict(list)
    for iid in by_img:
        im = imgs[iid]
        if im["clip"] and is_new(im["clip"]) == want_new:
            clip_frames[im["clip"]].append((im["frame"], iid))
    out = []
    for clip, frames in clip_frames.items():
        frames.sort()
        for i, (f_idx, iid) in enumerate(frames):
            im = imgs[iid]
            gate = MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
            nxt = next((frames[j2] for j2 in range(i + 1, len(frames))
                        if 0 < frames[j2][0] - f_idx <= GAP_CAP), None)
            if nxt is None:
                continue
            d = nxt[0] - f_idx
            dst = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in by_img[nxt[1]]]
            for b in by_img[iid]:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                best = min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in dst)
                if best <= gate:
                    out.append(best / d)
    return out


for sp in ["train", "valid"]:
    j = load(V16 / f"{sp}.json")
    print(f"\n[{sp}]")
    for cat in [2, 5]:
        for want_new, label in [(False, "v11舊"), (True, "新增56部")]:
            ds = displacements(j, cat, want_new)
            if not ds:
                print(f"  {CATS[cat]:8s} {label:9s}: (無資料)")
                continue
            ds.sort()
            n = len(ds)
            dyn = sum(1 for x in ds if x >= 20) / n
            static = sum(1 for x in ds if x < 10) / n
            print(f"  {CATS[cat]:8s} {label:9s}: {n:5d} 對  中位={ds[n//2]:6.1f}px/幀  "
                  f"動態(≥20px)={dyn*100:4.1f}%  靜止(<10px)={static*100:4.1f}%")

print("\n" + "=" * 78)
print("(4) 有多少新 clip 會被 SEED_THRESH=20 的自動門檻抓到？（估人工 review 量）")
print("=" * 78)
for sp in ["train"]:
    j = load(V16 / f"{sp}.json")
    imgs = {i["id"]: i for i in j["images"]}
    hit = defaultdict(bool)
    all_new = {im["clip"] for im in j["images"] if is_new(im["clip"])}
    all_old = {im["clip"] for im in j["images"] if not is_new(im["clip"])}
    for cat in [2, 5]:
        by_img = defaultdict(list)
        for a in j["annotations"]:
            if a["category_id"] == cat:
                by_img[a["image_id"]].append(a["bbox"])
        cf = defaultdict(list)
        for iid in by_img:
            im = imgs[iid]
            if im["clip"]:
                cf[im["clip"]].append((im["frame"], iid))
        for clip, frames in cf.items():
            frames.sort()
            for i, (f_idx, iid) in enumerate(frames):
                im = imgs[iid]
                gate = MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
                nxt = next((frames[j2] for j2 in range(i + 1, len(frames))
                            if 0 < frames[j2][0] - f_idx <= GAP_CAP), None)
                if nxt is None:
                    continue
                d = nxt[0] - f_idx
                dst = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in by_img[nxt[1]]]
                for b in by_img[iid]:
                    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                    best = min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in dst)
                    if best <= gate and best / d >= 20:
                        hit[clip] = True
                        break
    new_hit = sum(1 for c in all_new if hit[c])
    old_hit = sum(1 for c in all_old if hit[c])
    print(f"\n[{sp}] 新增 56 部: {new_hit}/{len(all_new)} 有動態種子 "
          f"→ 需人工看的無種子 clip = {len(all_new)-new_hit} 部")
    print(f"      v11 舊 44 部: {old_hit}/{len(all_old)} 有動態種子 "
          f"（第13次人工補了 4 部）")
