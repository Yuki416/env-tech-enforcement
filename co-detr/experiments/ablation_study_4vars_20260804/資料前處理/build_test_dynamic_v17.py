#!/usr/bin/env python3
"""變因③ test 更新：建 v17 標準的「動態子集」test（landing 20px），並產 review 影片供人工複核。

決策見 `../test_v17_update_決策.md`。landing 定義（精確規格）：
  - 主物件 = 每幀 litter/garbage 中面積最大的框。
  - 位移 d(f) = 主物件框心在 f 與「前一個有 litter/garbage 標註的幀」之間的歐氏距離(px)。
  - landing = 使 d(f) > T(=20px) 的最後一個幀。
  - 保留範圍 = [主物件第一個有標幀, landing+1]；其餘靜止長尾捨棄。
  - 全片位移都 ≤ T → 該片不進 dynamic。
  - GT = v17母集 test 5 類標註原樣搬入。

輸出（→ chiayi_v17_own_dynamic_head10_manual_statictail_ablation/annotations/）：
  test_dynamic_v17_8vids.json   8 部完整
  test_dynamic_v17_7vids.json   7 部（排 13m21s 洩漏片）
review 影片（HDD）：/path/to/data/ablation_test_dynamic_review/
  review_dyn_<slug>.mp4  每部 test 影片一支（全片；綠=主物件 黃=其他 lg 框；banner 標 d(f)/KEEP/DROP/LANDING）
  _landing報告.md

用法：PYTHONUTF8=1 python3 build_test_dynamic_v17.py            # 建 json + 報告
      PYTHONUTF8=1 python3 build_test_dynamic_v17.py --videos  # 另產 review 影片
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

SRC = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")
DST = Path("/path/to/data/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation")
VID_OUT = Path("/path/to/data/ablation_test_dynamic_review")
T = 20.0
LG = {2, 5}
LEAK = "20240710172351_13m21s"   # 洩漏片（與 train 06m27s 近重複）→ 7vids 版排除

# 使用者定案（2026-08-04）：這幾部沿用舊制手動修正的 landing「保留上界」。
# 只覆寫上界(訖=hi)，起點仍用「首個有標幀」（與其餘影片一致的乾淨起點）。
# key = 影片時間戳(14碼)；value = 保留上界幀號(= 舊 build_v13.py KEEP 值)。
KEEP_HI_OVERRIDE = {
    "20240911055458": 20,   # 舊手動修正 landing→19，保留至 20
    "20240803182230": 24,   # 舊手動修正 landing→23，保留至 24（起點仍 f18）
    "20240829185652": 24,   # 舊手動修正 landing→23，保留至 24
}
CATN = {1: "car", 2: "litter", 3: "moto", 4: "person", 5: "garbage"}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def parse(fn):
    m = frame_re.match(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def area(b):
    return b[2] * b[3]


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def ascii_slug(clip):
    s = "".join(c if c.isascii() else "" for c in clip)
    return re.sub(r"_+", "_", s).strip("_")[-42:]


def compute_landing(clip, frames_lg):
    """frames_lg: {frame: [bbox of litter/garbage...]}. 回傳 (landing, keep_lo, keep_hi, disp_map) 或 None。"""
    ann_frames = sorted(frames_lg)
    if not ann_frames:
        return None
    # 主物件 = 每幀最大 lg 框；位移 = 與前一個有標幀比
    main = {f: max(frames_lg[f], key=area) for f in ann_frames}
    disp = {}
    landing = None
    for i, f in enumerate(ann_frames):
        if i == 0:
            disp[f] = None
            continue
        pc = center(main[ann_frames[i - 1]])
        cc = center(main[f])
        d = ((pc[0] - cc[0]) ** 2 + (pc[1] - cc[1]) ** 2) ** 0.5
        disp[f] = d
        if d > T:
            landing = f
    if landing is None:
        return None
    keep_lo = ann_frames[0]
    keep_hi = landing + 1
    return landing, keep_lo, keep_hi, disp, main


def main_build(make_videos):
    j = json.load(open(SRC / "annotations" / "test.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in j["images"]}
    for im in j["images"]:
        im["clip"], im["frame"] = parse(im["file_name"])
    ann_by_img = defaultdict(list)
    for a in j["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    clip_frames = defaultdict(dict)   # clip -> {frame: image_id}
    clip_lg = defaultdict(lambda: defaultdict(list))  # clip -> {frame: [lg bbox]}
    for im in j["images"]:
        clip_frames[im["clip"]][im["frame"]] = im["id"]
    for a in j["annotations"]:
        if a["category_id"] in LG:
            im = imgs[a["image_id"]]
            clip_lg[im["clip"]][im["frame"]].append(a["bbox"])

    rows = []
    kept_img_ids = set()
    per_clip_keep = {}
    for clip in sorted(clip_frames):
        res = compute_landing(clip, clip_lg[clip])
        if res is None:
            rows.append((clip, None, None, None, 0, "無動態(不進dynamic)"))
            continue
        landing, lo, hi, disp, mainbox = res
        status = "OK"
        for ts, ov_hi in KEEP_HI_OVERRIDE.items():
            if ts in clip:
                hi = ov_hi
                status = f"OK(手動上界→{ov_hi})"
                break
        keep_frames = [f for f in clip_frames[clip] if lo <= f <= hi]
        per_clip_keep[clip] = (lo, hi, landing)
        for f in keep_frames:
            kept_img_ids.add(clip_frames[clip][f])
        rows.append((clip, landing, lo, hi, len(keep_frames), status))

    # 建 json（8vids / 7vids）
    def dump(name, exclude_leak):
        sel = []
        for im in j["images"]:
            if im["id"] not in kept_img_ids:
                continue
            if exclude_leak and LEAK in im["file_name"]:
                continue
            sel.append(im)
        sel_ids = {im["id"] for im in sel}
        out_imgs, out_anns = [], []
        old2new = {}
        for nid, im in enumerate(sorted(sel, key=lambda x: x["file_name"])):
            old2new[im["id"]] = nid
            out_imgs.append({"id": nid, "file_name": im["file_name"],
                             "width": im["width"], "height": im["height"]})
        aid = 0
        catcnt = defaultdict(int)
        for a in j["annotations"]:
            if a["image_id"] in sel_ids:
                out_anns.append({**a, "id": aid, "image_id": old2new[a["image_id"]]})
                catcnt[a["category_id"]] += 1
                aid += 1
        json.dump({"images": out_imgs, "annotations": out_anns, "categories": j["categories"]},
                  open(DST / "annotations" / name, "w", encoding="utf-8"), ensure_ascii=False)
        return len(out_imgs), len(out_anns), catcnt

    (DST / "annotations").mkdir(parents=True, exist_ok=True)
    n8 = dump("test_dynamic_v17_8vids.json", exclude_leak=False)
    n7 = dump("test_dynamic_v17_7vids.json", exclude_leak=True)

    # 報告
    print("=" * 92)
    print("變因③ test 動態子集（landing 20px）逐部 landing")
    print("=" * 92)
    md = ["# test 動態子集 landing 報告（landing 20px, v17）", "",
          "| 影片 | landing | 保留範圍[起,訖] | 保留幀 | 狀態 |",
          "|---|---|---|---|---|"]
    for clip, landing, lo, hi, nkeep, status in rows:
        rng = f"[{lo},{hi}]" if lo is not None else "-"
        leak = " ⚠️洩漏片" if LEAK in clip else ""
        print(f"  {clip[:52]:52} landing={str(landing):>4} 範圍={rng:>10} 保留={nkeep:>3}  {status}{leak}")
        md.append(f"| {clip}{leak} | {landing} | {rng} | {nkeep} | {status} |")
    print(f"\n  8vids: {n8[0]} 幀 / {n8[1]} 框  " + ", ".join(f"{CATN[k]}={n8[2][k]}" for k in sorted(n8[2])))
    print(f"  7vids: {n7[0]} 幀 / {n7[1]} 框  " + ", ".join(f"{CATN[k]}={n7[2][k]}" for k in sorted(n7[2])))
    md += ["", f"**8vids**: {n8[0]} 幀 / {n8[1]} 框（" + ", ".join(f"{CATN[k]}={n8[2][k]}" for k in sorted(n8[2])) + "）",
           f"**7vids**（排 13m21s）: {n7[0]} 幀 / {n7[1]} 框（" + ", ".join(f"{CATN[k]}={n7[2][k]}" for k in sorted(n7[2])) + "）"]
    VID_OUT.mkdir(parents=True, exist_ok=True)
    (VID_OUT / "_landing報告.md").write_text("\n".join(md), encoding="utf-8")

    if make_videos:
        print("\n產 review 影片…")
        for clip in sorted(clip_frames):
            frames = sorted(clip_frames[clip])
            keep = per_clip_keep.get(clip)
            f0 = frames[0]
            sample = cv2.imread(str(SRC / "test" / imgs[clip_frames[clip][f0]]["file_name"]))
            if sample is None:
                continue
            h, w = sample.shape[:2]
            vid = VID_OUT / f"review_dyn_{ascii_slug(clip)}.mp4"
            vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
            for f in frames:
                im = imgs[clip_frames[clip][f]]
                img = cv2.imread(str(SRC / "test" / im["file_name"]))
                if img is None:
                    continue
                lgb = clip_lg[clip].get(f, [])
                mainb = max(lgb, key=area) if lgb else None
                for b in lgb:
                    x, y, bw, bh = map(int, b)
                    col = (0, 255, 0) if b is mainb else (0, 255, 255)
                    cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 2)
                if keep:
                    lo, hi, landing = keep
                    inkeep = lo <= f <= hi
                    tag = "KEEP" if inkeep else "DROP"
                    tcol = (0, 200, 0) if inkeep else (0, 0, 255)
                    if f == landing:
                        tag += "  <<LANDING"
                else:
                    tag, tcol = "no-dynamic", (0, 0, 255)
                cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
                cv2.putText(img, f"f{f}/{frames[-1]}  {tag}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcol, 2)
                vw.write(img)
            vw.release()
            print(f"  {vid.name}")
    print(f"\n完成 → json 在 {DST}/annotations/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true")
    args = ap.parse_args()
    main_build(args.videos)
