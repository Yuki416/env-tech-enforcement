#!/usr/bin/env python3
"""把變因① 處理組(statictail_ablation) train 每個 clip 的標註做成 review 影片，供人工肉眼查核。

每支影片 = 一個 clip，逐幀畫 5 類標註框（litter/garbage 粗+亮、其餘細+暗），banner 標：
  clip、frame、以及該幀是 CONTROL（16th 原有）還是 TAIL（本次靜止尾新增）。
→ 一眼看出「靜止尾加了哪些落地靜止幀、標得對不對」。

輸出（HDD）：/path/to/data/ablation_train_annotation_review/
  review_train_NN_<slug>.mp4  每 clip 一支
  _clip清單.md                 clip↔編號 + 幀數 + tail 幀數

用法：PYTHONUTF8=1 python3 gen_train_annotation_review.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

DS = Path("/path/to/data/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation")
CTRL = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual")  # 16th=CONTROL
OUT = Path("/path/to/data/ablation_train_annotation_review")
FPS = 3
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
# BGR：litter/garbage 亮，其餘暗
CAT = {2: ("litter", (0, 255, 0), 2), 5: ("garbage", (0, 165, 255), 2),
       1: ("car", (150, 150, 150), 1), 4: ("person", (255, 150, 0), 1),
       3: ("moto", (200, 0, 200), 1)}


def parse(fn):
    m = frame_re.match(fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def ascii_slug(clip):
    return re.sub(r"_+", "_", "".join(c if c.isascii() else "" for c in clip)).strip("_")[-40:]


def main():
    j = json.load(open(DS / "annotations" / "train.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in j["images"]}
    ann_by_img = defaultdict(list)
    for a in j["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    ctrl_fnames = {im["file_name"] for im in
                   json.load(open(CTRL / "annotations" / "train.json", encoding="utf-8"))["images"]}

    clip_imgs = defaultdict(list)   # clip -> [(frame, image_id, file_name)]
    for im in j["images"]:
        c, f = parse(im["file_name"])
        clip_imgs[c].append((f, im["id"], im["file_name"]))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, clip in enumerate(sorted(clip_imgs), 1):
        frames = sorted(clip_imgs[clip])
        slug = ascii_slug(clip)
        vid = OUT / f"review_train_{idx:02d}_{slug}.mp4"
        first = cv2.imread(str(DS / "train" / frames[0][2]))
        if first is None:
            continue
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        n_tail = 0
        for f, iid, fn in frames:
            img = cv2.imread(str(DS / "train" / fn))
            if img is None:
                continue
            for a in ann_by_img[iid]:
                name, col, th = CAT.get(a["category_id"], ("?", (255, 255, 255), 1))
                x, y, bw, bh = map(int, a["bbox"])
                cv2.rectangle(img, (x, y), (x + bw, y + bh), col, th)
                if a["category_id"] in (2, 5):
                    cv2.putText(img, name, (x, max(y - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            is_tail = fn not in ctrl_fnames
            if is_tail:
                n_tail += 1
            tag = "TAIL(靜止尾新增)" if is_tail else "CONTROL(16th)"
            tcol = (0, 0, 255) if is_tail else (0, 200, 0)
            cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(img, f"#{idx:02d} f{f} [{tag}]", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcol, 2)
            vw.write(img)
        vw.release()
        rows.append((idx, clip, len(frames), n_tail, slug))
        print(f"[{idx:02d}] {vid.name}  幀={len(frames)} tail={n_tail}")

    md = ["# 處理組 train 標註 review — clip 清單", "",
          f"共 {len(rows)} 個 clip。banner：綠=CONTROL(16th原有)、紅=TAIL(靜止尾新增)。",
          "框：litter=綠 garbage=橘（粗）；car=灰 person=藍 moto=紫（細）。", "",
          "| # | 影片 | 總幀 | 其中TAIL新增 | clip |", "|---|---|---|---|---|"]
    for idx, clip, nfr, ntail, slug in rows:
        md.append(f"| {idx} | review_train_{idx:02d}_{slug}.mp4 | {nfr} | {ntail} | {clip} |")
    (OUT / "_clip清單.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n完成 → {OUT}（{len(rows)} 支 + _clip清單.md）")


if __name__ == "__main__":
    main()
