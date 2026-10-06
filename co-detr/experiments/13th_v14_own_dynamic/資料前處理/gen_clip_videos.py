#!/usr/bin/env python3
"""把 v14（第13次訓練資料）每個 clip 的連續 frame 串成一部影片，供人工逐部檢視。

- 每個 clip 一部 mp4；同 clip 內按 frame index 排序串接。
- 疊 GT 框：litter 綠(粗)、garbage 紅(粗)、car/moto/person 細框（灰/藍/橙）。
- 上方疊字：frame idx + 該幀 litter/garbage 數量。
- 輸出（HDD）：13th_expt_v14_own_dynamic_36e/連續片段影片/{train,valid}/<clip>.mp4

用法：python gen_clip_videos.py [LIMIT]   # LIMIT 可選，只產前 N 部（煙霧測試）
"""
import cv2
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/workspace/datasets/coco/chiayi_v14_own_dynamic_litter")
# 輸出寫 HDD：執行容器須掛載 -v /path/to/data:/workspace/hdd2_storage
OUT_BASE = Path("/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/連續片段影片")
FPS = 8
# category_id -> (名稱, BGR 顏色, 線寬)
CAT = {
    1: ("car", (160, 160, 160), 2),
    2: ("litter", (0, 255, 0), 3),
    3: ("motorcycle", (255, 128, 0), 2),
    4: ("person", (0, 165, 255), 2),
    5: ("garbage", (0, 0, 255), 3),
}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)


def run(split, budget):
    j = json.load(open(DATA / "annotations" / f"{split}.json"))
    imgs = {im["id"]: im for im in j["images"]}
    anns = defaultdict(list)
    for a in j["annotations"]:
        anns[a["image_id"]].append(a)
    clips = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            clips[m.group(1)].append((int(m.group(2)), im["id"]))

    outdir = OUT_BASE / split
    outdir.mkdir(parents=True, exist_ok=True)
    made = 0
    for clip, frames in sorted(clips.items()):
        if budget is not None and made >= budget:
            break
        frames.sort()
        first = cv2.imread(str(DATA / split / imgs[frames[0][1]]["file_name"]))
        h, w = first.shape[:2]
        name = sanitize(clip)[:90]
        vw = cv2.VideoWriter(str(outdir / f"{name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        nseed_lit = nseed_gar = 0
        for fidx, iid in frames:
            img = cv2.imread(str(DATA / split / imgs[iid]["file_name"]))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            n_lit = n_gar = 0
            for a in anns[iid]:
                nm, color, th = CAT[a["category_id"]]
                x, y, bw, bh = [int(v) for v in a["bbox"]]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, th)
                if a["category_id"] == 2:
                    n_lit += 1
                elif a["category_id"] == 5:
                    n_gar += 1
            nseed_lit = max(nseed_lit, n_lit); nseed_gar = max(nseed_gar, n_gar)
            cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(img, f"f{fidx}  litter={n_lit}  garbage={n_gar}  (green=litter red=garbage)",
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            vw.write(img)
        vw.release()
        made += 1
        print(f"[{split}] {name}.mp4  frames={len(frames)}  peak litter={nseed_lit} garbage={nseed_gar}",
              flush=True)
    return made


if __name__ == "__main__":
    budget = LIMIT
    for sp in ["train", "valid"]:
        n = run(sp, budget)
        if budget is not None:
            budget -= n
    print("DONE")
