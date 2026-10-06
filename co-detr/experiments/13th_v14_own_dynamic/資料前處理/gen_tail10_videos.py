#!/usr/bin/env python3
"""產『tail=10』比較版影片（動態最後一幀後 padding 10，頭段 HEAD=30 不變）。

- 不動 tail=5 的資料集與影片：import build 模組，只在 runtime 覆寫 b.TAIL_WINDOW=10，
  直接從 v11 原始幀算選片＋畫框產影片，輸出到獨立資料夾 連續片段影片_tail10/。
- 疊字標 [TAIL=10] 以便和 tail=5 版區分。
容器內執行（cv2）。
"""
import cv2
import re
from collections import defaultdict
from pathlib import Path
import importlib.util, sys

_HERE = "/workspace/project/experiments/13th_v14_own_dynamic/資料前處理"
spec = importlib.util.spec_from_file_location("b", f"{_HERE}/build_v14_dynamic_litter.py")
b = importlib.util.module_from_spec(spec); sys.modules["b"] = b; spec.loader.exec_module(b)

b.SRC = Path("/workspace/datasets/coco/chiayi_v11_garbage_litter_corrected")  # 容器內路徑
b.TAIL_WINDOW = 10  # ← 唯一改動：尾巴 5→10（頭段 30 不變）

# 輸出寫 HDD：執行容器須掛載 -v /path/to/data:/workspace/hdd2_storage
OUT_BASE = Path("/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/連續片段影片_tail10")
FPS = 8
CAT = {1: ("car", (160, 160, 160), 2), 2: ("litter", (0, 255, 0), 3),
       3: ("motorcycle", (255, 128, 0), 2), 4: ("person", (0, 165, 255), 2),
       5: ("garbage", (0, 0, 255), 3)}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)


for split in ["train", "valid"]:
    j, imgs, bc, seed_frames, selected, _ = b.select_frames(split)
    anns = defaultdict(list)
    for a in j["annotations"]:
        anns[a["image_id"]].append(a)
    clips = defaultdict(list)
    for iid in selected:
        clips[imgs[iid]["clip"]].append((imgs[iid]["frame"], iid))
    outdir = OUT_BASE / split
    outdir.mkdir(parents=True, exist_ok=True)
    for clip, frames in sorted(clips.items()):
        frames.sort()
        first = cv2.imread(str(b.SRC / split / imgs[frames[0][1]]["file_name"]))
        h, w = first.shape[:2]
        name = sanitize(clip)[:90]
        vw = cv2.VideoWriter(str(outdir / f"{name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for fidx, iid in frames:
            img = cv2.imread(str(b.SRC / split / imgs[iid]["file_name"]))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            nl = ng = 0
            for a in anns[iid]:
                nm, color, th = CAT[a["category_id"]]
                x, y, bw, bh = [int(v) for v in a["bbox"]]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, th)
                if a["category_id"] == 2:
                    nl += 1
                elif a["category_id"] == 5:
                    ng += 1
            cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(img, f"f{fidx}  litter={nl}  garbage={ng}  [TAIL=10]",
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            vw.write(img)
        vw.release()
        print(f"[{split}] {name}.mp4  frames={len(frames)}", flush=True)
print("DONE")
