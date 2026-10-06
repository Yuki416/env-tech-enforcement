#!/usr/bin/env python3
"""變因① 靜止尾補標：對「最後標註幀後仍有 ≥GAP_MIN 幀」的候選 clip，
把最後一個 litter/garbage bbox 往後延伸(propagate)到片尾，產出 review 影片供人工確認。

用途：母集靜止落地標註不齊（有的標到片尾、有的落地就停）。本腳本把「疑似落地靜止但未標」
的尾巴用最後靜止框往後複製、畫成影片，人工看影片標出「垃圾真正離開/看不見」的那一幀，
再據此補標、重建資料集。

輸出（HDD）：/path/to/data/ablation_static_tail_review/
  review_NN_<ascii_slug>.mp4   每部候選 clip 一支（綠=原標, 紅=延伸複製, 疊幀號）
  _候選清單.md                  clip↔編號對照 + 靜止/移動提示 + 待你填「離開幀」欄
  _propagated_draft.json        延伸段的草稿標註（clip->frame->bbox），供補標後快速定案

用法：PYTHONUTF8=1 python3 gen_static_tail_review.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

SRC = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")
OUT = Path("/path/to/data/ablation_static_tail_review")
GAP_MIN = 5          # 最後標註幀之後至少還有幾幀才算候選
STATIC_PX = 20       # 最後兩個 litter/garbage 幀中心位移 < 此值 → 視為靜止（與 SEED_THRESH 一致）
CONTEXT = 8          # 影片往前多帶幾幀當脈絡
FPS = 3
LG_CATS = {2, 5}
CATN = {2: "litter", 5: "garbage"}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def ascii_slug(clip):
    s = "".join(c if c.isascii() else "" for c in clip)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[-40:]


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def main():
    j = json.load(open(SRC / "annotations" / "train.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in j["images"]}
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        im["clip"], im["frame"] = (m.group(1), int(m.group(2)))

    # clip -> {frame: {"file": fn, "lg": [bbox...], "cat": [..]}}
    clip_data = defaultdict(dict)
    for im in j["images"]:
        clip_data[im["clip"]][im["frame"]] = {"file": im["file_name"], "lg": [], "cat": []}
    for a in j["annotations"]:
        if a["category_id"] in LG_CATS:
            im = imgs[a["image_id"]]
            clip_data[im["clip"]][im["frame"]]["lg"].append(a["bbox"])
            clip_data[im["clip"]][im["frame"]]["cat"].append(a["category_id"])

    # 找候選
    cands = []
    for clip, frames in clip_data.items():
        lg_frames = sorted(f for f, d in frames.items() if d["lg"])
        if not lg_frames:
            continue
        last_lg = lg_frames[-1]
        clip_end = max(frames)
        if clip_end - last_lg < GAP_MIN:
            continue
        # 靜止判定：最後一個 lg 幀 vs 前一個 lg 幀的最小中心位移
        static = None
        if len(lg_frames) >= 2:
            prev = lg_frames[-2]
            d1 = [center(b) for b in frames[last_lg]["lg"]]
            d0 = [center(b) for b in frames[prev]["lg"]]
            static = min(((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5 for a in d1 for b in d0)
        cands.append((clip, last_lg, clip_end, static, len(lg_frames)))

    cands.sort(key=lambda x: -(x[2]-x[1]))
    OUT.mkdir(parents=True, exist_ok=True)

    draft = {}
    rows = []
    for idx, (clip, last_lg, clip_end, static, n_lg) in enumerate(cands, 1):
        frames = clip_data[clip]
        prop_boxes = frames[last_lg]["lg"]
        prop_cats = frames[last_lg]["cat"]
        slug = ascii_slug(clip)
        vid = OUT / f"review_{idx:02d}_{slug}.mp4"

        # 草稿：延伸段每幀複製最後靜止框
        draft[clip] = {"last_lg": last_lg, "clip_end": clip_end,
                       "boxes": [[round(x, 1) for x in b] for b in prop_boxes],
                       "cats": prop_cats,
                       "frames": list(range(last_lg + 1, clip_end + 1))}

        # 讀第一張拿尺寸
        f0 = min(frames)
        sample = cv2.imread(str(SRC / "train" / frames[f0]["file"]))
        h, w = sample.shape[:2]
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        start = max(f0, last_lg - CONTEXT)
        for f in range(start, clip_end + 1):
            if f not in frames:
                continue
            img = cv2.imread(str(SRC / "train" / frames[f]["file"]))
            if img is None:
                continue
            if f <= last_lg:
                for b, c in zip(frames[f]["lg"], frames[f]["cat"]):
                    x, y, bw, bh = map(int, b)
                    cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                    cv2.putText(img, f"ORIG {CATN[c]}", (x, max(y-5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tag = "ORIG (annotated)"
            else:
                for b, c in zip(prop_boxes, prop_cats):
                    x, y, bw, bh = map(int, b)
                    cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
                    cv2.putText(img, f"PROP {CATN[c]}", (x, max(y-5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                tag = "PROPAGATED (needs review)"
            banner = f"#{idx:02d} f{f}/{clip_end}  last_annotated=f{last_lg}  {tag}"
            cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(img, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            vw.write(img)
        vw.release()

        shint = "?" if static is None else (f"STATIC({static:.0f}px)" if static < STATIC_PX
                                            else f"MOVING({static:.0f}px→拋射物?)")
        rows.append((idx, clip, last_lg, clip_end, clip_end-last_lg, n_lg, shint, slug))
        print(f"[{idx:02d}] {vid.name}  f{last_lg}->{clip_end} (+{clip_end-last_lg})  {shint}")

    # 候選清單
    md = ["# 變因① 靜止尾補標 — review 候選清單",
          "",
          f"共 {len(rows)} 部（母集 train 中「最後標註幀後仍有 ≥{GAP_MIN} 幀」）。",
          "綠框=母集原標、紅框=最後靜止框往後複製（待你確認）。",
          "看影片後，在「垃圾離開幀」填該部**垃圾最後一次還看得到**的幀號；",
          "全程都在→填 clip_end；紅框一出現就沒東西（拋射物飛走）→填 last_annotated。",
          "",
          "| # | 影片 | last_annotated | clip_end | 延伸幀數 | 靜止提示 | 垃圾離開幀(你填) | clip |",
          "|---|---|---|---|---|---|---|---|"]
    for idx, clip, last_lg, clip_end, gap, n_lg, shint, slug in rows:
        md.append(f"| {idx} | review_{idx:02d}_{slug}.mp4 | {last_lg} | {clip_end} | {gap} | {shint} |  | {clip} |")
    (OUT / "_候選清單.md").write_text("\n".join(md), encoding="utf-8")
    json.dump(draft, open(OUT / "_propagated_draft.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n完成 → {OUT}\n  影片 {len(rows)} 支 + _候選清單.md + _propagated_draft.json")


if __name__ == "__main__":
    main()
