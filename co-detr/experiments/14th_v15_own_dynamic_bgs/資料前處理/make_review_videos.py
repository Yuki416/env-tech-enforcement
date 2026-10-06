#!/usr/bin/env python3
"""把 valid / test 每部 clip 的 v15 changemap 用幀做成一支 review 影片。

輸出：<DST>/_review_changemap/{valid,test}/<clip>.mp4   （不分 per-clip 子資料夾）
每幀左右並排：
  左 = 原始 RGB + GT 框（litter=綠, garbage=藍）
  右 = changemap 原始黑白圖（白=前景, 黑=背景）+ 同 GT 框
上方資訊列（ASCII，cv2 無中文字型）：clip / frame idx / 暖機方式。
只放 v15 實際收錄的用幀（= 動態子集），忠實呈現模型會看到的 changemap。fps=4。
"""
import json, re
from pathlib import Path
import cv2, numpy as np

V14 = Path("/path/to/data/coco/chiayi_v14_own_dynamic_litter")
DST = Path("/path/to/data/coco/chiayi_v15_own_dynamic_bgs")
ALIGN = json.load(open(Path(__file__).resolve().parent / "alignment_v15.json"))
TAIL = {"train/20240713054438_00m00s"}          # 尾巴暖機的 clip（僅 train，此處不會用到）
OUTROOT = DST / "_review_changemap"
PANEL_W = 640
FPS = 4
GT_COLOR = {2: (0, 255, 0), 5: (255, 128, 0)}   # BGR: litter=綠, garbage=藍

def fnum(n): return int(re.search(r"_frame_(\d+)", n).group(1))

def clip_key(split, name):
    m = re.search(r"(\d{14})_(\d{2})m(\d{2})s", name)
    return f"{split}/{m.group(1)}_{m.group(2)}m{m.group(3)}s"

def panel(img):
    h = int(round(PANEL_W / img.shape[1] * img.shape[0]))
    return cv2.resize(img, (PANEL_W, h)), h

def draw_gt(vis, boxes, sx, sy):
    for c, (x, y, w, h) in boxes:
        if c not in GT_COLOR:
            continue
        cv2.rectangle(vis, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), GT_COLOR[c], 2)

def main():
    for split in ["valid", "test"]:
        outdir = OUTROOT / split
        outdir.mkdir(parents=True, exist_ok=True)
        d = json.load(open(V14 / f"annotations/{split}.json"))
        gt = {}
        for a in d["annotations"]:
            gt.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"]))
        clips = {}
        for im in d["images"]:
            clips.setdefault(clip_key(split, im["file_name"]), []).append(im)

        for key, ims in sorted(clips.items()):
            ims.sort(key=lambda im: fnum(im["file_name"]))
            info = ALIGN[key]
            warm = "tail-warmup" if key in TAIL else f"pre-roll={info['preroll']}"
            cname = key.split("/")[1]

            # 先讀一張定尺寸
            first = cv2.imread(str(V14 / split / ims[0]["file_name"]))
            _, ph = panel(first)
            HEAD = 34
            W, H = PANEL_W * 2, ph + HEAD
            vw = cv2.VideoWriter(str(outdir / f"{cname}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

            for im in ims:
                name = im["file_name"]
                rgb = cv2.imread(str(V14 / split / name))
                cm = cv2.imread(str(DST / f"{split}_changemap" / (name[:-4] + ".png")), 0)
                sx, sy = PANEL_W / rgb.shape[1], ph / rgb.shape[0]
                boxes = gt.get(im["id"], [])

                left, _ = panel(rgb)
                draw_gt(left, boxes, sx, sy)

                right, _ = panel(cv2.cvtColor(cm, cv2.COLOR_GRAY2BGR))   # 原始黑白 changemap
                draw_gt(right, boxes, sx, sy)

                canvas = np.zeros((H, W, 3), np.uint8)
                canvas[HEAD:HEAD+ph, :PANEL_W] = left
                canvas[HEAD:HEAD+ph, PANEL_W:] = right
                nlit = sum(1 for c, _ in boxes if c == 2)
                ngar = sum(1 for c, _ in boxes if c == 5)
                cv2.putText(canvas, f"{cname}  f{fnum(name):04d}  {warm}  GT litter={nlit} garbage={ngar}",
                            (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "RGB+GT", (8, HEAD+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "changemap B&W (white=foreground)", (PANEL_W+8, HEAD+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                vw.write(canvas)
            vw.release()
            print(f"[{split}] {cname}  {len(ims)} 幀  {warm}")
    print(f"\n✅ 完成 → {OUTROOT}")

if __name__ == "__main__":
    main()
