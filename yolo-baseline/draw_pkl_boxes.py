#!/usr/bin/env python3
"""步驟 5 驗收第 2 道：把 pkl 裡的預測框畫回原圖，人工確認位置正確。

刻意不經過 Ultralytics，只讀 pkl 與 COCO json，用最單純的方式畫：
綠色 = 正確答案框（COCO），其他顏色 = 預測框（只畫信心分數 ≥ 0.3 的，避免畫面太亂），旁邊標類別與分數。
如果座標尺度錯了（例如誤用 1024 縮放後的座標），預測框會整片縮在左上角，一眼就看得出來。

用法（容器內）：python draw_pkl_boxes.py --pkl <x.pkl> --ann <test_dynamic166.json> --img-dir <dir> --out <dir> [--n 5]
"""
import argparse
import json
import os
import pickle
import random

from PIL import Image, ImageDraw

NAMES = ["car", "litter", "motorcycle", "person", "garbage"]
COLORS = ["#3b82f6", "#ef4444", "#06b6d4", "#eab308", "#a855f7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.3)
    a = ap.parse_args()

    coco = json.load(open(a.ann, encoding="utf-8"))
    ims = {im["id"]: im for im in coco["images"]}
    gts = {}
    for g in coco["annotations"]:
        gts.setdefault(g["image_id"], []).append(g)
    recs = pickle.load(open(a.pkl, "rb"))
    # 優先挑有 litter 正確答案的圖，最能看出小框有沒有對準
    with_litter = [r for r in recs if any(g["category_id"] == 2 for g in gts.get(r["img_id"], []))]
    random.seed(0)
    pick = random.sample(with_litter, min(a.n, len(with_litter)))
    os.makedirs(a.out, exist_ok=True)
    for r in pick:
        im = ims[r["img_id"]]
        with Image.open(os.path.join(a.img_dir, im["file_name"])) as pil:
            pil = pil.convert("RGB")
            d = ImageDraw.Draw(pil)
            for g in gts.get(r["img_id"], []):
                x, y, w, h = g["bbox"]
                d.rectangle([x, y, x + w, y + h], outline="#22c55e", width=5)
            p = r["pred_instances"]
            for (x1, y1, x2, y2), s, l in zip(p["bboxes"], p["scores"], p["labels"]):
                if s < a.min_score:
                    continue
                d.rectangle([x1, y1, x2, y2], outline=COLORS[l], width=3)
                d.text((x1 + 3, max(0, y1 - 14)), f"{NAMES[l]} {s:.2f}", fill=COLORS[l])
            pil.thumbnail((1400, 1400))
            name = f"{os.path.splitext(im['file_name'])[0][-48:]}.jpg"
            pil.save(os.path.join(a.out, name), quality=85)
            print("saved", name)


if __name__ == "__main__":
    main()
