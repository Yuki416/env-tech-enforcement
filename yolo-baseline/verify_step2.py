#!/usr/bin/env python3
"""步驟 2 驗收（YOLO_baseline_定案規劃.md）：獨立於轉換腳本，從磁碟讀回 YOLO 標註檢查。

  1. 每個 split、每個類別的框數與 COCO 原檔完全一致
  2. YOLO 標註反算回像素座標，與 COCO 框逐一配對，最大誤差 < 1 像素
  3. 每類隨機抽 5 張，用「YOLO 標註」畫框存圖，供人工目視
  （第 4 項「容器內 symlink 與中文檔名」在 yolo-baseline 容器裡另外檢查）

另外順便確認：影像與標註一一對應、每個 symlink 都指得到檔案、
COCO json 記錄的寬高與實際影像寬高一致（Ultralytics 用實際寬高還原座標，兩者不符就會錯位）。

任何一項失敗就以非零狀態結束。
用法：python3 verify_step2.py
"""
import json
import os
import random
import sys
from collections import Counter

from PIL import Image, ImageDraw

COCO_ROOT = "/path/to/data/coco"
SRC = os.path.join(COCO_ROOT, "chiayi_v17_own_dynamic_litter_head10_manual")
DST = os.path.join(COCO_ROOT, "yolo_v17_own_dynamic_head10_manual")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_step2_samples")
NAMES = ["car", "litter", "motorcycle", "person", "garbage"]
COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#eab308", "#a855f7"]

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)
        print("  ✗", msg)


def read_yolo(path, W, H):
    boxes = []
    for line in open(path, encoding="utf-8").read().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        c, cx, cy, w, h = int(c), float(cx), float(cy), float(w), float(h)
        boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H))
    return boxes


def main():
    random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    for sp in ["train", "valid", "test"]:
        print(f"== {sp}")
        coco = json.load(open(os.path.join(SRC, "annotations", f"{sp}.json"), encoding="utf-8"))
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        img_files = set(os.listdir(os.path.join(DST, "images", sp)))
        lbl_files = set(os.listdir(os.path.join(DST, "labels", sp)))
        check(len(img_files) == len(coco["images"]), f"{sp} 影像數 {len(img_files)} ≠ COCO {len(coco['images'])}")
        check(len(lbl_files) == len(coco["images"]), f"{sp} 標註檔數 {len(lbl_files)} ≠ COCO {len(coco['images'])}")

        coco_cnt, yolo_cnt = Counter(), Counter()
        max_err, size_mismatch, broken = 0.0, 0, 0
        by_class_imgs = {n: [] for n in NAMES}
        for im in coco["images"]:
            fn, W, H = im["file_name"], im["width"], im["height"]
            link = os.path.join(DST, "images", sp, fn)
            if not os.path.exists(link):          # exists() 會跟隨 symlink；斷掉的連結回傳 False
                broken += 1
                continue
            with Image.open(link) as pil:
                if pil.size != (W, H):
                    size_mismatch += 1
            stem = os.path.splitext(fn)[0]
            yb = read_yolo(os.path.join(DST, "labels", sp, stem + ".txt"), W, H)
            cb = [(a["category_id"] - 1, *a["bbox"]) for a in anns.get(im["id"], [])]
            for b in cb:
                coco_cnt[NAMES[b[0]]] += 1
            for b in yb:
                yolo_cnt[NAMES[b[0]]] += 1
            # 每個 YOLO 框配給「同類別、座標最接近、還沒被配過」的 COCO 框。
            # 不能用兩邊各自排序後逐一配對：同一張圖裡兩個框起點相同時（例如 x 都是 714），
            # 小數四捨五入的微小差異會讓排序順序對調，配錯框而報出幾百像素的假誤差。
            check(len(yb) == len(cb), f"{sp}/{fn} 框數 {len(yb)} ≠ COCO {len(cb)}")
            unused = list(cb)
            for y in yb:
                same = [c for c in unused if c[0] == y[0]]
                if not same:
                    check(False, f"{sp}/{fn} YOLO 框類別 {y[0]} 在 COCO 找不到對應")
                    continue
                c = min(same, key=lambda c: max(abs(p - q) for p, q in zip(y[1:], c[1:])))
                unused.remove(c)
                max_err = max(max_err, *(abs(p - q) for p, q in zip(y[1:], c[1:])))
            for c in {b[0] for b in yb}:
                by_class_imgs[NAMES[c]].append((fn, W, H, yb))

        check(broken == 0, f"{sp} 有 {broken} 個斷掉的 symlink")
        check(size_mismatch == 0, f"{sp} 有 {size_mismatch} 張 COCO 寬高與實際影像不符")
        check(coco_cnt == yolo_cnt, f"{sp} 每類框數不一致：COCO {dict(coco_cnt)} vs YOLO {dict(yolo_cnt)}")
        check(max_err < 1.0, f"{sp} 反算座標最大誤差 {max_err:.4f} 像素 ≥ 1")
        print(f"  每類框數 COCO={dict(coco_cnt)}")
        print(f"  每類框數 YOLO={dict(yolo_cnt)}")
        print(f"  反算座標最大誤差 {max_err:.4f} 像素；斷掉的 symlink {broken}；寬高不符 {size_mismatch}")

        # 第 3 項：只在 train 抽樣畫圖（每類 5 張），畫的是 YOLO 標註
        if sp == "train":
            for ci, n in enumerate(NAMES):
                for fn, W, H, yb in random.sample(by_class_imgs[n], min(5, len(by_class_imgs[n]))):
                    with Image.open(os.path.join(DST, "images", sp, fn)) as pil:
                        pil = pil.convert("RGB")
                        d = ImageDraw.Draw(pil)
                        for c, x, y, w, h in yb:
                            d.rectangle([x, y, x + w, y + h], outline=COLORS[c], width=6 if c == ci else 3)
                        pil.thumbnail((1296, 1296))
                        pil.save(os.path.join(OUT, f"{n}__{os.path.splitext(fn)[0][-40:]}.jpg"), quality=85)
            print(f"  抽樣圖已存到 {OUT}（每類 5 張，該類框加粗）")

    print("\nVERIFY_STEP2:", "ALL_PASS" if not fail else f"FAIL ({len(fail)} 項)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
