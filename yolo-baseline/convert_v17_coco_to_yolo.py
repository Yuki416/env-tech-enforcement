#!/usr/bin/env python3
"""v17 資料集 COCO → YOLO 格式（YOLO_baseline_定案規劃.md 步驟 2）。

輸入：chiayi_v17_own_dynamic_litter_head10_manual/annotations/{train,valid,test}.json
輸出：yolo_v17_own_dynamic_head10_manual/
        images/{split}/<檔名>.jpg  → 相對路徑 symlink 指回 v17 原圖（不複製）
        labels/{split}/<檔名>.txt  → 每行「類別 cx cy w h」，座標除以該圖自己的寬高（0～1）
        data.yaml
        conversion_report.json     → 每個 split 每類框數，供驗收比對

類別編號 = COCO category_id − 1（car=0, litter=1, motorcycle=2, person=3, garbage=4），
與 metrics_from_preds.py 的 label 對應一致。

COCO 裡完全重複的標註（同圖、同類、同框）照原樣寫出，讓 labels 與 COCO 一對一；
Ultralytics 在建立訓練快取時會自動移除重複列（train 50 個、valid 24 個），記在 README。

用法（host 上執行即可，只需標準函式庫）：
  python3 convert_v17_coco_to_yolo.py
"""
import json
import os
from collections import Counter

COCO_ROOT = "/path/to/data/coco"
SRC = os.path.join(COCO_ROOT, "chiayi_v17_own_dynamic_litter_head10_manual")
DST = os.path.join(COCO_ROOT, "yolo_v17_own_dynamic_head10_manual")
SPLITS = ["train", "valid", "test"]
NAMES = ["car", "litter", "motorcycle", "person", "garbage"]


def main():
    report = {}
    for sp in SPLITS:
        coco = json.load(open(os.path.join(SRC, "annotations", f"{sp}.json"), encoding="utf-8"))
        # 類別名稱與編號必須和 NAMES 一致，否則整個對應會錯
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        assert cats == {i + 1: n for i, n in enumerate(NAMES)}, f"{sp} 類別不符：{cats}"

        img_dir = os.path.join(DST, "images", sp)
        lbl_dir = os.path.join(DST, "labels", sp)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        per_class = Counter()
        n_lines = 0
        for im in coco["images"]:
            fn, W, H = im["file_name"], im["width"], im["height"]
            src_img = os.path.join(SRC, sp, fn)
            assert os.path.isfile(src_img), f"找不到原圖：{src_img}"

            # 相對路徑 symlink：images/{sp}/X.jpg → ../../../<v17>/{sp}/X.jpg
            link = os.path.join(img_dir, fn)
            target = os.path.relpath(src_img, img_dir)
            if os.path.islink(link):
                os.remove(link)
            os.symlink(target, link)

            lines = []
            for a in anns_by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                # 事前已確認 v17 沒有越界或零面積的框；這裡再擋一次，有就停
                assert w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W + 1e-6 and y + h <= H + 1e-6, \
                    f"{sp}/{fn} 框越界或零面積：{a['bbox']}（影像 {W}x{H}）"
                cls = a["category_id"] - 1
                lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                per_class[NAMES[cls]] += 1
            stem = os.path.splitext(fn)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_lines += len(lines)

        report[sp] = {"images": len(coco["images"]), "boxes": n_lines,
                      "per_class": {n: per_class[n] for n in NAMES}}
        print(f"{sp}: {len(coco['images'])} 張影像、{n_lines} 個框  {report[sp]['per_class']}")

    # data.yaml 的 path 寫容器內路徑（訓練在 yolo-baseline 容器裡跑）
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# 由 baseline/yolo/convert_v17_coco_to_yolo.py 產生；path 為 yolo-baseline 容器內路徑\n")
        f.write("path: /workspace/datasets/coco/yolo_v17_own_dynamic_head10_manual\n")
        f.write("train: images/train\nval: images/valid\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    json.dump(report, open(os.path.join(DST, "conversion_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("CONVERT_DONE →", DST)


if __name__ == "__main__":
    main()
