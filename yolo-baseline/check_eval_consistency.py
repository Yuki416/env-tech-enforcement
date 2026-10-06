#!/usr/bin/env python3
"""步驟 5 驗收第 1 道：同一個 checkpoint，在 valid 上用 Ultralytics 內建評估與我們的評估各算一次。

兩者算法不同（Ultralytics 用自己的 AP 實作、而且會刪掉重複的正確答案框；我們用 pycocotools），
數字不會完全一樣，但每一類都應該在同一個量級。若我們的數字接近 0 而 Ultralytics 不是，
就表示 pkl 的座標尺度或影像編號對錯了。

用法（容器內）：python check_eval_consistency.py --ckpt <epochN.pt> --out <dir>
"""
import argparse
import contextlib
import io
import json
import os

from pycocotools.coco import COCO
from ultralytics import YOLO

from yolo_eval_common import HEAD_LABEL, VAL_KW, load_metrics_module, predict_to_records, save_pkl

DATA = "/workspace/datasets/coco/yolo_v17_own_dynamic_head10_manual/data.yaml"
VALID_ANN = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations/valid.json"
VALID_IMG = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/valid"
NAMES = ["car", "litter", "motorcycle", "person", "garbage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1) Ultralytics 內建評估（valid，imgsz 1024）。用哪個頭由 YOLO_HEAD 決定（VAL_KW），與下面我們的流程（PRED_KW）同一個頭：
    #    e2e → nms=False（一對一頭）；nms → 不設，Ultralytics 預設 nms=None＝一對多頭＋NMS
    m = YOLO(a.ckpt)
    r = m.val(data=DATA, split="val", imgsz=1024, batch=16, device=a.device, plots=False, verbose=False,
              project=a.out, name="ultralytics_val", exist_ok=True, **VAL_KW)
    ul = {}
    for i, n in enumerate(NAMES):
        p, rcl, ap50, ap = r.box.class_result(list(r.box.ap_class_index).index(i))
        ul[n] = {"AP@0.5": float(ap50), "AP@0.5:0.95": float(ap)}

    # 2) 我們的流程：predict_to_records → pkl → metrics_from_preds.load_preds → coco_ap
    recs = predict_to_records(YOLO(a.ckpt), VALID_ANN, VALID_IMG, device=a.device)
    pkl = os.path.join(a.out, "valid.pkl")
    save_pkl(recs, pkl)
    mfp = load_metrics_module()
    with contextlib.redirect_stdout(io.StringIO()):
        ours = mfp.coco_ap(COCO(VALID_ANN), mfp.load_preds(pkl))

    print(f"\n{'class':11s} {'UL AP50':>8s} {'ours AP50':>9s} {'UL AP':>7s} {'ours AP':>8s}")
    for n in NAMES:
        print(f"{n:11s} {ul[n]['AP@0.5']:8.3f} {ours[n]['AP@0.5']:9.3f} {ul[n]['AP@0.5:0.95']:7.3f} {ours[n]['AP@0.5:0.95']:8.3f}")
    json.dump({"ckpt": a.ckpt, "head": HEAD_LABEL, "ultralytics": ul, "ours": ours}, open(os.path.join(a.out, "consistency.json"), "w"), indent=2)
    print("CONSISTENCY_DONE")


if __name__ == "__main__":
    main()
