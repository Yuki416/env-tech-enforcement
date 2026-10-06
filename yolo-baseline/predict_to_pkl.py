#!/usr/bin/env python3
"""步驟 5：用選定的 checkpoint 推論，寫成 metrics_from_preds.py 可讀的 pkl，並直接算指標。

影像來源與第 16、18 次的正式評估相同：test 影像從 v18 資料夾讀（v18 test 是 v17 的整份複製，
test.json 已用 cmp 確認逐位元相同），標註用 v18 annotations 底下的 test_dynamic166.json / test_static594.json。

用法（容器內，需設 YOLO_HEAD；用哪個頭由它決定，見 yolo_eval_common.py）：
  python predict_to_pkl.py --ckpt <epochN.pt> --split dynamic166 --out <dir> --tag "YOLO26l seed0 dynamic166"
  --split 可為 dynamic166、static594、valid、train、test_full
產出：<out>/<split>.pkl 與 <out>/<split>_metrics.json（metrics_from_preds 的輸出格式）
"""
import argparse
import contextlib
import io
import json
import os

from pycocotools.coco import COCO
from ultralytics import YOLO

from yolo_eval_common import load_metrics_module, predict_to_records, save_pkl

V18 = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids"
V17 = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual"
SPLITS = {
    "dynamic166": (f"{V18}/annotations/test_dynamic166.json", f"{V18}/test"),
    "static594": (f"{V18}/annotations/test_static594.json", f"{V18}/test"),
    "valid": (f"{V17}/annotations/valid.json", f"{V17}/valid"),
    # 影片推論用：三個 split 的全部 frames（test_full = v17 test.json 的 655 張，dynamic166 / static594 都是它的子集）
    "train": (f"{V17}/annotations/train.json", f"{V17}/train"),
    "test_full": (f"{V17}/annotations/test.json", f"{V17}/test"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    ann, img_dir = SPLITS[a.split]
    records = predict_to_records(YOLO(a.ckpt), ann, img_dir, device=a.device)
    pkl = os.path.join(a.out, f"{a.split}.pkl")
    save_pkl(records, pkl)

    # 用 metrics_from_preds.py 自己的 load_preds() 讀回剛寫的 pkl 再算，確保走的是跟第 16、18 次完全相同的路徑
    mfp = load_metrics_module()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(ann)
        preds = mfp.load_preds(pkl)
        apr = mfp.coco_ap(coco_gt, preds)
    tff = mfp.tp_fp_fn(coco_gt, preds)
    rows = {}
    print(f"\n===== {a.tag} =====  (score{mfp.SCORE_THR}, IoU{mfp.IOU_THR}; ann={os.path.basename(ann)})")
    print(f"{'class':11s} {'AP@0.5':>7s} {'AP.5:.95':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>6s} {'Rec':>6s}")
    for c in mfp.CLASSES:
        s = tff[c]; ap50 = apr[c]["AP@0.5"]; apc = apr[c]["AP@0.5:0.95"]
        prec = s["TP"] / (s["TP"] + s["FP"]) if s["TP"] + s["FP"] else 0.0
        rec = s["TP"] / (s["TP"] + s["FN"]) if s["TP"] + s["FN"] else 0.0
        print(f"{c:11s} {ap50:7.3f} {apc:8.3f} {s['TP']:5d} {s['FP']:5d} {s['FN']:5d} {prec:6.3f} {rec:6.3f}")
        rows[c] = dict(AP50=round(ap50, 3), AP=round(apc, 3), **s, Prec=round(prec, 3), Rec=round(rec, 3))
    out_json = os.path.join(a.out, f"{a.split}_metrics.json")
    json.dump(dict(tag=a.tag, ann=ann, ckpt=a.ckpt, rows=rows), open(out_json, "w"), ensure_ascii=False, indent=2)
    print("→", out_json)


if __name__ == "__main__":
    main()
