#!/usr/bin/env python3
"""YOLO26l 影片推論：直接從預測檔（pkl）畫影片，確保影片與表格數字使用同一批預測、同一套判定。

預測來源：步驟 5 產生的 <YB>/seed<N>/full_preds/{train,valid,test_full}.pkl，<YB> 由環境變數 YOLO_HEAD 決定
（設定與步驟 5 相同：imgsz 1024、conf 0.001、max_det 300、各 seed 在 valid 選中的 epoch；
  YOLO_HEAD=e2e 是端到端（一對一頭、不做 NMS），YOLO_HEAD=nms 是一對多頭＋標準 NMS）。
影片畫面上方與摘要標題都會標出是哪一版（e2e / NMS），兩版影片檔名相同，靠這個區分。

兩種門檻（不要混淆）：
  信心分數門檻 --thr：分數 ≥ 門檻的框才畫、才算。0.5＝步驟 5～7 的 TP/FP/FN 與第 16、18 次正式數字的門檻；
                      0.3＝過去 Co-DETR 影片推論的門檻（較寬鬆）；
                      0.1＝接近 YOLO26l 在 valid 上 litter 的最佳 F1 門檻（實際值從 threshold_eval.json 讀出寫進摘要），
                      用來看 YOLO 自己最有利的門檻下長什麼樣子，不拿來和 Co-DETR 的影片對比。三種各做一套影片。
  IoU 門檻：固定 0.5，預測框與正確答案框重疊 ≥ 0.5 才算抓對。

每個框的判定用 frame_matching.match_image()，規則與 metrics_from_preds.tp_fp_fn() 相同，
而且開始畫之前先用 check_against_reference() 核對：逐框判定的加總必須等於 tp_fp_fn() 的結果。
所以影片裡每一幀 litter/garbage 的 TP/FP/FN 加總，會等於 thresholds_0.1_0.3_0.5/metrics_thr0.1_0.3_0.5.md 對應那一格的數字。

畫面：
  正確答案框（虛線）：綠色＝被抓到（TP）；紅色＝漏掉（FN，只針對 litter/garbage 標 FN 字樣）
  預測框（實線，顏色依類別）：litter 紅、garbage 橘、car 黃、motorcycle 青、person 洋紅；
    litter/garbage 的預測框標籤後面加「TP」或「FP」
  上方 HUD：split｜clip｜clip 分類｜模型與門檻；第二行是這一幀 litter、garbage 的 TP/FP/FN

clip 分類（只看 litter、garbage）：
  有 litter/garbage 正確答案框的 clip：任一幀有 litter/garbage 的 TP → TP，否則 FN
  沒有 litter/garbage 正確答案框的 clip：任一幀有 litter/garbage 預測框（必為 FP）→ FP，否則 TN
  ⚠️ clip 級分類只要任一幀命中就算，會失真；判斷模型好壞以逐框數字為準。

用法（容器內，需設 YOLO_HEAD）：python render_videos_from_pkl.py --seed 0 --thr 0.5
輸出：<YB>/seed<N>/video_inference/thr<T>/{train,valid,test}/*.mp4、video_classification.csv、video_classification_summary.md
"""
import argparse
import collections
import contextlib
import csv
import io
import os

import json

import cv2
from pycocotools.coco import COCO

from frame_matching import check_against_reference
from yolo_eval_common import HEAD, HEAD_LABEL, YB, load_metrics_module

V17 = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual"
SPLITS = {"train": "train", "valid": "valid", "test": "test_full"}       # 影片資料夾名 → pkl 名
TARGET = ("litter", "garbage")
OUT_FPS, OUT_W, OUT_H = 5, 1296, 972
COLORS = {"car": (0, 255, 255), "litter": (0, 0, 255), "motorcycle": (255, 255, 0),
          "person": (255, 0, 255), "garbage": (0, 165, 255)}             # BGR，與第 16、18 次影片相同
GT_HIT, GT_MISS = (0, 255, 0), (0, 0, 255)
HUD = {"TP": (0, 255, 0), "TN": (200, 200, 200), "FP": (0, 165, 255), "FN": (0, 0, 255)}


def dashed(img, p1, p2, color, th=2, d=12):
    (x1, y1), (x2, y2) = p1, p2
    for a, b, c, e in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x1, y2, x2, y2), (x1, y1, x1, y2)]:
        L = max(abs(c - a), abs(e - b), 1)
        for i in range(0, L, 2 * d):
            f0, f1 = i / L, min(i + d, L) / L
            cv2.line(img, (int(a + (c - a) * f0), int(b + (e - b) * f0)), (int(a + (c - a) * f1), int(b + (e - b) * f1)), color, th)


def label(img, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    ty = max(y - 6, th + 34)
    cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty), color, -1)
    cv2.putText(img, text, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--thr", type=float, required=True)
    a = ap.parse_args()
    mfp = load_metrics_module()
    out_base = f"{YB}/seed{a.seed}/video_inference/thr{a.thr}"
    rows, totals = [], {}

    for vdir, pname in SPLITS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(f"{V17}/annotations/{'test' if vdir == 'test' else vdir}.json")
        preds = mfp.load_preds(f"{YB}/seed{a.seed}/full_preds/{pname}.pkl")
        per_image, split_counts = check_against_reference(mfp, gt, preds, a.thr)   # 不一致會直接停止
        totals[vdir] = {c: split_counts[c] for c in TARGET}
        clips = collections.defaultdict(list)
        for img_id, im in gt.imgs.items():
            v, n = im["file_name"][:-4].rsplit("_frame_", 1)
            clips[v].append((int(n), img_id))
        od = f"{out_base}/{vdir}"
        os.makedirs(od, exist_ok=True)
        print(f"[seed{a.seed} 門檻{a.thr}] {vdir}：{len(gt.imgs)} 幀、{len(clips)} 部 clip；"
              f"整個 split 的 litter TP/FP/FN={tuple(split_counts['litter'].values())}、"
              f"garbage={tuple(split_counts['garbage'].values())}（已與 tp_fp_fn 核對一致）", flush=True)

        for v, fl in sorted(clips.items()):
            fl.sort()
            cc = {c: {"TP": 0, "FP": 0, "FN": 0} for c in TARGET}
            has_gt = has_tp = has_det = False
            for _, img_id in fl:
                gts, dets = per_image[img_id]
                for c, _, s in gts:
                    if c in TARGET:
                        has_gt = True
                        cc[c]["TP" if s == "TP" else "FN"] += 1
                        has_tp |= s == "TP"
                for c, _, _, s in dets:
                    if c in TARGET:
                        has_det = True
                        if s == "FP":
                            cc[c]["FP"] += 1
            cls = ("TP" if has_tp else "FN") if has_gt else ("FP" if has_det else "TN")

            path = f"{od}/{v}.mp4"
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (OUT_W, OUT_H))
            for _, img_id in fl:
                img0 = cv2.imread(f"{V17}/{'test' if vdir == 'test' else vdir}/{gt.imgs[img_id]['file_name']}")
                oh, ow = img0.shape[:2]
                img = cv2.resize(img0, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                sx, sy = OUT_W / ow, OUT_H / oh
                gts, dets = per_image[img_id]
                fc = {c: {"TP": 0, "FP": 0, "FN": 0} for c in TARGET}
                for c, b, s in gts:
                    col = GT_HIT if s == "TP" else (GT_MISS if c in TARGET else GT_HIT)
                    dashed(img, (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy)), col)
                    tag = f"GT:{c}" + (" FN" if (c in TARGET and s == "FN") else "")
                    cv2.putText(img, tag, (int(b[0] * sx), max(int(b[1] * sy) - 6, 44)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    if c in TARGET:
                        fc[c]["TP" if s == "TP" else "FN"] += 1
                for c, sc, b, s in dets:
                    x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2)
                    label(img, x1, y1, f"{c} {sc:.2f}" + (f" {s}" if c in TARGET else ""), COLORS[c])
                    if c in TARGET and s == "FP":
                        fc[c]["FP"] += 1
                cv2.rectangle(img, (0, 0), (OUT_W, 50), (0, 0, 0), -1)
                cv2.putText(img, f"{vdir.upper()} | {v[-34:]} | {cls} | YOLO26l {HEAD_LABEL} seed{a.seed} | score thr {a.thr}, IoU 0.5",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD[cls], 1, cv2.LINE_AA)
                cv2.putText(img, "this frame  litter TP/FP/FN {TP}/{FP}/{FN}".format(**fc["litter"]) +
                            "   garbage TP/FP/FN {TP}/{FP}/{FN}".format(**fc["garbage"]),
                            (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                w.write(img)
            w.release()
            rows.append(dict(split=vdir, video_name=v, num_frames=len(fl), has_gt_litter_garbage=has_gt,
                             classification=cls,
                             litter_TP=cc["litter"]["TP"], litter_FP=cc["litter"]["FP"], litter_FN=cc["litter"]["FN"],
                             garbage_TP=cc["garbage"]["TP"], garbage_FP=cc["garbage"]["FP"], garbage_FN=cc["garbage"]["FN"],
                             output_video=path))

        # 核對：各 clip 加總 = 整個 split 的數字
        for c in TARGET:
            for k in ["TP", "FP", "FN"]:
                ssum = sum(r[f"{c}_{k}"] for r in rows if r["split"] == vdir)
                assert ssum == totals[vdir][c][k], f"{vdir} {c} {k}：clip 加總 {ssum} ≠ split {totals[vdir][c][k]}"

    with open(f"{out_base}/video_classification.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    bf = json.load(open(f"{YB}/seed{a.seed}/test_eval/threshold_eval.json"))["best_f1_thr_from_valid"]["litter"]
    md = [f"# YOLO26l {HEAD_LABEL} seed{a.seed} 影片分類摘要（信心分數門檻 {a.thr}、IoU 門檻 0.5）\n\n",
          f"> **YOLO 版本**：{'端到端（一對一頭、不做 NMS），正式版' if HEAD == 'e2e' else '一對多頭＋標準 NMS，補充對照'}。\n"
          f"> **信心分數門檻 {a.thr}**：分數 ≥ {a.thr} 的預測框才畫、才算。"
          "0.5＝步驟 5～7 的 TP/FP/FN 與第 16、18 次正式數字使用的門檻；0.3＝過去 Co-DETR 影片推論使用的門檻（較寬鬆）；"
          f"0.1＝接近 YOLO26l 在 valid 上 litter 的最佳 F1 門檻（這個 seed 是 {bf:.2f}），只用來看 YOLO 自己最有利的情況，"
          "不和 Co-DETR 的影片對比。\n"
          "> **IoU 門檻 0.5**：預測框與正確答案框重疊 ≥ 0.5 才算抓對。\n"
          "> **逐框判定**與 `metrics_from_preds.py` 的 `tp_fp_fn()` 規則相同，各 clip 的 TP/FP/FN 加總已核對等於整個 split 的數字。\n"
          "> **clip 分類**：有 litter/garbage 正確答案的 clip，任一幀有 TP → TP，否則 FN；沒有的 clip，任一幀有預測 → FP，否則 TN。"
          "任一幀命中即算，會失真，判斷好壞以逐框數字為準。\n\n"]
    for vdir in SPLITS:
        ss = [r for r in rows if r["split"] == vdir]
        cnt = collections.Counter(r["classification"] for r in ss)
        t = totals[vdir]
        md.append(f"## {vdir.upper()}（{len(ss)} 部 clip）\n\n"
                  f"clip 分類：TP {cnt['TP']}、TN {cnt['TN']}、FP {cnt['FP']}、FN {cnt['FN']}　｜　"
                  f"逐框：litter TP/FP/FN = {t['litter']['TP']}/{t['litter']['FP']}/{t['litter']['FN']}，"
                  f"garbage = {t['garbage']['TP']}/{t['garbage']['FP']}/{t['garbage']['FN']}\n\n"
                  "| clip | frames | 分類 | litter TP / FP / FN | garbage TP / FP / FN |\n|---|---|---|---|---|\n")
        for r in sorted(ss, key=lambda r: (r["classification"], r["video_name"])):
            md.append(f"| {r['video_name']} | {r['num_frames']} | {r['classification']} | "
                      f"{r['litter_TP']} / {r['litter_FP']} / {r['litter_FN']} | {r['garbage_TP']} / {r['garbage_FP']} / {r['garbage_FN']} |\n")
        md.append("\n")
    open(f"{out_base}/video_classification_summary.md", "w", encoding="utf-8").writelines(md)
    for vdir in SPLITS:
        cnt = collections.Counter(r["classification"] for r in rows if r["split"] == vdir)
        print(f"  {vdir}: TP={cnt['TP']} TN={cnt['TN']} FP={cnt['FP']} FN={cnt['FN']}")
    print(f"RENDER_DONE seed{a.seed} thr{a.thr} → {out_base}")


if __name__ == "__main__":
    main()
