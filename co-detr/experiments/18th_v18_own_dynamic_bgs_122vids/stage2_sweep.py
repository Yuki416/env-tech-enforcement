#!/usr/bin/env python3
"""Stage 2 第二步：從逐幀預測掃出「抽幀間隔 → 事件命中率 / 誤報率」。不需要 GPU。

## 為什麼可以用子抽樣代替真的重跑

線上設計是 U=1（KNN 每幀 apply，只有 Co-DETR 依抽幀間隔挑幀），所以第 f 幀的
changemap 與抽幀間隔無關 —— 見 stage2_run_window.py 開頭。因此 stage2_out/*.json
裡每一幀的預測，就是「若該幀被抽中，模型會吐出什麼」。任何抽幀間隔都只是挑子集。

## 相位平均

抽幀間隔 S 有 S 種可能的起始相位（從窗口第 0、1、…、S-1 幀開始）。
只取一種相位等於只看一次擲骰。這裡對 **全部 S 種相位各算一次再平均**，
得到的是期望值而不是單一實現，小樣本下穩定得多。

## 兩個指標

**事件命中率**：該 clip 的事件窗內，被抽中的幀裡是否存在
「同類別、IoU≥IOU_THR、分數≥SCORE_THR」的預測。整支 clip 是 0 或 1，再對 clip 平均。

**誤報率**：事件窗之前的 150 幀（事件尚未發生）中，被抽中的幀吐出多少
litter/garbage 預測（分數≥SCORE_THR）。換算成「每小時影片的誤報框數」，
因為抽幀間隔會改變被檢查的幀數，用絕對數量比較沒有意義。

用法：python stage2_sweep.py [--score 0.5] [--iou 0.5]
"""
import argparse, json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "stage2_out"
STRIDES = [1, 2, 3, 5, 7, 10, 20, 35]
LITTER_CID, GARBAGE_CID = 2, 5      # COCO category_id（GT 用）
LITTER_LID, GARBAGE_LID = 1, 4      # 模型 label id（CLASSES 索引；預測用）
CID_OF_LID = {LITTER_LID: LITTER_CID, GARBAGE_LID: GARBAGE_CID}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_by_source_frame(rec):
    """dataset 幀號 → 原始影片幀號，回傳 {原始幀號: [(cat_id, xyxy)]}。

    公式取自 build_alignment_v18.py:98 —— idx = first_abs + round((fn - fn0) * slope)
    """
    out = {}
    for fn_s, boxes in rec["gt"].items():
        idx = rec["first_abs"] + int(round((int(fn_s) - rec["fn0"]) * rec["slope"]))
        if boxes:
            out[idx] = boxes
    return out


def analyse(rec, score_thr, iou_thr):
    preds = {int(k): v for k, v in rec["preds"].items()}
    gt = gt_by_source_frame(rec)
    fp_lo, ev_lo, ev_hi = rec["fp_start"], rec["event_start"], rec["end"]
    fps = rec["fps"]

    # 事件窗中「哪些幀單獨看得到這次丟棄」
    hit_frames = set()
    for f, boxes in gt.items():
        p = preds.get(f)
        if not p:
            continue
        for lid, sc, bb in p:
            if sc < score_thr or lid not in CID_OF_LID:
                continue
            cid = CID_OF_LID[lid]
            if any(cid == g_cid and iou(g_box, bb) >= iou_thr for g_cid, g_box in boxes):
                hit_frames.add(f)
                break

    # 誤報窗中每幀的 litter/garbage 預測數
    fp_count = {f: sum(1 for lid, sc, _ in preds.get(f, [])
                       if lid in CID_OF_LID and sc >= score_thr)
                for f in range(fp_lo, ev_lo)}

    res = {}
    for S in STRIDES:
        hits, fp_boxes, fp_frames = 0, 0, 0
        for phase in range(S):
            sampled = set(range(fp_lo + phase, ev_hi, S))
            hits += 1 if (sampled & hit_frames) else 0
            for f in sampled:
                if f < ev_lo:
                    fp_boxes += fp_count.get(f, 0)
                    fp_frames += 1
        res[S] = dict(hit=hits / S,                       # 0~1，對相位平均
                      fp_boxes=fp_boxes / S,              # 該 clip 誤報窗的平均誤報框數
                      fp_sec=(fp_frames / S) * S / fps)   # 誤報窗涵蓋的影片秒數（= 150/fps）
    return res, len(hit_frames), len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    files = sorted(OUT.glob("*.json"))
    if not files:
        raise SystemExit(f"{OUT} 底下沒有結果，先跑 stage2_run_window.py")

    groups = collections.defaultdict(list)   # 'test+valid' / 'train' -> [(key, res)]
    ms = []
    per_clip = []
    for fp in files:
        rec = json.load(open(fp))
        if "preds" not in rec:
            continue
        res, nhit, ngt = analyse(rec, a.score, a.iou)
        g = "train" if rec["split"] == "train" else "test+valid"
        groups[g].append(res)
        ms.append(rec["ms_per_frame"])
        per_clip.append((rec["key"], g, nhit, ngt, res))

    print(f"=== Stage 2 掃描 · score≥{a.score} · IoU≥{a.iou} ===")
    print(f"clip 數：test+valid {len(groups['test+valid'])}、train {len(groups['train'])}"
          f"（train 是模型訓練過的，數字偏樂觀，僅作對照）")
    print(f"逐幀處理速度：{min(ms):.0f}–{max(ms):.0f} ms/幀"
          f"（中位數 {sorted(ms)[len(ms)//2]:.0f}，含解碼＋KNN＋rm5＋Co-DETR）\n")

    for g in ("test+valid", "train"):
        rs = groups[g]
        if not rs:
            continue
        print(f"--- {g}（{len(rs)} 部 clip）---")
        print(f"{'抽幀間隔':>9}{'事件命中率':>13}{'誤報（框/小時）':>18}")
        for S in STRIDES:
            hit = sum(r[S]["hit"] for r in rs) / len(rs)
            fpb = sum(r[S]["fp_boxes"] for r in rs)
            fps_sec = sum(r[S]["fp_sec"] for r in rs)
            per_hr = fpb / fps_sec * 3600 if fps_sec else 0.0
            print(f"{S:>9}{hit*100:>12.1f}%{per_hr:>17.1f}")
        print()

    print("--- 逐 clip（事件命中率，對相位平均）---")
    hdr = "".join(f"{S:>7}" for S in STRIDES)
    print(f"{'clip':<44}{'組':<11}{'可命中幀/有GT幀':>16}{hdr}")
    for key, g, nhit, ngt, res in sorted(per_clip, key=lambda x: (x[1], x[0])):
        row = "".join(f"{res[S]['hit']:>7.2f}" for S in STRIDES)
        print(f"{key:<44}{g:<11}{f'{nhit}/{ngt}':>16}{row}")


if __name__ == "__main__":
    main()
