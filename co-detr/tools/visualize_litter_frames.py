#!/usr/bin/env python3
"""
對指定影片的所有 frame 跑推論，畫出 litter 的 GT 與預測結果並存圖。
顏色規則：
  綠色實線  = GT litter（有被偵測到，TP）
  紅色虛線  = GT litter（漏偵測，FN）
  藍色實線  = 預測到的 litter（TP，有對應 GT）
  橘色實線  = 預測到的 litter（FP，無對應 GT）

在容器內執行：
  python /workspace/project/visualize_litter_frames.py \
    --config     /workspace/project/configs/co_dino_chiayi_v4_orig_res_36e.py \
    --checkpoint /workspace/project/results/chiayi_v4_orig_res_36e/best_coco_car_precision_epoch_1.pth \
    --data-root  /workspace/datasets/coco/chiayi-v4-original_resolution \
    --output-dir /workspace/project/results/chiayi_v4_orig_res_36e/litter_vis \
    --timestamps 20240807184204 20240826063130 20240906054955 \
    --split train
"""

import os, sys, json, argparse, re
from collections import defaultdict
import numpy as np
import cv2

sys.path.insert(0, '/opt/mmdetection')

LITTER_IDX = 1
IOU_THR    = 0.5
SCORE_THR  = 0.5

# BGR colors
COLOR_GT_TP  = (0, 200, 0)      # 綠：GT 被偵測到
COLOR_GT_FN  = (0, 0, 220)      # 紅：GT 漏偵測
COLOR_PRED_TP = (220, 150, 0)   # 藍：預測到且對應到 GT
COLOR_PRED_FP = (0, 140, 255)   # 橘：誤報


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    u = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/u if u > 0 else 0.0


def match(gt_boxes, pred_boxes, pred_scores):
    preds = sorted([(s, i) for i, s in enumerate(pred_scores) if s >= SCORE_THR], reverse=True)
    matched_gt = set()
    tp_gt, tp_pred, fp_pred = set(), set(), []
    for score, pi in preds:
        best_iou, best_gi = 0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            v = iou(gb, pred_boxes[pi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= IOU_THR and best_gi >= 0:
            matched_gt.add(best_gi)
            tp_gt.add(best_gi)
            tp_pred.add(pi)
        else:
            fp_pred.append(pi)
    fn_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    return tp_gt, tp_pred, fp_pred, fn_gt


def draw_box(img, xywh, color, dashed=False, label=None, thickness=3):
    x, y, w, h = [int(v) for v in xywh]
    if dashed:
        # 手動畫虛線
        step = 12
        pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        for i in range(len(pts)-1):
            x0, y0 = pts[i]; x1, y1 = pts[i+1]
            dx = x1-x0; dy = y1-y0
            length = max(abs(dx), abs(dy))
            if length == 0:
                continue
            steps = length // step
            for s in range(steps):
                if s % 2 == 0:
                    sx0 = x0 + dx*s//steps
                    sy0 = y0 + dy*s//steps
                    sx1 = x0 + dx*(s+1)//steps
                    sy1 = y0 + dy*(s+1)//steps
                    cv2.line(img, (sx0, sy0), (sx1, sy1), color, thickness)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.7
        (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
        cv2.rectangle(img, (x, y-th-6), (x+tw+4, y), color, -1)
        cv2.putText(img, label, (x+2, y-4), font, fs, (255,255,255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--timestamps', nargs='+')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    from mmdet.apis import init_detector, inference_detector

    # 載入標註
    ann_path = os.path.join(args.data_root, args.split, '_annotations.coco.json')
    with open(ann_path) as f:
        data = json.load(f)
    sorted_cats = sorted(data['categories'], key=lambda c: c['id'])
    cat_map = {c['id']: idx for idx, c in enumerate(sorted_cats)}

    # 建立 file_name → img_id 與 GT 的對應
    fname_to_id = {img['file_name']: img['id'] for img in data['images']}
    gt_by_id = defaultdict(list)
    for ann in data['annotations']:
        if cat_map[ann['category_id']] == LITTER_IDX:
            gt_by_id[ann['image_id']].append(ann['bbox'])

    # 篩出目標 frame
    target_frames = []
    for fname in sorted(fname_to_id.keys()):
        if any(ts in fname for ts in args.timestamps):
            target_frames.append(fname)
    print(f'目標 frames：{len(target_frames)} 張')

    # 載入模型
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    # 按影片分組輸出
    def ts_from_fname(fname):
        m = re.search(r'(\d{14}_\d{2}m\d{2}s)', fname)
        return m.group(1) if m else 'unknown'

    total = len(target_frames)
    stats = defaultdict(lambda: {'TP':0,'FP':0,'FN':0,'GT':0})

    for i, fname in enumerate(target_frames):
        ts = ts_from_fname(fname)
        out_dir = os.path.join(args.output_dir, ts)
        os.makedirs(out_dir, exist_ok=True)

        img_path = os.path.join(args.data_root, args.split, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f'  讀取失敗：{fname}')
            continue

        img_id = fname_to_id[fname]
        gt_boxes = gt_by_id[img_id]

        # 推論
        result = inference_detector(model, img_path)
        preds = result.pred_instances
        litter_boxes, litter_scores = [], []
        for j in range(len(preds.bboxes)):
            if int(preds.labels[j]) == LITTER_IDX:
                x1, y1, x2, y2 = preds.bboxes[j].cpu().numpy()
                litter_boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
                litter_scores.append(float(preds.scores[j]))

        tp_gt, tp_pred, fp_pred, fn_gt = match(gt_boxes, litter_boxes, litter_scores)

        # 統計
        stats[ts]['GT'] += len(gt_boxes)
        stats[ts]['TP'] += len(tp_gt)
        stats[ts]['FP'] += len(fp_pred)
        stats[ts]['FN'] += len(fn_gt)

        # 畫圖
        vis = img.copy()

        # GT：TP 綠實線，FN 紅虛線
        for gi, gb in enumerate(gt_boxes):
            if gi in tp_gt:
                draw_box(vis, gb, COLOR_GT_TP, dashed=False, label='GT-TP')
            else:
                draw_box(vis, gb, COLOR_GT_FN, dashed=True, label='GT-FN')

        # Pred：TP 藍實線，FP 橘實線
        for pi, (pb, ps) in enumerate(zip(litter_boxes, litter_scores)):
            if ps < SCORE_THR:
                continue
            if pi in tp_pred:
                draw_box(vis, pb, COLOR_PRED_TP, label=f'Pred {ps:.2f}')
            else:
                draw_box(vis, pb, COLOR_PRED_FP, label=f'FP {ps:.2f}')

        # 左上角統計資訊
        info = f'GT={len(gt_boxes)} TP={len(tp_gt)} FN={len(fn_gt)} FP={len(fp_pred)}'
        cv2.putText(vis, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
        cv2.putText(vis, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        out_name = os.path.basename(fname)
        cv2.imwrite(os.path.join(out_dir, out_name), vis)

        if (i+1) % 20 == 0 or (i+1) == total:
            print(f'  進度 {i+1}/{total}  ({fname[:60]}...)')

    print('\n=== 各影片 litter 統計 ===')
    for ts, s in sorted(stats.items()):
        recall = s['TP']/s['GT'] if s['GT'] > 0 else 0
        print(f"  {ts}  GT={s['GT']} TP={s['TP']} FN={s['FN']} FP={s['FP']} Recall={recall:.3f}")

    print(f'\n完成。圖片存於：{args.output_dir}')


if __name__ == '__main__':
    main()
