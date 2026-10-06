#!/usr/bin/env python3
"""
針對 litter 類別，分析各影片的偵測表現（TP/FP/FN/Recall/F1）
在容器內執行：
  python /workspace/project/litter_video_analysis.py \
    --config   /workspace/project/configs/co_dino_chiayi_v4_orig_res_36e.py \
    --checkpoint /workspace/project/results/chiayi_v4_orig_res_36e/best_coco_car_precision_epoch_1.pth \
    --data-root  /workspace/datasets/coco/chiayi-v4-original_resolution \
    --output-dir /workspace/project/results/chiayi_v4_orig_res_36e/error_analysis \
    --splits train valid test
"""

import os, sys, json, argparse, re, csv
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/opt/mmdetection')
os.environ.setdefault('PYTHONPATH', '/opt/mmdetection:/workspace/project')

LITTER_IDX = 1   # CLASSES = ["car","litter","motorcycle","person"]
IOU_THR    = 0.5
SCORE_THR  = 0.5


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    u = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/u if u > 0 else 0.0


def match(gt_boxes, pred_boxes, pred_scores):
    """回傳 (tp_gt_indices, fp_pred_indices, fn_gt_indices)"""
    if not gt_boxes and not pred_boxes:
        return [], [], []
    preds = [(s, i) for i, s in enumerate(pred_scores) if s >= SCORE_THR]
    preds.sort(reverse=True)
    matched_gt = set()
    tp_gt, fp_pred = [], []
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
            tp_gt.append(best_gi)
        else:
            fp_pred.append(pi)
    fn_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    return tp_gt, fp_pred, fn_gt


def load_gt(data_root, split):
    ann_path = os.path.join(data_root, split, '_annotations.coco.json')
    with open(ann_path) as f:
        data = json.load(f)
    sorted_cats = sorted(data['categories'], key=lambda c: c['id'])
    cat_map = {c['id']: idx for idx, c in enumerate(sorted_cats)}
    img_info = {img['id']: img['file_name'] for img in data['images']}
    gt = defaultdict(lambda: {'litter': []})
    for ann in data['annotations']:
        cls_idx = cat_map[ann['category_id']]
        if cls_idx == LITTER_IDX:
            gt[ann['image_id']]['litter'].append(ann['bbox'])
    return img_info, gt


def run_inference(config_file, checkpoint, data_root, split):
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    model = init_detector(config_file, checkpoint, device='cuda:0')
    ann_path = os.path.join(data_root, split, '_annotations.coco.json')
    with open(ann_path) as f:
        data = json.load(f)
    results = {}
    total = len(data['images'])
    for i, img in enumerate(data['images']):
        if i % 100 == 0:
            print(f'  推論進度 {i}/{total} ...')
        img_path = os.path.join(data_root, split, img['file_name'])
        result = inference_detector(model, img_path)
        preds = result.pred_instances
        bboxes, classes, scores = [], [], []
        for j in range(len(preds.bboxes)):
            x1, y1, x2, y2 = preds.bboxes[j].cpu().numpy()
            bboxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
            classes.append(int(preds.labels[j].cpu()))
            scores.append(float(preds.scores[j].cpu()))
        results[img['id']] = {'bboxes': bboxes, 'classes': classes, 'scores': scores}
    return results


def video_name_from_file(fname):
    # 去掉 _frame_XXXX.jpg 後綴取影片名
    m = re.match(r'^(.+?)_frame_\d+\.jpg$', fname)
    if m:
        return m.group(1)
    return re.sub(r'\.[^.]+$', '', fname)


def timestamp_from_video(vname):
    # 抽出 "20240809180115_04m55s" 格式
    m = re.search(r'(\d{14}_\d{2}m\d{2}s)', vname)
    return m.group(1) if m else vname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--splits', nargs='+', default=['train','valid','test'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, 'litter_per_video.csv')

    all_rows = []
    for split in args.splits:
        print(f'\n=== {split.upper()} ===')
        img_info, gt = load_gt(args.data_root, split)
        preds = run_inference(args.config, args.checkpoint, args.data_root, split)

        video_stats = defaultdict(lambda: {'TP':0,'FP':0,'FN':0,'GT':0})
        for img_id, fname in img_info.items():
            vname = video_name_from_file(fname)
            gt_boxes = gt[img_id]['litter']
            pred = preds.get(img_id, {'bboxes':[],'classes':[],'scores':[]})
            litter_pred_boxes  = [b for b,c in zip(pred['bboxes'],pred['classes']) if c==LITTER_IDX]
            litter_pred_scores = [s for s,c in zip(pred['scores'],pred['classes']) if c==LITTER_IDX]
            tp_gt, fp_pred, fn_gt = match(gt_boxes, litter_pred_boxes, litter_pred_scores)
            video_stats[vname]['GT'] += len(gt_boxes)
            video_stats[vname]['TP'] += len(tp_gt)
            video_stats[vname]['FP'] += len(fp_pred)
            video_stats[vname]['FN'] += len(fn_gt)

        # 只輸出有 litter GT 的影片
        rows = []
        for vname, s in video_stats.items():
            if s['GT'] == 0:
                continue
            recall = s['TP'] / s['GT']
            prec = s['TP']/(s['TP']+s['FP']) if (s['TP']+s['FP'])>0 else 0.0
            f1 = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0
            ts = timestamp_from_video(vname)
            rows.append({'split':split,'timestamp':ts,'video':vname,
                         'GT':s['GT'],'TP':s['TP'],'FP':s['FP'],'FN':s['FN'],
                         'Recall':round(recall,4),'Precision':round(prec,4),'F1':round(f1,4)})
        rows.sort(key=lambda x: x['F1'], reverse=True)
        all_rows.extend(rows)

        print(f"{'#':<3} {'F1':>5} {'Recall':>7} {'GT':>4} {'TP':>4} {'FN':>4} {'FP':>4}  timestamp")
        for i, r in enumerate(rows):
            print(f"{i+1:<3} {r['F1']:>5.3f} {r['Recall']:>7.3f} {r['GT']:>4} {r['TP']:>4} {r['FN']:>4} {r['FP']:>4}  {r['timestamp']}")
        if not rows:
            print("  (無含 litter GT 的影片)")

    # 寫 CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split','timestamp','GT','TP','FP','FN','Recall','Precision','F1','video'])
        w.writeheader()
        w.writerows(all_rows)
    print(f'\n結果寫入：{out_csv}')


if __name__ == '__main__':
    main()
