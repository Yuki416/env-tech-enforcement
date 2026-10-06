"""
infer_12th_v13_4ch_stem_unfreeze_video_classification.py
第 12 次訓練（v13, 單類別 litter-only, RGB+changemap 4ch, 解凍 stem）影片推論 + 逐 frame 逐框 IoU 誤差分析。
推論邏輯與第 11 次完全相同，只換 config / ckpt / 輸出路徑（指向第 12 次）。

對 v13 dataset 指定 split 的影片跑 best ckpt inference：
- 每張 frame：預測框(score>=0.3) 用 IoU>=0.5 貪婪配對 GT → 逐框 TP / FP / FN
- 畫面：TP=綠實線、FP=橘實線、漏抓 GT(FN)=紅虛線；HUD 顯示該幀 TP/FP/FN 數
- 串成原解析度 mp4，並輸出 per-video / per-frame CSV + summary

★ 與 v12（第 10 次）唯一的差別：模型是 4 通道（RGB+changemap）。
  DetInferencer 內建只載 3ch RGB → 不能用。改成 init_detector + 自訂 4ch 推論 pipeline
  （LoadImageAndChangemap 會自動從 <split>_changemap/ 取第 4 通道）+ model.test_step。
  畫框 / IoU 配對 / HUD / 輸出格式皆與 v12 腳本一致，方便兩次直接對比。

用法（Docker 內，需 -v /path/to/data:/path/to/data）：
  python /workspace/project/infer_11th_v13_4ch_litter_video_classification.py [--splits test valid train] [--ckpt ...]
"""

import os
import json
import csv
import glob
import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet.apis import init_detector

# ── 路徑設定 ────────────────────────────────────────────────────────────────
CONFIG     = "/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
CKPT_DIR   = "/path/to/data/10th_expt_v12_other_team_litter_36e"
DATA_ROOT  = "/path/to/data/coco/chiayi_v13_other_team_litter_dynamic_test_frame"
OUTPUT_DIR = "/path/to/data/10th_expt_v12_other_team_litter_36e/video_inference_on_v13dynamic"

CLASSES        = ['litter']          # 單類別
TARGET_CLASSES = {'litter'}
SCORE_THRESH   = 0.3                  # 與 v12 一致
IOU_THRESH     = 0.5                  # 與 v12 一致
OUT_FPS        = 5

# 4ch 推論 pipeline 參數（對齊 config 的 test_pipeline，但拿掉 LoadAnnotations，GT 由 json 另讀）
IMAGE_SIZE = (1024, 1024)
PAD4       = (114, 114, 114, 0)

# 原生解析度輸出（不縮放）；框線/字體放大以配合大圖
BOX_THICK   = 5
GT_THICK    = 4
FONT_SCALE  = 1.4
FONT_THICK  = 3
HUD_H       = 56

# BGR 顏色
COLOR_TP = (0, 200, 0)      # green  正確偵測
COLOR_FP = (0, 165, 255)    # orange 誤報
COLOR_FN = (0, 0, 255)      # red    漏抓（GT 虛線）


def build_model_and_pipeline(ckpt, device):
    """3ch RGB 模型（第 10 次）+ 標準 3ch 推論 pipeline。"""
    init_default_scope('mmdet')
    cfg = Config.fromfile(CONFIG)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    model = init_detector(cfg, ckpt, device=device)
    infer_pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=IMAGE_SIZE, keep_ratio=True),
        dict(type='Pad', size=IMAGE_SIZE, pad_val=dict(img=(114, 114, 114))),
        dict(type='PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
    ])
    return model, infer_pipeline


def infer_one(model, pipeline, img_path, device):
    """回傳 list of (score, [x1,y1,x2,y2])，已過 score 門檻 + 限定 litter。"""
    data = pipeline({'img_path': str(img_path), 'img_id': 0})
    data = pseudo_collate([data])
    with torch.no_grad():
        results = model.test_step(data)
    pred = results[0].pred_instances
    boxes  = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    out = []
    for bb, sc, lid in zip(boxes, scores, labels):
        if sc >= SCORE_THRESH and lid < len(CLASSES) and CLASSES[lid] in TARGET_CLASSES:
            out.append((float(sc), [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]))
    return out


def compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def match_frame(gt_boxes, pred_boxes):
    gt_used = [False] * len(gt_boxes)
    tp_preds, fp_preds = [], []
    for score, pb in sorted(pred_boxes, key=lambda x: -x[0]):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = compute_iou(pb, gb)
            if iou >= IOU_THRESH and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            gt_used[best_j] = True
            tp_preds.append((score, pb))
        else:
            fp_preds.append((score, pb))
    fn_gts = [gb for j, gb in enumerate(gt_boxes) if not gt_used[j]]
    return tp_preds, fp_preds, fn_gts


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=22):
    x1, y1 = pt1
    x2, y2 = pt2
    for seg, axis in [((x1, y1, x2, y1), 'h'), ((x2, y1, x2, y2), 'v'),
                      ((x1, y2, x2, y2), 'h'), ((x1, y1, x1, y2), 'v')]:
        ax1, ay1, ax2, ay2 = seg
        length = abs(ax2 - ax1) if axis == 'h' else abs(ay2 - ay1)
        steps = max(length // (dash_len * 2), 1)
        for i in range(steps):
            if axis == 'h':
                sx = ax1 + i * 2 * dash_len
                ex = min(ax1 + (i * 2 + 1) * dash_len, ax2)
                cv2.line(img, (sx, ay1), (ex, ay2), color, thickness)
            else:
                sy = ay1 + i * 2 * dash_len
                ey = min(ay1 + (i * 2 + 1) * dash_len, ay2)
                cv2.line(img, (ax1, sy), (ax2, ey), color, thickness)


def put_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    ty = max(y - 8, th + 6)
    cv2.rectangle(img, (x, ty - th - 8), (x + tw + 6, ty), color, -1)
    cv2.putText(img, text, (x + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)


def annotate_frame(img, tp_preds, fp_preds, fn_gts):
    for gb in fn_gts:
        x1, y1, x2, y2 = map(int, gb)
        draw_dashed_rect(img, (x1, y1), (x2, y2), COLOR_FN, thickness=GT_THICK)
        put_label(img, "FN(miss)", x1, y1, COLOR_FN)
    for score, pb in tp_preds:
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TP, BOX_THICK)
        put_label(img, f"TP {score:.2f}", x1, y1, COLOR_TP)
    for score, pb in fp_preds:
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_FP, BOX_THICK)
        put_label(img, f"FP {score:.2f}", x1, y1, COLOR_FP)


def process_split(split, model, pipeline, gt_lookup, out_base, per_frame_rows, device):
    split_img_dir = Path(DATA_ROOT) / split
    frames = sorted(split_img_dir.glob("*.jpg"))
    print(f"\n[{split}] 共 {len(frames)} 張 frames")

    video_frames = collections.defaultdict(list)
    for f in frames:
        parts = f.stem.rsplit('_frame_', 1)
        vname = parts[0] if len(parts) == 2 else f.stem
        fnum  = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        video_frames[vname].append((fnum, f))
    for v in video_frames:
        video_frames[v].sort()

    out_dir = Path(out_base) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    total_videos = len(video_frames)

    for vidx, (vname, frame_list) in enumerate(sorted(video_frames.items()), 1):
        print(f"  [{vidx}/{total_videos}] {vname} ({len(frame_list)} frames)")

        frame_data = []   # (fpath, tp_preds, fp_preds, fn_gts)
        v_tp = v_fp = v_fn = v_gt = 0
        for fnum, fpath in frame_list:
            fname = fpath.name
            gt_boxes = [g['bbox'] for g in gt_lookup.get(fname, [])
                        if g['class'] in TARGET_CLASSES]

            pred_boxes = infer_one(model, pipeline, fpath, device)

            tp_preds, fp_preds, fn_gts = match_frame(gt_boxes, pred_boxes)
            n_tp, n_fp, n_fn = len(tp_preds), len(fp_preds), len(fn_gts)
            v_tp += n_tp; v_fp += n_fp; v_fn += n_fn; v_gt += len(gt_boxes)

            per_frame_rows.append({
                'split': split, 'video_name': vname, 'frame': fname,
                'n_gt': len(gt_boxes), 'TP': n_tp, 'FP': n_fp, 'FN': n_fn,
            })
            frame_data.append((fpath, tp_preds, fp_preds, fn_gts))

        has_gt = v_gt > 0
        has_tp = v_tp > 0
        has_fp = v_fp > 0
        if   has_gt and     has_tp: cls = 'TP'
        elif has_gt and not has_tp: cls = 'FN'
        elif not has_gt and has_fp: cls = 'FP'
        else:                       cls = 'TN'
        print(f"    → {cls}  (GT={v_gt} TP={v_tp} FP={v_fp} FN={v_fn})")

        out_video = str(out_dir / f"{vname}.mp4")
        writer = None
        out_w = out_h = None
        for fpath, tp_preds, fp_preds, fn_gts in frame_data:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            oh, ow = img.shape[:2]
            if writer is None:
                out_w, out_h = ow, oh
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_video, fourcc, OUT_FPS, (out_w, out_h))
            elif (ow, oh) != (out_w, out_h):
                img = cv2.resize(img, (out_w, out_h))

            annotate_frame(img, tp_preds, fp_preds, fn_gts)

            hud = f"{split.upper()} | {vname[-40:]} | {cls} | TP:{len(tp_preds)} FP:{len(fp_preds)} FN:{len(fn_gts)}"
            cv2.rectangle(img, (0, 0), (out_w, HUD_H), (0, 0, 0), -1)
            cv2.putText(img, hud, (10, HUD_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        {'TP': COLOR_TP, 'TN': (200,200,200),
                         'FP': COLOR_FP, 'FN': COLOR_FN}.get(cls, (255,255,255)),
                        2, cv2.LINE_AA)
            writer.write(img)
        if writer:
            writer.release()

        stats.append({
            'split': split, 'video_name': vname, 'num_frames': len(frame_list),
            'n_gt': v_gt, 'n_tp': v_tp, 'n_fp': v_fp, 'n_fn': v_fn,
            'classification': cls, 'output_video': out_video,
        })

    return stats


def load_gt(ann_file):
    with open(ann_file, encoding='utf-8') as f:
        data = json.load(f)
    cat_map = {c['id']: c['name'] for c in data['categories']}
    img_map = {img['id']: img['file_name'] for img in data['images']}
    gt = collections.defaultdict(list)
    for ann in data['annotations']:
        fname = img_map[ann['image_id']]
        x, y, w, h = ann['bbox']
        gt[fname].append({'bbox': [x, y, x+w, y+h], 'class': cat_map[ann['category_id']]})
    return dict(gt)


def find_best_ckpt():
    cands = sorted(glob.glob(f"{CKPT_DIR}/best_coco_bbox_mAP_epoch_*.pth"))
    if not cands:
        raise FileNotFoundError(f"找不到 best ckpt 於 {CKPT_DIR}")
    # 取 epoch 數字最大者
    return max(cands, key=lambda p: int(Path(p).stem.split('_')[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=['test', 'valid', 'train'])
    parser.add_argument('--ckpt', default=None, help='預設自動找 best ckpt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    ckpt = args.ckpt or find_best_ckpt()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"載入 4ch Co-DETR ({Path(ckpt).name}, score>={SCORE_THRESH}, IoU>={IOU_THRESH})...")
    model, pipeline = build_model_and_pipeline(ckpt, args.device)
    print("模型載入完成\n")

    all_stats = []
    per_frame_rows = []
    for split in args.splits:
        gt_lookup = load_gt(f"{DATA_ROOT}/annotations/{split}.json")
        all_stats.extend(process_split(split, model, pipeline, gt_lookup,
                                       OUTPUT_DIR, per_frame_rows, args.device))

    with open(f"{OUTPUT_DIR}/video_classification.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split','video_name','num_frames','n_gt',
                                          'n_tp','n_fp','n_fn','classification','output_video'])
        w.writeheader(); w.writerows(all_stats)

    with open(f"{OUTPUT_DIR}/per_frame_errors.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split','video_name','frame','n_gt','TP','FP','FN'])
        w.writeheader(); w.writerows(per_frame_rows)

    lines = [f'# 第 10 次訓練（純 RGB）在 v13 動態 test 上的影片分類摘要（{Path(ckpt).name}, '
             f'score>={SCORE_THRESH}, IoU>={IOU_THRESH}）\n',
             '> 影片層級：has_gt / has_tp（IoU>=0.5）；TP/FP/FN 為逐框累計\n\n']
    for split in args.splits:
        ss = [s for s in all_stats if s['split'] == split]
        counts = collections.Counter(s['classification'] for s in ss)
        tot_tp = sum(s['n_tp'] for s in ss); tot_fp = sum(s['n_fp'] for s in ss)
        tot_fn = sum(s['n_fn'] for s in ss); tot_gt = sum(s['n_gt'] for s in ss)
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp+tot_fp) else 0
        rec  = tot_tp / (tot_tp + tot_fn) if (tot_tp+tot_fn) else 0
        lines.append(f'## {split.upper()}\n')
        lines.append(f'- 影片層級：' + ' '.join(f'{c}={counts.get(c,0)}' for c in ['TP','TN','FP','FN']) + '\n')
        lines.append(f'- 逐框累計：GT={tot_gt} TP={tot_tp} FP={tot_fp} FN={tot_fn} '
                     f'| **precision={prec:.3f} recall={rec:.3f}**\n\n')
        lines.append('| 影片 | frames | GT | TP | FP | FN | 分類 |\n|---|---|---|---|---|---|---|\n')
        for s in sorted(ss, key=lambda x: x['classification']):
            lines.append(f"| {s['video_name'][:50]} | {s['num_frames']} | {s['n_gt']} "
                         f"| {s['n_tp']} | {s['n_fp']} | {s['n_fn']} | {s['classification']} |\n")
        lines.append('\n')
    with open(f"{OUTPUT_DIR}/video_classification_summary.md", 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print('\n=== 全部完成 ===')
    for split in args.splits:
        ss = [s for s in all_stats if s['split'] == split]
        counts = collections.Counter(s['classification'] for s in ss)
        tot_tp = sum(s['n_tp'] for s in ss); tot_fp = sum(s['n_fp'] for s in ss); tot_fn = sum(s['n_fn'] for s in ss)
        prec = tot_tp/(tot_tp+tot_fp) if (tot_tp+tot_fp) else 0
        rec  = tot_tp/(tot_tp+tot_fn) if (tot_tp+tot_fn) else 0
        print(f'{split}: 影片 TP={counts["TP"]} TN={counts["TN"]} FP={counts["FP"]} FN={counts["FN"]} '
              f'| 框 P={prec:.3f} R={rec:.3f} (TP={tot_tp} FP={tot_fp} FN={tot_fn})')
    print(f'\n輸出目錄：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
