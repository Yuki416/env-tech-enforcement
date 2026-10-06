"""
infer_17th_v17_2cls_video_classification.py  （第17次 v17 動態版，2 類 litter+garbage）
對 v17 dataset 全部影片（train/valid/test）跑 best-ckpt inference，
輸出帶 bounding box 的 MP4，並依 litter/garbage TP/TN/FP/FN 分類每支影片。
※ test = 原始 v11 靜止版（v14 沿用）→ test 推論即「動態訓練→靜止 test」的定性檢視。

★ 2 類版差異（相對第16次 5 類 infer）：
  1. CLASSES = ['litter','garbage']（與 17 次 config metainfo 同序 → 模型 pred 索引 0/1 直接對齊；
     5 類版硬編 CLASSES 會把 litter(0)→car、garbage(1)→litter 全錯位）。
  2. load_gt 只保留 litter/garbage 的 GT（car/person/motorcycle 不畫；使用者指定）。
  3. 路徑指向 17th。

用法（在 Docker container 內）：
  python /workspace/project/experiments/17th_v17_own_dynamic_2cls/infer_17th_v17_2cls_video_classification.py [--splits train valid test]
"""

import os
import re
import glob
import json
import csv
import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import DetInferencer

# ── 路徑設定（v14；CHECKPOINT 自動抓 best epoch）─────────────────────────────
CONFIG     = "/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py"
_CKPTS = sorted(glob.glob(
    "/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e/best_coco_litter_precision_epoch_*.pth"))
CHECKPOINT = _CKPTS[-1] if _CKPTS else "/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e/BEST_CKPT_NOT_FOUND.pth"
DATA_ROOT  = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual"
OUTPUT_DIR = "/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e/video_inference"

CLASSES       = ['litter', 'garbage']   # ★ 2 類，與 config metainfo 同序（pred 索引 0/1 直接對齊）
TARGET_CLASSES = {'litter', 'garbage'}
SCORE_THRESH  = 0.3
OUT_FPS       = 5
OUT_WIDTH     = 1296
OUT_HEIGHT    = 972   # half of original 2592×1944 (4:3)

# BGR 顏色（只剩 litter/garbage 兩類會被畫到）
PRED_COLORS = {
    'litter':     (0,   0,   255),   # red
    'garbage':    (0,   165, 255),   # orange
}
GT_COLOR = (0, 255, 0)  # green


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=12):
    """在圖上畫虛線矩形（GT 用）。"""
    x1, y1 = pt1
    x2, y2 = pt2
    for rect_pts in [[(x1, y1, x2, y1), 'h'], [(x2, y1, x2, y2), 'v'],
                     [(x1, y2, x2, y2), 'h'], [(x1, y1, x1, y2), 'v']]:
        seg, axis = rect_pts
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


def annotate_frame(img, gt_anns, preds, scale_x, scale_y):
    """在 frame 上畫 GT（虛線綠框）和 predictions（實線彩色框）。"""
    # GT
    for gt in gt_anns:
        x1, y1, x2, y2 = gt['bbox']
        x1, y1, x2, y2 = (int(x1*scale_x), int(y1*scale_y),
                           int(x2*scale_x), int(y2*scale_y))
        draw_dashed_rect(img, (x1, y1), (x2, y2), GT_COLOR, thickness=2)
        cv2.putText(img, f"GT:{gt['class']}", (x1, max(y1-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1, cv2.LINE_AA)

    # Predictions
    for label_id, score, bbox in preds:
        if score < SCORE_THRESH:
            continue
        class_name = CLASSES[label_id] if label_id < len(CLASSES) else str(label_id)
        color = PRED_COLORS.get(class_name, (200, 200, 200))
        x1, y1, x2, y2 = (int(bbox[0]*scale_x), int(bbox[1]*scale_y),
                           int(bbox[2]*scale_x), int(bbox[3]*scale_y))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, label_text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


def process_split(split, inferencer, gt_lookup, out_base):
    """處理一個 split，回傳 list of video stats dicts。"""
    split_img_dir = Path(DATA_ROOT) / split
    frames = sorted(split_img_dir.glob("*.jpg"))
    print(f"\n[{split}] 共 {len(frames)} 張 frames")

    # 依 video_name 分組
    video_frames = collections.defaultdict(list)
    for f in frames:
        parts = f.stem.rsplit('_frame_', 1)
        vname = parts[0] if len(parts) == 2 else f.stem
        fnum  = int(parts[1]) if len(parts) == 2 else 0
        video_frames[vname].append((fnum, f))
    for v in video_frames:
        video_frames[v].sort()

    out_dir = Path(out_base) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    total_videos = len(video_frames)

    for vidx, (vname, frame_list) in enumerate(sorted(video_frames.items()), 1):
        print(f"  [{vidx}/{total_videos}] {vname} ({len(frame_list)} frames)")

        has_gt_target   = False
        has_pred_target = False

        # ── Phase 1：推論，蒐集結果 ────────────────────────────────────────
        frame_data = []   # list of (frame_path, gt_anns, preds)
        for fnum, fpath in frame_list:
            fname = fpath.name
            gt_anns = gt_lookup.get(fname, [])
            if any(g['class'] in TARGET_CLASSES for g in gt_anns):
                has_gt_target = True

            result = inferencer(str(fpath), show=False, print_result=False)
            pred   = result['predictions'][0]
            preds  = list(zip(pred['labels'], pred['scores'], pred['bboxes']))

            if any(CLASSES[lid] in TARGET_CLASSES
                   for lid, sc, _ in preds if sc >= SCORE_THRESH and lid < len(CLASSES)):
                has_pred_target = True

            frame_data.append((fpath, gt_anns, preds))

        # ── 分類 ────────────────────────────────────────────────────────────
        if   has_gt_target and     has_pred_target: cls = 'TP'
        elif has_gt_target and not has_pred_target: cls = 'FN'
        elif not has_gt_target and has_pred_target: cls = 'FP'
        else:                                        cls = 'TN'
        print(f"    → {cls}  (gt_target={has_gt_target}, pred_target={has_pred_target})")

        # ── Phase 2：畫框並編碼影片 ─────────────────────────────────────────
        out_video = str(out_dir / f"{vname}.mp4")
        writer = None

        for fpath, gt_anns, preds in frame_data:
            img_orig = cv2.imread(str(fpath))
            if img_orig is None:
                continue
            oh, ow = img_orig.shape[:2]
            img = cv2.resize(img_orig, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA)
            sx, sy = OUT_WIDTH / ow, OUT_HEIGHT / oh

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_video, fourcc, OUT_FPS, (OUT_WIDTH, OUT_HEIGHT))

            annotate_frame(img, gt_anns, preds, sx, sy)

            # HUD 文字
            hud = f"{split.upper()} | {vname[-30:]} | {cls}"
            cv2.rectangle(img, (0, 0), (OUT_WIDTH, 28), (0, 0, 0), -1)
            cv2.putText(img, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        {'TP': (0,255,0), 'TN': (200,200,200),
                         'FP': (0,165,255), 'FN': (0,0,255)}.get(cls, (255,255,255)),
                        1, cv2.LINE_AA)
            writer.write(img)

        if writer:
            writer.release()

        stats.append({
            'split': split, 'video_name': vname,
            'num_frames': len(frame_list),
            'has_gt_litter_garbage': has_gt_target,
            'has_pred_litter_garbage': has_pred_target,
            'classification': cls,
            'output_video': out_video,
        })

    return stats


def load_gt(ann_file):
    """讀取 COCO annotation JSON，回傳 filename → list of GT dicts。
    ★ 2 類版：只保留 litter/garbage 的 GT（car/person/motorcycle 不畫；使用者指定）。
      分類邏輯本就只看 TARGET_CLASSES，故濾除不影響 TP/TN/FP/FN，只讓畫面乾淨聚焦。"""
    with open(ann_file, encoding='utf-8') as f:
        data = json.load(f)
    cat_map = {c['id']: c['name'] for c in data['categories']}
    img_map = {img['id']: img['file_name'] for img in data['images']}
    gt = collections.defaultdict(list)
    for ann in data['annotations']:
        cls_name = cat_map[ann['category_id']]
        if cls_name not in TARGET_CLASSES:   # 只留 litter/garbage
            continue
        fname = img_map[ann['image_id']]
        x, y, w, h = ann['bbox']
        gt[fname].append({'bbox': [x, y, x+w, y+h],
                          'class': cls_name})
    return dict(gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 由 checkpoint 檔名推導 epoch，避免標題寫死成前一次實驗的值
    _m = re.search(r'epoch_(\d+)', os.path.basename(CHECKPOINT))
    ckpt_ep = _m.group(1) if _m else '?'

    print(f"載入 Co-DETR inferencer (epoch {ckpt_ep})...")
    inferencer = DetInferencer(model=CONFIG, weights=CHECKPOINT, device=args.device)
    print("模型載入完成\n")

    all_stats = []
    for split in args.splits:
        ann_file = f"{DATA_ROOT}/annotations/{split}.json"
        gt_lookup = load_gt(ann_file)
        split_stats = process_split(split, inferencer, gt_lookup, OUTPUT_DIR)
        all_stats.extend(split_stats)

    # ── 輸出 CSV ─────────────────────────────────────────────────────────────
    csv_path = f"{OUTPUT_DIR}/video_classification.csv"
    fieldnames = ['split','video_name','num_frames','has_gt_litter_garbage',
                  'has_pred_litter_garbage','classification','output_video']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_stats)
    print(f"\nCSV 報告：{csv_path}")

    # ── 輸出 Markdown 摘要 ────────────────────────────────────────────────────
    md_path = f"{OUTPUT_DIR}/video_classification_summary.md"
    lines = [f'# v17 影片分類摘要（第17次訓練，2 類，Epoch {ckpt_ep}，score_thr={SCORE_THRESH}）\n',
             '> 分類依據：litter/garbage 類別是否有 GT 標註 vs 模型預測（Step1 粗分類，未做 IoU 校正）\n\n']

    for split in args.splits:
        ss = [s for s in all_stats if s['split'] == split]
        counts = collections.Counter(s['classification'] for s in ss)
        lines.append(f'## {split.upper()}\n')
        lines.append('| 分類 | 影片數 |\n|------|--------|\n')
        for cls in ['TP', 'TN', 'FP', 'FN']:
            lines.append(f'| {cls} | {counts.get(cls, 0)} |\n')
        lines.append('\n### 詳細清單\n')
        lines.append('| 影片名稱 | frames | 分類 |\n|---------|--------|------|\n')
        for s in sorted(ss, key=lambda x: x['classification']):
            lines.append(f"| {s['video_name']} | {s['num_frames']} | {s['classification']} |\n")
        lines.append('\n')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Markdown 摘要：{md_path}")

    # ── 終端摘要 ──────────────────────────────────────────────────────────────
    print('\n=== 全部完成 ===')
    for split in args.splits:
        ss = [s for s in all_stats if s['split'] == split]
        counts = collections.Counter(s['classification'] for s in ss)
        print(f'{split}: TP={counts["TP"]} TN={counts["TN"]} FP={counts["FP"]} FN={counts["FN"]}')
    print(f'\n影片輸出目錄：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
