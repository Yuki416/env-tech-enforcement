"""
infer_14th_v15_4ch_video_classification.py  （第14次 v15 = 動態子集 + BGS changemap 4ch）
對 v15 dataset 全部影片（train/valid/test）跑 best-ckpt inference，
輸出帶 bounding box 的 MP4，並依 litter/garbage TP/TN/FP/FN 分類每支影片。

★ 與第 13 次（v14 純 RGB）的唯一差別：模型是 **4 通道**（RGB + changemap）。
  `DetInferencer` 內建只載 3ch RGB → **不能用**。改成第 12 次的做法：
  init_detector + 自訂 4ch 推論 pipeline（LoadImageAndChangemap 會自動從 <split>_changemap/
  取第 4 通道）+ model.test_step。其餘（5 類、TP/TN/FP/FN 分類、畫框、CSV/MD 摘要）沿用第 13 次。

⚠️ DATA_ROOT 必須指向 **v15**（有 {split}_changemap/）。若誤指 v14，LoadImageAndChangemap 會
   **靜默補零通道、不報錯**，整批結果作廢。啟動時有防呆檢查。

用法（Docker container 內）：
  python /workspace/project/experiments/14th_v15_own_dynamic_bgs/infer_14th_v15_4ch_video_classification.py [--splits train valid test]
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
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmcv.transforms import Compose
from mmdet.apis import init_detector

# ── 路徑設定（v15；CHECKPOINT 自動抓 best epoch）─────────────────────────────
CONFIG     = "/workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py"
RESULT_DIR = "/workspace/hdd2_storage/14th_expt_v15_own_dynamic_bgs_4ch_36e"
_CKPTS = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
CHECKPOINT = _CKPTS[-1] if _CKPTS else f"{RESULT_DIR}/BEST_CKPT_NOT_FOUND.pth"
DATA_ROOT  = "/workspace/datasets/coco/chiayi_v15_own_dynamic_bgs"   # ★ 必須是 v15
OUTPUT_DIR = f"{RESULT_DIR}/video_inference"

CLASSES        = ['car', 'litter', 'motorcycle', 'person', 'garbage']
TARGET_CLASSES = {'litter', 'garbage'}
SCORE_THRESH   = 0.3
OUT_FPS        = 5
OUT_WIDTH      = 1296
OUT_HEIGHT     = 972   # half of original 2592×1944 (4:3)

# 4ch 推論 pipeline（對齊 config 的 test_pipeline，但拿掉 LoadAnnotations，GT 由 json 另讀）
IMAGE_SIZE = (1024, 1024)
PAD4       = (114, 114, 114, 0)

# BGR 顏色
PRED_COLORS = {
    'car':        (0,   255, 255),   # yellow
    'litter':     (0,   0,   255),   # red
    'motorcycle': (255, 255, 0  ),   # cyan
    'person':     (255, 0,   255),   # magenta
    'garbage':    (0,   165, 255),   # orange
}
GT_COLOR = (0, 255, 0)  # green


def check_changemap_or_die():
    """防呆：DATA_ROOT 必須有 {split}_changemap/，否則第4通道會靜默全零。"""
    missing = [s for s in ('train', 'valid', 'test')
               if not Path(f"{DATA_ROOT}/{s}_changemap").is_dir()]
    if missing:
        raise SystemExit(
            f"❌ DATA_ROOT={DATA_ROOT} 缺少 changemap 目錄：{missing}\n"
            f"   4ch 模型若讀不到 changemap 會靜默補零、結果全錯。請確認指向 v15。")
    print(f"✅ changemap 目錄檢查通過：{DATA_ROOT}")


def build_model_and_pipeline(ckpt, device):
    """4ch 模型 + 4ch 推論 pipeline（第 12 次做法）。"""
    init_default_scope('mmdet')
    cfg = Config.fromfile(CONFIG)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg['custom_imports'])
    model = init_detector(cfg, ckpt, device=device)
    pipeline = Compose([
        dict(type='LoadImageAndChangemap'),
        dict(type='Resize', scale=IMAGE_SIZE, keep_ratio=True),
        dict(type='Pad', size=IMAGE_SIZE, pad_val=dict(img=PAD4)),
        dict(type='PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
    ])
    return model, pipeline


def infer_one(model, pipeline, img_path):
    """回傳 list of (label_id, score, [x1,y1,x2,y2])（全類別，未過門檻；由呼叫端過濾）。"""
    data = pipeline({'img_path': str(img_path), 'img_id': 0})
    data = pseudo_collate([data])
    with torch.no_grad():
        results = model.test_step(data)
    pred = results[0].pred_instances
    boxes  = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    return [(int(lid), float(sc), [float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            for b, sc, lid in zip(boxes, scores, labels)]


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=12):
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


def annotate_frame(img, gt_anns, preds, scale_x, scale_y):
    for gt in gt_anns:
        x1, y1, x2, y2 = gt['bbox']
        x1, y1, x2, y2 = (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))
        draw_dashed_rect(img, (x1, y1), (x2, y2), GT_COLOR, thickness=2)
        cv2.putText(img, f"GT:{gt['class']}", (x1, max(y1-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1, cv2.LINE_AA)

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


def process_split(split, model, pipeline, gt_lookup, out_base):
    split_img_dir = Path(DATA_ROOT) / split
    frames = sorted(split_img_dir.glob("*.jpg"))
    print(f"\n[{split}] 共 {len(frames)} 張 frames")

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
        print(f"  [{vidx}/{total_videos}] {vname} ({len(frame_list)} frames)", flush=True)

        has_gt_target = False
        has_pred_target = False
        frame_data = []

        for fnum, fpath in frame_list:
            gt_anns = gt_lookup.get(fpath.name, [])
            if any(g['class'] in TARGET_CLASSES for g in gt_anns):
                has_gt_target = True

            preds = infer_one(model, pipeline, fpath)
            if any(CLASSES[lid] in TARGET_CLASSES
                   for lid, sc, _ in preds if sc >= SCORE_THRESH and lid < len(CLASSES)):
                has_pred_target = True

            frame_data.append((fpath, gt_anns, preds))

        if   has_gt_target and     has_pred_target: cls = 'TP'
        elif has_gt_target and not has_pred_target: cls = 'FN'
        elif not has_gt_target and has_pred_target: cls = 'FP'
        else:                                       cls = 'TN'
        print(f"    → {cls}  (gt_target={has_gt_target}, pred_target={has_pred_target})")

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
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                         OUT_FPS, (OUT_WIDTH, OUT_HEIGHT))
            annotate_frame(img, gt_anns, preds, sx, sy)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    check_changemap_or_die()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _m = re.search(r'epoch_(\d+)', os.path.basename(CHECKPOINT))
    ckpt_ep = _m.group(1) if _m else '?'

    print(f"載入 4ch Co-DETR (epoch {ckpt_ep}, score>={SCORE_THRESH})...")
    model, pipeline = build_model_and_pipeline(CHECKPOINT, args.device)
    print("模型載入完成\n")

    all_stats = []
    for split in args.splits:
        gt_lookup = load_gt(f"{DATA_ROOT}/annotations/{split}.json")
        all_stats.extend(process_split(split, model, pipeline, gt_lookup, OUTPUT_DIR))

    csv_path = f"{OUTPUT_DIR}/video_classification.csv"
    fieldnames = ['split','video_name','num_frames','has_gt_litter_garbage',
                  'has_pred_litter_garbage','classification','output_video']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_stats)
    print(f"\nCSV 報告：{csv_path}")

    md_path = f"{OUTPUT_DIR}/video_classification_summary.md"
    lines = [f'# v15 影片分類摘要（第14次訓練 BGS 4ch，Epoch {ckpt_ep}，score_thr={SCORE_THRESH}）\n',
             '> 分類依據：litter/garbage 類別是否有 GT 標註 vs 模型預測\n\n']
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

    print('\n=== 全部完成 ===')
    for split in args.splits:
        ss = [s for s in all_stats if s['split'] == split]
        c = collections.Counter(s['classification'] for s in ss)
        print(f'{split}: TP={c["TP"]} TN={c["TN"]} FP={c["FP"]} FN={c["FN"]}')
    print(f'\n影片輸出目錄：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
