"""
fix_video_classifications_v11.py
用 error_analysis_score03（IoU ≥ 0.5）重新計算每支影片的 TP/TN/FP/FN，
對分類有誤的影片重渲染 HUD 文字，並更新 CSV 和 Markdown 報告。
"""

import csv
import collections
from pathlib import Path

import cv2
import numpy as np

EA_DIR    = "/workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score03"
VIDEO_DIR = "/path/to/data/chiayi_v11_36e_65vids/video_inference"
OLD_CSV   = f"{VIDEO_DIR}/video_classification.csv"
NEW_CSV   = f"{VIDEO_DIR}/video_classification.csv"   # 原地覆蓋
MD_PATH   = f"{VIDEO_DIR}/video_classification_summary.md"

SPLITS = ['train', 'valid', 'test']

CLS_COLOR = {
    'TP': (0,   255, 0  ),
    'TN': (200, 200, 200),
    'FP': (0,   165, 255),
    'FN': (0,   0,   255),
}


def compute_correct_classifications():
    """從 error_analysis_score03 計算 IoU-based 影片分類。"""
    correct = {}
    for split in SPLITS:
        path = f"{EA_DIR}/{split}_per_frame_class_errors.csv"
        by_video = collections.defaultdict(lambda: {'gt': 0, 'tp': 0, 'fp': 0})
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['class'] not in ('litter', 'garbage'):
                    continue
                v = r['video_name']
                by_video[v]['gt'] += int(r['TP']) + int(r['FN'])
                by_video[v]['tp'] += int(r['TP'])
                by_video[v]['fp'] += int(r['FP'])

        for vname, s in by_video.items():
            has_gt = s['gt'] > 0
            has_tp = s['tp'] > 0
            has_fp = s['fp'] > 0
            if   has_gt and     has_tp: cls = 'TP'
            elif has_gt and not has_tp: cls = 'FN'
            elif not has_gt and has_fp: cls = 'FP'
            else:                       cls = 'TN'
            correct[(split, vname)] = cls
    return correct


def re_render_hud(video_path, split, vname, new_cls):
    """重新渲染影片 HUD（覆蓋舊分類文字）。"""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', dir='/tmp')
    import os as _os; _os.close(tmp_fd)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: 無法開啟 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    color = CLS_COLOR.get(new_cls, (255, 255, 255))
    hud   = f"{split.upper()} | {vname[-30:]} | {new_cls}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 覆蓋舊 HUD
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    import shutil
    shutil.move(tmp_path, video_path)
    print(f"  重渲染完成：{Path(video_path).name}")


def main():
    correct = compute_correct_classifications()

    rows = []
    with open(OLD_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    changed = 0

    for row in rows:
        key = (row['split'], row['video_name'])
        new_cls = correct.get(key, row['classification'])
        old_cls = row['classification']

        if old_cls != new_cls:
            print(f"[{row['split']}] {row['video_name']}: {old_cls} → {new_cls}")
            video_path = row['output_video']
            re_render_hud(video_path, row['split'], row['video_name'], new_cls)
            row['classification'] = new_cls
            row['has_gt_litter_garbage']  = str(correct.get(key) in ('TP', 'FN'))
            row['has_pred_litter_garbage'] = str(correct.get(key) in ('TP', 'FP'))
            changed += 1

    with open(NEW_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV 已更新：{NEW_CSV}")

    lines = [
        '# V11 影片分類摘要（Epoch 5，score_thr=0.3，IoU≥0.5 嚴格比對）\n',
        '> 分類依據：litter/garbage 是否有 IoU≥0.5 匹配的 TP 偵測\n\n',
    ]
    for split in SPLITS:
        ss = [r for r in rows if r['split'] == split]
        counts = collections.Counter(r['classification'] for r in ss)
        lines.append(f'## {split.upper()}\n')
        lines.append('| 分類 | 影片數 |\n|------|--------|\n')
        for cls in ['TP', 'TN', 'FP', 'FN']:
            lines.append(f'| {cls} | {counts.get(cls, 0)} |\n')
        lines.append('\n### 詳細清單\n')
        lines.append('| 影片名稱 | frames | 分類 |\n|---------|--------|------|\n')
        for r in sorted(ss, key=lambda x: x['classification']):
            lines.append(f"| {r['video_name']} | {r['num_frames']} | {r['classification']} |\n")
        lines.append('\n')

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Markdown 已更新：{MD_PATH}")

    print(f"\n=== 完成，共修正 {changed} 支影片 ===")
    for split in SPLITS:
        ss = [r for r in rows if r['split'] == split]
        counts = collections.Counter(r['classification'] for r in ss)
        print(f"{split}: TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}")


if __name__ == '__main__':
    main()
