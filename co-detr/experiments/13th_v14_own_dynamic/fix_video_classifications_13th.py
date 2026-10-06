"""
fix_video_classifications_13th.py  （第13次 v14 動態版）— 補做 video-inference Step 2
用 error_analysis_score03（IoU ≥ 0.5）重新計算每支影片的 TP/TN/FP/FN，
對分類有誤的影片重渲染 HUD 文字，並更新 CSV 與 Markdown 報告。

補做原因：
  第13次的 run_post_training_13th.sh 只跑了 Step 1（infer_13th_v14_video_classification.py），
  **漏掉 Step 2 的 IoU 校正**（見影片推論流程說明）。Step 1 的分類只看
  「有沒有預測 litter/garbage」，模型隨便吐一個框就算 TP → 高估。實測有 4 部影片被高估
  （valid 3 部、test 1 部，皆 TP → FN）。

⚠️ 不要用 tools/fix_video_classifications.py（通用版）：其路徑寫死指向 chiayi_v10（第8次），
   NEW_CSV 為原地覆蓋 → 直接跑會改到別次實驗的檔案。依專案慣例每次實驗各自獨立一支。

score 對齊：Step 1 用 SCORE_THRESH=0.3 → 對應 error_analysis_score03（IoU 門檻另固定 0.5）。

用法（容器內；需掛 hdd2_storage，CSV 內 output_video 是容器路徑）：
  python /workspace/project/experiments/13th_v14_own_dynamic/fix_video_classifications_13th.py
"""

import csv
import re
import glob
import os
import collections
from pathlib import Path

import cv2

RESULT_DIR = "/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
EA_DIR     = f"{RESULT_DIR}/error_analysis_score03"
VIDEO_DIR  = f"{RESULT_DIR}/video_inference"
OLD_CSV    = f"{VIDEO_DIR}/video_classification.csv"
NEW_CSV    = f"{VIDEO_DIR}/video_classification.csv"   # 原地覆蓋
MD_PATH    = f"{VIDEO_DIR}/video_classification_summary.md"

SPLITS = ['train', 'valid', 'test']
CLS_COLOR = {
    'TP': (0,   255, 0  ),
    'TN': (200, 200, 200),
    'FP': (0,   165, 255),
    'FN': (0,   0,   255),
}


def best_epoch():
    ck = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
    m = re.search(r"epoch_(\d+)", os.path.basename(ck[-1])) if ck else None
    return m.group(1) if m else '?'


def compute_correct_classifications():
    """從 error_analysis_score03 計算 IoU-based（≥0.5）影片分類。"""
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
            has_gt, has_tp, has_fp = s['gt'] > 0, s['tp'] > 0, s['fp'] > 0
            if   has_gt and     has_tp: cls = 'TP'
            elif has_gt and not has_tp: cls = 'FN'
            elif not has_gt and has_fp: cls = 'FP'
            else:                       cls = 'TN'
            correct[(split, vname)] = cls
    return correct


def re_render_hud(video_path, split, vname, new_cls):
    """重新渲染影片 HUD（覆蓋舊分類文字）。"""
    import tempfile, shutil
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4', dir='/tmp')
    os.close(tmp_fd)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: 無法開啟 {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    color = CLS_COLOR.get(new_cls, (255, 255, 255))
    hud   = f"{split.upper()} | {vname[-30:]} | {new_cls}"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()
    shutil.move(tmp_path, video_path)
    print(f"  重渲染完成：{Path(video_path).name}")


def main():
    ep = best_epoch()
    correct = compute_correct_classifications()

    with open(OLD_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    before = collections.Counter((r['split'], r['classification']) for r in rows)
    changed = 0
    for row in rows:
        key = (row['split'], row['video_name'])
        new_cls = correct.get(key, row['classification'])
        old_cls = row['classification']
        if old_cls != new_cls:
            print(f"[{row['split']}] {row['video_name']}: {old_cls} → {new_cls}")
            re_render_hud(row['output_video'], row['split'], row['video_name'], new_cls)
            row['classification'] = new_cls
            row['has_gt_litter_garbage']   = str(new_cls in ('TP', 'FN'))
            row['has_pred_litter_garbage'] = str(new_cls in ('TP', 'FP'))
            changed += 1

    with open(NEW_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV 已更新：{NEW_CSV}")

    lines = [
        f'# v14 影片分類摘要（第13次訓練，Epoch {ep}，score_thr=0.3，**IoU≥0.5 嚴格比對**）\n',
        '> 分類依據：litter/garbage 是否有 IoU≥0.5 匹配的 TP 偵測（已用 error_analysis_score03 校正）\n\n',
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
        c = collections.Counter(r['classification'] for r in ss)
        b = {k[1]: v for k, v in before.items() if k[0] == split}
        print(f"{split}:  校正前 TP={b.get('TP',0)} TN={b.get('TN',0)} FP={b.get('FP',0)} FN={b.get('FN',0)}"
              f"  →  校正後 TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}")


if __name__ == '__main__':
    main()
