"""變因① statictail — Video Inference 第二步（Step 2，沿用第16次做法）。
用 error_analysis（IoU≥0.5, score0.3）重算每支影片的 TP/TN/FP/FN，對分類有誤的影片重渲染 HUD，更新 CSV/MD。

**為什麼一定要跑**：Step 1（infer）分類只看「有沒有預測到 litter/garbage」→ 高估。13/14 次都曾漏此步。
前置：post_analysis_var1.sh（Error Analysis）需先跑完，產出 {split}_per_frame_class_errors.csv。

⚠️ 逐 split 指定 error_analysis 來源（涵蓋範圍要與影片一致，否則某片查無資料會靜默保留高估分類）：
   train/valid → error_analysis_score03（statictail train3551/valid373）
   test        → error_analysis_full655_score03（影片涵蓋 655/8 部；不可用 594 子集）
⚠️ clip 級指標本身失真（任一幀命中即 TP）→ 判讀以逐框 AP/TP/FN 為準，此處僅定性檢視。

用法（容器內）：python /workspace/project/experiments/ablation_study_4vars_20260804/fix_video_classifications_var1.py
"""
import csv, re, glob, os, collections
from pathlib import Path
import cv2

RESULT_DIR = "/workspace/hdd2_storage/ablation_var1_statictail_36e"
VIDEO_DIR = f"{RESULT_DIR}/video_inference"
CSV_PATH = f"{VIDEO_DIR}/video_classification.csv"          # 原地覆蓋
MD_PATH = f"{VIDEO_DIR}/video_classification_summary.md"

EA_DIR_BY_SPLIT = {
    'train': f"{RESULT_DIR}/error_analysis_score03",
    'valid': f"{RESULT_DIR}/error_analysis_score03",
    'test':  f"{RESULT_DIR}/error_analysis_full655_score03",   # 655/8 部，與影片一致
}
SPLITS = ['train', 'valid', 'test']
CLS_COLOR = {'TP': (0,255,0), 'TN': (200,200,200), 'FP': (0,165,255), 'FN': (0,0,255)}


def best_epoch():
    ck = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
    m = re.search(r"epoch_(\d+)", os.path.basename(ck[-1])) if ck else None
    return m.group(1) if m else '?'


def compute_correct():
    correct = {}
    for split in SPLITS:
        path = f"{EA_DIR_BY_SPLIT[split]}/{split}_per_frame_class_errors.csv"
        if not os.path.exists(path):
            raise SystemExit(f"❌ 找不到 {path}\n   post_analysis_var1.sh 是否已跑完？"
                             f"（test 需 error_analysis_full655_score03，因影片涵蓋 655/8 部）")
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
            cls = 'TP' if (has_gt and has_tp) else 'FN' if has_gt else 'FP' if has_fp else 'TN'
            correct[(split, vname)] = cls
    return correct


def re_render(video_path, split, vname, new_cls):
    import tempfile, shutil
    tmp_fd, tmp = tempfile.mkstemp(suffix='.mp4', dir='/tmp'); os.close(tmp_fd)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: 無法開啟 {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    color = CLS_COLOR.get(new_cls, (255,255,255))
    hud = f"{split.upper()} | {vname[-30:]} | {new_cls}"
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        cv2.rectangle(fr, (0,0), (w,28), (0,0,0), -1)
        cv2.putText(fr, hud, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        out.write(fr)
    cap.release(); out.release(); shutil.move(tmp, video_path)
    print(f"  重渲染：{Path(video_path).name}")


def main():
    ep = best_epoch()
    correct = compute_correct()
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    fieldnames = list(rows[0].keys())
    before = collections.Counter((r['split'], r['classification']) for r in rows)

    missing = [(r['split'], r['video_name']) for r in rows if (r['split'], r['video_name']) not in correct]
    if missing:
        print(f"❌ 有 {len(missing)} 部影片在 error_analysis 查無資料，無法校正：")
        for sp, v in missing[:10]:
            print(f"   [{sp}] {v}")
        raise SystemExit("→ 影片與 error_analysis 的 split 涵蓋不一致（test 應用 full655）。不可放行（會高估）。")
    print(f"✅ 校正涵蓋率：{len(rows)}/{len(rows)} 部皆有 error_analysis 對應")

    changed = 0
    for row in rows:
        new_cls = correct[(row['split'], row['video_name'])]
        if row['classification'] != new_cls:
            print(f"[{row['split']}] {row['video_name']}: {row['classification']} → {new_cls}")
            re_render(row['output_video'], row['split'], row['video_name'], new_cls)
            row['classification'] = new_cls
            row['has_gt_litter_garbage'] = str(new_cls in ('TP', 'FN'))
            row['has_pred_litter_garbage'] = str(new_cls in ('TP', 'FP'))
            changed += 1

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

    lines = [f'# 變因① statictail 影片分類（epoch {ep}, score≥0.3, **IoU≥0.5 校正**）\n\n',
             '> 分類依據：litter/garbage 是否有 IoU≥0.5 匹配 TP（已用 error_analysis 校正）\n\n']
    for split in SPLITS:
        ss = [r for r in rows if r['split'] == split]
        c = collections.Counter(r['classification'] for r in ss)
        lines.append(f'## {split.upper()}\n\n| 分類 | 影片數 |\n|---|---|\n')
        for k in ['TP','TN','FP','FN']:
            lines.append(f'| {k} | {c.get(k,0)} |\n')
        lines.append('\n| 影片 | frames | 分類 |\n|---|---|---|\n')
        for r in sorted(ss, key=lambda x: x['classification']):
            lines.append(f"| {r['video_name']} | {r['num_frames']} | {r['classification']} |\n")
        lines.append('\n')
    open(MD_PATH, 'w', encoding='utf-8').writelines(lines)

    print(f"\n=== 完成，修正 {changed} 支 ===")
    for split in SPLITS:
        ss = [r for r in rows if r['split'] == split]
        c = collections.Counter(r['classification'] for r in ss)
        b = {k[1]: v for k, v in before.items() if k[0] == split}
        print(f"{split}: 前 TP={b.get('TP',0)} TN={b.get('TN',0)} FP={b.get('FP',0)} FN={b.get('FN',0)}"
              f"  →  後 TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}")


if __name__ == '__main__':
    main()
