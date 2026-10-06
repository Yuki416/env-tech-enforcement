"""變因① statictail — Video Inference 第一步（Step 1，沿用第16次做法）。
對 best-ckpt 在 train/valid/test 每個 clip 跑推論，輸出帶 bbox 的 MP4（預測彩色實線 + GT 綠虛線），
並依 litter/garbage「有無預測」做**粗分類** TP/TN/FP/FN。

⚠️ 這裡的分類只看「有沒有預測到 litter/garbage」→ 會高估（模型吐一個位置全錯的框也算 TP）。
   **必須再跑 Step 2 `fix_video_classifications_var1.py`**（讀 error_analysis 的 per-frame CSV，IoU≥0.5 校正）。
   → 因此執行順序：先 post_analysis_var1.sh（Error Analysis）→ 本腳本 → fix。

用法（容器內）：
  python /workspace/project/experiments/ablation_study_4vars_20260804/infer_var1_statictail_video_classification.py [--splits train valid test]
"""
import os, re, glob, json, csv, argparse, collections
from pathlib import Path
import cv2
from mmdet.apis import DetInferencer

CONFIG = "/workspace/project/configs/co_dino_chiayi_v17_statictail_ablation_36e.py"
_CKPTS = sorted(glob.glob(
    "/workspace/hdd2_storage/ablation_var1_statictail_36e/best_coco_litter_precision_epoch_*.pth"))
CHECKPOINT = _CKPTS[-1] if _CKPTS else "/workspace/hdd2_storage/ablation_var1_statictail_36e/BEST_CKPT_NOT_FOUND.pth"
DATA_ROOT = "/workspace/datasets/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation"
OUTPUT_DIR = "/workspace/hdd2_storage/ablation_var1_statictail_36e/video_inference"

CLASSES = ['car', 'litter', 'motorcycle', 'person', 'garbage']
TARGET_CLASSES = {'litter', 'garbage'}
SCORE_THRESH = 0.3
OUT_FPS = 5
OUT_WIDTH, OUT_HEIGHT = 1296, 972
PRED_COLORS = {'car': (0,255,255), 'litter': (0,0,255), 'motorcycle': (255,255,0),
               'person': (255,0,255), 'garbage': (0,165,255)}
GT_COLOR = (0, 255, 0)
CLS_HUD = {'TP': (0,255,0), 'TN': (200,200,200), 'FP': (0,165,255), 'FN': (0,0,255)}


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=12):
    x1, y1 = pt1; x2, y2 = pt2
    for seg, axis in [((x1,y1,x2,y1),'h'), ((x2,y1,x2,y2),'v'), ((x1,y2,x2,y2),'h'), ((x1,y1,x1,y2),'v')]:
        ax1, ay1, ax2, ay2 = seg
        length = abs(ax2-ax1) if axis == 'h' else abs(ay2-ay1)
        for i in range(max(length // (dash_len*2), 1)):
            if axis == 'h':
                sx = ax1 + i*2*dash_len; ex = min(ax1 + (i*2+1)*dash_len, ax2)
                cv2.line(img, (sx, ay1), (ex, ay2), color, thickness)
            else:
                sy = ay1 + i*2*dash_len; ey = min(ay1 + (i*2+1)*dash_len, ay2)
                cv2.line(img, (ax1, sy), (ax2, ey), color, thickness)


def annotate(img, gt_anns, preds, sx, sy):
    for g in gt_anns:
        x1, y1, x2, y2 = [int(v*s) for v, s in zip(g['bbox'], (sx, sy, sx, sy))]
        draw_dashed_rect(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(img, f"GT:{g['class']}", (x1, max(y1-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1, cv2.LINE_AA)
    for lid, sc, bb in preds:
        if sc < SCORE_THRESH:
            continue
        cn = CLASSES[lid] if lid < len(CLASSES) else str(lid)
        col = PRED_COLORS.get(cn, (200,200,200))
        x1, y1, x2, y2 = [int(v*s) for v, s in zip(bb, (sx, sy, sx, sy))]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        t = f"{cn} {sc:.2f}"; (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = max(y1-6, th+4)
        cv2.rectangle(img, (x1, ty-th-4), (x1+tw+4, ty), col, -1)
        cv2.putText(img, t, (x1+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)


def load_gt(ann_file):
    data = json.load(open(ann_file, encoding='utf-8'))
    cat = {c['id']: c['name'] for c in data['categories']}
    img = {i['id']: i['file_name'] for i in data['images']}
    gt = collections.defaultdict(list)
    for a in data['annotations']:
        x, y, w, h = a['bbox']
        gt[img[a['image_id']]].append({'bbox': [x, y, x+w, y+h], 'class': cat[a['category_id']]})
    return dict(gt)


def process_split(split, inferencer, gt_lookup, out_base):
    frames = sorted((Path(DATA_ROOT)/split).glob("*.jpg"))
    print(f"\n[{split}] {len(frames)} frames")
    vids = collections.defaultdict(list)
    for f in frames:
        p = f.stem.rsplit('_frame_', 1)
        vids[p[0] if len(p) == 2 else f.stem].append((int(p[1]) if len(p) == 2 else 0, f))
    for v in vids:
        vids[v].sort()
    out_dir = Path(out_base)/split; out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for vidx, (vname, fl) in enumerate(sorted(vids.items()), 1):
        has_gt = has_pred = False
        fdata = []
        for fnum, fpath in fl:
            gt_anns = gt_lookup.get(fpath.name, [])
            if any(g['class'] in TARGET_CLASSES for g in gt_anns):
                has_gt = True
            r = inferencer(str(fpath), show=False, print_result=False)['predictions'][0]
            preds = list(zip(r['labels'], r['scores'], r['bboxes']))
            if any(CLASSES[l] in TARGET_CLASSES for l, s, _ in preds if s >= SCORE_THRESH and l < len(CLASSES)):
                has_pred = True
            fdata.append((fpath, gt_anns, preds))
        # 粗分類（只看有無預測；Step2 fix 會用 IoU≥0.5 校正）
        cls = 'TP' if (has_gt and has_pred) else 'FN' if has_gt else 'FP' if has_pred else 'TN'
        print(f"  [{vidx}/{len(vids)}] {vname[-40:]} ({len(fl)}f) → {cls} (粗分類)")
        writer = None
        for fpath, gt_anns, preds in fdata:
            o = cv2.imread(str(fpath))
            if o is None:
                continue
            oh, ow = o.shape[:2]
            img = cv2.resize(o, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir/f"{vname}.mp4"),
                                         cv2.VideoWriter_fourcc(*'mp4v'), OUT_FPS, (OUT_WIDTH, OUT_HEIGHT))
            annotate(img, gt_anns, preds, OUT_WIDTH/ow, OUT_HEIGHT/oh)
            cv2.rectangle(img, (0,0), (OUT_WIDTH, 28), (0,0,0), -1)
            cv2.putText(img, f"{split.upper()} | {vname[-30:]} | {cls}", (6,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLS_HUD.get(cls, (255,255,255)), 1, cv2.LINE_AA)
            writer.write(img)
        if writer:
            writer.release()
        stats.append({'split': split, 'video_name': vname, 'num_frames': len(fl),
                      'has_gt_litter_garbage': has_gt, 'has_pred_litter_garbage': has_pred,
                      'classification': cls, 'output_video': str(out_dir/f"{vname}.mp4")})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ep = (re.search(r'epoch_(\d+)', os.path.basename(CHECKPOINT)) or [None, '?'])[1]
    print(f"CKPT: {CHECKPOINT} (epoch {ep})")
    inferencer = DetInferencer(model=CONFIG, weights=CHECKPOINT, device=a.device)
    alls = []
    for sp in a.splits:
        alls += process_split(sp, inferencer, load_gt(f"{DATA_ROOT}/annotations/{sp}.json"), OUTPUT_DIR)
    csvp = f"{OUTPUT_DIR}/video_classification.csv"
    fn = ['split','video_name','num_frames','has_gt_litter_garbage','has_pred_litter_garbage','classification','output_video']
    with open(csvp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(alls)
    print(f"\nStep1 CSV（粗分類）：{csvp}\n→ 接著必跑 fix_video_classifications_var1.py 做 IoU≥0.5 校正")


if __name__ == '__main__':
    main()
