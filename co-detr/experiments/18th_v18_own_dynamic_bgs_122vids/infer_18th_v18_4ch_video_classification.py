"""
infer_18th_v18_4ch_video_classification.py （第18次 v18 = v17手動動態子集 + BGS changemap 4ch）
對 v18 dataset 全部影片（train/valid/test）跑 best-ckpt inference，輸出帶 bbox 的 MP4，
並依 litter/garbage 分類每支影片。改寫自第14次 4ch 版。

★ 4 通道推論：DetInferencer 只載 3ch → 不能用。用 init_detector + 自訂 4ch pipeline
  （LoadImageAndChangemap 自動從 {split}_changemap/ 取第 4 通道）+ model.test_step。

★ IoU≥0.5 校正「內建」（避免另跑 4ch error_analysis、也避免 Step2 漏做）：
  - 有 GT target 的 clip：**某幀有「score≥thr 且與同類 GT IoU≥0.5」的 litter/garbage 偵測** → TP；否則 FN
    （即「任一幀命中但定位不準(IoU<0.5)」會被降級為 FN，等同過往 fix 腳本的 Step 2）。
  - 無 GT target 的 clip：某幀有 litter/garbage 原始預測(score≥thr) → FP；否則 TN。
  ⚠️ clip 級指標仍會失真（任一幀即算）；判讀一律以逐框（post-analysis 任務2/3）為準。

⚠️ DATA_ROOT 必須是 **v18**（有 {split}_changemap/）；否則第4通道靜默補零、整批作廢（啟動有防呆）。
用法：python .../infer_18th_v18_4ch_video_classification.py [--splits train valid test]
"""
import os, re, glob, json, csv, argparse, collections
from pathlib import Path
import cv2, torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmcv.transforms import Compose
from mmdet.apis import init_detector

CONFIG     = "/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"
RESULT_DIR = "/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e"
_CKPTS = sorted(glob.glob(f"{RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"))
CHECKPOINT = _CKPTS[-1] if _CKPTS else f"{RESULT_DIR}/BEST_CKPT_NOT_FOUND.pth"
DATA_ROOT  = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids"   # ★ 必須 v18（有 changemap）
OUTPUT_DIR = f"{RESULT_DIR}/video_inference"

CLASSES        = ['car', 'litter', 'motorcycle', 'person', 'garbage']
TARGET_CLASSES = {'litter', 'garbage'}
SCORE_THRESH   = 0.3
IOU_THRESH     = 0.5          # ★ 內建校正門檻
OUT_FPS, OUT_WIDTH, OUT_HEIGHT = 5, 1296, 972
IMAGE_SIZE, PAD4 = (1024, 1024), (114, 114, 114, 0)

PRED_COLORS = {'car':(0,255,255),'litter':(0,0,255),'motorcycle':(255,255,0),
               'person':(255,0,255),'garbage':(0,165,255)}
GT_COLOR = (0, 255, 0)


def iou_xyxy(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def check_changemap_or_die():
    missing = [s for s in ('train','valid','test') if not Path(f"{DATA_ROOT}/{s}_changemap").is_dir()]
    if missing:
        raise SystemExit(f"❌ DATA_ROOT={DATA_ROOT} 缺 changemap 目錄：{missing}（4ch 會靜默補零→結果全錯）")
    print(f"✅ changemap 目錄檢查通過：{DATA_ROOT}")


def build_model_and_pipeline(ckpt, device):
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
             meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor')),
    ])
    return model, pipeline


def infer_one(model, pipeline, img_path):
    data = pseudo_collate([pipeline({'img_path': str(img_path), 'img_id': 0})])
    with torch.no_grad():
        results = model.test_step(data)
    pred = results[0].pred_instances
    b, s, l = pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()
    return [(int(li), float(sc), [float(x[0]),float(x[1]),float(x[2]),float(x[3])])
            for x, sc, li in zip(b, s, l)]


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=12):
    x1,y1 = pt1; x2,y2 = pt2
    for seg,axis in [((x1,y1,x2,y1),'h'),((x2,y1,x2,y2),'v'),((x1,y2,x2,y2),'h'),((x1,y1,x1,y2),'v')]:
        a1,b1,a2,b2 = seg; length = abs(a2-a1) if axis=='h' else abs(b2-b1)
        for i in range(max(length//(dash_len*2),1)):
            if axis=='h':
                sx=a1+i*2*dash_len; ex=min(a1+(i*2+1)*dash_len,a2); cv2.line(img,(sx,b1),(ex,b2),color,thickness)
            else:
                sy=b1+i*2*dash_len; ey=min(b1+(i*2+1)*dash_len,b2); cv2.line(img,(a1,sy),(a2,ey),color,thickness)


def annotate_frame(img, gt_anns, preds, sx, sy):
    for gt in gt_anns:
        x1,y1,x2,y2 = gt['bbox']
        draw_dashed_rect(img,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),GT_COLOR,2)
        cv2.putText(img,f"GT:{gt['class']}",(int(x1*sx),max(int(y1*sy)-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,GT_COLOR,1,cv2.LINE_AA)
    for lid,score,bbox in preds:
        if score < SCORE_THRESH: continue
        cname = CLASSES[lid] if lid < len(CLASSES) else str(lid)
        color = PRED_COLORS.get(cname,(200,200,200))
        x1,y1,x2,y2 = int(bbox[0]*sx),int(bbox[1]*sy),int(bbox[2]*sx),int(bbox[3]*sy)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        t=f"{cname} {score:.2f}"; (tw,th),_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
        ty=max(y1-6,th+4); cv2.rectangle(img,(x1,ty-th-4),(x1+tw+4,ty),color,-1)
        cv2.putText(img,t,(x1+2,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)


def frame_hit_iou(gt_anns, preds):
    """某幀是否有『score≥thr 且與同類 GT IoU≥0.5』的 litter/garbage 偵測（內建校正）。"""
    for lid, sc, box in preds:
        if sc < SCORE_THRESH or lid >= len(CLASSES): continue
        cname = CLASSES[lid]
        if cname not in TARGET_CLASSES: continue
        for g in gt_anns:
            if g['class'] == cname and iou_xyxy(box, g['bbox']) >= IOU_THRESH:
                return True
    return False


def frame_pred_raw(preds):
    """某幀是否有任何 litter/garbage 原始預測(score≥thr)——供無 GT clip 判 FP。"""
    return any(lid < len(CLASSES) and CLASSES[lid] in TARGET_CLASSES and sc >= SCORE_THRESH
               for lid, sc, _ in preds)


def process_split(split, model, pipeline, gt_lookup, out_base):
    frames = sorted((Path(DATA_ROOT)/split).glob("*.jpg"))
    print(f"\n[{split}] 共 {len(frames)} 張 frames")
    video_frames = collections.defaultdict(list)
    for f in frames:
        parts = f.stem.rsplit('_frame_', 1)
        vname = parts[0] if len(parts)==2 else f.stem
        fnum  = int(parts[1]) if len(parts)==2 else 0
        video_frames[vname].append((fnum, f))
    for v in video_frames: video_frames[v].sort()
    out_dir = Path(out_base)/split; out_dir.mkdir(parents=True, exist_ok=True)

    stats = []; total = len(video_frames)
    for vidx,(vname,frame_list) in enumerate(sorted(video_frames.items()),1):
        print(f"  [{vidx}/{total}] {vname} ({len(frame_list)} frames)", flush=True)
        has_gt = has_hit = has_raw = False
        frame_data = []
        for fnum,fpath in frame_list:
            gt_anns = gt_lookup.get(fpath.name, [])
            if any(g['class'] in TARGET_CLASSES for g in gt_anns): has_gt = True
            preds = infer_one(model, pipeline, fpath)
            if frame_hit_iou(gt_anns, preds): has_hit = True     # IoU≥0.5 命中
            if frame_pred_raw(preds):         has_raw = True     # 原始預測（供 FP）
            frame_data.append((fpath, gt_anns, preds))

        if has_gt:
            cls = 'TP' if has_hit else 'FN'      # ★ 任一幀命中但 IoU<0.5 → 降 FN（內建校正）
        else:
            cls = 'FP' if has_raw else 'TN'
        print(f"    → {cls}  (gt={has_gt}, hit_iou={has_hit}, raw_pred={has_raw})")

        out_video = str(out_dir/f"{vname}.mp4"); writer = None
        for fpath,gt_anns,preds in frame_data:
            img0 = cv2.imread(str(fpath))
            if img0 is None: continue
            oh,ow = img0.shape[:2]
            img = cv2.resize(img0,(OUT_WIDTH,OUT_HEIGHT),interpolation=cv2.INTER_AREA)
            sx,sy = OUT_WIDTH/ow, OUT_HEIGHT/oh
            if writer is None:
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), OUT_FPS,(OUT_WIDTH,OUT_HEIGHT))
            annotate_frame(img, gt_anns, preds, sx, sy)
            hud = f"{split.upper()} | {vname[-30:]} | {cls}"
            cv2.rectangle(img,(0,0),(OUT_WIDTH,28),(0,0,0),-1)
            cv2.putText(img,hud,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,
                        {'TP':(0,255,0),'TN':(200,200,200),'FP':(0,165,255),'FN':(0,0,255)}.get(cls,(255,255,255)),
                        1,cv2.LINE_AA)
            writer.write(img)
        if writer: writer.release()

        stats.append(dict(split=split, video_name=vname, num_frames=len(frame_list),
                          has_gt_litter_garbage=has_gt, has_pred_iou_hit=has_hit,
                          has_pred_raw=has_raw, classification=cls, output_video=out_video))
    return stats


def load_gt(ann_file):
    data = json.load(open(ann_file, encoding='utf-8'))
    cat = {c['id']:c['name'] for c in data['categories']}
    img = {i['id']:i['file_name'] for i in data['images']}
    gt = collections.defaultdict(list)
    for a in data['annotations']:
        x,y,w,h = a['bbox']
        gt[img[a['image_id']]].append({'bbox':[x,y,x+w,y+h],'class':cat[a['category_id']]})
    return dict(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train','valid','test'])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    check_changemap_or_die()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_ep = (re.search(r'epoch_(\d+)', os.path.basename(CHECKPOINT)) or [None,'?'])[1] \
              if re.search(r'epoch_(\d+)', os.path.basename(CHECKPOINT)) else '?'
    print(f"載入 4ch Co-DETR (epoch {ckpt_ep}, score>={SCORE_THRESH}, IoU>={IOU_THRESH} 內建校正)...")
    model, pipeline = build_model_and_pipeline(CHECKPOINT, args.device)

    all_stats = []
    for split in args.splits:
        gt = load_gt(f"{DATA_ROOT}/annotations/{split}.json")
        all_stats.extend(process_split(split, model, pipeline, gt, OUTPUT_DIR))

    csv_path = f"{OUTPUT_DIR}/video_classification.csv"
    fn = ['split','video_name','num_frames','has_gt_litter_garbage','has_pred_iou_hit','has_pred_raw','classification','output_video']
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(all_stats)
    print(f"\nCSV：{csv_path}")

    md = [f'# 第18次 v18 BGS 4ch 影片分類摘要（Epoch {ckpt_ep}, score_thr={SCORE_THRESH}, **IoU≥{IOU_THRESH} 內建校正**）\n',
          '> TP=有GT且某幀有IoU≥0.5命中；FN=有GT但無命中(含任一幀命中但定位不準已降級)；FP=無GT但有預測；TN=無GT無預測\n',
          '> ⚠️ clip 級失真，判讀以逐框 post-analysis 為準\n\n']
    for split in args.splits:
        ss = [s for s in all_stats if s['split']==split]
        c = collections.Counter(s['classification'] for s in ss)
        md.append(f'## {split.upper()}\n| 分類 | 影片數 |\n|------|--------|\n')
        for k in ['TP','TN','FP','FN']: md.append(f'| {k} | {c.get(k,0)} |\n')
        md.append('\n### 詳細清單\n| 影片名稱 | frames | 分類 |\n|---------|--------|------|\n')
        for s in sorted(ss, key=lambda x:x['classification']):
            md.append(f"| {s['video_name']} | {s['num_frames']} | {s['classification']} |\n")
        md.append('\n')
    open(f"{OUTPUT_DIR}/video_classification_summary.md",'w',encoding='utf-8').writelines(md)
    print(f"MD：{OUTPUT_DIR}/video_classification_summary.md")
    for split in args.splits:
        c = collections.Counter(s['classification'] for s in all_stats if s['split']==split)
        print(f'{split}: TP={c["TP"]} TN={c["TN"]} FP={c["FP"]} FN={c["FN"]}')
    print(f'\nVIDEO_INFERENCE_DONE  影片目錄：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
