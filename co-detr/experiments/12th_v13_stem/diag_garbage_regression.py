"""診斷第 12 次「大垃圾袋退化」：對動態 test 每個 GT 框，比較
(1) changemap 在框內的覆蓋率（BGS 有沒有點亮這個物件）
(2) 10th RGB 模型 對該框的最高信心（IoU>=0.5）
(3) 12th 4ch 模型 對該框的最高信心
→ 區分「BGS 沒點亮大物件」vs「4ch 學成只認小 motion」。
輸出：per-GT CSV + 依 garbage/litter、大小分組的摘要。
"""
import json, collections, csv, re
from pathlib import Path
import cv2, numpy as np, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet.apis import init_detector

V13   = "/path/to/data/coco/chiayi_v13_other_team_litter_dynamic_test_frame"
V11J  = "/path/to/data/coco/chiayi_v11_garbage_litter_corrected/annotations/test.json"
CFG10 = "/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
CK10  = "/path/to/data/10th_expt_v12_other_team_litter_36e/best_coco_bbox_mAP_epoch_17.pth"
CFG12 = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
CK12  = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/best_coco_bbox_mAP_epoch_17.pth"
OUT   = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/garbage_regression"
IMG   = (1024, 1024)

def iou(a, b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def best_score_for(gt, preds):
    best=0.0
    for sc,bb in preds:
        if iou(gt,bb)>=0.5: best=max(best,sc)
    return best

def preds_3ch(model, pipe, path):
    data=pipe({'img_path':path,'img_id':0}); data=pseudo_collate([data])
    with torch.no_grad(): r=model.test_step(data)[0].pred_instances
    return [(float(s),[float(b[0]),float(b[1]),float(b[2]),float(b[3])])
            for b,s in zip(r.bboxes.cpu().numpy(), r.scores.cpu().numpy())]

def main():
    init_default_scope('mmdet')
    Path(OUT).mkdir(parents=True, exist_ok=True)
    # v11 類別（garbage vs litter）供分組
    v11=json.load(open(V11J)); v11c={c['id']:c['name'] for c in v11['categories']}
    v11img={i['id']:i['file_name'] for i in v11['images']}
    cls_of=collections.defaultdict(dict)   # fname -> list of (box, class)
    v11gt=collections.defaultdict(list)
    for a in v11['annotations']:
        nm=v11c[a['category_id']]
        if nm in ('litter','garbage'):
            x,y,w,h=a['bbox']; v11gt[v11img[a['image_id']]].append(([x,y,x+w,y+h],nm))

    # 動態 test GT
    d=json.load(open(f"{V13}/annotations/test.json"))
    img={i['id']:i['file_name'] for i in d['images']}
    gt=collections.defaultdict(list)
    for a in d['annotations']:
        x,y,w,h=a['bbox']; gt[img[a['image_id']]].append([x,y,x+w,y+h])

    # 模型
    cfg12=Config.fromfile(CFG12); import_modules_from_strings(**cfg12['custom_imports'])
    m10=init_detector(Config.fromfile(CFG10), CK10, device='cuda:0')
    pipe10=Compose([dict(type='LoadImageFromFile'),
                    dict(type='Resize',scale=IMG,keep_ratio=True),
                    dict(type='Pad',size=IMG,pad_val=dict(img=(114,114,114))),
                    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])
    m12=init_detector(cfg12, CK12, device='cuda:0')
    pipe12=Compose([dict(type='LoadImageAndChangemap'),
                    dict(type='Resize',scale=IMG,keep_ratio=True),
                    dict(type='Pad',size=IMG,pad_val=dict(img=(114,114,114,0))),
                    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])

    rows=[]
    for fn in sorted(gt):
        vname=fn.split('_frame_')[0]
        img_path=f"{V13}/test/{fn}"
        cm_path=f"{V13}/test_changemap/{fn[:-4]}.png"
        cm=cv2.imread(cm_path,0)
        p10=preds_3ch(m10,pipe10,img_path)
        p12=preds_3ch(m12,pipe12,img_path)
        # v11 類別對照（用檔名+框 IoU 找 garbage/litter）
        for gbox in gt[fn]:
            # 找對應 v11 類別
            klass='?'
            for vb,nm in v11gt.get(fn,[]):
                if iou(gbox,vb)>=0.5: klass=nm; break
            area=(gbox[2]-gbox[0])*(gbox[3]-gbox[1])
            # changemap 框內覆蓋率
            x1,y1,x2,y2=[int(v) for v in gbox]
            cov=float((cm[y1:y2,x1:x2]==255).mean()) if cm is not None and x2>x1 and y2>y1 else -1
            rows.append(dict(video=vname[-24:], frame=fn.split('_frame_')[1][:-4],
                             klass=klass, area=int(area),
                             cm_cov=round(cov,3),
                             s10=round(best_score_for(gbox,p10),3),
                             s12=round(best_score_for(gbox,p12),3)))

    with open(f"{OUT}/per_gt_scores.csv",'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # 摘要：依 class × 大小
    def bucket(a): return 'large(>9216)' if a>=9216 else ('medium' if a>=1024 else 'small')
    print(f"\n{'group':28s} {'n':>3} {'cm_cov':>7} {'10th_det%':>9} {'12th_det%':>9} {'s10':>5} {'s12':>5}")
    agg=collections.defaultdict(list)
    for r in rows: agg[(r['klass'],bucket(r['area']))].append(r)
    for k in sorted(agg):
        rs=agg[k]; n=len(rs)
        cov=np.mean([r['cm_cov'] for r in rs if r['cm_cov']>=0])
        det10=np.mean([r['s10']>=0.3 for r in rs])*100
        det12=np.mean([r['s12']>=0.3 for r in rs])*100
        s10=np.mean([r['s10'] for r in rs]); s12=np.mean([r['s12'] for r in rs])
        print(f"{k[0]+'/'+k[1]:28s} {n:>3} {cov:>7.3f} {det10:>8.0f}% {det12:>8.0f}% {s10:>5.2f} {s12:>5.2f}")
    print(f"\nCSV: {OUT}/per_gt_scores.csv")

if __name__=='__main__':
    main()
