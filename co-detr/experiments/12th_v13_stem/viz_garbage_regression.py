"""視覺化大垃圾退化案例：對指定影片每幀，畫
左 = RGB + GT(綠) + 10th預測(藍,含分數) + 12th預測(紅,含分數)
右 = changemap + GT(綠)
存 PNG，肉眼判斷 4ch 為何漏掉 RGB 抓得到的大垃圾。
"""
import json, collections, re
from pathlib import Path
import cv2, numpy as np, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet.apis import init_detector

V13   = "/path/to/data/coco/chiayi_v13_other_team_litter_dynamic_test_frame"
CFG10 = "/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
CK10  = "/path/to/data/10th_expt_v12_other_team_litter_36e/best_coco_bbox_mAP_epoch_17.pth"
CFG12 = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
CK12  = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/best_coco_bbox_mAP_epoch_17.pth"
OUT   = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/garbage_regression/viz"
IMG   = (1024, 1024)
TARGET_TS = ['20240908065944', '20240911055458']   # 兩部退化 Garbage

def run(model, pipe, path):
    data=pipe({'img_path':path,'img_id':0}); data=pseudo_collate([data])
    with torch.no_grad(): r=model.test_step(data)[0].pred_instances
    return [(float(s),[int(b[0]),int(b[1]),int(b[2]),int(b[3])])
            for b,s in zip(r.bboxes.cpu().numpy(),r.scores.cpu().numpy()) if s>=0.15]

init_default_scope('mmdet'); Path(OUT).mkdir(parents=True,exist_ok=True)
cfg12=Config.fromfile(CFG12); import_modules_from_strings(**cfg12['custom_imports'])
m10=init_detector(Config.fromfile(CFG10),CK10,device='cuda:0')
pipe10=Compose([dict(type='LoadImageFromFile'),dict(type='Resize',scale=IMG,keep_ratio=True),
    dict(type='Pad',size=IMG,pad_val=dict(img=(114,114,114))),
    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])
m12=init_detector(cfg12,CK12,device='cuda:0')
pipe12=Compose([dict(type='LoadImageAndChangemap'),dict(type='Resize',scale=IMG,keep_ratio=True),
    dict(type='Pad',size=IMG,pad_val=dict(img=(114,114,114,0))),
    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])

d=json.load(open(f"{V13}/annotations/test.json"))
img={i['id']:i['file_name'] for i in d['images']}
gt=collections.defaultdict(list)
for a in d['annotations']:
    x,y,w,h=a['bbox']; gt[img[a['image_id']]].append([int(x),int(y),int(x+w),int(y+h)])

for ts in TARGET_TS:
    fns=sorted([fn for fn in gt if ts in fn], key=lambda s:int(re.search(r'_frame_(\d+)',s).group(1)))
    for fn in fns:
        rgb=cv2.imread(f"{V13}/test/{fn}")
        cm=cv2.imread(f"{V13}/test_changemap/{fn[:-4]}.png",0)
        cmc=cv2.cvtColor(cm,cv2.COLOR_GRAY2BGR) if cm is not None else np.zeros_like(rgb)
        for g in gt[fn]:
            cv2.rectangle(rgb,(g[0],g[1]),(g[2],g[3]),(0,220,0),4)
            cv2.rectangle(cmc,(g[0],g[1]),(g[2],g[3]),(0,220,0),4)
        for s,b in run(m10,pipe10,f"{V13}/test/{fn}"):
            cv2.rectangle(rgb,(b[0],b[1]),(b[2],b[3]),(255,120,0),3)
            cv2.putText(rgb,f"10th {s:.2f}",(b[0],b[1]-8),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,120,0),3)
        for s,b in run(m12,pipe12,f"{V13}/test/{fn}"):
            cv2.rectangle(rgb,(b[0],b[1]),(b[2],b[3]),(0,0,255),3)
            cv2.putText(rgb,f"12th {s:.2f}",(b[0],b[3]+34),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        canvas=np.hstack([rgb,cmc])
        h=70; cv2.rectangle(canvas,(0,0),(canvas.shape[1],h),(0,0,0),-1)
        cv2.putText(canvas,f"{ts} {fn.split('_frame_')[1][:-4]} | GT=green 10th=blue 12th=red | right=changemap",
                    (10,48),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),2)
        outp=f"{OUT}/{ts}_{fn.split('_frame_')[1][:-4]}.png"
        cv2.imwrite(outp,canvas)
    print(f"{ts}: {len(fns)} 幀 已存")
print("done:",OUT)
