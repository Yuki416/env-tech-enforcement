#!/usr/bin/env python3
"""暖機幀數掃描：KNN 背景模型到底需要幾幀才暖得起來？

方法：挑一部 pre-roll 充足的 clip，暖機幀數 W 從 0 掃到上限。
  對每個 W：餵 .avi[first_abs-W : first_abs] 建背景 → 凍結 → 對查詢幀取前景。
  量兩個指標：
    signal = litter GT 框內前景比例（越高越好，代表垃圾被點亮）
    noise  = 所有 GT 框外的前景比例（越低越好，代表背景乾淨）
  同時輸出一張查詢幀的疊圖 montage（各 W 並排），紅=前景、綠框=GT。
生成參數與正式 pipeline 一致（S=0.5、KNN 500/500/shadow、rm5）。
"""
import json, re, glob, sys
from pathlib import Path
import cv2, numpy as np

V14=Path('/path/to/data/coco/chiayi_v14_own_dynamic_litter')
SANHE='/path/to/data/raw_videos'
ALIGN=json.load(open(Path(__file__).resolve().parent/'alignment_v15.json'))
OUT=Path(__file__).resolve().parent/'warmup_probe'; OUT.mkdir(exist_ok=True)
S=0.5; RM=5
CLIP_TS='20240605180713'; SPLIT='test'
WARMS=[0,1,2,3,5,10,20,30,50,100,150,200]

def rm5(mask):
    fg=(mask==255).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(fg,8); o=np.zeros_like(fg)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=RM: o[lab==i]=1
    return (o*255).astype(np.uint8)

def main():
    key=next(k for k in ALIGN if ALIGN[k]['ts']==CLIP_TS and ALIGN[k]['split']==SPLIT)
    info=ALIGN[key]; avi=info['avi']; fa=info['first_abs']
    d=json.load(open(V14/f'{SPLIT}'.join(['annotations/','.json'])))
    imgs=[i for i in d['images'] if CLIP_TS in i['file_name']]
    imgs.sort(key=lambda i:int(re.search(r'_frame_(\d+)',i['file_name']).group(1)))
    gt={}   # image_id -> list of (cat,bbox)
    for a in d['annotations']:
        gt.setdefault(a['image_id'],[]).append((a['category_id'],a['bbox']))

    # 查詢幀：挑前段動態幀（含 litter GT）
    q=[im for im in imgs if any(c==2 for c,_ in gt.get(im['id'],[]))][:1]
    if not q: q=imgs[:1]
    qim=q[0]; qpath=V14/SPLIT/qim['file_name']
    qimg=cv2.imread(str(qpath)); H,W=qimg.shape[:2]
    qboxes=gt.get(qim['id'],[])

    # 預讀 pre-roll 全段（half-res）
    cap=cv2.VideoCapture(avi)
    pre=[]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,fa-max(WARMS)))
    for idx in range(max(0,fa-max(WARMS)), fa):
        ok,f=cap.read()
        if not ok: break
        pre.append(cv2.resize(f,None,fx=S,fy=S))
    cap.release()
    qsmall=cv2.resize(qimg,None,fx=S,fy=S)

    # 量所有 litter 查詢幀的平均 signal/noise（跨全 clip 更穩），但 montage 只畫 qim
    litter_ims=[im for im in imgs if any(c==2 for c,_ in gt.get(im['id'],[]))]
    tiles=[]; rows=[]
    for Wn in WARMS:
        knn=cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=500.0,detectShadows=True)
        for f in pre[len(pre)-Wn:] if Wn>0 else []:
            knn.apply(f)
        # signal/noise 跨所有 litter 幀
        sig=[]; noi=[]
        for im in litter_ims:
            img=cv2.imread(str(V14/SPLIT/im['file_name']))
            hh,ww=img.shape[:2]
            if Wn==0:
                m=np.zeros((hh,ww),np.uint8)
            else:
                m=cv2.resize(rm5(knn.apply(cv2.resize(img,None,fx=S,fy=S),learningRate=0)),(ww,hh),interpolation=cv2.INTER_NEAREST)
            boxmask=np.zeros((hh,ww),bool)
            for c,(x,y,w,h) in gt.get(im['id'],[]):
                xi,yi,wi,hi=[int(round(v)) for v in (x,y,w,h)]
                if c==2:
                    crop=m[max(0,yi):yi+hi,max(0,xi):xi+wi]
                    if crop.size: sig.append((crop>127).mean())
                boxmask[max(0,yi):yi+hi,max(0,xi):xi+wi]=True
            noi.append((m[~boxmask]>127).mean())
        rows.append((Wn,np.mean(sig) if sig else 0,np.mean(noi)))

        # montage tile：qim 的疊圖
        if Wn==0:
            mq=np.zeros((H,W),np.uint8)
        else:
            mq=cv2.resize(rm5(knn.apply(qsmall,learningRate=0)),(W,H),interpolation=cv2.INTER_NEAREST)
        ov=qimg.copy(); ov[mq>127]=(0,0,255)
        vis=cv2.addWeighted(qimg,0.5,ov,0.5,0)
        for c,(x,y,w,h) in qboxes:
            if c==2: cv2.rectangle(vis,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),3)
        vis=cv2.resize(vis,(384,288))
        cv2.putText(vis,f'W={Wn}',(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        tiles.append(vis)

    # 排 montage 4 列
    per=4; rowsimg=[]
    for i in range(0,len(tiles),per):
        r=tiles[i:i+per]
        while len(r)<per: r.append(np.zeros_like(tiles[0]))
        rowsimg.append(np.hstack(r))
    cv2.imwrite(str(OUT/f'montage_{CLIP_TS}.png'), np.vstack(rowsimg))

    print(f'掃描 clip {CLIP_TS}（pre-roll={info["preroll"]}, litter幀={len(litter_ims)}）查詢montage幀={qim["file_name"]}')
    print(f'{"W(暖機幀)":>10s} {"signal(框內前景)":>16s} {"noise(框外前景)":>16s} {"訊噪比":>8s}')
    for Wn,s,n in rows:
        snr=s/n if n>1e-6 else float("inf")
        print(f'{Wn:>10d} {s:>16.3f} {n:>16.4f} {snr:>8.1f}')
    print(f'\nmontage → {OUT}/montage_{CLIP_TS}.png')

if __name__=='__main__':
    main()
