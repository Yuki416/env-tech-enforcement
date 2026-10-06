"""量那 4 部抓不到的 clip，GT 框內的 changemap 前景比 —— 對照抓得到的那 8 部。"""
import json, sys, os, statistics
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/workspace/project/experiments/18th_v18_own_dynamic_bgs_122vids')
from stage2_sweep import gt_by_source_frame
OUT=Path('/workspace/project/experiments/18th_v18_own_dynamic_bgs_122vids/stage2_out')
S,HIST,DIST,RM,WARM=0.5,500,500.0,5,150
BAD={"valid/20240731024839_00m35s","valid/20240810181149_01m29s",
     "valid/20240829062428_01m39s","valid/20240908065812_02m13s"}
def rm5(m):
    fg=(m==255).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(fg,8)
    keep=np.zeros(n,np.uint8); keep[1:]=(st[1:,cv2.CC_STAT_AREA]>=RM)
    return (keep[lab]*255).astype(np.uint8)
print(f"{'clip':<40}{'組':<10}{'框內前景比中位數':>18}{'全黑框比例':>12}")
rows=[]
for fp in sorted(OUT.glob('*.json')):
    r=json.load(open(fp))
    if r['split']=='train': continue
    cap=cv2.VideoCapture(r['avi'])
    knn=cv2.createBackgroundSubtractorKNN(history=HIST,dist2Threshold=DIST,detectShadows=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES,r['warm_start'])
    for _ in range(WARM):
        ok,f=cap.read()
        if not ok: break
        knn.apply(cv2.resize(f,None,fx=S,fy=S))
    gt=gt_by_source_frame(r); ratios=[]
    idx=r['fp_start']
    while idx<r['end']:
        ok,f=cap.read()
        if not ok: break
        cm=knn.apply(cv2.resize(f,None,fx=S,fy=S),learningRate=0)
        if idx in gt:
            cmf=rm5(cm); H,W=cmf.shape
            for cid,(x1,y1,x2,y2) in gt[idx]:
                a,b=int(x1*S),int(y1*S); c,d=int(x2*S),int(y2*S)
                a,b=max(0,a),max(0,b); c,d=min(W,c),min(H,d)
                if c>a and d>b: ratios.append(float((cmf[b:d,a:c]>0).mean()))
        idx+=1
    cap.release()
    if not ratios: continue
    g='抓不到' if r['key'] in BAD else '抓得到'
    med=statistics.median(ratios); black=sum(1 for x in ratios if x==0)/len(ratios)
    rows.append((g,med,black))
    print(f"{r['key']:<40}{g:<10}{med*100:>17.1f}%{black*100:>11.0f}%")
print()
for g in ('抓得到','抓不到'):
    sub=[r for r in rows if r[0]==g]
    print(f"{g}（{len(sub)} 部）：框內前景比中位數的中位數 {statistics.median([x[1] for x in sub])*100:.1f}%、"
          f"全黑框比例中位數 {statistics.median([x[2] for x in sub])*100:.0f}%")
