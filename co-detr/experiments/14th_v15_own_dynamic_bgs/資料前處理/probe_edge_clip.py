"""0 pre-roll 的 edge clip 20240713054438：三種生法對照，疊圖看實際長相。
  A. 全黑（v15 目前做法：pre-roll=0 → 無背景 → 全黑）
  B. in-clip 整片暖機（餵 .avi 全 201 幀建背景 → 凍結 → 查詢 15 用幀；移動垃圾仍前景）
  C. tail 暖機（只餵用幀之後的尾巴 frame 15~200 建「垃圾落地後」背景 → 凍結 → 查詢）
輸出每個用幀的三聯疊圖（紅=前景、綠框=litter GT）與框內前景比例。
"""
import json, re, glob
from pathlib import Path
import cv2, numpy as np

V14=Path('/path/to/data/coco/chiayi_v14_own_dynamic_litter')
SANHE='/path/to/data/raw_videos'
OUT=Path(__file__).resolve().parent/'warmup_probe'; OUT.mkdir(exist_ok=True)
TS='20240713054438'; S=0.5; RM=5

def rm5(mask):
    fg=(mask==255).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(fg,8); o=np.zeros_like(fg)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=RM: o[lab==i]=1
    return (o*255).astype(np.uint8)

def knn_frozen(warm_idxs, avi):
    knn=cv2.createBackgroundSubtractorKNN(history=500,dist2Threshold=500.0,detectShadows=True)
    cap=cv2.VideoCapture(avi)
    for idx in warm_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,f=cap.read()
        if ok: knn.apply(cv2.resize(f,None,fx=S,fy=S))
    cap.release(); return knn

def fg_full(knn,img):
    hh,ww=img.shape[:2]
    return cv2.resize(rm5(knn.apply(cv2.resize(img,None,fx=S,fy=S),learningRate=0)),(ww,hh),interpolation=cv2.INTER_NEAREST)

def main():
    avi=glob.glob(f'{SANHE}/**/*{TS}*.avi',recursive=True)[0]
    d=json.load(open(V14/'annotations/train.json'))
    imgs=[i for i in d['images'] if TS in i['file_name']]
    imgs.sort(key=lambda i:int(re.search(r'_frame_(\d+)',i['file_name']).group(1)))
    gt={}
    for a in d['annotations']:
        gt.setdefault(a['image_id'],[]).append((a['category_id'],a['bbox']))
    # 用幀 clip 內編號 1..15 → .avi 0..14
    knnB=knn_frozen(range(0,201),avi)          # 整片
    knnC=knn_frozen(range(15,201),avi)         # 尾巴

    print(f'edge clip {TS}：15 用幀，litter GT，框內前景比例（>127 佔比）')
    print(f'{"frame":>6s} {"A全黑":>8s} {"B整片":>8s} {"C尾巴":>8s}')
    tiles=[]
    for k,im in enumerate(imgs):
        img=cv2.imread(str(V14/'train'/im['file_name']))
        boxes=[b for c,b in gt.get(im['id'],[]) if c==2]
        mA=np.zeros(img.shape[:2],np.uint8); mB=fg_full(knnB,img); mC=fg_full(knnC,img)
        def ratio(m):
            v=[]
            for x,y,w,h in boxes:
                xi,yi,wi,hi=[int(round(t)) for t in (x,y,w,h)]
                cr=m[max(0,yi):yi+hi,max(0,xi):xi+wi]
                if cr.size: v.append((cr>127).mean())
            return np.mean(v) if v else 0
        print(f'{k+1:>6d} {ratio(mA):>8.3f} {ratio(mB):>8.3f} {ratio(mC):>8.3f}')
        # 疊圖（只挑前 5 幀進 montage）
        if k<5:
            row=[]
            for tag,m in [('A black',mA),('B whole',mB),('C tail',mC)]:
                ov=img.copy(); ov[m>127]=(0,0,255); vis=cv2.addWeighted(img,0.5,ov,0.5,0)
                for x,y,w,h in boxes: cv2.rectangle(vis,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),3)
                vis=cv2.resize(vis,(384,288)); cv2.putText(vis,f'f{k+1} {tag}',(6,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                row.append(vis)
            tiles.append(np.hstack(row))
    cv2.imwrite(str(OUT/f'edge_{TS}.png'), np.vstack(tiles))
    print(f'\nmontage → {OUT}/edge_{TS}.png')

if __name__=='__main__':
    main()
