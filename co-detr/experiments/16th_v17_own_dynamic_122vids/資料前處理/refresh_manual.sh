#!/bin/bash
# 手動微調一條龍：改完 build_v17_manual_head10.py 的 RANGE_OVERRIDE 後跑這支。
# build 手動版 → 產 override clips 影片 → 更新動態區間 MD → 上傳手動調整夾到 gdrive。
set -e
P="$(cd "$(dirname "$0")" && pwd)"
DYN=/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual
VIDOUT=/path/to/data/16th_expt_v17_own_dynamic_122vids_36e/連續片段影片_手動調整
GD="gdrive:Co-DETR_training/16th_expt_v17_own_dynamic_122vids/連續片段影片_手動調整"

echo "--- 1. build 手動版 ---"
python3 "$P/build_v17_manual_head10.py" > /path/to/repo/build_v17_manual_head10.log 2>&1
echo "--- 2. 產 override clips 影片 ---"
PYTHONUTF8=1 python3 - "$P" "$DYN" "$VIDOUT" <<'PYEOF'
import importlib.util, sys
from pathlib import Path
P, DYN, VIDOUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def load(n):
    sp=importlib.util.spec_from_file_location(n[:-3], P/n); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
G=load("gen_review_videos_16th.py"); MB=load("build_v17_manual_head10.py")
km=lambda c,k: c==k or c==f"{k}_RoboFlow"
keys=set(MB.B.RANGE_OVERRIDE)
for split in ["train","valid"]:
    imgs,anns,clips=G.load(DYN,split)
    for clip,frames in clips.items():
        if any(km(clip,k) for k in keys):
            G.make_video(clip,frames,imgs,anns,DYN/split,VIDOUT)
PYEOF
echo "--- 3. 更新動態區間 MD ---"
PYTHONUTF8=1 python3 "$P/analyze_dynamic_intervals.py" > /dev/null
echo "--- 4. 上傳手動調整夾 ---"
rclone copy "$VIDOUT" "$GD/" --stats-one-line -v > /path/to/repo/upload_manual.log 2>&1
echo "=== 手動覆寫段（MD）==="
sed -n '/## 手動覆寫/,/## 逐 clip/p' "$P/動態區間對照_HEAD10_vs_HEAD15_vs_15th.md" | grep -E "^\| (split|train|valid)"
echo "gdrive 手動調整夾: $(rclone lsf "$GD/" 2>/dev/null | grep -c '\.mp4$') 部"
echo "REFRESH_DONE"
