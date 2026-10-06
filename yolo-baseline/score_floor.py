#!/usr/bin/env python3
"""端到端重跑步驟 5 驗收 ③：量每個預測檔實際的存檔下限（見 ../YOLO_baseline_端到端重跑規劃.md）。

存檔時有兩道關卡：分數要高於 conf=0.001，而且每張圖最多 max_det=300 個框。哪一道先擋住，就決定那張圖實際存到多低的分數。
  一張圖不到 300 個框 → 是 0.001 這道擋住的，這張圖的下限就是 0.001
  一張圖剛好 300 個框 → 是 300 這道擋住的，這張圖的下限是第 300 名的分數，比 0.001 高
AP 只能掃到存下來的最低分，所以「剛好 300 個框」的圖越多、它們的第 300 名分數越高，AP 曲線就越早結束。

對 <YB>/seed<N>/{test_eval,full_preds}/*.pkl 逐檔統計：影像數、最低分數、每張圖框數的最大／平均、
剛好 300 個框的影像數與比例、這些影像的第 300 名分數（中位數、最大值）。

用法（容器內，需設 YOLO_HEAD）：python score_floor.py
輸出：<YB>/score_floor.json 與 score_floor.md
"""
import glob
import json
import os
import pickle

import numpy as np

from yolo_eval_common import HEAD_LABEL, PRED_KW, YB

MAX_DET = PRED_KW["max_det"]


def main():
    rows = []
    for pkl in sorted(glob.glob(f"{YB}/seed*/test_eval/*.pkl") + glob.glob(f"{YB}/seed*/full_preds/*.pkl")):
        recs = pickle.load(open(pkl, "rb"))
        n = np.array([len(r["pred_instances"]["scores"]) for r in recs])
        mins = [float(r["pred_instances"]["scores"].min()) for r in recs if len(r["pred_instances"]["scores"])]
        capped = [float(r["pred_instances"]["scores"].min()) for r in recs if len(r["pred_instances"]["scores"]) == MAX_DET]
        rows.append(dict(
            pkl=os.path.relpath(pkl, YB), n_images=len(recs),
            min_score=round(min(mins), 6) if mins else None,
            boxes_per_image_max=int(n.max()), boxes_per_image_mean=round(float(n.mean()), 1),
            n_images_at_cap=len(capped), frac_at_cap=round(len(capped) / len(recs), 4),
            cap_floor_median=round(float(np.median(capped)), 4) if capped else None,
            cap_floor_max=round(float(np.max(capped)), 4) if capped else None,
        ))
        assert n.max() <= MAX_DET, f"{pkl}：有影像超過 {MAX_DET} 個框"
    json.dump(rows, open(f"{YB}/score_floor.json", "w"), ensure_ascii=False, indent=2)
    md = [f"# YOLO26l {HEAD_LABEL} 版的實際存檔下限\n\n",
          f"> 存檔設定：分數 > {PRED_KW['conf']}、每張圖最多 {MAX_DET} 個框。「剛好 {MAX_DET} 個框」的影像，實際下限是它第 {MAX_DET} 名的分數。\n\n",
          f"| 預測檔 | 影像數 | 最低分數 | 每張圖框數 最大 / 平均 | 剛好 {MAX_DET} 個框的影像（比例） | 這些影像第 {MAX_DET} 名分數 中位數 / 最大 |\n",
          "|---|---|---|---|---|---|\n"]
    for r in rows:
        cf = f"{r['cap_floor_median']} / {r['cap_floor_max']}" if r["n_images_at_cap"] else "—"
        md.append(f"| `{r['pkl']}` | {r['n_images']} | {r['min_score']} | {r['boxes_per_image_max']} / {r['boxes_per_image_mean']} | "
                  f"{r['n_images_at_cap']}（{r['frac_at_cap']:.1%}） | {cf} |\n")
    open(f"{YB}/score_floor.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))
    print("SCORE_FLOOR_DONE")


if __name__ == "__main__":
    main()
