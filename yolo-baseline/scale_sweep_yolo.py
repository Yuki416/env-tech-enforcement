"""縮放實驗（《數據彙整（End-to-End Rerun）》第 14 節）：YOLO26l 端到端版在 valid 上，以不同輸入尺寸推論。

目的：讓「同一批 litter」在模型輸入上變小或變大，其他條件（影像、場景、clip、權重）都不變，
看召回率隨尺寸變化的速度，和第 16 次 Co-DETR（scale_sweep_codetr16.sh）對照。
推論設定與正式結果相同（conf 0.001、max_det 300、端到端 nms=False），只把 imgsz 換成 512 / 768 / 1024 / 1536。
imgsz=1024 是正式設定，重跑一次用來確認與 e2e/seed<N>/test_eval/valid.pkl 一致。

用法（容器內，YOLO_HEAD 必須是 e2e）：
  YOLO_HEAD=e2e bash run_in_container.sh 0 scale_sweep_yolo.py
輸出：/workspace/hdd2_storage/baseline_yolo26l_v17/e2e/scale_sweep/yolo_seed<N>_<imgsz>/valid.pkl
"""
import json
import os

from ultralytics import YOLO

import yolo_eval_common as C

assert C.HEAD == "e2e", "縮放實驗只做端到端版"
V17 = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual"
SIZES = [512, 768, 1024, 1536]


def main():
    for seed in range(3):
        best = json.load(open(f"{C.YB}/seed{seed}/epoch_select/best_epoch.json"))["best_epoch"]
        model = YOLO(C.ckpt_path(seed, best))
        for sz in SIZES:
            out = f"{C.YB}/scale_sweep/yolo_seed{seed}_{sz}/valid.pkl"
            if os.path.isfile(out):
                print("skip", out); continue
            # predict_to_records 讀的是模組層級的 PRED_KW；這裡只換 imgsz，其他設定（含 nms=False）不動
            C.PRED_KW = dict(C.PRED_KW, imgsz=sz)
            recs = C.predict_to_records(model, f"{V17}/annotations/valid.json", f"{V17}/valid", batch=8)
            C.save_pkl(recs, out)
            json.dump(dict(seed=seed, epoch=best, ckpt=C.ckpt_path(seed, best), pred_kw=C.PRED_KW),
                      open(out.replace(".pkl", "_meta.json"), "w"), indent=2)
            print("→", out, flush=True)
    print("SCALE_SWEEP_YOLO_DONE")


if __name__ == "__main__":
    main()
