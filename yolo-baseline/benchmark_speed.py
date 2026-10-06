#!/usr/bin/env python3
"""步驟 8：量 YOLO26l 的推論速度、峰值 GPU 記憶體與參數量（yolo-baseline 容器）。

量測條件與 benchmark_speed_codetr.py 完全相同：同一張 RTX 4090、batch 1、FP32、
同一批 100 張 test 影像（dynamic166 依影像 id 排序的前 100 張），先跑 10 張暖機不計時。
記錄兩種時間：
  端到端：從讀進影像路徑到拿到最終框（含讀檔、前處理、模型、後處理），用 CUDA 同步後的牆鐘時間
  純模型：Ultralytics 自己回報的 inference 時間（只有網路前向）

用法（容器內，需設 YOLO_HEAD）：python benchmark_speed.py --ckpt <epochN.pt> --out <json>
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from ultralytics import YOLO

from yolo_eval_common import HEAD_LABEL, PRED_KW

ANN = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations/test_dynamic166.json"
IMG = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ims = sorted(json.load(open(ANN, encoding="utf-8"))["images"], key=lambda x: x["id"])[:110]
    paths = [os.path.join(IMG, im["file_name"]) for im in ims]
    model = YOLO(a.ckpt)
    kw = dict(**PRED_KW, device=0, half=False)               # 推論設定與評估完全相同（含 YOLO_HEAD 決定的頭）

    for p in paths[:10]:                       # 暖機
        model.predict(p, **kw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    e2e, pure = [], []
    for p in paths[10:110]:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = model.predict(p, **kw)[0]
        torch.cuda.synchronize(); e2e.append((time.perf_counter() - t0) * 1000)
        pure.append(r.speed["inference"])

    # 兩種參數量：checkpoint 全部（含兩個頭，舊版 JSON 的 params_M_inference_model 其實是這個，名稱有誤）、
    # predictor 實際執行的模型（fuse 後只剩用到的那個頭，BN 也併進卷積）
    n_params = sum(p.numel() for p in model.model.parameters())
    n_params_run = sum(p.numel() for p in model.predictor.model.model.parameters())   # predictor.model 是 AutoBackend 包裝層，本身沒有註冊參數
    res = {
        "model": "YOLO26l", "head": HEAD_LABEL, "ckpt": a.ckpt, "gpu": torch.cuda.get_device_name(0), "precision": "fp32",
        "n_images": len(e2e), "warmup": 10, "imgsz": 1024,
        "end_to_end_ms": {"mean": float(np.mean(e2e)), "median": float(np.median(e2e))},
        "model_only_ms": {"mean": float(np.mean(pure)), "median": float(np.median(pure))},
        "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "params_M_checkpoint_all": n_params / 1e6,
        "params_M_fused_predictor": n_params_run / 1e6,
        "end2end": bool(getattr(model.predictor.model, "end2end", False)),
    }
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
