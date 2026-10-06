#!/usr/bin/env python3
"""步驟 8：量第 16 次 Co-DETR 的推論速度、峰值 GPU 記憶體與參數量（co-detr 容器）。

量測條件與 benchmark_speed.py（YOLO）完全相同：同一張 RTX 4090、batch 1、FP32、
同一批 100 張 test 影像（dynamic166 依影像 id 排序的前 100 張），先跑 10 張暖機不計時。
記錄兩種時間：
  端到端：inference_detector() 一張圖（含讀檔、test_pipeline 前處理、模型、後處理含 soft-NMS），CUDA 同步後的牆鐘時間
  純模型：只計 model.test_step()（前處理已先做好），對應 YOLO 的「純模型」時間的口徑較接近，但仍含後處理

參數量分兩個：整體（含只在訓練時使用的輔助頭）與推論實際用到的部分（backbone + neck + query_head）。

用法（co-detr 容器內）：python benchmark_speed_codetr.py --out <json>
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate

CFG = "/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py"
CKPT = "/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth"
ANN = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations/test_dynamic166.json"
IMG = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Config.fromfile(CFG)
    cfg.custom_hooks = []                      # 避免 TrainingMarkdownLogger 動到第 16 次的 training_log.md
    model = init_detector(cfg, CKPT, device="cuda:0")
    ims = sorted(json.load(open(ANN, encoding="utf-8"))["images"], key=lambda x: x["id"])[:110]
    paths = [os.path.join(IMG, im["file_name"]) for im in ims]

    for p in paths[:10]:
        inference_detector(model, p)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    e2e = []
    for p in paths[10:110]:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inference_detector(model, p)
        torch.cuda.synchronize(); e2e.append((time.perf_counter() - t0) * 1000)

    # 純模型：先把前處理做好，只計 test_step
    test_cfg = model.cfg.test_dataloader.dataset.pipeline
    test_cfg[0] = dict(type="LoadImageFromFile")
    pipe = Compose(test_cfg)
    datas = [pseudo_collate([pipe(dict(img_path=p, img_id=0))]) for p in paths[10:110]]
    pure = []
    with torch.no_grad():
        for d in datas:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.test_step(d)
            torch.cuda.synchronize(); pure.append((time.perf_counter() - t0) * 1000)

    tot = sum(p.numel() for p in model.parameters())
    infer = sum(p.numel() for n in ["backbone", "neck", "query_head"] for p in getattr(model, n).parameters())
    res = {
        "model": "Co-DETR 第16次 (Co-DINO 5scale R50)", "ckpt": CKPT, "gpu": torch.cuda.get_device_name(0),
        "precision": "fp32", "n_images": len(e2e), "warmup": 10, "input": "長邊 1024（config test_pipeline）",
        "end_to_end_ms": {"mean": float(np.mean(e2e)), "median": float(np.median(e2e))},
        "model_only_ms": {"mean": float(np.mean(pure)), "median": float(np.median(pure)),
                          "note": "model.test_step()，含資料前處理器(正規化/pad)與後處理(soft-NMS)，不含讀檔與 pipeline"},
        "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "params_M_total": tot / 1e6, "params_M_inference_path": infer / 1e6,
    }
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
