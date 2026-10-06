"""
Co-DETR 推論 Smoke Test
-----------------------
用法（在 container 內）：
    python /workspace/project/test_inference.py \
        --img  /workspace/hdd2_storage/raw_videos/6月份影片/113.6.15/<任一>.png \
        --out  /workspace/project/results/smoke_test_output.jpg

預設 config / checkpoint 使用 Co-DINO 5scale + R50（COCO 預訓練）。
"""

import argparse
import os
import glob

import cv2
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer


# ── 預設路徑（container 內掛載後的路徑）────────────────────────────────────────
# mmdetection v3 的 Co-DETR config 位於 projects/CO_DETR/
DEFAULT_CONFIG = (
    "/opt/mmdetection/projects/CO-DETR/configs/codino/"
    "co_dino_5scale_r50_8xb2_1x_coco.py"
)
DEFAULT_CKPT = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"
DEFAULT_IMG_GLOB = (
    "/workspace/hdd2_storage/raw_videos/"
    "6月份影片/113.6.15/**/*.png"
)
DEFAULT_OUT = "/workspace/project/results/smoke_test_output.jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="Co-DETR Inference Smoke Test")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt",   default=DEFAULT_CKPT)
    parser.add_argument("--img",    default=None,
                        help="指定圖片路徑；不指定則自動從 hdd2_storage 找第一張")
    parser.add_argument("--out",    default=DEFAULT_OUT)
    parser.add_argument("--score-thr", type=float, default=0.3,
                        help="顯示 bbox 的信心分數門檻（預設 0.3）")
    return parser.parse_args()


def find_first_image(glob_pattern: str) -> str:
    """在 glob_pattern 中找第一張圖片（排除帶『複製』的檔案）"""
    candidates = [
        p for p in sorted(glob.glob(glob_pattern, recursive=True))
        if "複製" not in p
    ]
    if not candidates:
        raise FileNotFoundError(
            f"找不到任何圖片，請用 --img 指定路徑。\n搜尋範圍：{glob_pattern}"
        )
    return candidates[0]


def main():
    args = parse_args()

    # ── 自動選圖 ──────────────────────────────────────────────────────────────
    img_path = args.img or find_first_image(DEFAULT_IMG_GLOB)
    print(f"[INFO] 測試圖片：{img_path}")

    # ── 確保輸出目錄存在 ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── 環境確認 ──────────────────────────────────────────────────────────────
    print(f"[INFO] PyTorch  : {torch.__version__}")
    print(f"[INFO] CUDA 可用 : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")

    # ── 初始化模型 ────────────────────────────────────────────────────────────
    register_all_modules()
    print("[INFO] 載入模型中…（第一次會較慢）")
    model = init_detector(args.config, args.ckpt, device="cuda:0")
    print("[INFO] 模型載入完成")

    # ── 推論 ──────────────────────────────────────────────────────────────────
    print("[INFO] 執行推論…")
    result = inference_detector(model, img_path)
    print("[INFO] 推論完成")

    # ── 計算有信心分數 > threshold 的預測數量 ─────────────────────────────────
    pred_instances = result.pred_instances
    keep = pred_instances.scores >= args.score_thr
    n_det = int(keep.sum())
    print(f"[INFO] 偵測到 {n_det} 個物件（score ≥ {args.score_thr}）")

    # ── 視覺化輸出 ────────────────────────────────────────────────────────────
    visualizer = DetLocalVisualizer(
        vis_backends=[dict(type="LocalVisBackend")],
        save_dir=os.path.dirname(args.out),
    )
    visualizer.dataset_meta = model.dataset_meta

    img_bgr = cv2.imread(img_path)
    visualizer.add_datasample(
        name=os.path.basename(args.out),
        image=img_bgr[:, :, ::-1],   # BGR → RGB
        data_sample=result,
        draw_gt=False,
        pred_score_thr=args.score_thr,
        show=False,
        out_file=args.out,
    )
    print(f"[✓] Smoke Test 成功！輸出圖片：{args.out}")


if __name__ == "__main__":
    main()
