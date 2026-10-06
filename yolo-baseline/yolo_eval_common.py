"""YOLO 預測 → metrics_from_preds.py 可讀的 pkl（步驟 4、5 共用）。

metrics_from_preds.py（第 18 次 post-analysis 的正式評估腳本）讀的格式是 tools/test.py --out 的 pkl：
  list of {"img_id": COCO 影像 id,
           "pred_instances": {"bboxes": (N,4) 原圖像素座標 xyxy, "scores": (N,), "labels": (N,) 0..4}}
這裡把 YOLO 的推論結果轉成同樣格式，評估就能完全沿用同一支腳本、一行不改。

推論設定（YOLO_baseline_定案規劃.md 步驟 5）：
  imgsz=1024        對齊第 16 次
  conf=0.001        YOLO 預設只留 0.25 以上的框，AP 會算不完整；留下低分框，門檻由評估腳本自己套
  max_det=300       每張圖最多 300 個框，與 Co-DINO 的 test_cfg（max_per_img=300）一致
  nms               由環境變數 YOLO_HEAD 決定（必填，沒設就停止，不給預設值）：
                      YOLO_HEAD=e2e → nms=False：一對一頭、不做 NMS（端到端，正式版）
                      YOLO_HEAD=nms → 不設 nms：Ultralytics 預設 nms=None＝一對多頭＋標準 NMS（IoU 0.7）（補充對照）
                    ⚠️ 2026-10-02 更正：原本沒設 nms，以為預設就是端到端，實際走的是一對多頭＋NMS。
                    那批結果搬到 baseline_yolo26l_v17/nms/；經過見 ../YOLO_baseline_端到端重跑規劃.md

結果位置（容器內）：
  YB_SHARED = .../baseline_yolo26l_v17          兩版共用：seed<N>/weights、codetr16_valid
  YB        = .../baseline_yolo26l_v17/<YOLO_HEAD>  這一版的結果：seed<N>/{epoch_select,test_eval,...} 等
"""
import json
import os
import pickle

import numpy as np

METRICS_DIR = "/workspace/codetr_exp/18th_v18_own_dynamic_bgs_122vids"   # metrics_from_preds.py 所在（容器內唯讀掛載）

HEAD = os.environ.get("YOLO_HEAD")
if HEAD not in ("e2e", "nms"):
    raise SystemExit(f"環境變數 YOLO_HEAD 必須是 e2e 或 nms（目前是 {HEAD!r}）。不給預設值，避免再吃到 Ultralytics 的預設。")
HEAD_LABEL = {"e2e": "e2e", "nms": "NMS"}[HEAD]           # 影片、報表上標示的版本名稱

YB_SHARED = "/workspace/hdd2_storage/baseline_yolo26l_v17"
YB = f"{YB_SHARED}/{HEAD}"

# nms 版必須和 2026-10-02 以前的設定逐字相同（步驟 0 用它重現舊結果）
PRED_KW = dict(imgsz=1024, conf=0.001, max_det=300, verbose=False)
VAL_KW = {}                                               # 給 Ultralytics 內建評估 model.val() 用
if HEAD == "e2e":
    PRED_KW["nms"] = False
    VAL_KW["nms"] = False


def ckpt_path(seed, epoch):
    """選中的 epoch（1 起算）→ 權重檔路徑（Ultralytics 檔名從 0 起算）。"""
    return f"{YB_SHARED}/seed{seed}/weights/epoch{epoch - 1}.pt"


def predict_to_records(model, ann_json, img_dir, device=0, batch=16):
    """對 ann_json 裡的每張影像推論，回傳 metrics_from_preds 格式的 list。"""
    coco = json.load(open(ann_json, encoding="utf-8"))
    images = coco["images"]
    paths = [os.path.join(img_dir, im["file_name"]) for im in images]
    for p in paths:
        assert os.path.isfile(p), f"找不到影像：{p}"
    records = []
    for i in range(0, len(paths), batch):
        chunk = images[i:i + batch]
        results = model.predict([os.path.join(img_dir, im["file_name"]) for im in chunk],
                                device=device, batch=batch, **PRED_KW)
        for im, r in zip(chunk, results):
            # 防呆：結果的原圖尺寸必須等於 COCO 記錄的寬高，否則座標尺度一定錯
            assert tuple(r.orig_shape) == (im["height"], im["width"]), \
                f"{im['file_name']} 原圖尺寸 {r.orig_shape} ≠ COCO {(im['height'], im['width'])}"
            b = r.boxes
            records.append({
                "img_id": im["id"],
                "pred_instances": {
                    "bboxes": b.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4),   # 已是原圖像素座標
                    "scores": b.conf.cpu().numpy().astype(np.float32).reshape(-1),
                    "labels": b.cls.cpu().numpy().astype(np.int64).reshape(-1),
                },
            })
    assert len(records) == len(images)
    return records


def save_pkl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_metrics_module():
    """import metrics_from_preds.py（不修改原檔）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("metrics_from_preds", os.path.join(METRICS_DIR, "metrics_from_preds.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
