#!/usr/bin/env python3
"""端到端重跑步驟 0'：確認 YOLO_HEAD 真的決定了推論用哪個頭（見 ../YOLO_baseline_端到端重跑規劃.md）。

用同一個 checkpoint、dynamic166 前 20 張影像推論，檢查：
  1. predictor 裡 AutoBackend 的 end2end 旗標
  2. fuse 後偵測頭保留的是哪一組：cv2/cv3（一對多頭）還是 one2one_cv2/one2one_cv3（一對一頭）
  3. 後處理 nms.non_max_suppression 收到的 end2end 參數與輸入形狀（函式兩種模式都會被呼叫，
     end2end=True 時走 utils/nms.py 的分支：只按分數過濾、取前 max_det 個，不做重疊刪除；
     一對一頭的輸出形狀是 (B, 300, 6)，一對多頭是 (B, 4+類別數, 錨點數)）。
     只計入 predictor.postprocess 發出的呼叫：Ultralytics 在第一次推論前的暖機（stream_inference 的 warmup）
     也會用一個假張量呼叫一次，不帶 end2end 參數，那次不是真的推論，不算。
  4. 每張圖的框數（應 ≤ 300）
並和 YOLO_HEAD 的期望值比對，不符就停止。

用法（容器內，需設 YOLO_HEAD）：python verify_head.py --ckpt <epochN.pt>
"""
import argparse
import json
import os
import traceback

import ultralytics.utils.nms as nms_mod
from ultralytics import YOLO

from yolo_eval_common import HEAD, PRED_KW

ANN = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations/test_dynamic166.json"
IMG = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()

    calls = []
    orig = nms_mod.non_max_suppression

    def spy(prediction, *args, **kw):
        if any(f.name == "postprocess" for f in traceback.extract_stack()):   # 跳過暖機的那一次
            p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            calls.append((bool(kw.get("end2end", False)), tuple(p.shape)))
        return orig(prediction, *args, **kw)

    nms_mod.non_max_suppression = spy          # predict.py 以 nms.non_max_suppression 取用，換模組屬性即可攔到

    ims = sorted(json.load(open(ANN, encoding="utf-8"))["images"], key=lambda x: x["id"])[:20]
    model = YOLO(a.ckpt)
    n_boxes = []
    for im in ims:
        r = model.predict(os.path.join(IMG, im["file_name"]), device=0, **PRED_KW)[0]
        n_boxes.append(len(r.boxes))

    backend = model.predictor.model
    head = backend.model.model[-1]
    kept = [n for n in ("cv2", "cv3", "one2one_cv2", "one2one_cv3") if getattr(head, n, None) is not None]
    res = {
        "YOLO_HEAD": HEAD, "PRED_KW": PRED_KW,
        "autobackend_end2end": bool(backend.end2end),
        "head_end2end": bool(head.end2end),
        "kept_branches": kept,
        "postprocess_end2end_flags": sorted({c[0] for c in calls}), "n_postprocess_calls": len(calls),
        "postprocess_input_shapes": sorted({str(c[1]) for c in calls}),
        "boxes_per_image_max": max(n_boxes), "boxes_per_image_min": min(n_boxes),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))

    want_e2e = HEAD == "e2e"
    assert res["autobackend_end2end"] is want_e2e, "AutoBackend 的 end2end 與 YOLO_HEAD 不符"
    assert res["head_end2end"] is want_e2e, "偵測頭的 end2end 與 YOLO_HEAD 不符"
    assert kept == (["one2one_cv2", "one2one_cv3"] if want_e2e else ["cv2", "cv3"]), f"保留的分支不對：{kept}"
    assert res["n_postprocess_calls"] == len(ims), "後處理呼叫次數應等於影像數"
    assert res["postprocess_end2end_flags"] == [want_e2e], "後處理走的分支不對"
    assert all(c[1][-1] == 6 for c in calls) is want_e2e, "後處理輸入形狀與頭不符"
    assert res["boxes_per_image_max"] <= 300
    print("VERIFY_HEAD_OK", HEAD)


if __name__ == "__main__":
    main()
