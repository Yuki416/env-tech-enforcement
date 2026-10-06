import os
import glob
from mmdet.apis import DetInferencer

config_path = '/workspace/project/configs/co_dino_litter_12e.py'
checkpoint_path = '/workspace/project/results/chiayi_12e/epoch_10.pth' # 使用 epoch_10 因為它的 validation mAP 較高

print("初始化 Inferencer...")
inferencer = DetInferencer(model=config_path, weights=checkpoint_path, device='cuda:0')

img_dir = '/workspace/datasets/coco/chiayi-violation-detection.v2-first-annotation--late-night--dawn-time.coco-mmdetection/test/'
imgs = glob.glob(os.path.join(img_dir, '*.jpg'))

out_dir = '/workspace/project/results/chiayi_12e/inference_results_all'
os.makedirs(out_dir, exist_ok=True)

print(f"開始對 {len(imgs)} 張圖片進行推論...")
import json
results_summary = []

for img in imgs:
    # show=False, no_save_pred=False 會自動將劃好框的圖存到 out_dir/vis
    result = inferencer(img, out_dir=out_dir, show=False, print_result=False)
    # result['predictions'][0] contains 'labels', 'scores', 'bboxes'
    pred = result['predictions'][0]
    
    # 統計高信心的偵測框數量
    high_conf_idx = [i for i, score in enumerate(pred['scores']) if score > 0.3]
    
    results_summary.append({
        'image': os.path.basename(img),
        'num_detections': len(high_conf_idx),
        'max_score': float(max(pred['scores'])) if len(pred['scores']) > 0 else 0.0
    })

print(f"完成！視覺化結果已儲存至 {out_dir}/vis")

# 總結報告
total_images = len(results_summary)
images_with_dets = sum(1 for r in results_summary if r['num_detections'] > 0)
avg_max_score = sum(r['max_score'] for r in results_summary) / total_images if total_images > 0 else 0

print("\n=== 推論結果總結 (Confidence > 0.3) ===")
print(f"總圖片數: {total_images}")
print(f"有偵測出垃圾的圖片數: {images_with_dets} ({images_with_dets/total_images*100:.1f}%)")
print(f"平均最高信心分數: {avg_max_score:.3f}")

with open(f"{out_dir}/summary.json", 'w') as f:
    json.dump(results_summary, f, indent=4)
