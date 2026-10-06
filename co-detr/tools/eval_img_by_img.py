import os
import json
from mmengine.fileio import load

def main():
    prediction_path = '/workspace/project/results/chiayi_12e/test_preds.pkl'
    gt_path = '/workspace/datasets/coco/chiayi-violation-detection.v2-first-annotation--late-night--dawn-time.coco-mmdetection/test/_annotations.coco.json'
    score_thr = 0.3
    
    print("載入預測結果與 Ground Truth...")
    results = load(prediction_path)
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    # 建立回傳對照表
    # results 是對應到 test set 順序的推論結果（每個元素包含這張圖的所有 prediction）
    # MMDetection test loader 的順序與 coco image ID 可能不同，但通常照順序，保險起見我們抓取 img_id
    
    # 建立 gt 從 image_id 到其 annotations 的查找表
    img_id_to_name = {img['id']: img['file_name'] for img in gt_data['images']}
    img_id_to_anns = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)
        
    # MMeingie預測載入出來的是包含 data_samples 的 list
    
    stats = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    print(f"{'檔名':<70} | {'Ground Truth':<15} | {'模型預測 (>0.3)':<15} | {'結果分類'}")
    print("-" * 125)
    
    for res in results:
        # 在 MMDetection 3.x, result pkl 常見結構包含 pred_instances
        img_path = res['img_path']
        file_name = os.path.basename(img_path)
        
        # 找出對應的 image_id
        img_id = [k for k, v in img_id_to_name.items() if v == file_name]
        img_id = img_id[0] if len(img_id)>0 else None
        
        # GT
        has_gt = len(img_id_to_anns[img_id]) > 0 if img_id else False
        gt_str = "有 litter" if has_gt else "空白無標註"
        
        # Pred
        preds = res['pred_instances']['scores']
        has_pred = (preds.numpy() > score_thr).any() if len(preds) > 0 else False
        pred_str = "有偵測到" if has_pred else "無偵測"
        
        # Type
        if has_gt and has_pred:
            cls = "TP"
        elif not has_gt and not has_pred:
            cls = "TN"
        elif not has_gt and has_pred:
            cls = "FP"
        elif has_gt and not has_pred:
            cls = "FN"
            
        stats[cls] += 1
        print(f"{file_name:<70} | {gt_str:<15} | {pred_str:<17} | {cls}")
        
    print("-" * 125)
    print("=== 最終統計 ===")
    print(f"TP (True Positive - 有垃圾且抓到): {stats['TP']}")
    print(f"TN (True Negative - 沒垃圾且安靜): {stats['TN']}")
    print(f"FP (False Positive - 沒垃圾卻亂抓): {stats['FP']}")
    print(f"FN (False Negative - 有垃圾但漏抓): {stats['FN']}")

if __name__ == '__main__':
    main()
