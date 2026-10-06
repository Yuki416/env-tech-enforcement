import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mmengine import Config
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS

# 把分析工具路徑加進來以載入內部函式
sys.path.append('/opt/mmdetection/tools/analysis_tools')
from confusion_matrix import calculate_confusion_matrix

def main():
    config_path = '/workspace/project/configs/evaluate_test_set.py'
    prediction_path = '/workspace/project/results/chiayi_12e/test_preds.pkl'
    out_img = '/workspace/project/results/chiayi_12e/beautiful_confusion_matrix.png'

    print("Loading config and predictions...")
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    results = load(prediction_path)

    print("Building dataset...")
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    print("Calculating confusion matrix...")
    # class_names: "litter", + background
    confusion_matrix = calculate_confusion_matrix(dataset, results, score_thr=0.3)

    # plot with seaborn
    print("Plotting...")
    classes = dataset.metainfo['classes'] + ('background',)
    
    # 進行資料正規化處理 (以真實標籤總數為分母)
    # 如果 row sum = 0, 避免除以 0
    row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    norm_cm = np.divide(confusion_matrix, row_sums, out=np.zeros_like(confusion_matrix, dtype=float), where=row_sums!=0)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # 客製化調色盤，使用藍色漸層
    cmap = sns.color_palette("Blues", as_cmap=True)

    # 繪製 Heatmap
    ax = sns.heatmap(norm_cm, annot=True, fmt='.2%', cmap=cmap,
                     xticklabels=classes, yticklabels=classes,
                     cbar_kws={'label': 'Normalized Rate'},
                     annot_kws={"size": 16, "weight": "bold"})
                     
    # 在上方加一個原始數字的 annotation
    for t, count in zip(ax.texts, confusion_matrix.flatten()):
        text = t.get_text()
        t.set_text(f"{count}\n\n{int(float(text.strip('%'))*row_sums.flatten()[0]/100) if False else ''}({text})") 
        # 修改：為了清楚，我們直接把原始數值和百分比放一起
        
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = confusion_matrix[i, j]
            pct = norm_cm[i, j] * 100
            text = f"{int(val)}\n({pct:.1f}%)"
            ax.texts[i*len(classes)+j].set_text(text)

    # 設定標題和標籤
    plt.title('Detection Confusion Matrix (Confidence > 0.3)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Label (Ground Truth)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # 讓字體更好看
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Beautiful confusion matrix saved to: {out_img}")

if __name__ == '__main__':
    main()
