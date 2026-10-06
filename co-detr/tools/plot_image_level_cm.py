import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # 資料來自我們先前逐張圖片分析的結果
    # TP (True Positive - 有垃圾且抓到): 3
    # TN (True Negative - 沒垃圾且安靜): 51
    # FP (False Positive - 沒垃圾卻亂抓): 2
    # FN (False Negative - 有垃圾但漏抓): 3
    
    tp = 3
    tn = 51
    fp = 2
    fn = 3

    # 建構 2x2 混淆矩陣
    # 格式約定: 
    # [ [TN, FP],
    #   [FN, TP] ]
    confusion_matrix = np.array([[tn, fp], 
                                 [fn, tp]])
    
    classes = ['Background (No Litter)', 'Litter']
    
    # 計算正規化 (Normalized) 矩陣，用於著色或顯示百分比
    # 確保不會除以 0
    row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    norm_cm = np.divide(confusion_matrix, row_sums, out=np.zeros_like(confusion_matrix, dtype=float), where=row_sums!=0)

    # 開始畫圖
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    
    # 使用藍色漸層調色盤
    cmap = sns.color_palette("Blues", as_cmap=True)

    # 繪製 heatmap
    ax = sns.heatmap(norm_cm, annot=False, cmap=cmap,
                     xticklabels=classes, yticklabels=classes,
                     cbar_kws={'label': 'Recall Rate'},
                     vmin=0.0, vmax=1.0)
    
    # 手動加上客製化的文字：數量與百分比
    for i in range(2):
        for j in range(2):
            count = confusion_matrix[i, j]
            pct = norm_cm[i, j] * 100
            
            # 設定不同區塊的代表意義標籤
            label_name = ""
            if i == 0 and j == 0:
                label_name = "TN"
            elif i == 0 and j == 1:
                label_name = "FP"
            elif i == 1 and j == 0:
                label_name = "FN"
            elif i == 1 and j == 1:
                label_name = "TP"
                
            text = f"{label_name}\n\n{count} imgs\n({pct:.1f}%)"
            
            # 根據底色決定文字顏色 (深色底用白字，淺色底用黑字)
            text_color = "white" if norm_cm[i, j] > 0.5 else "black"
            
            ax.text(j + 0.5, i + 0.5, text, 
                    fontsize=14, fontweight='bold',
                    ha='center', va='center', color=text_color)

    # 加上標題及軸標籤
    plt.title('Image-Level Confusion Matrix (Confidence > 0.3)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=14, fontweight='bold')
    plt.xlabel('Model Prediction', fontsize=14, fontweight='bold')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    # 儲存圖片
    out_img = '/workspace/project/results/chiayi_12e/image_level_confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Image-level confusion matrix saved to: {out_img}")

if __name__ == '__main__':
    main()
