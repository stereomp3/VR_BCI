"""
根據之前程式碼已經跑完的數據，跑一張整理圖片，用於放在 PPT 上面，表格已經寫在 overleaf 上面ㄌ
下面是把其他地方的數值拿過來，然後繪製，一次開一個區塊
與 0518-2 基本一樣，不過 -2 他有 SD 的部分
"""

import matplotlib.pyplot as plt
import numpy as np

# 1. 準備數據
# metrics = ['Accuracy', 'WSI', 'MBSR', 'MSFI']
metrics = ['Accuracy', 'MBSR (Mu/Beta Spectral Learning)', 'MSFI (Sensorimotor Spatial Learning)']

# LE 和 AE 的效果數值 (Effect %)
# le_values = [4.87, 0.58, 8.86, 3.95]
le_values = [4.87, 8.86, 3.95]
# ae_values = [3.56, -1.74, 2.20, 1.32]
ae_values = [3.56, 2.20, 1.32]

############### T test area ######################################
# # 根據 Table 1 設定顯著性標註
# # p < 0.05 為 *, p < 0.01 為 **, 其他為 ns
# # le_significance = ['*', 'ns', '**', '*'] # p = [0.019, 0.743, 0.006, 0.020]
# le_significance = ['*', '**', '*']  # p = [0.019, 0.006, 0.020]
# # ae_significance = ['*', 'ns', 'ns', 'ns'] # # p = [0.049, 0.254, 0.529, 0.462] # 之前 * 是之前的算法算錯為 0.049
# ae_significance = ['ns', 'ns', 'ns']  # p = [0.096, 0.006, 0.020] # 後來

############### Wilcoxon rank test ######################################
le_significance = ['*', '*', '*']  # p = [0.0457, 0.0115, 0.0138]
ae_significance = ['ns', 'ns', 'ns']  # p = [0.1974, 0.4732, 0.9441]


# 2. 設定圖表參數
x = np.arange(len(metrics))  # X 軸的標籤位置
width = 0.35  # 長條圖的寬度

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# 3. 繪製兩組長條圖
rects1 = ax.bar(x - width / 2, le_values, width, label='LE', color='#4C72B0', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width / 2, ae_values, width, label='AE', color='#DD8452', edgecolor='black', alpha=0.8)

# 4. 加入標籤、標題與圖例
ax.set_ylabel('Effect (%)', fontsize=12)
ax.set_title('Effects Across Metrics (LE vs AE)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.axhline(0, color='black', linewidth=1)  # 加入 Y=0 的基準線
ax.legend(fontsize=12)


# 5. 定義函式：在長條圖上添加顯著性標註 (**, *, ns)
def add_significance_labels(rects, significances):
    for rect, sig in zip(rects, significances):
        height = rect.get_height()
        # 判斷數值是正還是負，來決定文字標註在長條上方還是下方
        y_offset = 3 if height >= 0 else -5
        va = 'bottom' if height >= 0 else 'top'

        ax.annotate(sig,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, y_offset),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=11, fontweight='bold', color='black')


# 執行標註
add_significance_labels(rects1, le_significance)
add_significance_labels(rects2, ae_significance)

# 7. 顯示並調整版面
fig.tight_layout()
plt.show()
