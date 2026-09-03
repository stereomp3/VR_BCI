"""
在 server 上面 run online (20260702)，然後拿取 log，提取看哪個參數條件比較好
測試，條件如下
* 把模型 online fine tune 上面設定，用 IRB bci data 測試 fine tune 的參數:
    * buffer size (10, 20, 40)
    * update 頻率 (4, 8)
    * epoch (4, 8, 16, 32)
    * batch (8, 16, 32)
    * lr: 10^{-4}
測試加入與不加入 validation
"""
import re
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns


def parse_log_file(file_path):
    # 讀取整個 txt 檔案
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 用來存放解析後的全部數據
    all_data = []

    # 1. 拆分不同參數的區塊 (以 "========== 模擬完成" 開頭)
    blocks = re.split(r'========== 模擬完成', content)

    # 2. 定義正規表達式來提取參數
    param_pattern = re.compile(
        r'update_freq:\s*(\d+),\s*num_epoch:\s*(\d+),\s*batch_size:\s*(\d+),\s*use_val:\s*(True|False)==========')

    for block in blocks:
        if not block.strip():
            continue

        # 尋找參數
        param_match = param_pattern.search(block)
        if param_match:
            update_freq = int(param_match.group(1))
            num_epoch = int(param_match.group(2))
            batch_size = int(param_match.group(3))
            use_val = param_match.group(4) == 'True'

            # 解析表格資料
            lines = block.split('\n')
            for line in lines:
                # 判斷是否為表格內容行 (特徵為包含 '|' 且第一欄通常為數字與括號)
                if '|' in line and 'id' not in line and '---' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 3:
                        subj_id = parts[0]
                        acc_str = parts[1]
                        condition = parts[2]

                        try:
                            # 將字串 "[0.5, 0.6, ...]" 轉換為 Python list
                            acc_list = ast.literal_eval(acc_str)
                            mean_acc = sum(acc_list) / len(acc_list) if acc_list else 0
                            max_acc = max(acc_list) if acc_list else 0

                            all_data.append({
                                'update_freq': update_freq,
                                'num_epoch': num_epoch,
                                'batch_size': batch_size,
                                'use_val': use_val,
                                'id': subj_id,
                                'condition': condition,
                                'mean_acc': mean_acc,
                                'max_acc': max_acc,
                                'raw_acc': acc_list
                            })
                        except Exception as e:
                            # 略過無法解析的行
                            pass

    return pd.DataFrame(all_data)


# ====== 執行解析 ======
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="解析 online_simulation 日誌尋找最佳參數")
parser.add_argument("--log_file", type=str, default="online_simulation_24_20260702_lr3.txt", help="Log 檔案路徑")
parser.add_argument("--output_csv", type=str, default="parsed_results.csv", help="輸出結果 CSV 路徑")
parser.add_argument("--output_plot", type=str, default="accuracy_comparison.png", help="輸出盒鬚圖路徑")
args, _ = parser.parse_known_args()

if not os.path.exists(args.log_file):
    print(f"⚠️ [提示] 找不到指定的 Log 檔案: '{args.log_file}'")
    print(f"👉 請使用 --log_file <檔案路徑> 指定欲分析的 online simulation log 檔案。")
    sys.exit(0)

df = parse_log_file(args.log_file)

# 預覽萃取出的資料表
print("萃取後的資料前 5 筆：")
print(df.head())

# 將結果儲存為 CSV 以便在 Excel 中查看
df.to_csv(args.output_csv, index=False)
print(f"\n已將完整表格儲存為 '{args.output_csv}'")


# ====== 建立分組標籤 ======
# 為了比較不同參數，我們將多個參數合併為一個標籤
df['params_group'] = "uf:" + df['update_freq'].astype(str) + \
                     "_ep:" + df['num_epoch'].astype(str) + \
                     "_bs:" + df['batch_size'].astype(str) + \
                     "_val:" + df['use_val'].astype(str)


# ====== 新增：找出效果最好的參數組合 ======
# 以參數群組進行 groupby，計算該參數下所有資料的 mean_acc 的平均值
group_performance = df.groupby(['update_freq', 'num_epoch', 'batch_size', 'use_val', 'params_group'])['mean_acc'].mean().reset_index()

# 依照平均準確度由高到低排序
group_performance = group_performance.sort_values(by='mean_acc', ascending=False)

print("\n=== 各參數組合的整體平均準確度 (由高到低) ===")
print(group_performance.to_string(index=False))

# 取得排名第一的最佳參數
best_combo = group_performance.iloc[0]
print("\n🏆 === 最佳參數組合 ===")
print(f"參數群組標籤 : {best_combo['params_group']}")
print(f"Update Freq  : {best_combo['update_freq']}")
print(f"Num Epoch    : {best_combo['num_epoch']}")
print(f"Batch Size   : {best_combo['batch_size']}")
print(f"Use Val      : {best_combo['use_val']}")
print(f"最高平均準確度: {best_combo['mean_acc']:.4f}")
print("========================\n")


# ====== 視覺化：觀察各參數對平均準確度的影響 ======
plt.figure(figsize=(12, 6))
# 使用 Boxplot (盒鬚圖) 來觀察每個參數組合下，所有 id 的準確度分佈差異
# 將 x 軸順序設定為依照整體平均表現排序 (由好到壞)，讓圖表更易讀
order = group_performance['params_group'].tolist()

sns.boxplot(data=df, x='params_group', y='mean_acc', order=order)
sns.stripplot(data=df, x='params_group', y='mean_acc', color='black', alpha=0.5, jitter=True, order=order)

plt.title('Impact of Hyperparameters on Mean Accuracy (Sorted by Overall Performance)')
plt.xlabel('Parameter Combinations (UpdateFreq_Epoch_BatchSize_UseVal)')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

# 儲存圖表
plt.savefig('accuracy_comparison.png')
# 若在無 GUI 環境執行可將 plt.show() 註解掉
# plt.show()