import re

# 1. 初始化三個空的 list 用來儲存資料
prediction_latency = []
tcp_latency = []
online_adaption = []

# 2. 定義正規表達式（Regular Expressions）
# [-+]?[\d.]+ 可以精準比對包含正負號的整數或浮點數
inf_pattern = re.compile(r'\[INFO\] \[Inference\] Count=\d+, Avg=([-+]?[\d.]+) ms')
tcp_pattern = re.compile(r'\[INFO\] \[TCP Latency\] ([-+]?[\d.]+) ms')
adapt_pattern = re.compile(r'\[INFO\] \[Online Adaptation\] Inference=([-+]?[\d.]+) ms')

log_file_path = 'log_20260629_011936_atc_8.txt'

try:
    # 3. 逐行讀取 log 檔案
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:

            # 檢查是否符合第一種形式 (Inference Avg)
            match_inf = inf_pattern.search(line)
            if match_inf:
                # 擷取第一個括號 group(1) 內的字串並轉成 float
                prediction_latency.append(float(match_inf.group(1)))
                continue

            # 檢查是否符合第二種形式 (TCP Latency)
            match_tcp = tcp_pattern.search(line)
            if match_tcp:
                # 擷取數值後，使用 abs() 取絕對值
                tcp_val = float(match_tcp.group(1))
                tcp_latency.append(abs(tcp_val))
                continue

            # 檢查是否符合第三種形式 (Online Adaptation Inference)
            match_adapt = adapt_pattern.search(line)
            if match_adapt:
                online_adaption.append(float(match_adapt.group(1)))
                continue

    # 4. 轉換完成後，使用 list 的方式印出結果
    print(f"prediction_latency = {prediction_latency}")
    # print(f"tcp_latency = {tcp_latency}")
    print(f"online_adaption = {online_adaption}")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{log_file_path}'，請確認檔案路徑是否正確。")