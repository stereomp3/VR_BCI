"""
把電腦上 20260302.py 搬過來，然後改位置，和使用 22 channel
"""

import os
import numpy as np
import datetime, re
import time
import torch


class EEGSelfDataLoader:
    def __init__(self, file_paths, log_paths, channel_index):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.x_data = None
        self.y_data = None
        self.x_data_resting = None
        self.y_data_resting = None
        self.failures_data = None

    def load_data(self):
        loader = EEGDataLoader(
            file_paths=self.file_paths,
            log_paths=self.log_paths,
            channel_index=self.channel_index
        )
        loader.load_and_preprocess_data()
        self.x_data, self.y_data, self.failures_data = loader.get_eeg_trial_channel_sample_np()

    def get_data(self):
        return self.x_data, self.y_data, self.failures_data

    def save_as_pt(self, name):
        self.load_data()
        data_name = name
        self.x_data, self.y_data, self.failures_data = arrange_by_label(self.x_data, self.y_data, self.failures_data)
        torch.save({'x_data': self.x_data, 'y_data': self.y_data, 'failures': self.failures_data}, data_name)
        # print(f"'x_data': {self.x_data}, 'y_data': {self.y_data}, 'failures': {self.failures_data}")
        print(f"mi data save as {data_name}")

    def save_resting_as_pt(self, name):
        loader = EEGDataLoader(
            file_paths=self.file_paths,
            log_paths=self.log_paths,
            channel_index=self.channel_index
        )
        self.x_data_resting, self.y_data_resting = loader.load_resting_eyes_segments()
        data_name = name
        torch.save({'x_data': self.x_data_resting, 'y_data': self.y_data_resting}, data_name)
        print(f"resting data save as {data_name}")


def arrange_by_label(x, y, f):
    """
    根據 x 和 y，把資料根據 y label 按照 0 1 0 1 排列，讓資料平衡 (train 的)
    如果資料本身不平衡，那就優先排列後面 (需要 train 的資料)，Val 給剩下的
    :param x: 輸入 data (trial, channel, sample)
    :param y: 輸入 label (0 or 1)
    :return: sorted_x, sorted_y 經過 y 排序 0,1,0,1 的 list
    """
    label_0_idx = np.where(y == 0)[0]
    label_1_idx = np.where(y == 1)[0]
    # 反轉索引，從後面開始交錯
    label_0_rev = label_0_idx[::-1]
    label_1_rev = label_1_idx[::-1]

    # 取能交錯的數量
    num_pairs = min(len(label_0_rev), len(label_1_rev))

    # 取出可交錯的 index
    paired_0 = label_0_rev[:num_pairs]
    paired_1 = label_1_rev[:num_pairs]

    # 從後面交錯 → 所以前面交錯順序要從最後一組開始
    # 所以還原順序
    paired_0 = paired_0[::-1]
    paired_1 = paired_1[::-1]

    # 交錯排列：0,1,0,1,...
    interleaved_idx = np.empty(num_pairs * 2, dtype=int)
    interleaved_idx[0::2] = paired_0
    interleaved_idx[1::2] = paired_1

    # 剩下的 index(不能配對的)
    remaining_0 = label_0_rev[num_pairs:][::-1]
    remaining_1 = label_1_rev[num_pairs:][::-1]
    remaining_idx = np.concatenate((remaining_0, remaining_1))

    # 最終順序：剩下的在前面
    final_idx = np.concatenate((remaining_idx, interleaved_idx))

    # 排列 x 和 y
    x_sorted = x[final_idx]
    y_sorted = y[final_idx]
    f_sorted = f[final_idx]
    return x_sorted, y_sorted, f_sorted


class EEGDataLoader:
    def __init__(self, file_paths, log_paths, channel_index, fs=500):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.fs = fs

        self.segments = []
        self.labels = []
        self.failures = []
        self.rec_epoch = None

    def load_and_preprocess_data(self):
        for i in range(len(self.file_paths)):
            timestamps, eeg = self._read_eeg_csv(self.file_paths[i])
            trials = self._parse_log_file(self.log_paths[i])
            self.rec_epoch = self._extract_record_epoch(self.file_paths[i])
            self._extract_segments(eeg, timestamps, trials)
            # print(f"self.segments {len(self.segments)}")

        if not self.segments:
            raise ValueError('No valid segments found')

    def _read_eeg_csv(self, path):
        timestamps_list, eeg_list = [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(11): next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= max(self.channel_index) + 1:
                    timestamps_list.append(float(parts[0]))
                    eeg_list.append([float(parts[x]) for x in self.channel_index])
        return np.array(timestamps_list), np.array(eeg_list)

    def _parse_log_file(self, path):
        trials = {}
        pat = re.compile(r'Trial\s+(\d+)\s+(START|CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.match(line.strip())
                if m:
                    idx = int(m.group(1))
                    typ = m.group(2).lower()
                    ts = float(m.group(3))
                    label = int(m.group(4)) if m.group(4) is not None else None

                    trial_entry = trials.setdefault(idx, {})
                    if typ == 'cut':
                        trial_entry.setdefault('cut', []).append(ts)
                    else:
                        trial_entry[typ] = ts
                    if label is not None:
                        trial_entry['label'] = label
        return trials

    def _extract_record_epoch(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(10):
                line = f.readline()
                if 'Record datetime' in line:
                    m = re.search(r'Record datetime:\s*([0-9\- :\.]+)', line)
                    # tz = datetime.timezone(datetime.timedelta(hours=8))
                    # 獲取當前本地時區的偏移量（秒）
                    local_timezone_offset = time.localtime().tm_gmtoff
                    # 計算本地時區偏移量（以小時為單位）
                    tz = datetime.timezone(datetime.timedelta(seconds=local_timezone_offset))

                    dt = datetime.datetime.strptime(m.group(1).strip(), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz)
                    return dt.timestamp()
        raise ValueError('找不到 Record datetime')

    def _extract_segments(self, eeg, timestamps, trials):
        # 在提取 Segment 時判斷並儲存 failure 狀態
        for idx in sorted(trials):
            t = trials[idx]
            if 'start' in t and 'end' in t and 'label' in t:
                # --- 判斷是否失敗 ---
                cut_list = t.get('cut', [])
                is_fail = (len(cut_list) < 5)  # 如果小於 5 次，標記為 True (失敗)
                # ------------------
                start_rel = t['start'] - self.rec_epoch
                end_rel = t['end'] - self.rec_epoch
                i0 = np.searchsorted(timestamps, start_rel, side='left')
                i1 = np.searchsorted(timestamps, end_rel, side='right') + 1
                # print(f"i0: {i0}, i1: {i1}")
                if i1 - i0 < 700:  # 加上限制，可拿掉
                    i1 = 700 + i0
                if i1 - i0 > 1700:  # 加上限制，可拿掉，這邊是因為後面有 4s 的
                    i1 = 2000 + i0

                seg = eeg[i0:i1]  # 從 start 開始往後取 4 秒
                # seg = eeg[max(0, i1 - 2000):i1]  # 從 end 開始往前取 4 秒
                # print(f"en(seg): {len(seg)}")

                if seg.size:
                    # b, a = butter(4, [1 / (0.5 * self.fs), 20 / (0.5 * self.fs)], btype='band')
                    # seg = filtfilt(b, a, seg, axis=0)
                    self.segments.append(seg)
                    self.labels.append(t['label'])
                    # 同步加入 failure 狀態
                    self.failures.append(is_fail)

    def get_eeg_trial_channel_sample_np(self, slide_windows=None,
                                        slide_windows_stride=20):  # get data x (trail, channel, sample)
        """
        這個 function 基本上就是直接讀取 trial 然後存檔，沒有做額外的處裡，slide window 也沒做，slide windows 基本可以忽略
        :param slide_windows:
        :param slide_windows_stride:
        :return:
        """
        min_len = min(s.shape[0] for s in self.segments)
        # print(f"min sample: {min_len}")
        self.segments = [s[:min_len] for s in self.segments]  # 取最小 size 當作 sample size 可以註解
        if slide_windows is not None:
            segment_len = slide_windows  # 每個 sample 長度
        else:
            segment_len = min_len  # 選擇直接讀取一個，不切
        stride = slide_windows_stride  # overlap 大小

        augmented_segments_train = []
        augmented_labels_train = []
        augmented_failures_train = []  # 新增 list

        labels_np = np.array(self.labels)
        failures_np = np.array(self.failures)  # 轉成 numpy 方便索引

        for i, s in enumerate(self.segments):
            label = labels_np[i]
            is_fail = failures_np[i]  # 取得該 trial 的 failure 狀態
            if s.shape[0] < segment_len:
                continue  # 忽略太短的資料段

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                # bandpass(s[start:start + segment_len])  # shape: (500, n_channels)
                window = s[start:start + segment_len]
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)
                augmented_failures_train.append(is_fail)  # 同步擴增
        x_data = np.transpose(np.stack(augmented_segments_train), (0, 2, 1))  # (trail, channel, sample)
        y_data = np.array(augmented_labels_train)  # (trail,)
        failures_data = np.array(augmented_failures_train)  # 轉成 numpy array (bool)
        # return self.remove_mean(x_data), y_data # 20250901 註解，之前訓練的都有 remove mean，但是 demean 效果比較好，之後可以考慮看看如何使用
        return x_data, y_data, failures_data  # 回傳原始檔案

    def load_resting_eyes_segments(self):
        """
        [20260207 新增功能] 額外讀取 Close eyes (Label 10) 和 Open eyes (Label 11) 的區段。
        固定擷取 20 秒長度。

        Returns:
            x_data: numpy array (trial, channel, time)
            y_data: numpy array (trial,)
        """
        # 設定固定參數
        DURATION = 20
        TARGET_SAMPLES = int(DURATION * self.fs)

        pat_eyes = re.compile(r'(Close eyes|Open eyes):\s*([\d\.]+)', re.IGNORECASE)

        # 暫存這次讀取到的 segments，用來製作回傳的 numpy array
        new_segments = []
        new_labels = []

        for i in range(len(self.file_paths)):
            log_path = self.log_paths[i]
            csv_path = self.file_paths[i]

            # 1. 解析 Log
            rest_events = {}
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = pat_eyes.match(line.strip())
                    if m:
                        action = m.group(1).lower()
                        ts = float(m.group(2))
                        if 'close' in action:
                            rest_events['close_start'] = ts
                        elif 'open' in action:
                            rest_events['open_start'] = ts

            if not rest_events:
                continue

            timestamps, eeg = self._read_eeg_csv(csv_path)
            current_rec_epoch = self._extract_record_epoch(csv_path)

            # 定義 helper 函數來統一提取邏輯
            def extract_fixed_segment(start_ts, label_val):
                start_rel = start_ts - current_rec_epoch

                # 檢查起始點是否在錄製範圍內
                if start_rel < timestamps[-1]:
                    i0 = np.searchsorted(timestamps, start_rel, side='left')
                    i1 = i0 + TARGET_SAMPLES  # 強制取固定長度

                    # 檢查是否超出邊界
                    if i1 <= len(eeg):
                        seg = eeg[i0:i1]
                        # 加入暫存列表
                        new_segments.append(seg)
                        new_labels.append(label_val)
                        self.labels.append(label_val)

            # A. 處理 Close Eyes (Label 10)
            if 'close_start' in rest_events:
                extract_fixed_segment(rest_events['close_start'], 10)

            # B. 處理 Open Eyes (Label 11)
            if 'open_start' in rest_events:
                extract_fixed_segment(rest_events['open_start'], 11)

        print(f"Resting segments added: {len(new_segments)}")

        # 轉換成 Numpy Array 回傳
        if len(new_segments) > 0:
            # new_segments 是一個 list of (Samples, Channels)
            # np.stack 之後變成 (Trials, Samples, Channels)
            # np.transpose(0, 2, 1) 轉成 (Trials, Channels, Samples)
            x_data = np.transpose(np.stack(new_segments), (0, 2, 1))
            y_data = np.array(new_labels)
            return x_data, y_data
        else:
            # 如果沒抓到資料，回傳空陣列
            return np.array([]), np.array([])

    def remove_mean(self, x_data):
        # 移除每段的 channel 均值（去 DC）
        if x_data is not None:
            return x_data - np.mean(x_data, axis=2, keepdims=True)
        else:
            return x_data


# 設定參數
# channel_index = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 28] # 13 channel
# 22 channel, all
channel_index = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33]
base_dir = r"/mnt/middle/tmp/2025/MIEXP"

print(f"開始遍歷並處理資料夾: {base_dir}...\n")

# 使用 os.walk 遍歷 base_dir 底下的所有層級目錄
for root, dirs, files in os.walk(base_dir):
    # ---------------------------------------------------------
    # 步驟 1: 在當前資料夾 (root) 中尋找目標檔案
    # ---------------------------------------------------------
    # 篩選出開頭是 "eeg_record" 結尾是 ".csv" 的檔案
    found_csvs = [f for f in files if f.startswith("eeg_record") and f.endswith(".csv")]

    # 篩選出開頭是 "log" 結尾是 ".txt" 的檔案
    found_logs = [f for f in files if f.startswith("log") and f.endswith(".txt")]

    # ---------------------------------------------------------
    # 步驟 2: 檢查是否找到成對的資料
    # ---------------------------------------------------------
    # 只有當 CSV 和 Log 都存在時才執行處理
    if found_csvs and found_logs:
        # 假設每個 run 資料夾下只有一組數據，取第一個找到的
        target_csv = found_csvs[0]
        target_log = found_logs[0]

        # ---------------------------------------------------------
        # 步驟 3: 組合絕對路徑
        # ---------------------------------------------------------
        # 這是給 file_paths 和 log_paths 用的完整路徑
        full_csv_path = os.path.join(root, target_csv)
        full_log_path = os.path.join(root, target_log)

        mi_name = "mi_22.pt"  # mi_e4 # mi.pt
        rest_name = "rest.pt"
        # 設定輸出的 save_name (在同一資料夾下存成 .pt)
        save_name_mi = os.path.join(root, mi_name)
        save_name_rest = os.path.join(root, rest_name)

        # 檢查目標檔案是否已經存在
        mi_exists = os.path.exists(save_name_mi)
        rest_exists = os.path.exists(save_name_rest)

        # 【優化】如果兩個檔案都已經存在，直接跳過，節省載入資料的時間
        if mi_exists and rest_exists:
            print(f"處理資料夾: {root}")
            print(f"  ├── {mi_name} 與 {rest_name} 皆已存在，跳過處理。\n")
            continue

        print(f"處理資料夾: {root}")
        print(f"  ├── 載入 EEG: {target_csv}")
        print(f"  ├── 載入 Log: {target_log}")

        try:
            # ---------------------------------------------------------
            # 步驟 4: 執行資料載入與轉換
            # ---------------------------------------------------------
            data = EEGSelfDataLoader(
                file_paths=[full_csv_path],  # 轉成 list 傳入
                log_paths=[full_log_path],  # 轉成 list 傳入
                channel_index=channel_index
            )

            data.save_as_pt(save_name_mi)
            print(f"  └── 成功儲存: {mi_name}\n")
            data.save_resting_as_pt(save_name_rest)
            print(f"  └── 成功儲存: {rest_name}\n")

        except Exception as e:
            print(f"  [錯誤] 轉換失敗: {e}\n")

print("所有資料夾處理完成。")
