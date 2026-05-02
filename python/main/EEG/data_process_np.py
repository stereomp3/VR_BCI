"""
20251015 從之前的 data_process_np 改過來，把多餘內容刪除
"""
import numpy as np
import datetime, re
import time
import main.Utils.config as config


# trials, channel, sample # modify from train_withSCCNet_2
class EEGDataLoader:
    def __init__(self, file_paths, log_paths, channel_index, fs=500):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.fs = fs

        self.segments = []
        self.labels = []
        self.failures = []  # 20260209 新增，用於 CUT 判斷這個 trial 是否成功
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
                is_fail = (len(cut_list) < config.group_note_num)  # 如果小於 5 次，標記為 True (失敗)
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
                seg = eeg[i0:i1]
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

    def remove_mean(self, x_data):
        # 移除每段的 channel 均值（去 DC）
        if x_data is not None:
            return x_data - np.mean(x_data, axis=2, keepdims=True)
        else:
            return x_data


def main():
    file_paths = ['../real_time_data/eeg_record_20260209_230657.csv']  # 多個 file 對應多個 log
    log_paths = ['../real_time_data/log_20260209_230657.txt']
    # Fp1,Fp2,AF3,AF4,F7(6),F3,Fz,F4,F8,FT7(11),FC3,FCz,FC4,FT8,T7(16),C3,Cz,C4,T8,TP7(21),CP3,CPz,CP4,TP8,P7(26),P3,Pz,P4,P8,O1(31),Oz,O2(33)
    # channel_index = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33]
    channel_index = [17, 18, 19]  # 0.
    # channel_index = [7, 9, 17, 18, 19, 28] # 0.64

    data_loader = EEGDataLoader(
        file_paths=file_paths,
        log_paths=log_paths,
        # channel_index=channel_index,
        channel_index=config.channel_index,
    )
    data_loader.load_and_preprocess_data()
    x_np, y_np, f = data_loader.get_eeg_trial_channel_sample_np()
    print(f"x_np {x_np.shape}")  # (T, C, S)
    print(f"x_np {y_np.shape}")


if __name__ == '__main__':
    main()
