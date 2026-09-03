"""
資料前處理：將原始 EEG CSV 與 Log 檔轉換為 MI 和 Resting 的 .pt 檔案。
支援命令列參數自訂 base_dir、output_dir 與是否按標籤平衡重排 (arrange_by_label)。
"""

import os
import sys
import numpy as np
import datetime
import re
import time
import argparse
import torch

# 引入共用函式庫
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in [current_dir, project_root, os.path.join(project_root, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common_utils import arrange_by_label, CHANNEL_CONFIGS


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

        if not self.segments:
            raise ValueError('No valid segments found')

    def _read_eeg_csv(self, path):
        timestamps_list, eeg_list = [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(11):
                next(f)
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
                    local_timezone_offset = time.localtime().tm_gmtoff
                    tz = datetime.timezone(datetime.timedelta(seconds=local_timezone_offset))
                    dt = datetime.datetime.strptime(m.group(1).strip(), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=tz)
                    return dt.timestamp()
        raise ValueError('找不到 Record datetime')

    def _extract_segments(self, eeg, timestamps, trials):
        for idx in sorted(trials):
            t = trials[idx]
            if 'start' in t and 'end' in t and 'label' in t:
                cut_list = t.get('cut', [])
                is_fail = (len(cut_list) < 5)
                start_rel = t['start'] - self.rec_epoch
                end_rel = t['end'] - self.rec_epoch
                i0 = np.searchsorted(timestamps, start_rel, side='left')
                i1 = np.searchsorted(timestamps, end_rel, side='right') + 1
                if i1 - i0 < 700:
                    i1 = 700 + i0
                if i1 - i0 > 1700:
                    i1 = 2000 + i0

                seg = eeg[i0:i1]
                if seg.size:
                    self.segments.append(seg)
                    self.labels.append(t['label'])
                    self.failures.append(is_fail)

    def get_eeg_trial_channel_sample_np(self, slide_windows=None, slide_windows_stride=20):
        min_len = min(s.shape[0] for s in self.segments)
        self.segments = [s[:min_len] for s in self.segments]
        segment_len = slide_windows if slide_windows is not None else min_len
        stride = slide_windows_stride

        augmented_segments_train = []
        augmented_labels_train = []
        augmented_failures_train = []

        labels_np = np.array(self.labels)
        failures_np = np.array(self.failures)

        for i, s in enumerate(self.segments):
            label = labels_np[i]
            is_fail = failures_np[i]
            if s.shape[0] < segment_len:
                continue

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                window = s[start:start + segment_len]
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)
                augmented_failures_train.append(is_fail)

        x_data = np.transpose(np.stack(augmented_segments_train), (0, 2, 1))  # (trial, channel, sample)
        y_data = np.array(augmented_labels_train)
        failures_data = np.array(augmented_failures_train)
        return x_data, y_data, failures_data

    def load_resting_eyes_segments(self):
        DURATION = 20
        TARGET_SAMPLES = int(DURATION * self.fs)
        pat_eyes = re.compile(r'(Close eyes|Open eyes):\s*([\d\.]+)', re.IGNORECASE)

        new_segments = []
        new_labels = []

        for i in range(len(self.file_paths)):
            log_path = self.log_paths[i]
            csv_path = self.file_paths[i]

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

            def extract_fixed_segment(start_ts, label_val):
                start_rel = start_ts - current_rec_epoch
                if start_rel < timestamps[-1]:
                    i0 = np.searchsorted(timestamps, start_rel, side='left')
                    i1 = i0 + TARGET_SAMPLES
                    if i1 <= len(eeg):
                        seg = eeg[i0:i1]
                        new_segments.append(seg)
                        new_labels.append(label_val)
                        self.labels.append(label_val)

            if 'close_start' in rest_events:
                extract_fixed_segment(rest_events['close_start'], 10)
            if 'open_start' in rest_events:
                extract_fixed_segment(rest_events['open_start'], 11)

        print(f"Resting segments added: {len(new_segments)}")
        if len(new_segments) > 0:
            x_data = np.transpose(np.stack(new_segments), (0, 2, 1))
            y_data = np.array(new_labels)
            return x_data, y_data
        else:
            return np.array([]), np.array([])


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

    def save_as_pt(self, name, do_arrange_by_label=True):
        self.load_data()
        data_name = name
        if do_arrange_by_label:
            self.x_data, self.y_data, self.failures_data = arrange_by_label(self.x_data, self.y_data, self.failures_data)
        os.makedirs(os.path.dirname(os.path.abspath(data_name)), exist_ok=True)
        torch.save({'x_data': self.x_data, 'y_data': self.y_data, 'failures': self.failures_data}, data_name)
        print(f"  └── 成功儲存 MI: {data_name}")

    def save_resting_as_pt(self, name):
        loader = EEGDataLoader(
            file_paths=self.file_paths,
            log_paths=self.log_paths,
            channel_index=self.channel_index
        )
        self.x_data_resting, self.y_data_resting = loader.load_resting_eyes_segments()
        data_name = name
        os.makedirs(os.path.dirname(os.path.abspath(data_name)), exist_ok=True)
        torch.save({'x_data': self.x_data_resting, 'y_data': self.y_data_resting}, data_name)
        print(f"  └── 成功儲存 Resting: {data_name}")


def process_data_dir(base_dir=r"/mnt/project/MIEXP/DATA_Cygnus", output_dir=None,
                     channels="22", mi_name=None, rest_name="rest.pt",
                     do_arrange_by_label=True, overwrite=False):
    """
    遍歷 base_dir 尋找成對的 eeg_record*.csv 與 log*.txt，並轉換儲存為 .pt 檔。
    
    :param base_dir: 原始 CSV 與 Log 所在的目錄
    :param output_dir: 目標輸出目錄 (若為 None 則存於原始同層目錄)
    :param channels: 通道模式 ("13", "22") 或自訂 list
    :param mi_name: 輸出 MI 檔名 (預設依 channels 自動設為 mi.pt 或 mi_22.pt)
    :param rest_name: 輸出 Resting 檔名 (預設 rest.pt)
    :param do_arrange_by_label: 是否按照標籤排列平衡 (預設 True)
    :param overwrite: 是否覆蓋已存在的 .pt 檔案 (預設 False)
    """
    if str(channels) in CHANNEL_CONFIGS:
        channel_index = CHANNEL_CONFIGS[str(channels)]["raw_channel_index"]
        default_mi_name = CHANNEL_CONFIGS[str(channels)]["mi_filename"]
    else:
        channel_index = [int(c.strip()) for c in str(channels).split(',') if c.strip()]
        default_mi_name = "mi.pt"

    if mi_name is None:
        mi_name = default_mi_name

    print("=" * 70)
    print(f"🚀 開始資料轉換處理")
    print(f"  ├── 原始資料目錄 (base_dir): {base_dir}")
    print(f"  ├── 輸出目錄 (output_dir): {output_dir if output_dir else '(原目錄同層)'}")
    print(f"  ├── 通道數量: {len(channel_index)} (模式: {channels})")
    print(f"  ├── arrange_by_label: {do_arrange_by_label}")
    print(f"  └── 輸出檔名: {mi_name}, {rest_name}")
    print("=" * 70)

    if not os.path.exists(base_dir):
        print(f"[錯誤] base_dir 不存在: {base_dir}")
        return

    processed_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(base_dir):
        found_csvs = [f for f in files if f.startswith("eeg_record") and f.endswith(".csv")]
        found_logs = [f for f in files if f.startswith("log") and f.endswith(".txt")]

        if found_csvs and found_logs:
            target_csv = found_csvs[0]
            target_log = found_logs[0]

            full_csv_path = os.path.join(root, target_csv)
            full_log_path = os.path.join(root, target_log)

            if output_dir:
                rel_path = os.path.relpath(root, base_dir)
                target_root = os.path.join(output_dir, rel_path)
            else:
                target_root = root

            save_name_mi = os.path.join(target_root, mi_name)
            save_name_rest = os.path.join(target_root, rest_name)

            mi_exists = os.path.exists(save_name_mi)
            rest_exists = os.path.exists(save_name_rest)

            if mi_exists and rest_exists and not overwrite:
                print(f"處理資料夾: {root}")
                print(f"  ├── {mi_name} 與 {rest_name} 已存在，跳過。\n")
                skipped_count += 1
                continue

            print(f"處理資料夾: {root}")
            print(f"  ├── 載入 EEG: {target_csv}")
            print(f"  ├── 載入 Log: {target_log}")

            try:
                data = EEGSelfDataLoader(
                    file_paths=[full_csv_path],
                    log_paths=[full_log_path],
                    channel_index=channel_index
                )
                data.save_as_pt(save_name_mi, do_arrange_by_label=do_arrange_by_label)
                data.save_resting_as_pt(save_name_rest)
                processed_count += 1
                print()
            except Exception as e:
                print(f"  [錯誤] 轉換失敗: {e}\n")

    print(f"🎉 處理完畢！成功處理: {processed_count} 個資料夾，跳過: {skipped_count} 個資料夾。")


def main():
    parser = argparse.ArgumentParser(description="將原始 EEG CSV 與 Log 轉換為 .pt 格式")
    parser.add_argument("--base_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="CSV 與 LOG 存放位置 (base_dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="指定輸出 .pt 的資料夾 (若不指定則儲存在原 CSV 同層目錄)")
    parser.add_argument("--channels", type=str, default="22",
                        help="通道數量或設定 ('13', '22', 或逗號分隔 channel 索引)")
    parser.add_argument("--mi_name", type=str, default=None,
                        help="輸出 MI 檔名 (預設 22 channel 為 mi_22.pt, 13 channel 為 mi.pt)")
    parser.add_argument("--rest_name", type=str, default="rest.pt",
                        help="輸出 Resting 檔名 (預設 rest.pt)")
    parser.add_argument("--arrange_by_label", action="store_true", default=True,
                        help="是否依照標籤進行 0,1,0,1 平衡排序 (預設: True)")
    parser.add_argument("--no_arrange_by_label", dest="arrange_by_label", action="store_false",
                        help="關閉依照標籤排序")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="若目標檔案已存在是否覆蓋")

    args = parser.parse_args()

    process_data_dir(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        mi_name=args.mi_name,
        rest_name=args.rest_name,
        do_arrange_by_label=args.arrange_by_label,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
