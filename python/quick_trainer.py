"""
從 20251025 複製和 test/cross_validation 複製
20251104:
    測試使用不同 channel 的準確度 (看看可以只使用多少 channel)，然後進行 cross validation 驗證
    測試不同 MI 時間區端 (使用 1s 然後滑過去)
    測試使用不同 subject 前面還不太會玩的 data，與後面比較會玩的 data 進行訓練，紀錄在 20251025 裡面，
    從這裡開始，加入讀取 存成 np 的 pt 檔案，節省讀取時間
    寫一個去頭去尾的 function，用來改 MI 時間 (crop_center)
    加入 import torch.backends.cudnn as cudnn，讓實驗可以完全重現不會因為 CNN 導致結果不一樣，不過會變慢一咪咪
20251106: 測試每個 subject 每個 run，然後使用 3 fold cross validation 進行測試
20251204:
    修改 SCCNet 把 forward 那邊可以讀取三維的內容，所以 SCCNet 也可以跑在 BraindecodeTrainer
    然後改使用 XBrainLab 的 SCCNet，對應的 load param 需要根據輸入的 fs 進行更改，這個和其他 load 不太一樣
20251218:
    可以寫入多個 .pt (data)，載入多點資料，加入 cat_all_data function
"""

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from scipy.signal import butter, filtfilt, decimate  # decimate 用來 down sample，這部分如果要寫論文可以寫使用的數學，非整數降維使用 resample
import torch
import torch.nn.functional as F
import sys
from functools import wraps
from datetime import datetime
import random
from main.EEG.data_process_np import EEGDataLoader
from main.EEG.MI_train import BraindecodeTrainer, load_sccnet_params, load_shallowfbcsp_params
from main.EEG.models import SCCNet
from braindecode.models import EEGConformer, CTNet, EEGNetv4, ShallowFBCSPNet, EEGNeX
import torch.backends.cudnn as cudnn


# https://braindecode.org/dev/generated/braindecode.models.CTNet.html (2024 MI)
# https://braindecode.org/dev/generated/braindecode.models.EEGNeX.html (2024 )
# https://braindecode.org/dev/generated/braindecode.models.SCCNet.html

# 自動儲存 log
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def tee_log(log_file=None):
    """裝飾器：將 print 輸出同時存檔與顯示"""
    if log_file is None:
        log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(log_file, "w", encoding="utf-8") as f:
                sys.stdout = Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout
            print(f"✅ 輸出已保存到 {log_file}")
            return result

        return wrapper

    return decorator


def balance_labels(x_self, y_self, n_start=3):
    assert len(x_self) == len(y_self), "x_self and y_self must have the same number of samples"

    label_0_idx = np.where(y_self == 0)[0]
    label_1_idx = np.where(y_self == 1)[0]

    count_0 = len(label_0_idx)
    count_1 = len(label_1_idx)

    if count_0 == count_1:
        print("Labels are already balanced.")
        return x_self.copy(), y_self.copy()

    # Determine majority and minority
    majority_label = 0 if count_0 > count_1 else 1
    minority_count = min(count_0, count_1)
    majority_idx = np.where(y_self == majority_label)[0]

    # How many need to be deleted
    num_to_delete = abs(count_0 - count_1)

    # Prepare deletion index
    delete_indices = []

    data_len = len(y_self)
    step = data_len // n_start

    for i in range(n_start):
        start = i * step
        end = data_len if i == n_start - 1 else (i + 1) * step

        # Search in this range for majority labels to delete
        for idx in range(start, end):
            if y_self[idx] == majority_label and idx not in delete_indices:
                delete_indices.append(idx)
                if len(delete_indices) >= num_to_delete:
                    break
        if len(delete_indices) >= num_to_delete:
            break

    # Create mask for indices to keep
    all_indices = set(range(data_len))
    keep_indices = sorted(list(all_indices - set(delete_indices)))

    x_balanced = x_self[keep_indices]
    y_balanced = y_self[keep_indices]

    return x_balanced, y_balanced


# =========================
# Class 1: 自己錄的 EEG CSV/TXT 資料
# =========================
class EEGSelfDataLoader:
    def __init__(self, file_paths, log_paths, channel_index):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.x_data = None
        self.y_data = None

    def load_data(self):
        loader = EEGDataLoader(
            file_paths=self.file_paths,
            log_paths=self.log_paths,
            channel_index=self.channel_index
        )
        loader.load_and_preprocess_data()
        self.x_data, self.y_data = loader.get_eeg_trial_channel_sample_np()

    def get_data(self):
        return self.x_data, self.y_data


# --- Motor Imagery preprocessing ---
def bandpass(data, fs=500, low=1, high=40):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)


def down_sample(data, new_fs=125):  # data: (n_samples, n_channels)
    decimation_factor = 500 // new_fs  # = 4
    return decimate(data, decimation_factor, axis=0, zero_phase=True)


def prepare_datasets(x_np_data, y_np_data, valid_num=0, segment_len=500, stride=20):  # (T, C, S)
    # stride overlap 大小, segment_len 每個 sample 長度
    x_np_data = np.transpose(x_np_data, (0, 2, 1))  # (T, C, S) -> (T, S, C)
    augmented_segments_valid = []
    augmented_labels_valid = []
    augmented_segments_train = []
    augmented_labels_train = []

    for i, s in enumerate(x_np_data):
        label = y_np_data[i]
        if s.shape[0] < segment_len:
            continue  # 忽略太短的資料段

        for start in range(0, s.shape[0] - segment_len + 1, stride):
            # shape: (500, n_channels) # down sample: (125, n_channels)
            # window = down_sample(bandpass(s[start:start + segment_len]))
            data = s[start:start + segment_len]
            window = bandpass(data - np.mean(data, axis=1, keepdims=True))  # demean
            # window = data
            if i < valid_num:
                augmented_segments_valid.append(window)
                augmented_labels_valid.append(label)
            else:
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)
    print(f"augmented_segments_train {np.stack(augmented_segments_train).shape}")  # (trial, sample, channel)
    if valid_num > 0:
        print(f"augmented_segments_valid {np.stack(augmented_segments_valid).shape}")

    # 轉換為 TensorDataset 格式
    def to_dataset(segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (N, C, T) # (sample, channel, trial)
        # data_x = np.transpose(np.stack(segment_list), (0, 1, 2))  # (T, C, N) # (sample, channel, trial)
        X = torch.tensor(data_x).unsqueeze(1)  # (N, 1, C, T)
        # X = torch.tensor(data_x)  # (N, C, T)
        y = F.one_hot(torch.tensor(label_list).long())
        return TensorDataset(X, y)

    dataset = to_dataset(augmented_segments_train, augmented_labels_train)
    print(f"Train dataset size: {len(dataset)}")
    if valid_num > 0:
        dataset_valid = to_dataset(augmented_segments_valid, augmented_labels_valid)
        print(f"Valid dataset size: {len(dataset_valid)}")
        return dataset, dataset_valid
    else:
        return dataset



def run_braindecode_training(model_class, dataset, dataset_valid, epochs=1000, batch_size=16, lr=1e-4,
                             freeze_layers=False, seed=42, params=None):
    if params is None:
        raise RuntimeError(f"please set the params of the {model_class}")
    print('seed is ' + str(seed))
    print(f"model {model_class}, epochs {epochs}, batch_size {batch_size}, lr {lr}, freeze_layers {freeze_layers}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = BraindecodeTrainer(
        dataset=dataset,
        val_dataset=dataset_valid,
        model_class=model_class,  # 這裡可以改成 EEGNetv4, ShallowFBCSPNet 等
        model_kwargs=params,  # 對應模型參數
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr
    )

    hist = trainer.train(freeze_layer=freeze_layers)
    best_loss_epoch = np.argmin(hist["val_loss"])
    print(f"best_epoch_ft: {best_loss_epoch}, "
          f"acc: {hist['acc'][best_loss_epoch]:.4f}, loss: {hist['loss'][best_loss_epoch]}, "
          f"val acc: {hist['val_acc'][best_loss_epoch]:.4f}, val loss: {hist['val_loss'][best_loss_epoch]}")
    best_epoch = np.argmax(hist["val_acc"])
    print(f"best_acc: {best_epoch}, acc: {hist['acc'][best_epoch]:.4f}, loss: {hist['loss'][best_epoch]}, "
          f"val acc: {hist['val_acc'][best_epoch]:.4f}, val loss: {hist['val_loss'][best_epoch]}")





def arrange_by_label(x, y):
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
    print(f"label: num_pairs {num_pairs}, total label: {len(y)}")
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
    # print(f"y_sorted: {y_sorted}")
    return x_sorted, y_sorted


def save_as_pt(data_x, data_y):  # 2025/10/31 加入，儲存模型
    torch.save({'data_x': data_x, 'data_y': data_y}, "data.pt")
    print("model save as data.pt")
    # use
    # train_data = torch.load("data.pt")
    # data_x = train_data['data_x']
    # data_y = train_data['data_y']
    # data_x = data_x.numpy() if data_x.is_cuda == False else data_x.cpu().numpy()
    # data_y = data_y.numpy() if data_y.is_cuda == False else data_y.cpu().numpy()


def cat_all_data(data_list):
    """
    使用方法
    # datas = ["1.pt", "6.pt"]
    # x_data, y_self = cat_all_data(datas)
    :param data_list: str
    :return: all x np, all y np
    """
    # 儲存合併後的資料
    all_x_data = []
    all_y_data = []

    # 載入並合併 1.pt 到 6.pt
    for i in data_list:
        print(i)
        file_name = i
        train_data = torch.load(file_name)  # 載入 .pt 檔案

        # 提取 x_data 和 y_data
        x_data = train_data['x_data']
        y_data = train_data['y_data']

        # 合併數據 (假設 x_data 和 y_data 是 PyTorch tensors 或者列表，可以進行直接合併)
        all_x_data.append(x_data)
        all_y_data.append(y_data)

    # 合併為單一 NumPy 陣列
    all_x_data_np = np.concatenate(all_x_data, axis=0)  # 合併所有 x_data
    all_y_data_np = np.concatenate(all_y_data, axis=0)  # 合併所有 y_data
    return all_x_data_np, all_y_data_np


cudnn.deterministic = True
cudnn.benchmark = False


@tee_log("training_log_20251204.txt")
def main():
    batch_size = 8
    channel_index = list(range(32))  # 0~31 # 用於直接儲存的資料 .pt # 32 channel
    lr = 1e-3
    epochs = 100

    id = "1"
    session = "s1"
    base_dir = r"D:/CECNL_lab/lab_project/VR/VR-BCI_beat_saber_python/main/tmp_data/MIEXP"
    datas = [
        f"{base_dir}/{id}/{session}/run7/mi.pt", f"{base_dir}/{id}/{session}/run6/mi.pt",
        f"{base_dir}/{id}/{session}/run5/mi.pt", f"{base_dir}/{id}/{session}/run4/mi.pt",
    ]
    x_data, y_self = cat_all_data(datas)
    strides = 100

    # k_folds = 3  # 5
    count = 0

    x_self = x_data
    # x_self, y_self = balance_labels(x_self, y_self, n_start=3)  # 平衡 label
    x_self, y_self = arrange_by_label(x_self, y_self)
    x_self = x_self[:, channel_index, :]  # 取出對應 channel
    # save_as_pt(x_self, y_self)  # save to data.pt
    print("Self data shape:", x_self.shape, y_self.shape)  # (trials, channels, samples)
    print(f"channel index: {channel_index}")

    # 設定為 1/3 的 data use in normal train
    valid_nums = x_self.shape[0] // 3 + 1 if (x_self.shape[0] // 3) % 2 != 0 else x_self.shape[0] // 3
    print(f"valid_nums: {valid_nums}")
    dataset_self, dataset_self_val = prepare_datasets(x_self, y_self, valid_num=valid_nums, segment_len=500,
                                                      stride=strides)
    # use in cross validation

    params = load_sccnet_params(dataset_self)
    # run_braindecode_CV_training(ShallowFBCSPNet, dataset_self, epochs=epochs,
    #                             batch_size=batch_size, lr=lr, freeze_layers=False, seed=42, params=params,
    #                             k_folds=k_folds)
    run_braindecode_training(SCCNet, dataset_self, dataset_self_val, epochs=epochs,
                             batch_size=batch_size, lr=lr, freeze_layers=False, seed=42, params=params)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    main()
