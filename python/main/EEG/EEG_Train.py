"""
VR-BCI 線上微調與模型訓練模組 (EEG Training & Fine-Tuning Pipeline)
用於 Calibration 場景蒐集資料後進行即時在線微調 (Fine-tune) 與重新訓練。
"""

import os
import sys
import time
import shutil
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# 自動路徑解析，允許直接以 python python/main/EEG/EEG_Train.py 執行
_current_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.dirname(_current_dir)
_python_dir = os.path.dirname(_main_dir)
for p in [_python_dir, _main_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# UTF-8 輸出防護
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import main.Utils.config as config
import main.Utils.preprocess as preprocess
import main.Utils.global_value as global_value
from main.EEG.data_process_np import EEGDataLoader
from main.EEG.MI_train import BraindecodeTrainer, OnlineCalibrationTrainer
from main.Utils.some_functions import rename_file_with_time, get_next_version_path


class EEGSelfDataLoader:
    """受試者自身 EEG 與標籤數據載入與存檔器"""
    def __init__(self, file_paths, log_paths, channel_index):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.x_data = None
        self.y_data = None
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

    def save_as_pt_with_name(self, name):
        """將資料依標籤排序平衡後存成 .pt 檔案"""
        self.load_data()
        self.x_data, self.y_data, self.failures_data = arrange_by_label(
            self.x_data, self.y_data, self.failures_data
        )
        torch.save({
            'x_data': self.x_data,
            'y_data': self.y_data,
            'failures': self.failures_data
        }, name)
        print(f"✅ 模型資料已儲存為: {name}")

    def save_as_pt(self):
        """自動生成時間戳檔名存檔，並將路徑登錄至全域訓練資料清單"""
        data_name = rename_file_with_time(config.getRunPtFilename())
        global_value.train_np_data.append(data_name)
        self.save_as_pt_with_name(data_name)


def arrange_by_label(x, y, f):
    """
    將資料依 y 標籤按照 0 1 0 1 交錯排列以平衡訓練集；若不平衡則將配對放後面，剩下的放前面。
    :param x: (trial, channel, sample)
    :param y: (trial,)
    :param f: (trial,) failures 標籤
    :return: 排序平衡後的 (x_sorted, y_sorted, f_sorted)
    """
    label_0_idx = np.where(y == 0)[0]
    label_1_idx = np.where(y == 1)[0]

    label_0_rev = label_0_idx[::-1]
    label_1_rev = label_1_idx[::-1]

    num_pairs = min(len(label_0_rev), len(label_1_rev))
    paired_0 = label_0_rev[:num_pairs][::-1]
    paired_1 = label_1_rev[:num_pairs][::-1]

    # 交錯排列：0, 1, 0, 1...
    interleaved_idx = np.empty(num_pairs * 2, dtype=int)
    interleaved_idx[0::2] = paired_0
    interleaved_idx[1::2] = paired_1

    # 無法配對的剩餘索引放在最前面
    remaining_0 = label_0_rev[num_pairs:][::-1]
    remaining_1 = label_1_rev[num_pairs:][::-1]
    remaining_idx = np.concatenate((remaining_0, remaining_1))

    final_idx = np.concatenate((remaining_idx, interleaved_idx))
    return x[final_idx], y[final_idx], f[final_idx]


def prepare_calibration_dataset(data_x_list, data_y_list, data_failures, segment_len=500, stride=100):
    """將校正資料以滑動視窗切片並前處理，轉換為含失敗權重之 TensorDataset"""
    augmented_segments = []
    augmented_labels = []
    augmented_failures = []

    for idx in range(len(data_x_list)):
        x_np = np.transpose(data_x_list[idx], (0, 2, 1))  # (T, C, S) -> (T, S, C)
        y_np = data_y_list[idx]
        f_np = data_failures[idx]

        for i, s in enumerate(x_np):
            if s.shape[0] < segment_len:
                continue
            label = y_np[i]
            failure = f_np[i]

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))

                augmented_segments.append(window)
                augmented_labels.append(label)
                augmented_failures.append(failure)

    if not augmented_segments:
        return None

    data_x = np.transpose(np.stack(augmented_segments), (0, 2, 1))  # (trial, channel, sample)
    X = torch.tensor(data_x).unsqueeze(1).float()  # (T, 1, C, N)
    y_tensor = torch.tensor(augmented_labels).long()
    n_class = config.N_Class if hasattr(config, 'N_Class') else 2
    y = F.one_hot(y_tensor, num_classes=n_class).float()
    f = torch.tensor(augmented_failures).float()

    return TensorDataset(X, y, f)


def prepare_datasets(data_x_list, data_y_list, valid_ratio=0.3, segment_len=500, stride=100):
    """
    將資料集列表按比例劃分為訓練集與驗證集，並進行帶通濾波與去均值處理
    """
    augmented_segments_valid, augmented_labels_valid = [], []
    augmented_segments_train, augmented_labels_train = [], []

    for idx in range(len(data_x_list)):
        x_np = np.transpose(data_x_list[idx], (0, 2, 1))  # (T, C, S) -> (T, S, C)
        y_np = data_y_list[idx]
        valid_num = int(len(x_np) * valid_ratio)

        for i, s in enumerate(x_np):
            if s.shape[0] < segment_len:
                continue
            label = y_np[i]

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))

                if i < valid_num:
                    augmented_segments_valid.append(window)
                    augmented_labels_valid.append(label)
                else:
                    augmented_segments_train.append(window)
                    augmented_labels_train.append(label)

    def _to_dataset(seg_list, lbl_list):
        data_x = np.transpose(np.stack(seg_list), (0, 2, 1))
        X = torch.tensor(data_x).unsqueeze(1).float()
        y = F.one_hot(torch.tensor(lbl_list).long(), num_classes=config.N_Class).float()
        return TensorDataset(X, y)

    train_ds = _to_dataset(augmented_segments_train, augmented_labels_train)
    if valid_ratio > 0 and augmented_segments_valid:
        val_ds = _to_dataset(augmented_segments_valid, augmented_labels_valid)
        return train_ds, val_ds
    return train_ds, None


# 向後相容別名
prepare_datasets_v2 = prepare_datasets


def run_braindecode_training(model_class, dataset, dataset_valid, epochs=100, batch_size=16, lr=1e-4,
                             freeze_layers=False, seed=42, params=None, ft=False, tcp_server=None):
    """執行 Braindecode 模型訓練與 Checkpoint 儲存"""
    if params is None:
        raise RuntimeError(f"請設定模型 {model_class} 的超參數 params")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = BraindecodeTrainer(
        dataset=dataset,
        val_dataset=dataset_valid,
        model_class=model_class,
        model_kwargs=params,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr,
        ft=ft
    )

    name = "ft" if ft else "train"
    if ft:
        trainer.load_checkpoint(global_value.NOW_TRAINED_CHECKPOINT)

    hist = trainer.train(freeze_layer=freeze_layers, tcp_server=tcp_server, patience=30)
    best_loss_epoch = int(np.argmin(hist["val_loss"]))

    if config.adaption_use_val:
        best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-best.pth")
    else:
        best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-epoch{best_loss_epoch}.pth")

    save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)
    shutil.copyfile(best_loss_path, save_path)
    global_value.NOW_TRAINED_CHECKPOINT = save_path

    print(f"\n{config.TAGS.INFO.value} 訓練完成，最佳模型已複製至: {save_path}")


class EEGFineTunePipeline:
    """線上 Calibration 快速微調 Pipeline（使用 Replay Buffer 增量學習）"""

    def __init__(self, tcp_server=None):
        self.batch_size = config.adaption_batch_size
        self.channel_index = config.channel_index
        self.lr = config.adaption_learning_rate
        self.epochs = config.adaption_epochs
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.ft = True
        self.tcp_server = tcp_server
        self.trainer = None
        self.save_path = global_value.NOW_TRAINED_CHECKPOINT

    def init_pipeline(self):
        """初始化在線微調器與 Checkpoint 路徑"""
        self.trainer = OnlineCalibrationTrainer(
            model_class=config.USE_MODEL,
            model_kwargs=config.LOAD_MODEL_PARAM,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            lr=self.lr,
            ft=self.ft
        )
        self.save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)

    def run_calibration(self, seed=42):
        if self.trainer is None:
            self.init_pipeline()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        name = "ft" if self.ft else "train"
        if self.ft:
            self.trainer.load_checkpoint(global_value.NOW_TRAINED_CHECKPOINT)

        if not global_value.train_np_data:
            print(f"{config.TAGS.WARNING.value} 無可用訓練資料 (train_np_data 為空)")
            return

        latest_data_path = global_value.train_np_data[-1]
        print(f"{config.TAGS.INFO.value} 載入最新校正資料: {latest_data_path}")
        train_data = torch.load(latest_data_path, map_location="cpu")

        dataset = prepare_calibration_dataset(
            [train_data['x_data']],
            [train_data['y_data']],
            [train_data['failures']]
        )

        loss = self.trainer.online_train(dataset)
        print(f"{config.TAGS.INFO.value} 線上微調完成，最新 Loss: {loss:.4f}")

        if config.adaption_use_val:
            best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-best.pth")
        else:
            best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-epoch{self.epochs - 1}.pth")

        shutil.copyfile(best_loss_path, self.save_path)
        global_value.NOW_TRAINED_CHECKPOINT = self.save_path
        print(f"\n{config.TAGS.INFO.value} 校正微調完成，更新主模型至: {self.save_path}")

        if self.tcp_server:
            time.sleep(0.5)
            self.tcp_server.broadcast(config.CALIBRATION_FINISH_STR)


class EEGTrainingPipeline:
    """離線 / Run 結束後常規訓練 Pipeline"""

    def __init__(self, epochs=150, tcp_server=None):
        self.batch_size = 8
        self.channel_index = config.channel_index
        self.lr = 1e-4
        self.epochs = epochs
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.tcp_server = tcp_server

    def set_the_ft_set(self):
        """讀取最新資料並劃分 70% 訓練集 / 30% 驗證集"""
        if not global_value.train_np_data:
            raise RuntimeError("無可用訓練資料 (train_np_data 為空)")

        latest_data_path = global_value.train_np_data[-1]
        print(f"{config.TAGS.INFO.value} 載入訓練資料: {latest_data_path}")
        train_data = torch.load(latest_data_path, map_location="cpu")

        data_x_list = [train_data['x_data']]
        data_y_list = [train_data['y_data']]

        train_ds, val_ds = prepare_datasets(
            data_x_list, data_y_list,
            valid_ratio=0.3,
            segment_len=self.segment_len,
            stride=self.strides
        )
        return train_ds, val_ds

    def run_training(self):
        dataset, dataset_val = self.set_the_ft_set()
        params = config.LOAD_MODEL_PARAM

        run_braindecode_training(
            model_class=config.USE_MODEL,
            dataset=dataset,
            dataset_valid=dataset_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            freeze_layers=False,
            params=params,
            seed=self.seed,
            ft=True,
            tcp_server=self.tcp_server
        )


def main():
    pipeline = EEGFineTunePipeline()
    pipeline.run_calibration()


if __name__ == "__main__":
    main()
