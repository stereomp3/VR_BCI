"""
用於 Calibration 場景 蒐集出來的資料做 fine-tune
和 training
"""
import numpy as np
from main.EEG.data_process_np import EEGDataLoader
from torch.utils.data import TensorDataset
import torch
import torch.nn.functional as F
import random
from main.EEG.MI_train import BraindecodeTrainer, OnlineCalibrationTrainer, load_shallowfbcsp_params, load_sccnet_params
from braindecode.models import ShallowFBCSPNet
from main.EEG.models import SCCNet
import main.Utils.config as config
import main.Utils.preprocess as preprocess
import shutil  # use for copy model
import main.Utils.global_value as global_value
import main.Utils.LSL as LSL
from main.Utils.some_functions import rename_file_with_time, get_next_version_path
import time


class EEGSelfDataLoader:
    def __init__(self, file_paths, log_paths, channel_index):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.x_data = None
        self.y_data = None
        self.failures_data = None  # 20260209 新增

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

    def save_as_pt(self):
        self.load_data()
        data_name = rename_file_with_time(config.getRunPtFilename())
        global_value.train_np_data.append(data_name)  # add the data name in the train np list
        self.x_data, self.y_data, self.failures_data = arrange_by_label(self.x_data, self.y_data, self.failures_data)
        torch.save({'x_data': self.x_data, 'y_data': self.y_data, 'failures': self.failures_data}, data_name)
        # print(f"'x_data': {self.x_data}, 'y_data': {self.y_data}, 'failures': {self.failures_data}")
        print(f"model save as {data_name}")

    def save_as_pt_with_name(self, name):
        self.load_data()
        data_name = name
        self.x_data, self.y_data, self.failures_data = arrange_by_label(self.x_data, self.y_data, self.failures_data)
        torch.save({'x_data': self.x_data, 'y_data': self.y_data, 'failures': self.failures_data}, data_name)
        # print(f"'x_data': {self.x_data}, 'y_data': {self.y_data}, 'failures': {self.failures_data}")
        print(f"model save as {data_name}")


def prepare_calibration_dataset(data_x_list, data_y_list, data_failures, segment_len=500, stride=100):
    augmented_segments_train = []
    augmented_labels_train = []
    augmented_failures_train = []

    # 遍歷每個資料集，這裡 data_x_list 和 data_y_list 都是列表
    for idx in range(len(data_x_list)):
        x_np_data = data_x_list[idx]  # 獲取每一個資料集的 x_data
        y_np_data = data_y_list[idx]  # 獲取對應的 y_data
        f_np_data = data_failures[idx]  # 獲取對應的 y_data

        # 假設 x_np_data 是 (T, C, S) 形式的資料，y_np_data 是對應的標籤
        x_np_data = np.transpose(x_np_data, (0, 2, 1))  # (T, C, S) -> (T, S, C)
        # 計算當前資料集應該分配給驗證集和訓練集的數量
        total_samples = len(x_np_data)

        # 將每一個資料集中的樣本按比例劃分為訓練集和驗證集
        for i, s in enumerate(x_np_data):
            label = y_np_data[i]
            failure = f_np_data[i]
            if s.shape[0] < segment_len:
                continue  # 忽略太短的資料段

            # 設定起始點，根據 stride 和 segment_len 截取資料
            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))  # demean

                augmented_segments_train.append(window)
                augmented_labels_train.append(label)
                augmented_failures_train.append(failure)

    # 轉換為 TensorDataset 格式
    def to_dataset(segment_list, label_list, failures_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (T, C, N) -> (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)  # (T, 1, C, N)
        # 先轉成 Long Tensor
        y_tensor = torch.tensor(label_list).long()

        # 獲取類別總數 (預設為 2，從 config 讀取)
        n_class = config.N_Class if hasattr(config, 'N_Class') else 2
        # 強制指定 num_classes，確保輸出一定是 [Batch, n_class]
        # 即使 label_list 全是 0，這裡也會產出 [1., 0.] 的格式
        y = F.one_hot(y_tensor, num_classes=n_class).float()

        f = torch.tensor(failures_list).float()

        return TensorDataset(X, y, f)

    # 構建訓練集
    dataset = to_dataset(augmented_segments_train, augmented_labels_train, augmented_failures_train)
    return dataset


# 20251111 新加入讓資料進去是使用各自的 30% 當作 val，而不是全部加起來然後前 30 %
def prepare_datasets_v2(data_x_list, data_y_list, valid_ratio=0.3, segment_len=500,
                        stride=100):  # valid_ratio 介於 0~1
    augmented_segments_valid = []
    augmented_labels_valid = []
    augmented_segments_train = []
    augmented_labels_train = []

    # 遍歷每個資料集，這裡 data_x_list 和 data_y_list 都是列表
    for idx in range(len(data_x_list)):
        x_np_data = data_x_list[idx]  # 獲取每一個資料集的 x_data
        y_np_data = data_y_list[idx]  # 獲取對應的 y_data

        # 假設 x_np_data 是 (T, C, S) 形式的資料，y_np_data 是對應的標籤
        x_np_data = np.transpose(x_np_data, (0, 2, 1))  # (T, C, S) -> (T, S, C)

        # 計算當前資料集應該分配給驗證集和訓練集的數量
        total_samples = len(x_np_data)
        valid_num = int(total_samples * valid_ratio)  # 根據比例計算驗證集的數量

        # 將每一個資料集中的樣本按比例劃分為訓練集和驗證集
        for i, s in enumerate(x_np_data):
            label = y_np_data[i]
            if s.shape[0] < segment_len:
                continue  # 忽略太短的資料段

            # 設定起始點，根據 stride 和 segment_len 截取資料
            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))  # demean

                # 根據該資料集在資料集列表中的位置來決定該資料應該屬於訓練集還是驗證集
                if i < valid_num:
                    augmented_segments_valid.append(window)
                    augmented_labels_valid.append(label)
                else:
                    augmented_segments_train.append(window)
                    augmented_labels_train.append(label)

    # 轉換為 TensorDataset 格式
    def to_dataset(segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (T, C, N) -> (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)  # (T, 1, C, N)
        y = F.one_hot(torch.tensor(label_list).long())  # trail, n class
        return TensorDataset(X, y)

    # 構建訓練集
    dataset = to_dataset(augmented_segments_train, augmented_labels_train)
    if valid_ratio > 0:
        # 構建驗證集
        dataset_valid = to_dataset(augmented_segments_valid, augmented_labels_valid)
        return dataset, dataset_valid
    else:
        return dataset


def prepare_datasets(x_np_data, y_np_data, valid_num=0, segment_len=500, stride=100):  # (T, C, S)
    # stride overlap 大小, segment_len 每個 sample 長度 # !! val 是從資料前面開始取 !!
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
            # window = preprocess.down_sample(bandpass(s[start:start + segment_len]))
            data = s[start:start + segment_len]
            window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))  # demean
            if i < valid_num:
                augmented_segments_valid.append(window)
                augmented_labels_valid.append(label)
            else:
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)

    # print(f"augmented_segments_train {np.stack(augmented_segments_train).shape}")  # (trial, sample, channel)
    # if valid_num > 0:
    #     print(f"augmented_segments_valid {np.stack(augmented_segments_valid).shape}")

    # 轉換為 TensorDataset 格式
    def to_dataset(segment_list, label_list):  # 這之前都是錯的，現在才對
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (T, C, N) # (trial, channel, sample)
        # data_x = np.transpose(np.stack(segment_list), (0, 1, 2))  # (N, C, T) # (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)  # (T, 1, C, N)
        # X = torch.tensor(data_x)  # (N, C, T)
        y = F.one_hot(torch.tensor(label_list).long())  # trail, n class
        # print(f"X.shape################# {X.shape}")
        # print(f"y.shape################# {y.shape}")
        return TensorDataset(X, y)

    dataset = to_dataset(augmented_segments_train, augmented_labels_train)
    # print(f"Train dataset size: {len(dataset)}")
    if valid_num > 0:
        dataset_valid = to_dataset(augmented_segments_valid, augmented_labels_valid)
        # print(f"Valid dataset size: {len(dataset_valid)}")
        return dataset, dataset_valid
    else:
        return dataset


def run_braindecode_training(model_class, dataset, dataset_valid, epochs=1000, batch_size=16, lr=1e-4,
                             freeze_layers=False, seed=42, params=None, ft=False, tcp_server=None):
    if params is None:
        raise RuntimeError(f"please set the params of the {model_class}")
    # print('seed is ' + str(seed))
    # print(f"model {model_class}, epochs {epochs}, batch_size {batch_size}, lr {lr}, freeze_layers {freeze_layers}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = BraindecodeTrainer(
        dataset=dataset,
        val_dataset=dataset_valid,
        model_class=model_class,  # 使用模型
        model_kwargs=params,  # 對應模型參數
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr,
        ft=ft  # ft 目前沒甚麼用
    )
    name = "train"
    if ft is True:  # fine-tune
        # trainer.load_checkpoint(config.MAIN_CHECKPOINT)
        trainer.load_checkpoint(global_value.NOW_TRAINED_CHECKPOINT)
        name = "ft"
    hist = trainer.train(freeze_layer=freeze_layers, tcp_server=tcp_server, patience=30)  # patience 代表 early stop 的判斷次數
    best_loss_epoch = np.argmin(hist["val_loss"])
    best_loss_epoch_path = f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}{name}-epoch{best_loss_epoch}.pth"
    save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)
    shutil.copyfile(best_loss_epoch_path, save_path)
    global_value.NOW_TRAINED_CHECKPOINT = save_path
    # shutil.copyfile(best_loss_epoch_path, config.TRAINED_CHECKPOINT)

    print(f"\n{config.TAGS.INFO.value} Calibration done, "
          f"copy model {name}-epoch{best_loss_epoch}.pth to the {save_path}")
    # print(f"best_epoch_ft: {best_loss_epoch}, "
    #       f"acc: {hist['acc'][best_loss_epoch]:.4f}, loss: {hist['loss'][best_loss_epoch]}, "
    #       f"val acc: {hist['val_acc'][best_loss_epoch]:.4f}, val loss: {hist['val_loss'][best_loss_epoch]}")
    # best_epoch = np.argmax(hist["val_acc"])
    # print(f"best_acc: {best_epoch}, acc: {hist['acc'][best_epoch]:.4f}, loss: {hist['loss'][best_epoch]}, "
    #       f"val acc: {hist['val_acc'][best_epoch]:.4f}, val loss: {hist['val_loss'][best_epoch]}")

    # return best_epoch


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


class EEGFineTunePipeline:
    """
    這個之前邏輯與 TrainPipeline 一樣，不過把 ft 和 Freezelayer 做修改
    20260208 把這個變成讀取存在 buffer 裡面的 calibration scene data，然後 update modell
    """

    def __init__(self, tcp_server=None):  # old: lsl_outlet=None # update one
        self.batch_size = 8
        # self.channel_index = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 27, 28, 29]
        self.channel_index = config.channel_index
        self.lr = 1e-4
        self.epochs = 2  # 少量資料 fine-tune 1~3 test
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.ft = True
        self.tcp_server = tcp_server
        self.trainer = None
        self.save_path = global_value.NOW_TRAINED_CHECKPOINT # tmp

    # 根據上面的 run 下去改，這邊為初始化 trainer，因為 trainer 有 buffer，所以需要最開始初始化一次
    def init_pipeline(self):
        params = dict(
            n_chans=len(config.channel_index),
            n_outputs=config.N_Class,
            n_times=config.SAMPLE_RATE,
        )
        self.trainer = OnlineCalibrationTrainer(
            model_class=ShallowFBCSPNet,  # 使用模型
            model_kwargs=params,  # 對應模型參數
            batch_size=self.batch_size,
            num_epochs=self.epochs,  # test 1~3
            lr=self.lr,
            ft=self.ft
        )
        self.save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)

    def run_calibration(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        name = "train"
        if self.ft is True:  # fine-tune
            # trainer.load_checkpoint(config.MAIN_CHECKPOINT)
            self.trainer.load_checkpoint(global_value.NOW_TRAINED_CHECKPOINT)
            name = "ft"

        print(f"{config.TAGS.INFO.value} train_np_data: {global_value.train_np_data[-1]}")
        train_data = torch.load(global_value.train_np_data[-1])  # load newest data
        x_new_np = train_data['x_data']
        y_new_np = train_data['y_data']
        failures_list = train_data['failures']
        dataset = prepare_calibration_dataset([x_new_np], [y_new_np], [failures_list])

        loss = self.trainer.online_train(dataset)

        print(f"{config.TAGS.INFO.value} Online update complete. Loss: {loss:.4f}")
        print(f"{config.TAGS.INFO.value} 模型已根據最新狀況 (包含失敗經驗) 進行修正。")

        best_loss_epoch_path = f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}{name}-epoch{self.epochs - 1}.pth"
        # save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)
        shutil.copyfile(best_loss_epoch_path, self.save_path)
        global_value.NOW_TRAINED_CHECKPOINT = self.save_path  # 都存在同樣的地方太多 unity 那邊會很難看，fine tune 存在同一個地方

        print(f"\n{config.TAGS.INFO.value} Calibration done, "
              f"copy model {name}-epoch{self.epochs - 1}.pth to the {self.save_path}")
        if self.tcp_server:
            time.sleep(0.5)  # 讓系統可以判斷
            self.tcp_server.broadcast(config.CALIBRATION_FINISH_STR)


class EEGTrainingPipeline:
    def __init__(self, epochs=150, tcp_server=None):
        self.batch_size = 8
        # self.channel_index = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 27, 28, 29]
        self.channel_index = config.channel_index
        self.lr = 1e-4
        self.epochs = epochs  # 少量資料 fine-tune
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.tcp_server = tcp_server

    def set_the_ft_set(self):
        # if config.is_simulated_unity:
        #     train_datas = [config.FT_CSV_FILENAME]  # config.CSV_FILENAME
        #     train_logs = [config.FT_LOG_FILENAME]  # config.LOG_FILENAME
        # else:  # 找出 calibration 場景的 Data。Fine tune (csv, log)
        #     train_datas = [x[0] for x in global_value.data_lookup_table[config.GameSTATE.MI.value]]
        #     train_logs = [x[1] for x in global_value.data_lookup_table[config.GameSTATE.MI.value]]
        #
        # print(f"{config.TAGS.INFO.value} train_datas: {train_datas}")
        # print(f"{config.TAGS.INFO.value} train_logs: {train_logs}")
        # ft_loader = EEGSelfDataLoader(
        #     file_paths=train_datas,
        #     log_paths=train_logs,
        #     channel_index=self.channel_index
        #     # [x[0] for x in global_value.data_lookup_table[config.GameSTATE.Calibration.value][0]]
        # )
        # ft_loader.load_data()
        # data_x, data_y = ft_loader.get_data()  # (30, 15, 2000) (trial, channel, samples)
        # data_x, data_y = arrange_by_label(data_x, data_y)
        data_x_list, data_y_list = [], []
        # print(f"{config.TAGS.INFO.value} train_np_data: {global_value.train_np_data}")
        # for name in global_value.train_np_data: # add all data
        #     train_data = torch.load(name)
        #     data_x_list.append(train_data['x_data'])
        #     data_y_list.append(train_data['y_data'])
        print(f"{config.TAGS.INFO.value} train_np_data: {global_value.train_np_data[-1]}")
        train_data = torch.load(global_value.train_np_data[-1])  # load newest data
        data_x_list.append(train_data['x_data'])
        data_y_list.append(train_data['y_data'])
        # data_x = np.concatenate(data_x_list, axis=0)
        # data_y = np.concatenate(data_y_list, axis=0)
        # print(f"data_x.shape: {data_x.shape}")
        # # self.valid_nums 為總資料數量的 3.333 趴，如果除出來不是偶數，那就 +1
        # # 預計會有 20 以上，14 拿來 fine tune，前面 6 組 val 20//3 = 6 val 佔 30 %，下面讀到 data 後會調整
        # valid_nums = data_x.shape[0] // 3 + 1 if (data_x.shape[0] // 3) % 2 != 0 else data_x.shape[0] // 3
        # print(f"valid_nums: {valid_nums}")
        # dataset, dataset_val = prepare_datasets(data_x, data_y, valid_num=valid_nums,
        #                                         segment_len=self.segment_len, stride=self.strides)
        print("valid_ratio: 0.3")
        dataset, dataset_val = prepare_datasets_v2(data_x_list, data_y_list, valid_ratio=0.3,
                                                   segment_len=self.segment_len, stride=self.strides)
        return dataset, dataset_val

    def run_training(self):
        dataset, dataset_val = self.set_the_ft_set()
        # params = load_sccnet_params(dataset)
        params = load_shallowfbcsp_params(dataset)

        # ft is use the pretrain model to continue train
        run_braindecode_training(ShallowFBCSPNet, dataset, dataset_val, epochs=self.epochs,
                                 batch_size=self.batch_size, lr=self.lr, freeze_layers=False, params=params,
                                 seed=self.seed, ft=True, tcp_server=self.tcp_server)  # ft=False


def main():
    pipeline = EEGFineTunePipeline()
    pipeline.run_calibration()


if __name__ == "__main__":
    main()
