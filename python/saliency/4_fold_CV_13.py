"""
繼承 20251206 的內容 3 fold cross validation，然後根據 MIEXP 的 data 進行調整。整合全部 session
"""
import numpy as np
import pandas as pd
from data_process_np import EEGDataLoader
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt, decimate  # decimate 用來 down sample，這部分如果要寫論文可以寫使用的數學，非整數降維使用 resample
import torch
import torch.nn.functional as F
import sys
from functools import wraps
from datetime import datetime
import random
from MI_train import EEGTrainerFineTune, BraindecodeTrainer, load_sccnet_params, load_eegnetv4_params, \
    load_conformer_params, load_shallowfbcsp_params, load_ctnet_params, load_eegnex_params, BraindecodeTrainerCV
from braindecode.models import EEGConformer, CTNet, EEGNetv4, ShallowFBCSPNet, EEGNeX
import torch.backends.cudnn as cudnn
import os
import pickle
import mne
from captum.attr import Saliency, NoiseTunnel
import shutil 
# 引入 XBrainLab 相關模組 (請確認路徑與您的專案一致)
from XBrainLab.load_data import Raw
from XBrainLab.dataset import Epochs
from XBrainLab.training.record import EvalRecord
from Models import SCCNet

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

    majority_label = 0 if count_0 > count_1 else 1
    minority_count = min(count_0, count_1)
    majority_idx = np.where(y_self == majority_label)[0]

    num_to_delete = abs(count_0 - count_1)
    delete_indices = []

    data_len = len(y_self)
    step = data_len // n_start

    for i in range(n_start):
        start = i * step
        end = data_len if i == n_start - 1 else (i + 1) * step

        for idx in range(start, end):
            if y_self[idx] == majority_label and idx not in delete_indices:
                delete_indices.append(idx)
                if len(delete_indices) >= num_to_delete:
                    break
        if len(delete_indices) >= num_to_delete:
            break

    all_indices = set(range(data_len))
    keep_indices = sorted(list(all_indices - set(delete_indices)))

    return x_self[keep_indices], y_self[keep_indices]


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


def bandpass(data, fs=500, low=1, high=40):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)


def down_sample(data, new_fs=125): 
    decimation_factor = 500 // new_fs 
    return decimate(data, decimation_factor, axis=0, zero_phase=True)


def prepare_datasets(x_np_data, y_np_data, valid_num=0, segment_len=500, stride=20): 
    x_np_data = np.transpose(x_np_data, (0, 2, 1)) 
    augmented_segments_valid = []
    augmented_labels_valid = []
    augmented_segments_train = []
    augmented_labels_train = []

    for i, s in enumerate(x_np_data):
        label = y_np_data[i]
        if s.shape[0] < segment_len:
            continue 

        for start in range(0, s.shape[0] - segment_len + 1, stride):
            data = s[start:start + segment_len]
            window = bandpass(data - np.mean(data, axis=1, keepdims=True)) 
            if i < valid_num:
                augmented_segments_valid.append(window)
                augmented_labels_valid.append(label)
            else:
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)

    print(f"augmented_segments_train {np.stack(augmented_segments_train).shape}") 

    def to_dataset(segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1)) 
        X = torch.tensor(data_x).unsqueeze(1) 
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

# ---------------------------------------------------------
# 修改點：讓 CV_training 回傳 hist 與 device，以便後續抓最佳模型
# ---------------------------------------------------------
def run_braindecode_CV_training(model_class, dataset, dataset_valid=None, epochs=1000, batch_size=16, lr=1e-4,
                                freeze_layers=False, seed=42, params=None, k_folds=5):
    if params is None:
        raise RuntimeError(f"please set the params of the {model_class}")
    print('seed is ' + str(seed))
    print(f"model {model_class}, epochs {epochs}, batch_size {batch_size}, lr {lr}, freeze_layers {freeze_layers}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = BraindecodeTrainerCV(
        dataset=dataset,
        val_dataset=dataset_valid,
        model_class=model_class, 
        model_kwargs=params, 
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr,
        k_folds=k_folds
    )

    hist = trainer.train_kfold(freeze_layer=freeze_layers)
    best_loss_epoch = np.argmin(hist["val_loss"])
    print(f"best_epoch_ft: {best_loss_epoch}, "
          f"acc: {hist['acc'][best_loss_epoch]:.4f}, loss: {hist['loss'][best_loss_epoch]}, "
          f"val acc: {hist['val_acc'][best_loss_epoch]:.4f}, val loss: {hist['val_loss'][best_loss_epoch]}")
    best_epoch = np.argmax(hist["val_acc"])
    print(f"best_acc: {best_epoch}, acc: {hist['acc'][best_epoch]:.4f}, loss: {hist['loss'][best_epoch]}, "
          f"val acc: {hist['val_acc'][best_epoch]:.4f}, val loss: {hist['val_loss'][best_epoch]}")
    avg_val_acc = sum(hist['avg_val_acc']) / k_folds
    print(f"avg_val acc: {avg_val_acc}, all val acc: {hist['avg_val_acc']}")
    
    return hist, trainer.device


def arrange_by_label(x, y):
    label_0_idx = np.where(y == 0)[0]
    label_1_idx = np.where(y == 1)[0]
    label_0_rev = label_0_idx[::-1]
    label_1_rev = label_1_idx[::-1]

    num_pairs = min(len(label_0_rev), len(label_1_rev))
    print(f"label: num_pairs {num_pairs}, total label: {len(y)}")
    
    paired_0 = label_0_rev[:num_pairs][::-1]
    paired_1 = label_1_rev[:num_pairs][::-1]

    interleaved_idx = np.empty(num_pairs * 2, dtype=int)
    interleaved_idx[0::2] = paired_0
    interleaved_idx[1::2] = paired_1

    remaining_0 = label_0_rev[num_pairs:][::-1]
    remaining_1 = label_1_rev[num_pairs:][::-1]
    remaining_idx = np.concatenate((remaining_0, remaining_1))

    final_idx = np.concatenate((remaining_idx, interleaved_idx))
    return x[final_idx], y[final_idx]


def cat_all_data(data_list):
    all_x_data = []
    all_y_data = []

    for i in data_list:
        if os.path.exists(i):
            train_data = torch.load(i)
        else:
            print(f"File not found: {i}")
            continue

        all_x_data.append(train_data['x_data'])
        all_y_data.append(train_data['y_data'])
        
    if len(all_x_data) == 0:
        return None, None
    return np.concatenate(all_x_data, axis=0), np.concatenate(all_y_data, axis=0)


def crop_center(data: np.ndarray, target_length: int) -> np.ndarray:
    if target_length > data.shape[-1]:
        raise ValueError(f"target_length ({target_length}) 不能大於資料長度 ({data.shape[-1]})")
    total = data.shape[-1]
    cut = (total - target_length) // 2
    remainder = (total - target_length) % 2
    return data[..., cut: total - cut - remainder]


cudnn.deterministic = True
cudnn.benchmark = False

# ==========================================
# 新增功能：計算與儲存 Saliency Map
# ==========================================
def compute_and_save_saliency(model, dataset, ch_names, save_dir, device, batch_size=16):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取完整數據與標籤
    tensor_x = dataset.tensors[0].to(device)
    tensor_y = dataset.tensors[1].to(device)
    
    # MNE 轉換 (N, 1, C, T) -> (N, C, T)
    np_x = tensor_x.squeeze(1).cpu().numpy()
    np_y = tensor_y.argmax(dim=1).cpu().numpy()
    
    # 建立 XBrainLab 的 Epochs 物件
    sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    events = np.column_stack((np.arange(len(np_y)) * sfreq, np.zeros(len(np_y), dtype=int), np_y))
    mne_epochs = mne.EpochsArray(np_x, info, events=events, verbose=False)
    
    raw_wrapper = Raw("dummy_data.fif", mne_epochs)
    raw_wrapper.set_subject_name(save_dir.split('/')[-2] if len(save_dir.split('/')) > 1 else "Sub")
    raw_wrapper.set_session_name(save_dir.split('/')[-1])
    xb_epochs = Epochs([raw_wrapper])
    
    # 計算 Saliency
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    saliency_inst = Saliency(model)
    noise_tunnel_inst = NoiseTunnel(saliency_inst)

    output_list, label_list = [], []
    saliency_maps = {'gradient': [], 'gradient_input': [], 'smoothgrad': [], 'smoothgrad_sq': [], 'vargrad': []}
    saliency_params = {'nt_samples': 5, 'stdevs': 1.0}

    print(f"正在計算 Saliency Map，儲存至: {save_dir}...")
    for inputs, labels in dataloader:
        inputs.requires_grad = True
        inputs_sq = inputs.float().squeeze(1).to(device) # Braindecode 吃 3D

        outputs = model(inputs_sq)
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        labels = labels.to(device)
        # 如果是 one-hot
        if labels.ndim == 2: # Captum 
            labels = labels.argmax(dim=1)
            
        grad = saliency_inst.attribute(inputs_sq, target=labels, abs=False).detach().cpu().numpy()
        saliency_maps['gradient'].append(grad)
        saliency_maps['gradient_input'].append(inputs_sq.detach().cpu().numpy() * grad)
        saliency_maps['smoothgrad'].append(noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='smoothgrad', **saliency_params).detach().cpu().numpy())
        saliency_maps['smoothgrad_sq'].append(noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='smoothgrad_sq', **saliency_params).detach().cpu().numpy())
        saliency_maps['vargrad'].append(noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='vargrad', **saliency_params).detach().cpu().numpy())

    outputs_arr = np.concatenate(output_list)
    labels_arr = np.concatenate(label_list).argmax(axis=1) 
    n_classes = outputs_arr.shape[1]
    
    final_grads = {}
    for key in saliency_maps:
        concatenated = np.concatenate(saliency_maps[key])
        final_grads[key] = {c: concatenated[labels_arr == c] for c in range(n_classes)}

    eval_record = EvalRecord(
        label=labels_arr,
        output=outputs_arr,
        gradient=final_grads['gradient'],
        gradient_input=final_grads['gradient_input'],
        smoothgrad=final_grads['smoothgrad'],
        smoothgrad_sq=final_grads['smoothgrad_sq'],
        vargrad=final_grads['vargrad']
    )

    with open(os.path.join(save_dir, "13_eval_xb_epochs.pkl"), 'wb') as f:
        pickle.dump(xb_epochs, f)
    with open(os.path.join(save_dir, "13_eval_record.pkl"), 'wb') as f:
        pickle.dump(eval_record, f)
        
    print(f"✅ Saliency 儲存完成！")


@tee_log("training_log_20260416_13.txt")
def main():
    batch_size = 8
    lr = 1e-3
    epochs = 100
    k_folds = 4  

    channel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    ch_names = ['F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'Pz']

    ids = ["35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", "51", "52", "54", "55", "57", "58",
           "63", "64", "65", "68", "69", "70"]
    sessions = ["s1", "s2"]
    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"

    for i in ids:
        for s in sessions:
            print(f"=====================subject {i}, session {s} start================================")
            subject_session_dir = f"{base_dir}/{i}/{s}"
            datas = [
                f"{subject_session_dir}/run1/mi.pt",
                f"{subject_session_dir}/run2/mi.pt",
                f"{subject_session_dir}/run3/mi.pt",
                f"{subject_session_dir}/run4/mi.pt",
                f"{subject_session_dir}/run5/mi.pt",
                f"{subject_session_dir}/run6/mi.pt",
                f"{subject_session_dir}/run7/mi.pt",
            ]
            
            strides = 100
            count = 0
            # ---------------------------------------------------------
            # 逐一 Run 獨立訓練與計算 Saliency Map
            # ---------------------------------------------------------
            for run_idx, data_path in enumerate(datas, 1):
                if not os.path.exists(data_path):
                    print(f"File not found: {data_path}")
                    continue
                run_save_dir = f"{subject_session_dir}/run{run_idx}"
                print(f"=====================new data {data_path}================================")
                train_data = torch.load(data_path)
                x_self, y_self = arrange_by_label(train_data['x_data'], train_data['y_data'])
                x_self = x_self[:, channel_index, :]
                print("Self data shape:", x_self.shape, y_self.shape)  # (trials, channels, samples)
                
                dataset_self = prepare_datasets(x_self, y_self, valid_num=0, segment_len=500, stride=strides)
                params = load_sccnet_params(dataset_self)
                print(f"=====================start================================")
                print(f"=====================run ShallowNet, count: {count}================================")
                hist, device = run_braindecode_CV_training(SCCNet, dataset_self, epochs=epochs, 
                                                           batch_size=batch_size, lr=lr, freeze_layers=False, 
                                                           seed=42, params=params, k_folds=k_folds)
                                                    
                count += 1
                print(f"=====================end================================")

                best_step = np.argmin(hist["val_loss"])
                best_model_path = os.path.join("checkpoints", f"train-epoch{best_step}.pth")
                shutil.copyfile(best_model_path, os.path.join(run_save_dir, "scc_13_model.pth"))

                best_model = SCCNet(**params).to(device)
                best_model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
                
                compute_and_save_saliency(best_model, dataset_self, ch_names, save_dir=run_save_dir, device=device)

if __name__ == "__main__":
    main()