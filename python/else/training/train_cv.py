"""
統一交叉驗證模型訓練與 Saliency Map 運算核心 (Unified Cross-Validation & Saliency Map Runner)
支援 13 與 22 通道、單 Run 獨立訓練 (per_run) 與 全 Run 合併訓練 (all_runs)。
"""

import os
import sys
import shutil
import pickle
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import mne
from captum.attr import Saliency, NoiseTunnel

# 自動處理路徑以載入專案相依模組
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
search_dirs = [current_dir, parent_dir, os.path.join(parent_dir, "utils"),
               os.path.join(parent_dir, "test"), os.path.join(parent_dir, "main", "EEG"),
               os.path.join(parent_dir, "20260526")]
for d in search_dirs:
    if d not in sys.path and os.path.exists(d):
        sys.path.insert(0, d)

from common_utils import (
    tee_log, arrange_by_label, prepare_datasets, cat_all_data,
    CHANNEL_CONFIGS, DEFAULT_IDS, DEFAULT_SESSIONS
)

from MI_train import BraindecodeTrainerCV, load_sccnet_params
from Models import SCCNet
from XBrainLab.load_data import Raw
from XBrainLab.dataset import Epochs
from XBrainLab.training.record import EvalRecord

cudnn.deterministic = True
cudnn.benchmark = False


def run_braindecode_CV_training(model_class, dataset, dataset_valid=None, epochs=100, batch_size=8, lr=1e-3,
                                freeze_layers=False, seed=42, params=None, k_folds=4):
    """執行 Braindecode 交叉驗證訓練"""
    if params is None:
        raise RuntimeError(f"Please set the params of {model_class}")
    print(f"seed: {seed}")
    print(f"model: {model_class}, epochs: {epochs}, batch_size: {batch_size}, lr: {lr}, freeze_layers: {freeze_layers}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
          f"acc: {hist['acc'][best_loss_epoch]:.4f}, loss: {hist['loss'][best_loss_epoch]:.4f}, "
          f"val acc: {hist['val_acc'][best_loss_epoch]:.4f}, val loss: {hist['val_loss'][best_loss_epoch]:.4f}")
    best_epoch = np.argmax(hist["val_acc"])
    print(f"best_acc: {best_epoch}, acc: {hist['acc'][best_epoch]:.4f}, loss: {hist['loss'][best_epoch]:.4f}, "
          f"val acc: {hist['val_acc'][best_epoch]:.4f}, val loss: {hist['val_loss'][best_epoch]:.4f}")
    avg_val_acc = sum(hist['avg_val_acc']) / k_folds
    print(f"avg_val acc: {avg_val_acc}, all val acc: {hist['avg_val_acc']}")

    return hist, trainer.device


def compute_and_save_saliency(model, dataset, ch_names, save_dir, device, batch_size=16, file_prefix="13"):
    """計算並儲存 Saliency Map 及評估紀錄"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    tensor_x = dataset.tensors[0].to(device)
    tensor_y = dataset.tensors[1].to(device)

    np_x = tensor_x.squeeze(1).cpu().numpy()
    np_y = tensor_y.argmax(dim=1).cpu().numpy()

    sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    events = np.column_stack((np.arange(len(np_y)) * sfreq, np.zeros(len(np_y), dtype=int), np_y))
    mne_epochs = mne.EpochsArray(np_x, info, events=events, verbose=False)

    sub_name = save_dir.replace('\\', '/').split('/')[-2] if len(save_dir.replace('\\', '/').split('/')) > 1 else "Sub"
    sess_name = save_dir.replace('\\', '/').split('/')[-1]
    raw_wrapper = Raw("dummy_data.fif", mne_epochs)
    raw_wrapper.set_subject_name(sub_name)
    raw_wrapper.set_session_name(sess_name)
    xb_epochs = Epochs([raw_wrapper])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    saliency_inst = Saliency(model)
    noise_tunnel_inst = NoiseTunnel(saliency_inst)

    output_list, label_list = [], []
    saliency_maps = {'gradient': [], 'gradient_input': [], 'smoothgrad': [], 'smoothgrad_sq': [], 'vargrad': []}
    saliency_params = {'nt_samples': 5, 'stdevs': 1.0}

    print(f"正在計算 Saliency Map，儲存至: {save_dir}...")
    for inputs, labels in dataloader:
        inputs.requires_grad = True
        inputs_sq = inputs.float().squeeze(1).to(device)

        outputs = model(inputs_sq)
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        labels = labels.to(device)
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)

        grad = saliency_inst.attribute(inputs_sq, target=labels, abs=False).detach().cpu().numpy()
        saliency_maps['gradient'].append(grad)
        saliency_maps['gradient_input'].append(inputs_sq.detach().cpu().numpy() * grad)
        saliency_maps['smoothgrad'].append(
            noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='smoothgrad', **saliency_params).detach().cpu().numpy())
        saliency_maps['smoothgrad_sq'].append(
            noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='smoothgrad_sq', **saliency_params).detach().cpu().numpy())
        saliency_maps['vargrad'].append(
            noise_tunnel_inst.attribute(inputs_sq, target=labels, nt_type='vargrad', **saliency_params).detach().cpu().numpy())

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

    xb_pkl = os.path.join(save_dir, f"{file_prefix}_eval_xb_epochs.pkl")
    record_pkl = os.path.join(save_dir, f"{file_prefix}_eval_record.pkl")
    with open(xb_pkl, 'wb') as f:
        pickle.dump(xb_epochs, f)
    with open(record_pkl, 'wb') as f:
        pickle.dump(eval_record, f)

    print(f"✅ Saliency 儲存完成！({xb_pkl})")


def train_cross_validation(channels="22", mode="per_run", base_dir=r"/mnt/project/MIEXP/DATA_Cygnus",
                           save_dir=None, ids=None, sessions=None, batch_size=8, lr=1e-3, epochs=100,
                           k_folds=4, strides=100, seed=42, log_file=None):
    """
    主要交叉驗證訓練執行流程
    :param channels: 通道模式 ("13" 或 "22")
    :param mode: 訓練模式 ("per_run": 每個 run 獨立訓練, "all_runs": 同 session 內所有 run 合併訓練)
    :param base_dir: 資料存放根目錄
    :param save_dir: 模型與評估結果儲存根目錄 (若為 None 則存於 base_dir 的對應子目錄中)
    """
    ch_str = str(channels)
    if ch_str not in CHANNEL_CONFIGS:
        raise ValueError(f"不支援的 channels: {channels}，請選擇 13 或 22")

    cfg = CHANNEL_CONFIGS[ch_str]
    channel_index = cfg["channel_index"]
    ch_names = cfg["ch_names"]
    mi_filename = cfg["mi_filename"]
    model_filename = cfg["model_filename"]
    file_prefix = ch_str

    if ids is None or len(ids) == 0:
        ids = DEFAULT_IDS
    elif isinstance(ids, str):
        ids = [x.strip() for x in ids.split(',') if x.strip()]

    if sessions is None or len(sessions) == 0:
        sessions = DEFAULT_SESSIONS
    elif isinstance(sessions, str):
        sessions = [x.strip() for x in sessions.split(',') if x.strip()]

    if log_file is None:
        mode_suffix = "_all" if mode == "all_runs" else ""
        log_file = f"training_log_20260416_{ch_str}{mode_suffix}.txt"

    @tee_log(log_file)
    def _run_training():
        print("=" * 75)
        print(f"🎯 開始 Cross-Validation 訓練")
        print(f"  ├── 通道: {ch_str} channels ({len(ch_names)} 組)")
        print(f"  ├── 模式: {mode}")
        print(f"  ├── base_dir: {base_dir}")
        print(f"  ├── 受試者數量: {len(ids)} ({ids})")
        print(f"  ├── Session: {sessions}")
        print(f"  ├── 參數: batch_size={batch_size}, lr={lr}, epochs={epochs}, k_folds={k_folds}, strides={strides}")
        print(f"  └── Log 檔: {log_file}")
        print("=" * 75)

        for i in ids:
            for s in sessions:
                print(f"\n===================== subject {i}, session {s} start ================================")
                subject_session_dir = os.path.join(base_dir, i, s)
                datas = [os.path.join(subject_session_dir, f"run{r}", mi_filename) for r in range(1, 8)]

                # 若指定自訂 save_dir，鏡像其路徑
                if save_dir:
                    out_subject_session_dir = os.path.join(save_dir, i, s)
                else:
                    out_subject_session_dir = subject_session_dir

                if mode == "per_run":
                    count = 0
                    for run_idx, data_path in enumerate(datas, 1):
                        if not os.path.exists(data_path):
                            print(f"File not found: {data_path}")
                            continue

                        if save_dir:
                            run_save_dir = os.path.join(save_dir, i, s, f"run{run_idx}")
                        else:
                            run_save_dir = os.path.join(subject_session_dir, f"run{run_idx}")

                        print(f"===================== new data {data_path} ================================")
                        train_data = torch.load(data_path, map_location='cpu')
                        x_self, y_self = arrange_by_label(train_data['x_data'], train_data['y_data'])
                        x_self = x_self[:, channel_index, :]
                        print("Self data shape:", x_self.shape, y_self.shape)

                        dataset_self = prepare_datasets(x_self, y_self, valid_num=0, segment_len=500, stride=strides)
                        params = load_sccnet_params(dataset_self)

                        print("===================== start ================================")
                        print(f"===================== run SCCNet, count: {count} ================================")
                        hist, device = run_braindecode_CV_training(
                            SCCNet, dataset_self, epochs=epochs, batch_size=batch_size, lr=lr,
                            freeze_layers=False, seed=seed, params=params, k_folds=k_folds
                        )
                        count += 1
                        print("===================== end ================================")

                        best_step = np.argmin(hist["val_loss"])
                        best_model_path = os.path.join("checkpoints", f"train-epoch{best_step}.pth")
                        os.makedirs(run_save_dir, exist_ok=True)
                        if os.path.exists(best_model_path):
                            shutil.copyfile(best_model_path, os.path.join(run_save_dir, model_filename))

                        best_model = SCCNet(**params).to(device)
                        if os.path.exists(best_model_path):
                            best_model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])

                        compute_and_save_saliency(best_model, dataset_self, ch_names, save_dir=run_save_dir,
                                                  device=device, file_prefix=file_prefix)

                elif mode == "all_runs":
                    x_all, y_all = cat_all_data(datas)
                    if x_all is not None:
                        x_self, y_self = arrange_by_label(x_all, y_all)
                        x_self = x_self[:, channel_index, :]
                        print("Self data shape (All runs):", x_self.shape, y_self.shape)

                        dataset_self = prepare_datasets(x_self, y_self, valid_num=0, segment_len=500, stride=strides)
                        params = load_sccnet_params(dataset_self)

                        print("===================== run SCCNet (All runs) ================================")
                        hist, device = run_braindecode_CV_training(
                            SCCNet, dataset_self, epochs=epochs, batch_size=batch_size, lr=lr,
                            freeze_layers=False, seed=seed, params=params, k_folds=k_folds
                        )
                        print("===================== end ================================")

                        best_step = np.argmin(hist["val_loss"])
                        best_model_path = os.path.join("checkpoints", f"train-epoch{best_step}.pth")
                        os.makedirs(out_subject_session_dir, exist_ok=True)
                        if os.path.exists(best_model_path):
                            shutil.copyfile(best_model_path, os.path.join(out_subject_session_dir, model_filename))

                        best_model = SCCNet(**params).to(device)
                        if os.path.exists(best_model_path):
                            best_model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])

                        compute_and_save_saliency(best_model, dataset_self, ch_names, save_dir=out_subject_session_dir,
                                                  device=device, file_prefix=file_prefix)

    _run_training()


def parse_train_args(default_channels="22", default_mode="per_run"):
    """解析交叉驗證訓練命令列參數"""
    parser = argparse.ArgumentParser(description="Cross-Validation 交叉驗證訓練與 Saliency 特徵生成")
    parser.add_argument("--channels", type=str, default=default_channels, choices=["13", "22"],
                        help="通道設定 ('13' 或 '22')")
    parser.add_argument("--mode", type=str, default=default_mode, choices=["per_run", "all_runs"],
                        help="訓練模式: per_run (各 run 獨立訓練) 或 all_runs (同 session 合併訓練)")
    parser.add_argument("--base_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集存放根目錄")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="模型與 Saliency 儲存目錄 (預設為 base_dir 內各 run 資料夾)")
    parser.add_argument("--ids", type=str, default=None,
                        help="受試者清單，以逗號分隔 (如 '35,37,38')，留空為預設 24 人")
    parser.add_argument("--sessions", type=str, default="s1,s2",
                        help="Session 清單，以逗號分隔 (如 's1,s2')")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小 (預設 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率 (預設 1e-3)")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch 數量 (預設 100)")
    parser.add_argument("--k_folds", type=int, default=4, help="K 折數量 (預設 4)")
    parser.add_argument("--strides", type=int, default=100, help="滑動視窗步伐步長 (預設 100)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")
    parser.add_argument("--log_file", type=str, default=None, help="指定輸出 Log 檔案路徑")

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_train_args()
    train_cross_validation(
        channels=args.channels,
        mode=args.mode,
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        ids=args.ids,
        sessions=args.sessions,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        k_folds=args.k_folds,
        strides=args.strides,
        seed=args.seed,
        log_file=args.log_file
    )
