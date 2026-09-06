"""
快速模型訓練與驗證工具 (Quick Model Trainer & Verification Tool)
用於快速載入任意 .pt 資料集或合成 Demo 資料，測試各模型架構收斂性與即時微調效果，
無需執行繁重的全受試者 4-Fold CV 流程。
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# 確保路徑導入
_tools_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(_tools_dir)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import main.Utils.config as config
from main.EEG.models import SCCNet
from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, ATCNet
from main.EEG.MI_train import BraindecodeTrainer, load_sccnet_params, load_shallowfbcsp_params

MODEL_DICT = {
    "SCCNet": SCCNet,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "EEGNetv4": EEGNetv4,
    "EEGConformer": EEGConformer,
    "ATCNet": ATCNet
}


def load_or_cat_pt_files(file_list):
    """載入並串接多個 .pt 檔案"""
    all_x = []
    all_y = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"⚠️ [警告] 找不到檔案: {f}，跳過")
            continue
        data = torch.load(f, map_location="cpu")
        if isinstance(data, dict):
            # 常見 dict 儲存格式 {"x": ..., "y": ...} 或 {"data": ..., "label": ...}
            x = data.get("x", data.get("data", data.get("X", None)))
            y = data.get("y", data.get("label", data.get("Y", None)))
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            x, y = data[0], data[1]
        elif isinstance(data, TensorDataset):
            x, y = data.tensors[0], data.tensors[1]
        else:
            print(f"⚠️ [警告] 無法識別的資料結構: {type(data)} in {f}")
            continue

        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return None, None

    x_combined = np.concatenate(all_x, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    return x_combined, y_combined


def create_demo_data(n_trials=60, n_channels=32, n_samples=500):
    """產生合成平衡腦波資料供測試 (30 左手, 30 右手)"""
    print(f"🧪 正在生成 Demo 腦波合成資料: {n_trials} Trials, {n_channels} Channels, {n_samples} Samples...")
    t = np.linspace(0, 1, n_samples)
    x_data = np.random.randn(n_trials, n_channels, n_samples) * 5.0
    y_data = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        label = i % 2
        y_data[i] = label
        # 注入不同頻率特徵以供模型學習 (左手 12Hz, 右手 20Hz)
        freq = 12.0 if label == 1 else 20.0
        signal = np.sin(2 * np.pi * freq * t) * 15.0
        x_data[i, :, :] += signal

    return x_data, y_data


def prepare_tensors(x_np, y_np, val_ratio=0.25, channel_indices=None):
    """整理資料維度並切分訓練集與驗證集"""
    # 擷取通道
    if channel_indices is not None and x_np.shape[1] > len(channel_indices):
        valid_indices = [idx for idx in channel_indices if idx < x_np.shape[1]]
        x_np = x_np[:, valid_indices, :]

    # 確保維度為 (N, 1, C, T)
    if x_np.ndim == 3:
        x_tensor = torch.tensor(x_np).unsqueeze(1).float()
    elif x_np.ndim == 4:
        x_tensor = torch.tensor(x_np).float()
    else:
        raise ValueError(f"不支援的資料維度: {x_np.shape}")

    # 確保標籤為 One-hot
    if y_np.ndim == 1:
        y_tensor = F.one_hot(torch.tensor(y_np).long(), num_classes=2).float()
    else:
        y_tensor = torch.tensor(y_np).float()

    total_samples = len(x_tensor)
    val_size = int(total_samples * val_ratio)
    indices = list(range(total_samples))
    random.shuffle(indices)

    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(x_tensor[val_idx], y_tensor[val_idx]) if val_size > 0 else None

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="VR-BCI 快速模型訓練與驗證工具")
    parser.add_argument("--pt_files", nargs="+", default=None,
                        help="欲訓練之 .pt 檔案路徑清單")
    parser.add_argument("--demo", action="store_true",
                        help="使用合成平衡 Demo 資料進行快速端到端測試")
    parser.add_argument("--channels", type=int, default=int(config.ACTIVE_CHANNEL_MODE),
                        choices=[8, 13, 22, 32], help="通道數設定")
    parser.add_argument("--model", type=str, default="SCCNet",
                        choices=list(MODEL_DICT.keys()), help="模型架構")
    parser.add_argument("--epochs", type=int, default=30, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    parser.add_argument("--save_path", type=str, default=None,
                        help="訓練後權重存檔路徑 (預設 python/main/EEG/checkpoint_main/<model>_quick.pth)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    # 1. 載入或生成資料
    channel_indices = config.CHANNEL_DEFINITIONS.get(str(args.channels), list(range(args.channels)))
    if args.pt_files:
        print(f"📂 正在載入指定的 {len(args.pt_files)} 個 .pt 檔案...")
        x_raw, y_raw = load_or_cat_pt_files(args.pt_files)
        if x_raw is None:
            print("❌ 載入失敗，查無可用資料。")
            return 1
    else:
        print("ℹ️ 未指定 --pt_files，自動啟動 --demo 模式驗證訓練流程...")
        x_raw, y_raw = create_demo_data(n_trials=60, n_channels=args.channels, n_samples=config.SAMPLE_RATE)

    # 2. 建立 Dataset
    train_ds, val_ds = prepare_tensors(x_raw, y_raw, val_ratio=0.25, channel_indices=channel_indices)
    print(f"📊 資料集準備完成: 訓練集={len(train_ds)} 筆, 驗證集={len(val_ds) if val_ds else 0} 筆, 通道數={args.channels}")

    # 3. 準備模型參數
    model_cls = MODEL_DICT[args.model]
    if args.model == "SCCNet":
        params = dict(samples=config.SAMPLE_RATE, channels=args.channels, n_classes=config.N_Class, sfreq=config.SAMPLE_RATE)
    else:
        params = dict(n_chans=args.channels, n_outputs=config.N_Class, n_times=config.SAMPLE_RATE)

    save_path = args.save_path or os.path.join(
        config.EEG_CHECKPOINT_MAIN_BASE_FILE,
        f"{args.channels}c_{args.model}_quick.pth"
    )

    # 4. 啟動訓練器
    print(f"\n🚀 開始訓練 {args.model} (Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr})...")
    trainer = BraindecodeTrainer(
        dataset=train_ds,
        val_dataset=val_ds,
        model_class=model_cls,
        model_kwargs=params,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )

    hist = trainer.train()

    # 5. 輸出訓練成果與存檔
    best_val_epoch = int(np.argmax(hist["val_acc"]))
    best_val_acc = hist["val_acc"][best_val_epoch]
    print(f"\n🏆 訓練完成！最佳驗證集準確率出現在第 {best_val_epoch+1} 輪: Acc = {best_val_acc*100:.2f}%")

    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "channels": args.channels,
        "model_name": args.model,
        "best_val_acc": best_val_acc,
        "history": hist
    }
    torch.save(checkpoint, save_path)
    print(f"💾 最佳模型權重已儲存至: {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
