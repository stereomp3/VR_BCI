"""
隨機初始化模型權重產生工具 (Random Model Weight Generator)
用於依據指定通道數量 (8, 13, 22, 32) 建立隨機權重並保存，解決切換通道數時的 size mismatch 問題。
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 確保路徑能導入 main 模組
_current_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.dirname(_current_dir)
_python_dir = os.path.dirname(_main_dir)
for p in [_python_dir, _main_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, ATCNet
from main.EEG.models import SCCNet
import main.Utils.config as config


def build_random_models(n_chans=32, n_outputs=2, n_times=500, seed=42):
    """根據指定通道數與長度建立所有支援架構的隨機模型"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    models = {
        "ShallowFBCSPNet": ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        ),
        "EEGNetv4": EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        ),
        "EEGConformer": EEGConformer(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        ),
        "ATCNet": ATCNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        ),
        "SCCNet": SCCNet(
            samples=n_times,
            channels=n_chans,
            n_classes=n_outputs,
            sfreq=500,
        )
    }
    return models


def generate_and_save_checkpoints(n_chans=32, n_outputs=2, n_times=500, output_dir=None,
                                  update_main_checkpoint=True, seed=42):
    """產生並儲存隨機權重至指定資料夾"""
    if output_dir is None:
        output_dir = config.EEG_CHECKPOINT_MAIN_BASE_FILE

    os.makedirs(output_dir, exist_ok=True)
    models = build_random_models(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, seed=seed)

    saved_files = []
    for name, model in models.items():
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": n_times,
            "model_name": name,
        }
        # 存成特定通道與模型命名的檔案，例如 13c_SCCNet_random.pth
        ch_named_file = os.path.join(output_dir, f"{n_chans}c_{name}_random.pth")
        torch.save(checkpoint, ch_named_file)
        saved_files.append(ch_named_file)
        print(f"✅ 已儲存隨機模型: {ch_named_file}")

        # 若為預設模型 (SCCNet) 且開啟 update_main_checkpoint，同步覆蓋 c_000.pth 與 model.pth
        if update_main_checkpoint and name == "SCCNet":
            c_000_path = os.path.join(output_dir, "c_000.pth")
            model_path = os.path.join(output_dir, "model.pth")
            torch.save(checkpoint, c_000_path)
            torch.save(checkpoint, model_path)
            print(f"🎯 已同步更新即時推論與 Calibration 基礎模型: {c_000_path} 及 {model_path} ({n_chans} 通道)")

    return saved_files


def create_matching_random_checkpoint(target_path, model_class, n_chans, n_outputs=2, n_times=500, seed=42):
    """專供 EEGPrediction 與 MI_train 在偵測到 size mismatch 時呼叫之自動修復函式"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = getattr(model_class, "__name__", str(model_class))
    print(f"🔧 [自動修復] 正為 {model_name} 建立符合 {n_chans} 通道的全新隨機初始權重...")

    if model_name == "SCCNet":
        model = SCCNet(samples=n_times, channels=n_chans, n_classes=n_outputs, sfreq=500)
    elif model_name == "ShallowFBCSPNet":
        model = ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    elif model_name == "EEGNetv4":
        model = EEGNetv4(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    elif model_name == "EEGConformer":
        model = EEGConformer(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    elif model_name == "ATCNet":
        model = ATCNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    else:
        # 通用 fallback
        try:
            model = model_class(channels=n_chans, samples=n_times, n_classes=n_outputs, sfreq=500)
        except Exception:
            model = model_class(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_chans": n_chans,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "model_name": model_name,
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        torch.save(checkpoint, target_path)
        print(f"💾 [自動修復] 隨機權重已成功寫入: {target_path}")
    except Exception as e:
        print(f"⚠️ [自動修復] 寫入檔案失敗 ({e})，返回記憶體中的權重字典")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="產生各通道與模型架構之隨機初始權重")
    parser.add_argument("--channels", type=int, default=None, choices=[8, 13, 22, 32],
                        help="指定通道數量 (預設讀取 config.N_CHANNELS)")
    parser.add_argument("--all_channels", action="store_true",
                        help="一鍵為 8, 13, 22, 32 全數產生隨機模型")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="模型儲存資料夾 (預設 python/main/EEG/checkpoint_main)")
    parser.add_argument("--no_update_main", action="store_true",
                        help="不覆蓋 c_000.pth 與 model.pth")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")
    args = parser.parse_args()

    out_dir = args.output_dir or config.EEG_CHECKPOINT_MAIN_BASE_FILE

    if args.all_channels:
        print("🚀 正在為所有支援通道 (8, 13, 22, 32) 建立隨機模型...")
        for ch in [8, 13, 22, 32]:
            generate_and_save_checkpoints(n_chans=ch, output_dir=out_dir,
                                          update_main_checkpoint=(ch == config.N_CHANNELS),
                                          seed=args.seed)
    else:
        ch = args.channels if args.channels is not None else config.N_CHANNELS
        print(f"🚀 正在為 {ch} 通道建立隨機模型...")
        generate_and_save_checkpoints(n_chans=ch, output_dir=out_dir,
                                      update_main_checkpoint=(not args.no_update_main),
                                      seed=args.seed)

    print("\n🎉 隨機模型建立完成！")


if __name__ == "__main__":
    main()
