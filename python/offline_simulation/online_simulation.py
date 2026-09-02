"""
================================================================================
線上 BCI 模擬與模型更新評估工具 (Online BCI Simulation & Model Update Heatmap)
================================================================================

【主要功能與升級】：
1. 逐 Trial 線上模擬 (Online Simulation)：
   - 模擬真實 Session 逐 Trial 即時推論 (自動相容 6-Run 240 Trials 與 7-Run 280 Trials)。
   - 支援 Condition A (Run 1~3 Calibration/Adaptive, Run 4~7 Formal/Static) 與 Condition B (全 Adaptive)。
   - 自動適應 6-Run 模式（跳過 Run 3，呈現 Run 1, Run 2, Run 4, Run 5, Run 6, Run 7）。
   - 依據線上準確率閾值 (< 75%) 自動觸發 SCCNet 線上微調 (Online Calibration)。

2. 色盲友善 Cross-Entropy 模型更新 vs. 全 Session Trial 預測表現熱圖：
   - X 軸 (橫向)：全 Session 所有 Trial（標註為 "Session Trials"）。
   - Y 軸 (縱向)：模型更新時間步（標註為 "Model Update Timeline"）。
   - 色彩編碼 (色盲友善 + Cross-Entropy 損失深淺)：
     * 藍色系 (Correct, 正確預測)：Cross-Entropy 越低 (信心度越高) 呈現深寶石藍，接近決策邊界呈現淺冰藍。
     * 橘紅系 (Incorrect, 錯誤預測)：Cross-Entropy 越高 (嚴重錯誤) 呈現深朱紅/深橘，接近決策邊界呈現淺粉橘。
     * 決策邊界 (Chance / 0.5)：中性淺灰色。
   - 視覺化引導線與標記：
     * 從橘色更新點延伸貫穿全圖的橫向引導虛線，清晰勾勒出每個更新模型的作用區間。
     * 右側子圖：同步展示隨模型更新的全 Session 整體平均準確率 (Global Acc %) 演化。
     * 圖例位置置於右上空白處，完全避免遮擋熱圖資訊。

3. 額外自動生成 $K \times N$ 精簡版模型快照對比圖 (Model Snapshots Heatmap)：
   - 縱軸標註各模型快照更新點（例如：Initial Model (M0)、M1 (Trial 4)、M2 (Trial 16)...）。
   - 一目了然每次更新前後對全 Session 各 Trial 預測的翻轉與改進。
================================================================================
"""

import os
import sys
import copy
import random
import argparse
from datetime import datetime
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.stats import mode

import matplotlib
matplotlib.use('Agg')  # 無 GUI 後端，適合批次生成與伺服器環境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 引入自定義模組
import global_value as global_value
import config as config
from Models import SCCNet
from MI_train import OnlineCalibrationTrainer


# ==============================================================================
# 0. 受試者對照表與實驗設計條件
# ==============================================================================
ALL_SUBJECT_IDS = [
    35, 37, 38, 40, 41, 42, 43, 44, 45, 47,
    48, 50, 51, 52, 54, 55, 57, 58, 63, 64,
    65, 68, 69, 70
]

SUBJECT_MAP = {
    35: "S1", 37: "S2", 38: "S3", 40: "S4", 41: "S5",
    42: "S6", 43: "S7", 44: "S8", 45: "S9", 47: "S10",
    48: "S11", 50: "S12", 51: "S13", 52: "S14", 54: "S15",
    55: "S16", 57: "S17", 58: "S18", 63: "S19", 64: "S20",
    65: "S21", 68: "S22", 69: "S23", 70: "S24"
}

# 論文官方 Crossover 實驗條件指派表
SUBJECT_CONDITIONS = {
    35: {'s1': 'B', 's2': 'A'},
    37: {'s1': 'A', 's2': 'B'},
    38: {'s1': 'B', 's2': 'A'},
    40: {'s1': 'B', 's2': 'A'},
    41: {'s1': 'B', 's2': 'A'},
    42: {'s1': 'A', 's2': 'B'},
    43: {'s1': 'A', 's2': 'B'},
    44: {'s1': 'B', 's2': 'A'},
    45: {'s1': 'B', 's2': 'A'},
    47: {'s1': 'B', 's2': 'A'},
    48: {'s1': 'B', 's2': 'A'},
    50: {'s1': 'A', 's2': 'B'},
    51: {'s1': 'B', 's2': 'A'},
    52: {'s1': 'B', 's2': 'A'},
    54: {'s1': 'B', 's2': 'A'},
    55: {'s1': 'A', 's2': 'B'},
    57: {'s1': 'B', 's2': 'A'},
    58: {'s1': 'A', 's2': 'B'},
    63: {'s1': 'A', 's2': 'B'},
    64: {'s1': 'A', 's2': 'B'},
    65: {'s1': 'A', 's2': 'B'},
    68: {'s1': 'A', 's2': 'B'},
    69: {'s1': 'A', 's2': 'B'},
    70: {'s1': 'A', 's2': 'B'}
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except UnicodeEncodeError:
                try:
                    f.write(obj.encode('utf-8', errors='replace').decode('utf-8'))
                    f.flush()
                except Exception:
                    f.write(obj.encode('ascii', errors='replace').decode('ascii'))
                    f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def tee_log(log_file=None):
    if log_file is None:
        log_file = f"online_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
            print(f"[INFO] Output log saved to: {log_file}")
            return result
        return wrapper
    return decorator


# ==============================================================================
# 1. 訊號預處理與 Dataset 工具
# ==============================================================================
def bandpass(data, fs=500, low=1, high=40):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)


def extract_windows_for_trial(x_trial, segment_len=500, stride=100, fs=500):
    """
    從單一 Trial 訊號 (channels, samples) 切出滑動視窗並濾波
    回傳: Tensor of shape (num_windows, 1, channels, segment_len)
    """
    x_trial_t = x_trial.T  # (samples, channels)
    if x_trial_t.shape[0] < segment_len:
        pad_len = segment_len - x_trial_t.shape[0]
        x_trial_t = np.pad(x_trial_t, ((0, pad_len), (0, 0)), mode='edge')

    windows = []
    for start in range(0, x_trial_t.shape[0] - segment_len + 1, stride):
        data_window = x_trial_t[start:start + segment_len, :]
        window_filtered = bandpass(data_window - np.mean(data_window, axis=0, keepdims=True), fs=fs)
        windows.append(window_filtered)

    if not windows:
        data_window = x_trial_t[:segment_len, :]
        window_filtered = bandpass(data_window - np.mean(data_window, axis=0, keepdims=True), fs=fs)
        windows.append(window_filtered)

    windows_np = np.stack(windows)
    windows_np = np.transpose(windows_np, (0, 2, 1))
    return torch.tensor(windows_np, dtype=torch.float32).unsqueeze(1)


def prepare_buffer_dataset(x_buffer, y_buffer, correctness_buffer, segment_len=500, stride=100):
    """
    將最新收集到的 Trial 轉換為 TensorDataset (包含 failures 權重判斷)
    """
    augmented_segments = []
    augmented_labels = []
    augmented_failures = []

    for i in range(len(x_buffer)):
        x_trial = x_buffer[i]
        label = y_buffer[i]
        is_fail = not correctness_buffer[i]

        x_trial_t = x_trial.T
        if x_trial_t.shape[0] < segment_len:
            pad_len = segment_len - x_trial_t.shape[0]
            x_trial_t = np.pad(x_trial_t, ((0, pad_len), (0, 0)), mode='edge')

        for start in range(0, x_trial_t.shape[0] - segment_len + 1, stride):
            data_window = x_trial_t[start:start + segment_len, :]
            window_filtered = bandpass(data_window - np.mean(data_window, axis=0, keepdims=True))
            augmented_segments.append(window_filtered)
            augmented_labels.append(label)
            augmented_failures.append(1.0 if is_fail else 0.0)

    if not augmented_segments:
        return None

    data_x = np.transpose(np.stack(augmented_segments), (0, 2, 1))
    X = torch.tensor(data_x).unsqueeze(1).float()
    y = F.one_hot(torch.tensor(augmented_labels).long(), num_classes=2).float()
    f = torch.tensor(augmented_failures).float()

    return TensorDataset(X, y, f)


# ==============================================================================
# 2. 色盲友善色彩映射與 Cross-Entropy 損失計算
# ==============================================================================
def create_colorblind_ce_colormap():
    """
    建立色盲友善 (Colorblind-Safe) 且結合 Cross-Entropy 損失深淺的色彩映射：
    - 正向 (Correct, 預測正確)：寶石藍系 (Deep Sapphire Blue -> Soft Ice Blue)
      Cross-Entropy 越低 (預測信心度越強、Loss 越小)，顏色越深藍。
    - 負向 (Incorrect, 預測錯誤)：朱紅/琥珀橘系 (Deep Vermilion/Rust -> Soft Peach)
      Cross-Entropy 越高 (預測嚴重偏離真實標籤、Loss 越大)，顏色越深橘紅。
    - 決策中界線 (Decision Boundary / Chance 0.5)：中性淺灰色 (#F0F0F0)
    """
    colors = [
        (0.00, "#7F2704"),  # 嚴重錯誤 (Very high CE loss, highly confident wrong)
        (0.20, "#BD0026"),  # 明顯錯誤 (High CE loss)
        (0.35, "#F16913"),  # 中度錯誤 (Moderate CE loss)
        (0.48, "#FDD0A2"),  # 臨界錯誤 (Marginal error, close to boundary)
        (0.50, "#F2F2F2"),  # 決策邊界 (Decision boundary, p=0.5, CE=0.693)
        (0.52, "#C6DBEF"),  # 臨界正確 (Marginal correct, close to boundary)
        (0.65, "#6BAED6"),  # 中度正確 (Moderate CE confidence)
        (0.80, "#2171B5"),  # 高度正確 (Low CE loss, high confidence)
        (1.00, "#08306B")   # 極高確信正確 (Minimal CE loss, peak confidence)
    ]
    positions = [c[0] for c in colors]
    color_codes = [c[1] for c in colors]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Colorblind_CE_Colormap",
        list(zip(positions, color_codes)),
        N=512
    )
    return cmap


def evaluate_trial_prediction(model, windows_tensor, true_label, device):
    """
    評估單一 Trial 在給定模型下的預測輸出、多數決類別、Cross-Entropy 損失與真實類別機率
    """
    model.eval()
    with torch.no_grad():
        w_tensor = windows_tensor.to(device)
        outputs = model(w_tensor)  # (K, n_classes)
        probs = F.softmax(outputs, dim=1)  # (K, n_classes)
        mean_prob = probs.mean(dim=0).cpu().numpy()  # (n_classes,)
        preds = outputs.argmax(dim=1).cpu().numpy()

    final_pred = int(mode(preds, keepdims=False)[0])
    is_correct = bool(final_pred == true_label)

    # 計算對 True Label 的 Cross-Entropy Loss
    p_true = float(np.clip(mean_prob[true_label], 1e-7, 1.0 - 1e-7))
    ce_loss = float(-np.log(p_true))
    score = p_true

    return final_pred, is_correct, ce_loss, score


def evaluate_all_trials_for_matrix(model, all_windows_tensors, all_labels, device):
    """
    快速批次評估當前模型在全 Session 所有 Trials 上的表現
    回傳: (preds, correctness_arr, ce_loss_arr, score_arr)
    """
    n_trials = len(all_labels)
    preds = np.zeros(n_trials, dtype=np.int32)
    correctness = np.zeros(n_trials, dtype=bool)
    ce_losses = np.zeros(n_trials, dtype=np.float32)
    scores = np.zeros(n_trials, dtype=np.float32)

    for j in range(n_trials):
        p, corr, ce, sc = evaluate_trial_prediction(
            model=model,
            windows_tensor=all_windows_tensors[j],
            true_label=all_labels[j],
            device=device
        )
        preds[j] = p
        correctness[j] = corr
        ce_losses[j] = ce
        scores[j] = sc

    return preds, correctness, ce_losses, scores


# ==============================================================================
# 3. 核心模擬函數 (支援動態 Run 偵測、6-Run 自適應與快照記錄)
# ==============================================================================
def simulate_online_experiment(
    subject_id,
    session_name,
    condition,
    base_dir,
    pretrain_model_path,
    params,
    device='cuda',
    update_freq=4,
    num_epochs=4,
    batch_size=8,
    lr=1e-3,
    use_val=True,
    demo_mode=False
):
    """
    模擬單一受試者之單一 Session 的線上實驗，並記錄 NxN 模型更新預測矩陣
    """
    print(f"\n================================================================================")
    print(f"[{subject_id} - {session_name}] 開始線上模擬 | 條件: {condition} | Demo: {demo_mode}")
    print(f"================================================================================")

    # 確保暫存 Checkpoint 目錄存在
    os.makedirs(config.EEG_CHECKPOINT_TMP_BASE_FILE, exist_ok=True)
    os.makedirs(config.EEG_CHECKPOINT_MAIN_BASE_FILE, exist_ok=True)

    # 重置全域 Replay Buffer
    global_value.replay_buffer = {0: [], 1: []}
    if hasattr(global_value, 'replay_buffer_val'):
        global_value.replay_buffer_val = {0: [], 1: []}
    config.REPLAY_BUFFER_LIMIT = 320

    # 載入 Pretrain Model
    model = SCCNet(**params).to(device)
    if os.path.exists(pretrain_model_path) and not demo_mode:
        checkpoint = torch.load(pretrain_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[{config.TAGS.INFO.value}] 成功載入預訓練權重: {pretrain_model_path}")
    else:
        print(f"[{config.TAGS.WARNING.value}] 未載入外部預訓練權重，使用隨機初始化模型 (Demo/測試模式)")

    # 初始化 Online Trainer
    trainer = OnlineCalibrationTrainer(
        model_class=SCCNet,
        model_kwargs=params,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        ft=True,
        use_val=use_val
    )
    trainer.model = model

    # --------------------------------------------------------------------------
    # A. 依據硬碟實際檔案動態載入各 Run (自動辨識 6-Run 或 7-Run)
    # --------------------------------------------------------------------------
    all_x_trials = []
    all_y_trials = []
    run_trial_counts = []
    run_trial_offsets = [0]
    loaded_run_indices = []

    for run_idx in range(1, 8):
        if demo_mode:
            # Demo 模式：若受試者為 S23 (ID 69) 或 S24，預設示範 6-Run [1, 2, 4, 5, 6, 7]
            if str(subject_id) in ['69', '68', '65', '64'] and run_idx == 3:
                continue  # 跳過 Run 3，形成 6 個 Run 的案例
            n_demo_trials = 40
            t_samples = 1500
            x_run = np.random.randn(n_demo_trials, params['channels'], t_samples).astype(np.float32)
            y_run = np.random.choice([0, 1], size=n_demo_trials).astype(np.int64)
        else:
            run_path = os.path.join(base_dir, str(subject_id), session_name, f"run{run_idx}", "mi.pt")
            if not os.path.exists(run_path):
                print(f"  [Info] Run {run_idx} 檔案不存在 (可能為 2-Run Calibration)，跳過此 Run")
                continue
            data = torch.load(run_path, weights_only=False)
            x_run = data['x_data']
            y_run = data['y_data']

        loaded_run_indices.append(run_idx)
        for tid in range(len(y_run)):
            all_x_trials.append(x_run[tid])
            all_y_trials.append(int(y_run[tid]))
        run_trial_counts.append(len(y_run))
        run_trial_offsets.append(run_trial_offsets[-1] + len(y_run))

    total_trials = len(all_y_trials)
    if total_trials == 0:
        print(f"  [Error] 沒有載入到任何 Trial 數據！")
        return None

    # 自動判斷 6-Run 映射 (若載入 6 個 Run 且索引為 1..6，顯示編號對應為 1, 2, 4, 5, 6, 7)
    if len(loaded_run_indices) == 6 and loaded_run_indices == [1, 2, 3, 4, 5, 6]:
        display_run_indices = [1, 2, 4, 5, 6, 7]
    else:
        display_run_indices = list(loaded_run_indices)

    run_names_str = ", ".join([f"Run {r}" for r in display_run_indices])
    print(f"  --> 實際載入 {len(loaded_run_indices)} 個 Run: [{run_names_str}] | 總 Trial 數: {total_trials}")

    # 預先對全 Session 的所有 Trial 切出 sliding windows
    print(f"  --> 正在預處理並提取全 Session {total_trials} 個 Trial 的特徵視窗...")
    precomputed_windows = []
    for i in range(total_trials):
        w_tensor = extract_windows_for_trial(
            x_trial=all_x_trials[i],
            segment_len=params['samples'],
            stride=100,
            fs=params['sfreq']
        )
        precomputed_windows.append(w_tensor)

    # --------------------------------------------------------------------------
    # B. 初始化 NxN 熱圖矩陣與模型快照追蹤
    # --------------------------------------------------------------------------
    matrix_correctness = np.zeros((total_trials, total_trials), dtype=bool)
    matrix_ce_loss = np.zeros((total_trials, total_trials), dtype=np.float32)
    matrix_score = np.zeros((total_trials, total_trials), dtype=np.float32)

    update_event_indices = []
    run_accuracies = []

    # 初始模型評估 (Model Snapshot 0: 預訓練初始模型 M0)
    curr_preds, curr_corr, curr_ce, curr_sc = evaluate_all_trials_for_matrix(
        model=trainer.model,
        all_windows_tensors=precomputed_windows,
        all_labels=all_y_trials,
        device=device
    )

    # 記錄模型快照 (供 K x N 快照熱圖使用，移除 @ 符號，改用簡潔格式)
    model_snapshots = [curr_sc.copy()]
    snapshot_labels = ["Initial Model (M0)"]
    snapshot_update_trials = [0]

    # --------------------------------------------------------------------------
    # C. 線上即時模擬迴圈 (逐 Trial 進行)
    # --------------------------------------------------------------------------
    recent_x_buffer = []
    recent_y_buffer = []
    recent_correctness_buffer = []
    recent_N_preds = []
    recent_N_labels = []

    current_loaded_idx = 0  # index in loaded_run_indices
    run_start_idx = 0
    run_correct_count = 0

    for t in range(total_trials):
        # 判斷當前 Trial 屬於第幾個載入的 Run
        for r_i in range(len(run_trial_counts)):
            if run_trial_offsets[r_i] <= t < run_trial_offsets[r_i + 1]:
                if current_loaded_idx != r_i:
                    # 進入新的 Run，結算前一個 Run 的 Acc
                    run_acc = run_correct_count / max(1, (t - run_start_idx))
                    run_accuracies.append(run_acc)
                    prev_run_num = display_run_indices[current_loaded_idx]
                    print(f"  --> Run {prev_run_num} 結束 | 模式: {'Adaptive' if is_adaptive_run else 'Static'} | 準確率: {run_acc:.3f}")
                    # 重置 Run 相關計數
                    current_loaded_idx = r_i
                    run_start_idx = t
                    run_correct_count = 0
                break

        # 取得當前 Run 的真實顯示編號 (例如 1, 2, 4, 5, 6, 7)
        curr_run_num = display_run_indices[current_loaded_idx]

        # 判斷當前 Run 是否為 Adaptive 模式
        # 6-Run 時：前 2 個 Run (Run 1, Run 2) 為 Calibration/Adaptive，後 4 個 Run (Run 4,5,6,7) 為 Static
        # 7-Run 時：前 3 個 Run (Run 1, Run 2, Run 3) 為 Calibration/Adaptive，後 4 個 Run (Run 4,5,6,7) 為 Static
        if condition.upper() in ['B', 'ADAPTIVE']:
            is_adaptive_run = True
        elif condition.upper() == 'A':
            is_adaptive_run = (current_loaded_idx < (2 if len(display_run_indices) == 6 else 3))
        elif condition.upper() == 'STATIC':
            is_adaptive_run = False
        else:
            is_adaptive_run = (current_loaded_idx < (2 if len(display_run_indices) == 6 else 3))

        # 線上推論 (使用當前最新的模型)
        online_pred = curr_preds[t]
        is_correct = curr_corr[t]

        if is_correct:
            run_correct_count += 1

        recent_N_preds.append(online_pred)
        recent_N_labels.append(all_y_trials[t])

        recent_x_buffer.append(all_x_trials[t])
        recent_y_buffer.append(all_y_trials[t])
        recent_correctness_buffer.append(is_correct)

        # 判斷是否需要觸發線上更新
        trial_in_run = t - run_start_idx + 1

        if trial_in_run % update_freq == 0:
            recent_acc = np.mean(np.array(recent_N_preds) == np.array(recent_N_labels))
            current_run_acc = run_correct_count / trial_in_run

            # 更新條件判斷：處於 Adaptive Run 且近 N 次 < 0.75 且 當前 Run Acc < 0.75
            if is_adaptive_run and (recent_acc < 0.75) and (current_run_acc < 0.75):
                update_dataset = prepare_buffer_dataset(recent_x_buffer, recent_y_buffer, recent_correctness_buffer)
                if update_dataset is not None and len(update_dataset) > 0:
                    loss = trainer.online_train(update_dataset)

                    # 載入更新後的最佳權重
                    latest_checkpoint = f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}ft-best.pth"
                    if os.path.exists(latest_checkpoint):
                        ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
                        trainer.model.load_state_dict(ckpt['model_state_dict'])

                    update_event_indices.append(t)

                    # 模型權重已更新 -> 重新評估全 Session 所有 Trial 的預測矩陣列向量
                    curr_preds, curr_corr, curr_ce, curr_sc = evaluate_all_trials_for_matrix(
                        model=trainer.model,
                        all_windows_tensors=precomputed_windows,
                        all_labels=all_y_trials,
                        device=device
                    )

                    model_snapshots.append(curr_sc.copy())
                    snapshot_labels.append(f"M{len(model_snapshots)-1} (Trial {t+1})")
                    snapshot_update_trials.append(t + 1)

                    print(f"    [⚡ Update M{len(model_snapshots)-1}] Run {curr_run_num} Trial {trial_in_run} (總 Trial {t+1}) | "
                          f"近{update_freq}次 Acc: {recent_acc:.2f} | 目前 Run Acc: {current_run_acc:.2f} | Loss: {loss:.4f}")

            # 清空 update 頻率暫存器
            recent_x_buffer.clear()
            recent_y_buffer.clear()
            recent_correctness_buffer.clear()
            recent_N_preds.clear()
            recent_N_labels.clear()

        # 將此時間步的模型預測紀錄填入 NxN 矩陣的第 t 列 (Row t)
        matrix_correctness[t, :] = curr_corr
        matrix_ce_loss[t, :] = curr_ce
        matrix_score[t, :] = curr_sc

    # 結算最後一個 Run
    last_run_len = total_trials - run_start_idx
    run_acc = run_correct_count / max(1, last_run_len)
    run_accuracies.append(run_acc)
    last_run_num = display_run_indices[-1]
    print(f"  --> Run {last_run_num} 結束 | 模式: {'Adaptive' if is_adaptive_run else 'Static'} | 準確率: {run_acc:.3f}")

    # 計算整體線上準確率 (對角線上的預測結果)
    online_diag_correct = np.diag(matrix_correctness)
    overall_online_acc = np.mean(online_diag_correct)
    print(f"\n✅ [{subject_id} - {session_name}] 模擬完成 | 總更新次數: {len(update_event_indices)} | 線上實時 Acc: {overall_online_acc:.3f}")

    return {
        'subject_id': subject_id,
        'session_name': session_name,
        'condition': condition,
        'total_trials': total_trials,
        'loaded_run_indices': loaded_run_indices,
        'display_run_indices': display_run_indices,
        'run_trial_counts': run_trial_counts,
        'run_trial_offsets': run_trial_offsets,
        'update_event_indices': update_event_indices,
        'run_accuracies': run_accuracies,
        'overall_online_acc': overall_online_acc,
        'matrix_correctness': matrix_correctness,
        'matrix_ce_loss': matrix_ce_loss,
        'matrix_score': matrix_score,
        'model_snapshots': np.array(model_snapshots),
        'snapshot_labels': snapshot_labels,
        'snapshot_update_trials': snapshot_update_trials
    }


# ==============================================================================
# 4. 全幅演進熱圖繪製模組 (優化排版、去冗餘、橘點延伸橫線、右上圖例、純淨標題)
# ==============================================================================
def plot_online_simulation_heatmap(sim_results, output_dir):
    """
    繪製 NxN 模型更新 vs. 全 Session Trial 預測表現熱圖 (簡潔標籤 + 橘點橫向導引線)
    """
    sub_id = sim_results['subject_id']
    sess_name = sim_results['session_name']
    cond = sim_results['condition']
    total_trials = sim_results['total_trials']
    offsets = sim_results['run_trial_offsets']
    display_runs = sim_results['display_run_indices']
    update_pts = sim_results['update_event_indices']
    online_acc = sim_results['overall_online_acc']
    matrix_score = sim_results['matrix_score']
    matrix_corr = sim_results['matrix_correctness']
    run_accs = sim_results['run_accuracies']

    sub_label = SUBJECT_MAP.get(int(sub_id) if str(sub_id).isdigit() else sub_id, f"S_{sub_id}")

    # 建立輸出目錄
    sub_dir = os.path.join(output_dir, str(sub_id))
    os.makedirs(sub_dir, exist_ok=True)

    # 建立色盲友善 Colormap
    cmap_cb = create_colorblind_ce_colormap()

    # 計算全域統計：每一列 (Model Snapshot) 在全 Session 所有 Trial 上的平均準確率
    row_global_acc = np.mean(matrix_corr, axis=1) * 100.0

    # --------------------------------------------------------------------------
    # 建立單列複合畫布 (主熱圖 + 右側演進曲線，已移除頂部副圖與灰色副標題)
    # --------------------------------------------------------------------------
    fig, (ax_main, ax_right) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(14.0, 10.5),
        dpi=200,
        gridspec_kw={
            'width_ratios': [10.0, 2.0],
            'left': 0.08,
            'right': 0.88,
            'bottom': 0.08,
            'top': 0.90,
            'wspace': 0.03
        }
    )

    # --------------------------------------------------------------------------
    # 1. 中央主熱圖: NxN 模型更新 vs. Session Trial 預測矩陣
    # --------------------------------------------------------------------------
    im = ax_main.imshow(
        matrix_score,
        cmap=cmap_cb,
        origin='upper',
        aspect='auto',
        extent=[0.5, total_trials + 0.5, total_trials + 0.5, 0.5],
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest'
    )

    # 繪製 Run 邊界格線 (橫線與豎線) 與交錯淡色背景
    for r_i in range(len(offsets) - 1):
        x_start = offsets[r_i] + 0.5
        x_end = offsets[r_i + 1] + 0.5
        if r_i % 2 == 1:
            ax_main.axvspan(x_start, x_end, color='#F8F9FA', alpha=0.15, zorder=0)

    for r_i in range(1, len(offsets) - 1):
        boundary = offsets[r_i] + 0.5
        ax_main.axvline(boundary, color='#333333', linestyle='--', linewidth=1.1, alpha=0.65)
        ax_main.axhline(boundary, color='#333333', linestyle='--', linewidth=1.1, alpha=0.65)

    # 從橘色更新點延伸貫穿熱圖與右側圖的橫向導引虛線
    if len(update_pts) > 0:
        pts_y = np.array(update_pts) + 1.0

        for y_val in pts_y:
            ax_main.axhline(y_val, color='#FF8C00', linestyle='--', linewidth=0.9, alpha=0.75, zorder=4)
            ax_right.axhline(y_val, color='#FF8C00', linestyle='--', linewidth=0.9, alpha=0.75, zorder=4)

        # 在主圖左側邊緣點出金色圓點
        ax_main.scatter(
            np.ones_like(pts_y) * 2.0,
            pts_y,
            color='#FF8C00',
            edgecolor='#FFFFFF',
            s=45,
            marker='o',
            label=f'Model Update ({len(update_pts)})',
            zorder=6
        )

    # 設定坐標軸刻度與標籤 (自動適應 6-Run 或 7-Run)
    run_centers = [(offsets[i] + offsets[i + 1]) / 2.0 for i in range(len(offsets) - 1)]
    run_labels_x = [f"Run {display_runs[i]}\n({run_accs[i]*100:.1f}%)" if i < len(run_accs) else f"Run {display_runs[i]}" for i in range(len(run_centers))]
    run_labels_y = [f"Run {display_runs[i]}" for i in range(len(run_centers))]

    ax_main.set_xticks(run_centers)
    ax_main.set_xticklabels(run_labels_x, fontsize=9.5, fontweight='bold')
    ax_main.set_yticks(run_centers)
    ax_main.set_yticklabels(run_labels_y, fontsize=9.5, fontweight='bold')

    # X 軸只寫 "Session Trials"，不加後綴
    ax_main.set_xlabel("Session Trials", fontsize=11, fontweight='bold', labelpad=8)
    ax_main.set_ylabel("Model Update Timeline", fontsize=11, fontweight='bold', labelpad=8)

    # --------------------------------------------------------------------------
    # 2. 右側邊界圖: 隨模型更新的全 Session 整體平均準確率演化曲線
    # --------------------------------------------------------------------------
    trials_y = np.arange(1, total_trials + 1)
    ax_right.plot(row_global_acc, trials_y, color='#1F77B4', linewidth=2.0, label='Global Acc %')
    ax_right.axvline(75.0, color='#D9534F', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_right.axvline(50.0, color='#6C757D', linestyle=':', linewidth=1.0, alpha=0.7)

    if len(update_pts) > 0:
        ax_right.scatter(
            row_global_acc[update_pts],
            np.array(update_pts) + 1.0,
            color='#FF8C00',
            edgecolor='#FFFFFF',
            s=40,
            zorder=5
        )

    ax_right.set_ylim(total_trials + 0.5, 0.5)
    ax_right.set_xlim(30, 100)
    ax_right.set_xlabel("Global Acc %", fontsize=9.5, fontweight='bold', color='#333333')
    ax_right.tick_params(axis='y', labelleft=False, length=2)
    ax_right.tick_params(axis='x', labelsize=8.5)
    ax_right.grid(True, linestyle=':', alpha=0.4)

    # --------------------------------------------------------------------------
    # 3. 色彩條 (Colorbar)
    # --------------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.05, 0.25, 0.50, 0.75, 0.95])
    cbar.set_ticklabels([
        "Severe Error\n(High CE Loss)",
        "Incorrect\n(Low CE)",
        "Decision\nBoundary (0.5)",
        "Correct\n(Low CE)",
        "Peak Confident\n(Minimal CE)"
    ], fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Prediction Confidence & Cross-Entropy Depth\n[Colorblind: Vermilion (Wrong) ↔ Blue (Correct)]", fontsize=9, fontweight='bold', labelpad=10)

    # --------------------------------------------------------------------------
    # 4. 圖表標題與右上圖例 (不標示 ID、不寫 Trial 數、不遮擋熱圖、Condition A -> Static, B -> Adaptive)
    # --------------------------------------------------------------------------
    cond_display = "Static" if str(cond).upper() in ['A', 'STATIC'] else "Adaptive" if str(cond).upper() in ['B', 'ADAPTIVE'] else str(cond)
    title_main = f"Online BCI Simulation Trial-Model Evolution Heatmap | {sub_label} - Session {sess_name.upper()} ({cond_display})"
    fig.suptitle(title_main, fontsize=13, fontweight='bold', y=0.96, color='#111111')

    # 圖例置於右上空白處 (橫向排版)
    handles = [
        mpatches.Patch(color='#08306B', label='Correct (Low CE)'),
        mpatches.Patch(color='#7F2704', label='Incorrect (High CE)'),
        mpatches.Patch(color='#F2F2F2', label='Boundary (p=0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8C00', markersize=7, label=f'Model Update ({len(update_pts)})')
    ]
    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.88, 0.94),
        ncol=4,
        fontsize=8.5,
        framealpha=0.95,
        facecolor='#FFFFFF',
        edgecolor='#D0D0D0'
    )

    out_filename = f"{sub_id}_{sess_name}_{cond}_model_update_trial_heatmap.png"
    out_filepath = os.path.join(sub_dir, out_filename)
    plt.savefig(out_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  🖼️  主熱圖已儲存至: {out_filepath}")

    # 同步儲存純矩陣數據
    npz_filename = f"{sub_id}_{sess_name}_{cond}_matrix_data.npz"
    npz_filepath = os.path.join(sub_dir, npz_filename)
    np.savez_compressed(
        npz_filepath,
        matrix_correctness=matrix_corr,
        matrix_ce_loss=sim_results['matrix_ce_loss'],
        matrix_score=matrix_score,
        update_points=np.array(update_pts),
        loaded_run_indices=np.array(sim_results['loaded_run_indices']),
        display_run_indices=np.array(display_runs),
        run_accuracies=np.array(run_accs),
        overall_online_acc=online_acc,
        subject_id=sub_id,
        session_name=sess_name,
        condition=cond
    )
    print(f"  💾 矩陣數值已壓縮儲存至: {npz_filepath}")

    # 額外繪製精簡版模型快照對比圖 (K x N Snapshot Heatmap)
    plot_model_snapshots_heatmap(sim_results, output_dir=output_dir)

    return out_filepath


# ==============================================================================
# 5. K x N 模型快照對比圖 (Model Snapshots Heatmap，不標示 ID、簡潔標籤)
# ==============================================================================
def plot_model_snapshots_heatmap(sim_results, output_dir):
    """
    繪製 K x N 精簡模型快照對比圖 (每一列為一次具體更新後的模型版本)
    """
    sub_id = sim_results['subject_id']
    sess_name = sim_results['session_name']
    cond = sim_results['condition']
    total_trials = sim_results['total_trials']
    offsets = sim_results['run_trial_offsets']
    display_runs = sim_results['display_run_indices']
    snapshots = sim_results['model_snapshots']  # (K, N)
    labels = sim_results['snapshot_labels']

    if len(snapshots) <= 1:
        return None  # 沒有觸發任何更新，無需獨立快照圖

    sub_label = SUBJECT_MAP.get(int(sub_id) if str(sub_id).isdigit() else sub_id, f"S_{sub_id}")
    sub_dir = os.path.join(output_dir, str(sub_id))
    os.makedirs(sub_dir, exist_ok=True)

    cmap_cb = create_colorblind_ce_colormap()

    fig_height = max(4.0, 0.45 * len(snapshots) + 2.5)
    fig, ax = plt.subplots(figsize=(13.0, fig_height), dpi=200)

    im = ax.imshow(
        snapshots,
        cmap=cmap_cb,
        origin='upper',
        aspect='auto',
        extent=[0.5, total_trials + 0.5, len(snapshots) - 0.5, -0.5],
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest'
    )

    # 繪製 Run 邊界格線
    for r_i in range(1, len(offsets) - 1):
        boundary = offsets[r_i] + 0.5
        ax.axvline(boundary, color='#333333', linestyle='--', linewidth=1.1, alpha=0.7)

    # 繪製每列模型間的水平分隔線
    for k in range(1, len(snapshots)):
        ax.axhline(k - 0.5, color='#FFFFFF', linestyle='-', linewidth=1.2, alpha=0.9)

    run_centers = [(offsets[i] + offsets[i + 1]) / 2.0 for i in range(len(offsets) - 1)]
    run_labels_x = [f"Run {display_runs[i]}" for i in range(len(run_centers))]

    ax.set_xticks(run_centers)
    ax.set_xticklabels(run_labels_x, fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(len(snapshots)))
    ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold')

    # X 軸標籤只寫 "Session Trials"
    ax.set_xlabel("Session Trials", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Distinct Model Checkpoints", fontsize=11, fontweight='bold', labelpad=8)

    # 標題不標示 ID，Condition A -> Static, B -> Adaptive
    cond_display = "Static" if str(cond).upper() in ['A', 'STATIC'] else "Adaptive" if str(cond).upper() in ['B', 'ADAPTIVE'] else str(cond)
    title = f"Model Update Discrete Snapshots Heatmap | {sub_label} - Session {sess_name.upper()} ({cond_display})"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

    # 色彩條
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0.05, 0.50, 0.95])
    cbar.set_ticklabels(["Severe Error", "Boundary", "Peak Correct"], fontsize=8)

    out_filename = f"{sub_id}_{sess_name}_{cond}_model_snapshots_heatmap.png"
    out_filepath = os.path.join(sub_dir, out_filename)
    plt.savefig(out_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  🖼️  精簡模型快照圖已儲存至: {out_filepath}")
    return out_filepath


# ==============================================================================
# 6. 主程式入口與命令列解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="BCI Online Simulation & Model-Update vs Trial Heatmap Generator"
    )
    parser.add_argument(
        "--data_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
        help="Path to EEG dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./simulation_heatmaps",
        help="Directory to save generated heatmaps and matrix data"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="checkpoints/best_model_24_pilot_Train.pth",
        help="Path to pretrained SCCNet model checkpoint (.pth)"
    )
    parser.add_argument(
        "--subject", type=str, default="all",
        help="Specific subject ID to run (e.g., '35', '69') or 'all'"
    )
    parser.add_argument(
        "--session", type=str, default="all",
        help="Specific session to run ('s1', 's2') or 'all'"
    )
    parser.add_argument(
        "--condition", type=str, default="auto",
        help="Condition mode: 'auto', 'A', 'B', 'Static', or 'Adaptive'"
    )
    parser.add_argument(
        "--update_freq", type=int, default=4,
        help="Trial update evaluation frequency (default: 4)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=4,
        help="Fine-tuning epochs per update (default: 4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Fine-tuning batch size (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Online fine-tuning learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--use_val", action="store_true", default=True,
        help="Whether to use validation buffer"
    )
    parser.add_argument(
        "--demo", action="store_true", default=False,
        help="Run in demo synthetic mode"
    )
    return parser.parse_args()


@tee_log(f"online_simulation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config.TAGS.INFO.value}] 運算設備: {device}")
    print(f"[{config.TAGS.INFO.value}] 輸出目錄: {os.path.abspath(args.output_dir)}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.subject.lower() == 'all':
        target_subjects = ALL_SUBJECT_IDS
    else:
        try:
            target_subjects = [int(args.subject)]
        except ValueError:
            target_subjects = [args.subject]

    if args.session.lower() == 'all':
        target_sessions = ['s1', 's2']
    else:
        target_sessions = [args.session.lower()]

    channel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    params = dict(
        samples=500,
        channels=len(channel_index),
        n_classes=2,
        sfreq=500
    )

    all_results_summary = []

    is_demo = args.demo
    if not is_demo and not os.path.exists(args.data_dir):
        print(f"[{config.TAGS.WARNING.value}] 找不到資料集路徑: {args.data_dir}")
        print(f"[{config.TAGS.INFO.value}] 自動切換至 --demo 擬真模擬模式以生成示範熱圖！")
        is_demo = True

    for sub_id in target_subjects:
        for sess in target_sessions:
            if args.condition.lower() == 'auto':
                sub_int = int(sub_id) if str(sub_id).isdigit() else sub_id
                cond = SUBJECT_CONDITIONS.get(sub_int, {}).get(sess, 'B')
            else:
                cond = args.condition.upper()

            try:
                sim_res = simulate_online_experiment(
                    subject_id=sub_id,
                    session_name=sess,
                    condition=cond,
                    base_dir=args.data_dir,
                    pretrain_model_path=args.pretrain_model_path,
                    params=params,
                    device=device,
                    update_freq=args.update_freq,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    use_val=args.use_val,
                    demo_mode=is_demo
                )

                if sim_res is not None:
                    img_path = plot_online_simulation_heatmap(sim_res, output_dir=args.output_dir)

                    all_results_summary.append({
                        'subject': sub_id,
                        'sub_label': SUBJECT_MAP.get(int(sub_id) if str(sub_id).isdigit() else sub_id, f"S_{sub_id}"),
                        'session': sess,
                        'condition': cond,
                        'online_acc': sim_res['overall_online_acc'],
                        'run_accs': sim_res['run_accuracies'],
                        'updates_count': len(sim_res['update_event_indices']),
                        'heatmap_path': img_path
                    })
            except Exception as e:
                print(f"❌ 執行 [{sub_id} - {sess}] 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n================================================================================")
    print(f"📊 全部模擬與熱圖生成完成！結果總結 (Total Sessions: {len(all_results_summary)})")
    print(f"================================================================================")
    print("| **Subject** | **ID** | **Session** | **Condition** | **Updates** | **Online Acc** | **Run Accuracies** |")
    print("| :---------: | :----: | :---------: | :-----------: | :---------: | :------------: | :-----------------: |")

    for item in all_results_summary:
        acc_str = "[" + ", ".join([f"{a:.3f}" for a in item['run_accs']]) + "]"
        print(f"| {item['sub_label']} | {item['subject']} | {item['session'].upper()} | {item['condition']} | {item['updates_count']} | **{item['online_acc']*100:.2f}%** | {acc_str} |")

    print(f"\n📁 所有熱圖與矩陣數據已保存在: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()