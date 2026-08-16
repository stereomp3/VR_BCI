import numpy as np
import torch
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps
from datetime import datetime
import sys

# === Packages for Signal Processing and Statistics ===
from scipy.signal import find_peaks
from scipy.stats import sem, ttest_ind

# ==========================================
# Utility Functions
# ==========================================
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
            print(f"✅ Output successfully saved to {log_file}")
            return result
        return wrapper
    return decorator

def cat_all_data(data_list):
    all_x_data = []
    all_y_data = []
    for i in data_list:
        if os.path.exists(i):
            train_data = torch.load(i)
        else:
            continue
        if len(train_data['x_data']) > 0:
            all_x_data.append(train_data['x_data'])
            all_y_data.append(train_data['y_data'])
    if len(all_x_data) == 0:
        return None, None
    return np.concatenate(all_x_data, axis=0), np.concatenate(all_y_data, axis=0)

# ==========================================
# EOG Event Detection (Robust MAD & Sustained Gaze)
# ==========================================
def detect_eog_events(x_data, ch_names, fs=500, blink_k=5.0, saccade_k=3.0):
    if 'Fp1' not in ch_names or 'Fp2' not in ch_names:
        return None
        
    fp1_idx = ch_names.index('Fp1')
    fp2_idx = ch_names.index('Fp2')
    n_trials = x_data.shape[0]
    
    fp1_raw = x_data[:, fp1_idx, :].copy()
    fp2_raw = x_data[:, fp2_idx, :].copy()
    
    fp1_filtered = mne.filter.filter_data(fp1_raw, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
    fp2_filtered = mne.filter.filter_data(fp2_raw, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)

    veog_all = (fp1_filtered + fp2_filtered) / 2.0
    heog_all = fp1_filtered - fp2_filtered

    def compute_robust_threshold(data_flat, k_multiplier):
        median_val = np.median(data_flat)
        mad = np.median(np.abs(data_flat - median_val))
        robust_std = mad * 1.4826 
        return median_val + k_multiplier * robust_std

    blink_th = compute_robust_threshold(veog_all.flatten(), blink_k)
    heog_median = np.median(heog_all.flatten())
    heog_robust_std = np.median(np.abs(heog_all.flatten() - heog_median)) * 1.4826
    saccade_th_pos = heog_median + saccade_k * heog_robust_std
    saccade_th_neg = heog_median - saccade_k * heog_robust_std

    blink_counts = np.zeros(n_trials)
    look_left_duration = np.zeros(n_trials)
    look_right_duration = np.zeros(n_trials)
    
    min_blink_dist = int(fs * 0.4) 
    base_samples = int(fs * 0.5)

    for i in range(n_trials):
        fp1_t = fp1_filtered[i, :]
        fp2_t = fp2_filtered[i, :]
        veog_t = veog_all[i, :]
        heog_t = heog_all[i, :]
        
        base_fp1 = np.mean(fp1_t[:base_samples])
        base_fp2 = np.mean(fp2_t[:base_samples])
        base_veog = np.mean(veog_t[:base_samples])
        base_heog = np.mean(heog_t[:base_samples])
        
        fp1_t = fp1_t - base_fp1
        fp2_t = fp2_t - base_fp2
        veog_t = veog_t - base_veog
        heog_t = heog_t - base_heog
        
        b_peaks, _ = find_peaks(veog_t, height=blink_th, distance=min_blink_dist)
        blink_counts[i] = len(b_peaks)
        
        is_left = (heog_t > saccade_th_pos) & (fp1_t > 0) & (fp2_t < 0)
        look_left_duration[i] = np.sum(is_left) / fs 
        
        is_right = (heog_t < saccade_th_neg) & (fp1_t < 0) & (fp2_t > 0)
        look_right_duration[i] = np.sum(is_right) / fs

    return {
        'Blinks': blink_counts,
        'Look Left': look_left_duration,
        'Look Right': look_right_duration
    }

def evaluate_cheating_suspicion(stats_A, stats_B, task_name_A, task_name_B):
    suspicions = []
    for event_type in ['Look Left', 'Look Right']:
        counts_A = stats_A[event_type]
        counts_B = stats_B[event_type]
        
        if np.std(counts_A) == 0 and np.std(counts_B) == 0:
            continue
            
        t_stat, p_val = ttest_ind(counts_A, counts_B, equal_var=False)
        mean_A, mean_B = np.mean(counts_A), np.mean(counts_B)
        mean_diff = abs(mean_A - mean_B)
        
        if p_val < 0.01 and mean_diff > 0.15:
            dominant_task = task_name_A if mean_A > mean_B else task_name_B
            suspicions.append(f"顯著 [{event_type}] (於 '{dominant_task}' 時頻發, p={p_val:.4f}, 差距={mean_diff:.2f}秒/trial)")
            
    return suspicions

# ==========================================
# 📊 圖表：客製化 4 個 Trial Raw EEG 比較圖
# ==========================================
def plot_custom_four_trials(x_cls_L, x_cls_R, fs, ch_names, mapped_subject, session, save_dir):
    """
    配置:
      - 總標題: 英文 Raw EEG 描述
      - 子圖標題: Trial 1, Trial 2, Trial 3, Trial 4
      - 左上 (Trial 1): T070 (Index 69) Right Hand, 藍色, 標註 1.0 ~ 2.0s
      - 右上 (Trial 2): T070 (Index 69) Left Hand, 紅色, 無標註
      - 左下 (Trial 3): T081 (Index 80) Right Hand, 藍色, 標註 0.5 ~ 1.5s
      - 右下 (Trial 4): T081 (Index 80) Left Hand, 紅色, 無標註
    """
    if 'Fp1' not in ch_names or 'Fp2' not in ch_names:
        print("⚠️ 找不到 Fp1 或 Fp2，無法繪製單一 Trial 比較圖。")
        return
        
    fp1_idx, fp2_idx = ch_names.index('Fp1'), ch_names.index('Fp2')
    time_axis = np.arange(x_cls_L.shape[2]) / fs
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=False)
    fig.suptitle(f"Raw EEG Waveforms across 4 Selected Motor Imagery Trials (Subject {mapped_subject}, Session {session})", 
                 fontsize=18, fontweight='bold', y=0.98)

    def get_filtered_trial_with_baseline(x_data, t_idx):
        if t_idx >= len(x_data):
            return None, None
        fp1_t = x_data[t_idx, fp1_idx, :]
        fp2_t = x_data[t_idx, fp2_idx, :]
        fp1_t = mne.filter.filter_data(fp1_t, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        fp2_t = mne.filter.filter_data(fp2_t, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        return fp1_t - np.mean(fp1_t), fp2_t - np.mean(fp2_t)

    # 4 個 Panel 的客製配置: (ax, title, x_data, index, color, task_name, highlight_span)
    panels_config = [
        (axes[0, 0], "Trial 1", x_cls_R, 69, 'blue', 'Right Hand', (0.5, 1.5)),
        (axes[0, 1], "Trial 2", x_cls_L, 69, 'red',  'Left Hand',  None),
        (axes[1, 0], "Trial 3", x_cls_R, 80, 'blue', 'Right Hand', (1.0, 2.0)),
        (axes[1, 1], "Trial 4", x_cls_L, 80, 'red',  'Left Hand',  None),
    ]

    for ax, title, data_src, idx, line_color, task_name, span in panels_config:
        fp1, fp2 = get_filtered_trial_with_baseline(data_src, idx)
        
        if fp1 is None or fp2 is None:
            ax.text(0.5, 0.5, "Trial Not Available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # 計算垂直偏移量
        max_amp = max(np.max(np.abs(fp1)), np.max(np.abs(fp2)))
        offset = max_amp * 1.4 if max_amp > 0 else 60.0

        ax.axhline(offset, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        # 繪製波形 (Fp1 帶 Offset, Fp2 位於 0)
        ax.plot(time_axis, fp1 + offset, color=line_color, linewidth=1.5, label=f'{task_name} (Fp1)')
        ax.plot(time_axis, fp2, color=line_color, linewidth=1.5, label=f'{task_name} (Fp2)')

        # 透明紅色區塊標註
        if span is not None:
            ax.axvspan(span[0], span[1], color='red', alpha=0.2, label=f'Artifact Interval ({span[0]}–{span[1]}s)')

        ax.set_yticks([0, offset])
        ax.set_yticklabels(['Fp2', 'Fp1'], fontsize=14, fontweight='bold')
        ax.set_ylim(-offset * 0.9, offset * 1.9)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_filename = f"Sub{mapped_subject}_Sess{session}_Raw_EEG_4Trials.png"
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ 圖片已成功儲存: {save_path}")

# ==========================================
# Main Execution Block
# ==========================================
@tee_log("eog_erp_analysis_sub45_s2.txt")
def main():
    fs = 500 
    ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz',
                'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"
    tfa_root_dir = "EOG"
    os.makedirs(tfa_root_dir, exist_ok=True)
    
    ids = ["45"]
    sessions = ["s2"]

    subjectMap = {
        "35": "01", "37": "02", "38": "03", "40": "04", "41": "05",
        "42": "06", "43": "07", "44": "08", "45": "09", "47": "10",
        "48": "11", "50": "12", "51": "13", "52": "14", "54": "15",
        "55": "16", "57": "17", "58": "18", "63": "19", "64": "20",
        "65": "21", "68": "22", "69": "23", "70": "24"
    }

    global_cheating_report = {}

    for subject in ids:
        mapped_subject = subjectMap.get(subject, subject)
        subject_save_dir = os.path.join(tfa_root_dir, mapped_subject)
        os.makedirs(subject_save_dir, exist_ok=True)
        
        for session in sessions:
            print(f"\n{'='*20} Subject {mapped_subject} (Raw ID: {subject}), Session {session} Started {'='*20}")
            
            subject_session_dir = f"{base_dir}/{subject}/{session}"
            mi_datas = [f"{subject_session_dir}/run{r}/mi_22.pt" for r in range(1, 8)]
            x_mi, y_mi = cat_all_data(mi_datas)

            if x_mi is None:
                print(f"⚠️ No sufficient data found, skipping.")
                continue
            
            print("Applying Base 1-40 Hz Bandpass Filter...")
            x_mi_filtered = mne.filter.filter_data(x_mi, sfreq=fs, l_freq=1.0, h_freq=40.0, n_jobs=-1, verbose=False)

            classes = np.unique(y_mi)
            if 0 in classes and 1 in classes:
                cls_Left = 1
                cls_Right = 0
                
                x_mi_cls_Left = x_mi_filtered[y_mi == cls_Left]
                x_mi_cls_Right = x_mi_filtered[y_mi == cls_Right]
                
                task_name_Left = "Left Hand"
                task_name_Right = "Right Hand"

                print("Detecting EOG events using Baseline Correction & Sustained Gaze Logic...")
                stats_L = detect_eog_events(x_mi_cls_Left, ch_names, fs=fs)
                stats_R = detect_eog_events(x_mi_cls_Right, ch_names, fs=fs)
                
                suspicions = evaluate_cheating_suspicion(stats_L, stats_R, task_name_Left, task_name_Right)
                
                if suspicions:
                    print(f"    🚨 警告：偵測到作弊嫌疑！")
                    for s in suspicions:
                        print(f"       -> {s}")
                    global_cheating_report.setdefault(mapped_subject, []).append(f"Sess {session}: " + ", ".join(suspicions))
                else:
                    print(f"    ✅ 通過眼動檢定：未偵測到明顯的眼動作弊行為。")

                # ⭐ 繪製指定的 4 個 Trial Raw EEG 圖表
                plot_custom_four_trials(
                    x_mi_cls_Left, x_mi_cls_Right, 
                    fs, ch_names, 
                    mapped_subject, session, 
                    save_dir=subject_save_dir
                )

    print("\n\n" + "!"*60)
    print(" 🚨 受試者作弊嫌疑總結報告 (Sustained Gaze Detection) 🚨")
    print("!"*60)
    if not global_cheating_report:
        print("太棒了！沒有人表現出依賴眼動的作弊嫌疑。")
    else:
        print("以下受試者存在高度統計關聯 (p<0.01)，可能使用持續凝視 (Sustained Gaze) 作弊：\n")
        for sub, msgs in global_cheating_report.items():
            print(f"📌 Subject {sub}:")
            for msg in msgs:
                print(f"    - {msg}")
    print("!"*60 + "\n")

if __name__ == "__main__":
    main()