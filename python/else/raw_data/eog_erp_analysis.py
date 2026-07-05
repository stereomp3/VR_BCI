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
# ⭐ EOG Event Detection (Robust MAD & Sustained Gaze)
# ==========================================
def detect_eog_events(x_data, ch_names, fs=500, blink_k=5.0, saccade_k=3.0):
    if 'Fp1' not in ch_names or 'Fp2' not in ch_names:
        return None
        
    fp1_idx = ch_names.index('Fp1')
    fp2_idx = ch_names.index('Fp2')
    n_trials = x_data.shape[0]
    
    fp1_raw = x_data[:, fp1_idx, :].copy()
    fp2_raw = x_data[:, fp2_idx, :].copy()
    
    # Lowpass filter for stable event detection
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
        
        # Baseline Correction
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

# ==========================================
# Cheating Evaluation
# ==========================================
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
# ⭐ Unified Comprehensive EOG Dashboard
# ==========================================
def plot_comprehensive_eog_dashboard(x_cls_L, x_cls_R, stats_L, stats_R, fs, ch_names, mapped_subject, session, cls_names, save_dir):
    if 'Fp1' not in ch_names or 'Fp2' not in ch_names:
        print("⚠️ 找不到 Fp1 或 Fp2，無法繪製儀表板。")
        return
        
    fp1_idx, fp2_idx = ch_names.index('Fp1'), ch_names.index('Fp2')
    time_axis = np.arange(x_cls_L.shape[2]) / fs
    
    fig = plt.figure(figsize=(20, 16))
    # ⭐ 標題使用 mapped_subject
    fig.suptitle(f"[Subject {mapped_subject} - Session {session}] Comprehensive EOG Artifact Dashboard", fontsize=20, fontweight='bold', y=0.95)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.6, wspace=0.2)

    # ---------------------------------------------------------
    # ① 左上 (Panel 1): Stacked 顯示
    # ---------------------------------------------------------
    ax_single = fig.add_subplot(gs[0:2, 0])
    check_trial = 70
    t_idx_L = min(check_trial, len(x_cls_L)-1) if len(x_cls_L) > 0 else 0
    t_idx_R = min(check_trial, len(x_cls_R)-1) if len(x_cls_R) > 0 else 0
    
    def get_filtered_trial_with_baseline(x_data, t_idx):
        fp1_t = x_data[t_idx, fp1_idx, :]
        fp2_t = x_data[t_idx, fp2_idx, :]
        fp1_t = mne.filter.filter_data(fp1_t, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        fp2_t = mne.filter.filter_data(fp2_t, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        return fp1_t - np.mean(fp1_t), fp2_t - np.mean(fp2_t)

    fp1_L_single, fp2_L_single = get_filtered_trial_with_baseline(x_cls_L, t_idx_L) if len(x_cls_L) > 0 else (None, None)
    fp1_R_single, fp2_R_single = get_filtered_trial_with_baseline(x_cls_R, t_idx_R) if len(x_cls_R) > 0 else (None, None)

    max_amp = 0
    for sig in [fp1_L_single, fp2_L_single, fp1_R_single, fp2_R_single]:
        if sig is not None:
            max_amp = max(max_amp, np.max(np.abs(sig)))
    offset = max_amp * 1.5 if max_amp > 0 else 80.0

    ax_single.axhline(offset, color='gray', linestyle=':', alpha=0.6)
    ax_single.axhline(0, color='gray', linestyle=':', alpha=0.6)

    if fp1_L_single is not None:
        ax_single.plot(time_axis, fp1_L_single + offset, color='red', linewidth=1.5, label=f'{cls_names[0]}')
        ax_single.plot(time_axis, fp2_L_single, color='red', linewidth=1.5, alpha=0.7)
    if fp1_R_single is not None:
        ax_single.plot(time_axis, fp1_R_single + offset, color='blue', linewidth=1.5, label=f'{cls_names[1]}')
        ax_single.plot(time_axis, fp2_R_single, color='blue', linewidth=1.5, alpha=0.7)

    ax_single.set_yticks([0, offset])
    ax_single.set_yticklabels(['Fp2', 'Fp1'], fontsize=16, fontweight='bold')
    ax_single.set_ylim(-max_amp * 1.2, offset + max_amp * 1.2)
    
    ax_single.set_title(f"1. Single Trial (Trial {check_trial}) EOG (10Hz Lowpass) Stacked", fontsize=14, fontweight='bold')
    ax_single.set_xlabel("Time (s)")
    ax_single.legend(loc='upper right')
    ax_single.grid(True, linestyle='--', alpha=0.3)

    # ---------------------------------------------------------
    # ② 右上 (Panel 2): EOG 統計長條圖
    # ---------------------------------------------------------
    ax_stats = fig.add_subplot(gs[0:2, 1])
    event_types = ['Blinks', 'Look Left', 'Look Right']
    
    means_L = [np.mean(stats_L[k]) for k in event_types]
    errs_L = [sem(stats_L[k]) if len(stats_L[k])>0 else 0 for k in event_types]
    means_R = [np.mean(stats_R[k]) for k in event_types]
    errs_R = [sem(stats_R[k]) if len(stats_R[k])>0 else 0 for k in event_types]
    
    x_pos = np.arange(len(event_types))
    width = 0.35  
    
    ax_stats.bar(x_pos - width/2, means_L, width, yerr=errs_L, label=cls_names[0], color='red', alpha=0.7, capsize=5)
    ax_stats.bar(x_pos + width/2, means_R, width, yerr=errs_R, label=cls_names[1], color='blue', alpha=0.7, capsize=5)
    
    ax_stats.set_title("2. EOG Event Statistics (Sustained Gaze Logic)", fontsize=14, fontweight='bold')
    ax_stats.set_ylabel("Blinks (Count) / Saccades (Seconds)")
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels(['Blinks\n(Count)', 'Look Left\n(Duration in sec)', 'Look Right\n(Duration in sec)'])
    ax_stats.legend(loc='upper right')
    ax_stats.grid(axis='y', linestyle='--', alpha=0.6)

    # ---------------------------------------------------------
    # ③ 左下 (Panel 3): Average Raw Fp1 / Fp2 (拆成上下兩圖)
    # ---------------------------------------------------------
    ax_avg_fp1 = fig.add_subplot(gs[2, 0])
    ax_avg_fp2 = fig.add_subplot(gs[3, 0], sharex=ax_avg_fp1)
    
    fp1_L_mean = np.mean(x_cls_L[:, fp1_idx, :], axis=0) if len(x_cls_L) > 0 else np.zeros_like(time_axis)
    fp1_R_mean = np.mean(x_cls_R[:, fp1_idx, :], axis=0) if len(x_cls_R) > 0 else np.zeros_like(time_axis)
    fp2_L_mean = np.mean(x_cls_L[:, fp2_idx, :], axis=0) if len(x_cls_L) > 0 else np.zeros_like(time_axis)
    fp2_R_mean = np.mean(x_cls_R[:, fp2_idx, :], axis=0) if len(x_cls_R) > 0 else np.zeros_like(time_axis)

    ax_avg_fp1.plot(time_axis, fp1_L_mean, color='red', label=cls_names[0], linewidth=2)
    ax_avg_fp1.plot(time_axis, fp1_R_mean, color='blue', label=cls_names[1], linewidth=2)
    ax_avg_fp1.set_title("3-A. Fp1 Average Raw Signal", fontweight='bold')
    ax_avg_fp1.set_ylabel("Amplitude (µV)")
    ax_avg_fp1.grid(True, linestyle='--', alpha=0.5)

    ax_avg_fp2.plot(time_axis, fp2_L_mean, color='red', label=cls_names[0], linewidth=2)
    ax_avg_fp2.plot(time_axis, fp2_R_mean, color='blue', label=cls_names[1], linewidth=2)
    ax_avg_fp2.set_title("3-B. Fp2 Average Raw Signal", fontweight='bold')
    ax_avg_fp2.set_xlabel("Time (s)")
    ax_avg_fp2.set_ylabel("Amplitude (µV)")
    ax_avg_fp2.grid(True, linestyle='--', alpha=0.5)

    # ---------------------------------------------------------
    # ④ 右下 (Panel 4): Average EOG ERP (HEOG / VEOG 拆成上下兩圖)
    # ---------------------------------------------------------
    ax_erp_heog = fig.add_subplot(gs[2, 1])
    ax_erp_veog = fig.add_subplot(gs[3, 1], sharex=ax_erp_heog)

    def process_erp(x_data):
        fp1 = x_data[:, fp1_idx, :]
        fp2 = x_data[:, fp2_idx, :]
        
        fp1_filt = mne.filter.filter_data(fp1, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        fp2_filt = mne.filter.filter_data(fp2, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
        
        heog = fp1_filt - fp2_filt
        veog = (fp1_filt + fp2_filt) / 2.0
        
        base_samples = int(fs * 0.5)
        heog = heog - np.mean(heog[:, :base_samples], axis=1, keepdims=True)
        veog = veog - np.mean(veog[:, :base_samples], axis=1, keepdims=True)
        return heog, veog

    heog_L, veog_L = process_erp(x_cls_L)
    heog_R, veog_R = process_erp(x_cls_R)

    h_mean_L, h_sem_L = np.mean(heog_L, axis=0), sem(heog_L, axis=0)
    h_mean_R, h_sem_R = np.mean(heog_R, axis=0), sem(heog_R, axis=0)
    v_mean_L, v_sem_L = np.mean(veog_L, axis=0), sem(veog_L, axis=0)
    v_mean_R, v_sem_R = np.mean(veog_R, axis=0), sem(veog_R, axis=0)

    # HEOG
    ax_erp_heog.plot(time_axis, h_mean_L, color='red', label=cls_names[0], linewidth=2)
    ax_erp_heog.fill_between(time_axis, h_mean_L - h_sem_L, h_mean_L + h_sem_L, color='red', alpha=0.2)
    ax_erp_heog.plot(time_axis, h_mean_R, color='blue', label=cls_names[1], linewidth=2)
    ax_erp_heog.fill_between(time_axis, h_mean_R - h_sem_R, h_mean_R + h_sem_R, color='blue', alpha=0.2)
    ax_erp_heog.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_erp_heog.set_title("4-A. Horizontal EOG ERP (HEOG = Fp1 - Fp2)", fontweight='bold')
    ax_erp_heog.set_ylabel("Amplitude (µV)")
    ax_erp_heog.grid(True, linestyle='--', alpha=0.5)

    # VEOG
    ax_erp_veog.plot(time_axis, v_mean_L, color='red', label=cls_names[0], linewidth=2)
    ax_erp_veog.fill_between(time_axis, v_mean_L - v_sem_L, v_mean_L + v_sem_L, color='red', alpha=0.2)
    ax_erp_veog.plot(time_axis, v_mean_R, color='blue', label=cls_names[1], linewidth=2)
    ax_erp_veog.fill_between(time_axis, v_mean_R - v_sem_R, v_mean_R + v_sem_R, color='blue', alpha=0.2)
    ax_erp_veog.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_erp_veog.set_title("4-B. Vertical EOG ERP (VEOG = (Fp1+Fp2)/2)", fontweight='bold')
    ax_erp_veog.set_xlabel("Time (s)")
    ax_erp_veog.set_ylabel("Amplitude (µV)")
    ax_erp_veog.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # ⭐ 儲存檔名使用 mapped_subject
    save_path = os.path.join(save_dir, f"Sub{mapped_subject}_Sess{session}_EOG_Comprehensive_Dashboard.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ 綜合眼動診斷儀表板已儲存: {save_path}")

# ==========================================
# Main Execution Block
# ==========================================
@tee_log("eog_erp_analysis.txt")
def main():
    fs = 500 
    ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz',
                'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"
    tfa_root_dir = "EOG"
    os.makedirs(tfa_root_dir, exist_ok=True)
    
    ids = [
        "35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", 
        "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"
    ]
    # ids = [
    #     "45" # 需要 80 check_trial 看的比較明顯
    # ]
    sessions = ["s1", "s2"]

    # ⭐ 加入 ID 映射字典
    subjectMap = {
        "35": "01", "37": "02", "38": "03", "40": "04", "41": "05",
        "42": "06", "43": "07", "44": "08", "45": "09", "47": "10",
        "48": "11", "50": "12", "51": "13", "52": "14", "54": "15",
        "55": "16", "57": "17", "58": "18", "63": "19", "64": "20",
        "65": "21", "68": "22", "69": "23", "70": "24"
    }

    global_cheating_report = {}

    for subject in ids:
        # ⭐ 取得對應的 Map ID (若無則維持原始 ID)
        mapped_subject = subjectMap.get(subject, subject)
        
        # ⭐ 資料夾名稱使用轉換後的 ID
        subject_save_dir = os.path.join(tfa_root_dir, mapped_subject)
        os.makedirs(subject_save_dir, exist_ok=True)
        
        for session in sessions:
            print(f"\n{'='*20} Subject {mapped_subject} (Raw ID: {subject}), Session {session} Started {'='*20}")
            
            # 讀取資料必須使用原始的 subject ID
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

                # 直接存入 subject_save_dir，如您先前的設定
                plot_comprehensive_eog_dashboard(
                    x_mi_cls_Left, x_mi_cls_Right, 
                    stats_L, stats_R, 
                    fs, ch_names, 
                    mapped_subject, session, # ⭐ 傳入 mapping 過的 ID 供繪圖使用
                    cls_names=(task_name_Left, task_name_Right), 
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