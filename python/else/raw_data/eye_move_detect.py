import numpy as np
import torch
import os
import mne
from mne.time_frequency import psd_array_welch
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps
from datetime import datetime
import sys

# === Packages for Signal Processing and Statistics ===
from scipy.signal import hilbert, find_peaks
from scipy.stats import sem

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
    """Decorator to redirect standard output to both console and a log file."""
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
    """Concatenate data from all experimental runs."""
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
# ⭐ UPGRADED: Paper-Level EOG Detection
# ==========================================
def detect_eog_events(x_data, ch_names, fs=500, blink_k=5.0, saccade_k=3.0):
    """
    Paper-grade EOG event detector using VEOG/HEOG, Lowpass filtering, 
    MAD robust thresholding, and peak detection for exact event counting.
    """
    if 'Fp1' not in ch_names or 'Fp2' not in ch_names:
        print("⚠️ Fp1 or Fp2 not found. Cannot perform EOG analysis.")
        return None
        
    fp1_idx = ch_names.index('Fp1')
    fp2_idx = ch_names.index('Fp2')
    n_trials = x_data.shape[0]
    
    # 1. Isolate Fp1/Fp2 and apply 10Hz Lowpass specifically for EOG detection
    # This prevents Alpha (8-13Hz) or EMG from triggering false positives
    fp1_raw = x_data[:, fp1_idx, :].copy()
    fp2_raw = x_data[:, fp2_idx, :].copy()
    
    fp1_filtered = mne.filter.filter_data(fp1_raw, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)
    fp2_filtered = mne.filter.filter_data(fp2_raw, sfreq=fs, l_freq=None, h_freq=10.0, verbose=False)

    # 2. Compute Virtual EOG Channels
    # VEOG (Vertical) = (Fp1 + Fp2) / 2  --> Best for Blinks
    # HEOG (Horizontal) = Fp1 - Fp2      --> Best for Saccades (Left/Right)
    veog_all = (fp1_filtered + fp2_filtered) / 2.0
    heog_all = fp1_filtered - fp2_filtered

    # 3. Robust Thresholding using MAD (Median Absolute Deviation)
    # MAD is highly resistant to extreme artifact outliers compared to standard deviation (std).
    def compute_robust_threshold(data_flat, k_multiplier):
        median_val = np.median(data_flat)
        mad = np.median(np.abs(data_flat - median_val))
        robust_std = mad * 1.4826 # Conversion factor for normal distribution
        return median_val + k_multiplier * robust_std

    # Calculate global thresholds based on flattened data across all trials
    veog_flat = veog_all.flatten()
    heog_flat = heog_all.flatten()
    
    # Assuming blinks generate POSITIVE peaks in standard EEG montages. 
    # If your amp is inverted (blinks go down), change blink_k to negative and look for valleys.
    blink_th = compute_robust_threshold(veog_flat, blink_k)
    
    # Saccades can be positive (Left) or negative (Right)
    heog_median = np.median(heog_flat)
    heog_robust_std = np.median(np.abs(heog_flat - heog_median)) * 1.4826
    saccade_th_pos = heog_median + saccade_k * heog_robust_std
    saccade_th_neg = heog_median - saccade_k * heog_robust_std

    print(f"    [Robust Thresholds] VEOG Blink: > {blink_th:.2f} µV")
    print(f"    [Robust Thresholds] HEOG Left: > {saccade_th_pos:.2f} µV | Right: < {saccade_th_neg:.2f} µV")

    # Arrays to store true EVENT counts
    blink_counts = np.zeros(n_trials)
    look_left_counts = np.zeros(n_trials)
    look_right_counts = np.zeros(n_trials)
    
    # Refractory period: events must be separated by at least 200ms
    min_distance_samples = int(fs * 0.2) 

    for i in range(n_trials):
        veog_trial = veog_all[i, :]
        heog_trial = heog_all[i, :]
        
        # Detrending (Zero-mean) per trial to fix local baseline drifts
        veog_trial = veog_trial - np.mean(veog_trial)
        heog_trial = heog_trial - np.mean(heog_trial)
        
        # --- Event Detection using find_peaks ---
        
        # 1. Blinks (Positive peak on VEOG)
        b_peaks, _ = find_peaks(veog_trial, height=blink_th, distance=min_distance_samples)
        blink_counts[i] = len(b_peaks)
        
        # 2. Look Left (Positive peak on HEOG: Fp1 > Fp2)
        l_peaks, _ = find_peaks(heog_trial, height=saccade_th_pos, distance=min_distance_samples)
        look_left_counts[i] = len(l_peaks)
        
        # 3. Look Right (Negative peak on HEOG: Fp2 > Fp1 -> equivalent to positive peak on inverted HEOG)
        r_peaks, _ = find_peaks(-heog_trial, height=-saccade_th_neg, distance=min_distance_samples)
        look_right_counts[i] = len(r_peaks)

    return {
        'Blinks': blink_counts,
        'Look Left': look_left_counts,
        'Look Right': look_right_counts
    }


def plot_eog_statistics(stats_left, stats_right, subject, session, cls_names=('Left Hand', 'Right Hand'), save_dir="TFA"):
    """
    Bar chart visualization for EOG event counts.
    """
    if stats_left is None or stats_right is None:
        return
        
    event_types = ['Blinks', 'Look Left', 'Look Right']
    
    means_left = [np.mean(stats_left[k]) for k in event_types]
    errs_left = [np.std(stats_left[k])/np.sqrt(len(stats_left[k])) if len(stats_left[k])>0 else 0 for k in event_types]
    
    means_right = [np.mean(stats_right[k]) for k in event_types]
    errs_right = [np.std(stats_right[k])/np.sqrt(len(stats_right[k])) if len(stats_right[k])>0 else 0 for k in event_types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(event_types))
    width = 0.35  
    
    ax.bar(x - width/2, means_left, width, yerr=errs_left, label=cls_names[0], color='red', alpha=0.7, capsize=5)
    ax.bar(x + width/2, means_right, width, yerr=errs_right, label=cls_names[1], color='blue', alpha=0.7, capsize=5)
    
    ax.set_ylabel('Mean Event Count per Trial')
    ax.set_title(f'[Subject {subject} - Session {session}] EOG Event Count Statistics', fontweight='bold')
    ax.set_xticks(x)
    # Updated labels to reflect the new HEOG/VEOG methodology
    ax.set_xticklabels(['Blinks (VEOG Peak)', 'Look Left (HEOG+)', 'Look Right (HEOG-)'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Sub{subject}_Sess{session}_EOG_Event_Counts.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ EOG Statistics Plot saved to: {save_path}")


# ==========================================
# Main Execution Block
# ==========================================
@tee_log("eye_move_detect.txt")
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
    sessions = ["s1", "s2"]

    for subject in ids:
        subject_save_dir = os.path.join(tfa_root_dir, subject)
        os.makedirs(subject_save_dir, exist_ok=True)
        
        for session in sessions:
            print(f"\n{'='*20} Subject {subject}, Session {session} Started {'='*20}")
            subject_session_dir = f"{base_dir}/{subject}/{session}"
            
            mi_datas = [f"{subject_session_dir}/run{r}/mi_22.pt" for r in range(1, 8)]
            x_mi, y_mi = cat_all_data(mi_datas)

            if x_mi is None:
                print(f"⚠️ No sufficient data found, skipping.")
                continue
            
            # --- 1. Bandpass Filter (1~40 Hz) ---
            print("Applying 1-40 Hz Bandpass Filter...")
            x_mi_filtered = mne.filter.filter_data(x_mi, sfreq=fs, l_freq=1.0, h_freq=40.0, n_jobs=-1, verbose=False)

            classes = np.unique(y_mi)
            
            if 0 in classes and 1 in classes:
                # User defined: Label 1 is Left Hand, Label 0 is Right Hand
                cls_Left = 1
                cls_Right = 0
                
                x_mi_cls_Left = x_mi_filtered[y_mi == cls_Left]
                x_mi_cls_Right = x_mi_filtered[y_mi == cls_Right]
                
                task_name_Left = "Left Hand"
                task_name_Right = "Right Hand"

                # 1. Calculate EOG Statistics (Pass 'fs' so find_peaks can compute 200ms distance)
                print("Detecting EOG events using Robust Thresholds (MAD) and VEOG/HEOG...")
                stats_L = detect_eog_events(x_mi_cls_Left, ch_names, fs=fs, blink_k=5.0, saccade_k=3.0)
                stats_R = detect_eog_events(x_mi_cls_Right, ch_names, fs=fs, blink_k=5.0, saccade_k=3.0)
                
                group_save_dir = os.path.join(subject_save_dir, "Eye_Movement")
                os.makedirs(group_save_dir, exist_ok=True)
                
                # 2. Plot Bar Charts
                plot_eog_statistics(stats_L, stats_R, subject, session, cls_names=(task_name_Left, task_name_Right), save_dir=group_save_dir) 

if __name__ == "__main__":
    main()