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
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir, os.path.join(parent_dir, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# === Packages for Signal Processing and Statistics ===
from scipy.signal import hilbert
from scipy.stats import sem

# ==========================================
# Experimental Flags (Boolean)
# ==========================================
DO_ICA = False                  # Whether to perform ICA for automatic EOG artifact removal (using Fp1 as ref)
DO_CAR = True                  # Whether to apply Common Average Reference (CAR)
CHECK_TRIAL = 70               # Select the middle trial (280 / 2 / 2 = 70) for time-domain visualization
# ==========================================

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
# ⭐ Feature Evaluation Helper
# ==========================================
def evaluate_feature_sig(x_L, x_R, ch_name, band, fs, ch_names):
    """
    計算指定通道在特定頻段的能量，並使用與畫圖相同的 SEM 邏輯來判斷顯著性方向。
    回傳值: "L>R" (Left顯著大於Right), "R>L" (Right顯著大於Left), 或 "-" (無顯著差異)
    """
    if ch_name not in ch_names: 
        return "-"
    
    ch_idx = ch_names.index(ch_name)
    data_L = x_L[:, ch_idx:ch_idx+1, :]
    data_R = x_R[:, ch_idx:ch_idx+1, :]
    
    # 計算 PSD
    p_L, f = psd_array_welch(data_L, sfreq=fs, fmin=1.0, fmax=40.0, n_fft=fs, n_jobs=-1, verbose=False)
    p_R, _ = psd_array_welch(data_R, sfreq=fs, fmin=1.0, fmax=40.0, n_fft=fs, n_jobs=-1, verbose=False)
    
    # 處理維度
    p_L = p_L.squeeze(axis=1) if p_L.ndim == 3 else p_L
    p_R = p_R.squeeze(axis=1) if p_R.ndim == 3 else p_R
    if p_L.ndim == 1: p_L = p_L[np.newaxis, :]
    if p_R.ndim == 1: p_R = p_R[np.newaxis, :]
    
    # 擷取頻段平均能量
    mask = (f >= band[0]) & (f <= band[1])
    pow_L = np.mean(p_L[:, mask], axis=1)
    pow_R = np.mean(p_R[:, mask], axis=1)
    
    mL, sL = np.mean(pow_L), sem(pow_L) if len(pow_L)>1 else 0
    mR, sR = np.mean(pow_R), sem(pow_R) if len(pow_R)>1 else 0
    
    # 與畫圖模組使用相同的顯著性判斷標準
    if abs(mL - mR) > (sL + sR):
        return "L>R" if mL > mR else "R>L"
    return "-"

# ==========================================
# === Plotting Module: 4-Panel Comparison ===
# ==========================================
def plot_4_panel_comparison(x_cls1, x_cls2, fs, ch_names, target_ch, subject, session, 
                            cls_names=('Left Hand', 'Right Hand'), save_dir="TFA", 
                            focus_band=(8, 13), focus_name="Alpha", psd_xmin=8.0):
    if target_ch not in ch_names:
        print(f"⚠️ Target channel {target_ch} not found, skipping plot.")
        return
        
    ch_idx = ch_names.index(target_ch)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    fig.suptitle(f"[Subject {subject} - Session {session}] Channel {target_ch} Analysis: {cls_names[0]} vs {cls_names[1]}", fontsize=18, fontweight='bold')

    trial_idx_1 = min(CHECK_TRIAL, len(x_cls1)-1) if len(x_cls1) > 0 else 0
    trial_idx_2 = min(CHECK_TRIAL, len(x_cls2)-1) if len(x_cls2) > 0 else 0
    
    time_axis = np.arange(x_cls1.shape[2]) / fs
    l_freq, h_freq = float(focus_band[0]), float(focus_band[1])

    # --- Panel 1: Raw EEG Quality Check ---
    ax_raw = axs[0, 0]
    if len(x_cls1) > 0: ax_raw.plot(time_axis, x_cls1[trial_idx_1, ch_idx, :], color='red', label=cls_names[0], alpha=0.8)
    if len(x_cls2) > 0: ax_raw.plot(time_axis, x_cls2[trial_idx_2, ch_idx, :], color='blue', label=cls_names[1], alpha=0.8)
    ax_raw.set_title("1. Raw EEG Quality Check (Single Trial Ref)", fontsize=14, fontweight='bold')
    ax_raw.set_xlabel("Time (s)")
    ax_raw.set_ylabel("Amplitude (µV)")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(True, linestyle='--', alpha=0.5)

    # --- Panel 3: Time-Domain Dynamics ---
    ax_env = axs[1, 0]
    if len(x_cls1) > 0:
        filtered_all_1 = mne.filter.filter_data(x_cls1[:, ch_idx, :], sfreq=fs, l_freq=l_freq, h_freq=h_freq, verbose=False)
        env_all_1 = np.abs(hilbert(filtered_all_1))
        env_mean_1 = np.mean(env_all_1, axis=0) 
        ax_env.plot(time_axis, env_mean_1, color='red', label=cls_names[0], linewidth=2)
    
    if len(x_cls2) > 0:
        filtered_all_2 = mne.filter.filter_data(x_cls2[:, ch_idx, :], sfreq=fs, l_freq=l_freq, h_freq=h_freq, verbose=False)
        env_all_2 = np.abs(hilbert(filtered_all_2))
        env_mean_2 = np.mean(env_all_2, axis=0)
        ax_env.plot(time_axis, env_mean_2, color='blue', label=cls_names[1], linewidth=2)
        
    ax_env.set_title(f"2. Time-Domain Dynamics: {focus_name} ERD/ERS Trend", fontsize=14, fontweight='bold')
    ax_env.set_xlabel("Time (s)")
    ax_env.set_ylabel("Amplitude (µV)")
    ax_env.legend(loc="upper right")
    ax_env.grid(True, linestyle='--', alpha=0.5)

    # --- Calculate PSD for all trials ---
    psds_1, freqs = psd_array_welch(x_cls1[:, ch_idx:ch_idx+1, :], sfreq=fs, fmin=1.0, fmax=40.0, n_fft=fs, n_jobs=-1, verbose=False)
    psds_2, _ = psd_array_welch(x_cls2[:, ch_idx:ch_idx+1, :], sfreq=fs, fmin=1.0, fmax=40.0, n_fft=fs, n_jobs=-1, verbose=False)
    psds_1, psds_2 = psds_1.squeeze(), psds_2.squeeze()
    
    if psds_1.ndim == 1: psds_1 = psds_1[np.newaxis, :]
    if psds_2.ndim == 1: psds_2 = psds_2[np.newaxis, :]

    # --- Panel 4: Power Spectral Density ---
    ax_psd = axs[1, 1]
    mean_psd_1 = np.mean(psds_1, axis=0)
    mean_psd_2 = np.mean(psds_2, axis=0)
    
    ax_psd.plot(freqs, mean_psd_1, color='red', label=cls_names[0], linewidth=2)
    ax_psd.plot(freqs, mean_psd_2, color='blue', label=cls_names[1], linewidth=2)
    ax_psd.axvspan(l_freq, h_freq, color='gray', alpha=0.2, label=f'{focus_name} Band')
    
    ax_psd.set_title("3. Power Spectral Density (PSD)", fontsize=14, fontweight='bold')
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.set_xlim(psd_xmin, 40)
    ax_psd.legend(loc="upper right")
    ax_psd.grid(True, linestyle='--', alpha=0.5)

    # --- Panel 2: Feature Significance ---
    ax_bar = axs[0, 1]
    focus_mask = np.logical_and(freqs >= l_freq, freqs <= h_freq)
    
    power_1 = np.mean(psds_1[:, focus_mask], axis=1)
    power_2 = np.mean(psds_2[:, focus_mask], axis=1)
    
    mean_1, sem_1 = np.mean(power_1), sem(power_1)
    mean_2, sem_2 = np.mean(power_2), sem(power_2)
    
    bars = ax_bar.bar([cls_names[0], cls_names[1]], [mean_1, mean_2], yerr=[sem_1, sem_2], 
                      color=['red', 'blue'], alpha=0.7, capsize=10)
    
    ax_bar.set_title(f"4. Feature Significance: Mean {focus_name} Power ({int(l_freq)}-{int(h_freq)} Hz)", fontsize=14, fontweight='bold')
    ax_bar.set_ylabel("Absolute Power")
    
    if abs(mean_1 - mean_2) > (sem_1 + sem_2):
        y_max = max(mean_1 + sem_1, mean_2 + sem_2) + (max(mean_1, mean_2) * 0.1)
        ax_bar.plot([0, 1], [y_max, y_max], color='black', linewidth=1.5)
        ax_bar.text(0.5, y_max, '*', ha='center', va='bottom', color='black', fontsize=16)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"Sub{subject}_Sess{session}_Ch{target_ch}_4Panel_{focus_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ 4-Panel comparison plot saved to: {save_path}")


# ==========================================
# Main Execution Block
# ==========================================
def run_tfa_analysis(base_dir=r"/mnt/project/MIEXP/DATA_Cygnus", output_dir="TFA",
                     channels="22", ids=None, sessions=None,
                     do_ica=False, do_car=True, log_file="TFA2.txt"):
    fs = 500  
    ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz',
                'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

    os.makedirs(output_dir, exist_ok=True)
    
    if ids is None:
        ids = [
            "35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", 
            "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"
        ]
    elif isinstance(ids, str):
        ids = [x.strip() for x in ids.split(",") if x.strip()]

    if sessions is None:
        sessions = ["s1", "s2"]
    elif isinstance(sessions, str):
        sessions = [x.strip() for x in sessions.split(",") if x.strip()]
    
    sorted_ids = sorted(ids, key=lambda x: int(x) if x.isdigit() else x)
    subjectMap = {sub: f"{i+1:02d}" for i, sub in enumerate(sorted_ids)}

    # ⭐ 建立一個 List 來收集所有受試者的診斷結果
    summary_report = []

    print("=" * 75)
    print("🚀 TFA2 時頻分析與特徵診斷系統")
    print(f"  ├── 資料目錄: {base_dir}")
    print(f"  ├── 輸出目錄: {output_dir}")
    print(f"  ├── 通道模式: {channels} 通道")
    print(f"  ├── ICA 去雜訊: {do_ica} | CAR 空間濾波: {do_car}")
    print(f"  └── 處理受試者: {len(ids)} 人")
    print("=" * 75)

    for subject in ids:
        mapped_subject = subjectMap.get(subject, subject)
        subject_save_dir = os.path.join(output_dir, mapped_subject)
        os.makedirs(subject_save_dir, exist_ok=True)
        
        for session in sessions:
            print(f"\n{'='*20} Subject {mapped_subject} (Raw ID: {subject}), Session {session} Started {'='*20}")
            subject_session_dir = f"{base_dir}/{subject}/{session}"
            
            ch_str = str(channels)
            mi_datas = []
            for r in range(1, 8):
                candidates = [
                    f"{subject_session_dir}/run{r}/mi_{ch_str}.pt",
                    f"{subject_session_dir}/run{r}/mi_22.pt",
                    f"{subject_session_dir}/run{r}/mi_13.pt",
                    f"{subject_session_dir}/run{r}/data.pt"
                ]
                found = False
                for c in candidates:
                    if os.path.exists(c):
                        mi_datas.append(c)
                        found = True
                        break
                if not found:
                    mi_datas.append(candidates[0])

            x_mi, y_mi = cat_all_data(mi_datas)

            if x_mi is None:
                print(f"⚠️ No sufficient data found, skipping.")
                continue
            
            print("Applying 1-40 Hz Bandpass Filter...")
            x_mi_filtered = mne.filter.filter_data(x_mi, sfreq=fs, l_freq=1.0, h_freq=40.0, n_jobs=-1, verbose=False)

            if do_ica:
                print("Performing ICA for artifact removal...")
                info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
                info.set_montage('standard_1020')
                epochs = mne.EpochsArray(x_mi_filtered, info, verbose=False)
                ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')
                ica.fit(epochs, verbose=False)
                eog_indices, _ = ica.find_bads_eog(epochs, ch_name='Fp1', verbose=False)
                ica.exclude = eog_indices
                epochs_clean = ica.apply(epochs.copy(), verbose=False)
                x_mi_filtered = epochs_clean.get_data(copy=False)

            if do_car:
                print("Applying CAR (Common Average Reference) re-referencing...")
                x_mi_filtered = x_mi_filtered - np.mean(x_mi_filtered, axis=1, keepdims=True)

            classes = np.unique(y_mi)
            
            if len(classes) >= 2:
                cls_Left = 1
                cls_Right = 0
                
                x_mi_cls_Left = x_mi_filtered[y_mi == cls_Left]
                x_mi_cls_Right = x_mi_filtered[y_mi == cls_Right]
                
                task_name_Left = "Left Hand"
                task_name_Right = "Right Hand"

                # ==========================================
                # ⭐ ERD & Alpha 自動判斷邏輯 (新增 CP3 / CP4)
                # ==========================================
                print("Evaluating ERD (Mu_Beta) and Alpha significance...")
                
                # 1. 判斷 ERD (8-30 Hz)
                sig_fc3 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'FC3', (8, 30), fs, ch_names)
                sig_fc4 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'FC4', (8, 30), fs, ch_names)
                sig_c3  = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'C3',  (8, 30), fs, ch_names)
                sig_c4  = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'C4',  (8, 30), fs, ch_names)
                sig_cp3 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'CP3', (8, 30), fs, ch_names) 
                sig_cp4 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'CP4', (8, 30), fs, ch_names) 
                
                erd_fc_ok = (sig_fc3 == "L>R" and sig_fc4 == "R>L")
                erd_c_ok  = (sig_c3 == "L>R" and sig_c4 == "R>L")
                erd_cp_ok = (sig_cp3 == "L>R" and sig_cp4 == "R>L") 
                
                # 只要這三個區域有其中一個符合典型的對側 ERD 現象，就算 Detected
                erd_detected = erd_fc_ok or erd_c_ok or erd_cp_ok 

                # 2. 判斷 Alpha (8-13 Hz)
                sig_o1 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'O1', (8, 13), fs, ch_names)
                sig_oz = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'Oz', (8, 13), fs, ch_names)
                sig_o2 = evaluate_feature_sig(x_mi_cls_Left, x_mi_cls_Right, 'O2', (8, 13), fs, ch_names)
                
                alpha_sigs = [sig_o1, sig_oz, sig_o2]
                alpha_detected = (alpha_sigs.count("L>R") >= 2) or (alpha_sigs.count("R>L") >= 2)

                # 3. 綜合診斷
                if erd_detected and alpha_detected:
                    overall_result = "Both"
                elif erd_detected:
                    overall_result = "ERD (Mu_Beta)"
                elif alpha_detected:
                    overall_result = "Alpha"
                else:
                    overall_result = "Neither"

                summary_report.append({
                    "Sub": mapped_subject,
                    "Sess": session,
                    "FC3": sig_fc3,
                    "FC4": sig_fc4,
                    "C3": sig_c3,
                    "C4": sig_c4,
                    "CP3": sig_cp3, 
                    "CP4": sig_cp4,
                    "O1": sig_o1,
                    "Oz": sig_oz,
                    "O2": sig_o2,
                    "Detected Feature": overall_result
                })

                print(f"--- Generating binary classification plots ({task_name_Left} vs {task_name_Right}) ---")

                # ==========================================
                # Plotting Loop
                # ==========================================
                analysis_groups = [
                    {'channels': ['O1', 'Oz', 'O2'], 'band': (8, 13), 'name': 'Alpha', 'psd_xmin': 8.0},
                    {'channels': ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'], 'band': (8, 30), 'name': 'ERD (Mu_Beta)', 'psd_xmin': 8.0}
                ]

                for group in analysis_groups:
                    print(f"Processing Region: {group['name']} -> Target Channels: {group['channels']}")
                    group_save_dir = os.path.join(subject_save_dir, group['name'])
                    os.makedirs(group_save_dir, exist_ok=True)
                    
                    for mi_channel in group['channels']:
                        if mi_channel in ch_names:
                            plot_4_panel_comparison(
                                x_mi_cls_Left, x_mi_cls_Right, fs, ch_names, 
                                target_ch=mi_channel, 
                                subject=mapped_subject, session=session,
                                cls_names=(task_name_Left, task_name_Right), 
                                save_dir=group_save_dir,           
                                focus_band=group['band'],          
                                focus_name=group['name'],          
                                psd_xmin=group['psd_xmin']         
                            )
                
            print(f"{'='*60}\n")

    # ==========================================
    # ⭐ 最終輸出: 分 Session 的綜合特徵診斷表格
    # ==========================================
    print("\n" + "✨" * 30)
    print("      📊 BCI Feature Detection Summary")
    print("✨" * 30)
    
    if summary_report:
        df_report = pd.DataFrame(summary_report)
        unique_sessions = sorted(df_report['Sess'].unique())
        
        for sess in unique_sessions:
            formatted_sess = sess.replace('s', '') if sess.startswith('s') else sess
            print(f"\n Session {formatted_sess}")
            df_sess = df_report[df_report['Sess'] == sess].drop(columns=['Sess'])
            print(df_sess.to_markdown(index=False, tablefmt="github"))
    else:
        print("No valid data processed.")
        
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="TFA2: BCI 時頻分析 4-Panel 比較與特徵自動檢驗系統")
    parser.add_argument("--base_dir", "--data_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集存放根目錄 (預設: /mnt/project/MIEXP/DATA_Cygnus)")
    parser.add_argument("--output_dir", type=str, default="TFA",
                        help="TFA 圖表輸出根目錄 (預設: TFA)")
    parser.add_argument("--channels", type=str, default="22", choices=["13", "22"],
                        help="通道設定 (預設: 22)")
    parser.add_argument("--ids", type=str, default=None,
                        help="指定分析受試者清單，逗號分隔 (留空為預設 24 人)")
    parser.add_argument("--sessions", type=str, default="s1,s2",
                        help="指定 Session 清單 (預設: s1,s2)")
    parser.add_argument("--log_file", type=str, default="TFA2.txt",
                        help="輸出 Log 檔名 (預設: TFA2.txt)")
    parser.add_argument("--do_ica", action="store_true", default=False,
                        help="是否啟用 ICA 進行眼電偽影自動濾除")
    parser.add_argument("--no_car", action="store_true", default=False,
                        help="關閉 CAR (Common Average Reference) 空間濾波")

    args = parser.parse_args()

    @tee_log(args.log_file)
    def _exec():
        run_tfa_analysis(
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            channels=args.channels,
            ids=args.ids,
            sessions=args.sessions,
            do_ica=args.do_ica,
            do_car=not args.no_car,
            log_file=args.log_file
        )

    _exec()


if __name__ == "__main__":
    main()