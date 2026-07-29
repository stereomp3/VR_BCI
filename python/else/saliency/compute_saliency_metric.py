import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
from scipy import signal
from PIL import Image
# XBrainLab 基礎類別引用
from XBrainLab.visualization.base import Visualizer
from collections import defaultdict
import scipy.stats as stats
from functools import wraps
from datetime import datetime
import sys

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
            print(f"✅ 輸出已保存到 {log_file}")
            return result

        return wrapper

    return decorator

class SaliencyPSDVisualizer(Visualizer):
    """
    客製化視覺化器：
    提供 plot_combined_for_label() 繪製特定 Label 的複合圖表
    (包含 Alpha/Beta Topomap 與 Saliency PSD)
    """

    def _compute_saliency_psd(self, method, label_idx, n_fft=None):
        """計算指定類別 Saliency 的 PSD"""
        saliency = self.get_saliency(method, label_idx)
        sfreq = self.epoch_data.sfreq

        if len(saliency) == 0:
            return None, None

        if n_fft is None:
            n_fft = int(sfreq)  # 1Hz 解析度

        # 使用 Welch 方法計算 PSD
        freqs, psd = signal.welch(saliency, fs=sfreq, nperseg=n_fft, axis=-1)
        return freqs, psd

    def plot_combined_for_label(self, label_idx, method="SmoothGrad", fmin=1, fmax=40,
                                save_path=None, use_abs=True, font_size=16,
                                show_y_axis=True, normalize=False):
        """
        繪製單一 Label 的綜合圖表
        上方：Alpha (8-13Hz) 與 Beta (13-30Hz) 的 Saliency Topomap
        下方：Saliency PSD
        """
        plt.rcParams.update({'font.size': font_size})

        try:
            class_name = self.epoch_data.label_map[label_idx]
        except (KeyError, IndexError):
            return

        positions = self.epoch_data.get_montage_position()
        chs = self.epoch_data.get_channel_names()

        # 計算 PSD
        freqs, psd = self._compute_saliency_psd(method, label_idx)
        if psd is None:
            print(f"警告：Label {class_name} 沒有 Saliency 資料可以繪製。")
            return

        # ==========================================
        # 設定版面配置 (精算比例以放大 Topomap 並容納大字體)
        # ==========================================
        fig = plt.figure(figsize=(12, 10))

        # 1. 下方長方形 PSD 
        # [left, bottom, width, height]
        ax_psd = fig.add_axes([0.12, 0.12, 0.84, 0.25])     
        
        # 2. 左上 Alpha (X 軸範圍：0.12 ~ 0.54，左側與 PSD 切齊)
        ax_alpha = fig.add_axes([0.12, 0.38, 0.42, 0.60])   
        
        # 3. 右上 Beta (X 軸範圍：0.54 ~ 0.96，右側與 PSD 切齊)
        ax_beta = fig.add_axes([0.54, 0.38, 0.42, 0.60])    

        cmap = 'Reds' if use_abs else 'RdBu_r'

        # ------------------------------------------
        # 1. 繪製左上角：Alpha Band (8-13Hz) Topomap
        # ------------------------------------------
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = psd[:, :, alpha_mask].mean(axis=-1).mean(axis=0)

        if normalize:
            a_min, a_max = np.min(alpha_power), np.max(alpha_power)
            if a_max > a_min:
                alpha_power = (alpha_power - a_min) / (a_max - a_min)
            else:
                alpha_power = np.zeros_like(alpha_power)

        im_alpha, _ = mne.viz.plot_topomap(
            alpha_power, pos=positions[:, :2], axes=ax_alpha,
            show=False, cmap=cmap, names=chs
        )

        # ------------------------------------------
        # 2. 繪製右上角：Beta Band (13-30Hz) Topomap
        # ------------------------------------------
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = psd[:, :, beta_mask].mean(axis=-1).mean(axis=0)

        if normalize:
            b_min, b_max = np.min(beta_power), np.max(beta_power)
            if b_max > b_min:
                beta_power = (beta_power - b_min) / (b_max - b_min)
            else:
                beta_power = np.zeros_like(beta_power)

        im_beta, _ = mne.viz.plot_topomap(
            beta_power, pos=positions[:, :2], axes=ax_beta,
            show=False, cmap=cmap, names=chs
        )

        # ------------------------------------------
        # 3. 繪製下方長方形：Saliency PSD
        # ------------------------------------------
        avg_psd = psd.mean(axis=0)

        if normalize:
            p_min, p_max = np.min(avg_psd), np.max(avg_psd)
            if p_max > p_min:
                avg_psd = (avg_psd - p_min) / (p_max - p_min)
            else:
                avg_psd = np.zeros_like(avg_psd)

        mask = (freqs >= fmin) & (freqs <= fmax)

        # 畫所有 Channel 的灰色細線
        ax_psd.plot(freqs[mask], avg_psd[:, mask].T, color='gray', alpha=0.3, linewidth=0.5)
        # 畫平均紅色粗線
        ax_psd.plot(freqs[mask], avg_psd[:, mask].mean(axis=0), color='red', linewidth=2.5, label='Mean Saliency')

        # 美化與背景色塊
        ax_psd.axvspan(8, 13, color='skyblue', alpha=0.15, label='Alpha (8-13)')
        ax_psd.axvspan(13, 30, color='salmon', alpha=0.1, label='Beta (13-30)')

        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_xlim(fmin, fmax)

        if normalize:
            ax_psd.set_ylim(-0.05, 1.05)
            y_label = "Power"
        else:
            y_label = "Power"

        if show_y_axis:
            ax_psd.set_ylabel(y_label)
        else:
            ax_psd.set_ylabel("")
            ax_psd.set_yticks([])

        ax_psd.legend(loc='upper right')
        ax_psd.grid(True, alpha=0.3)

        # ------------------------------------------
        # 4. 輸出與儲存
        # ------------------------------------------
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
    def compute_quantitative_metrics(self, label_idx, method="SmoothGrad"):
        """
        計算量化指標：
        1. MBSR (Motor-Band Saliency Ratio): 全腦區中，8-30Hz 頻段佔 1-40Hz 的比例
        2. MSFI (Motor Saliency Focus Index): 僅限 8-30Hz 頻段內，運動腦區特徵佔全腦區的比例
        """
        # 計算 PSD: (Trials, Channels, Freqs)
        freqs, psd = self._compute_saliency_psd(method, label_idx)
        if psd is None:
            return None, None

        # 設定頻帶遮罩
        mask_1_40 = (freqs >= 1) & (freqs <= 40)
        mask_8_30 = (freqs >= 8) & (freqs <= 30)

        # ==========================================
        # 1. 計算 MSFI (嚴格限制在 8~30 Hz)
        # ==========================================
        # 擷取 8-30Hz 的頻率與 PSD
        freqs_8_30 = freqs[mask_8_30]
        psd_8_30 = psd[:, :, mask_8_30]

        # 針對 8-30Hz 進行積分，計算每個 Channel 的能量 W_c (先積分，再對 Trial 取平均)
        W_c_trials = np.trapz(psd_8_30, freqs_8_30, axis=-1)  # Shape: (Trials, Channels)
        W_c = np.mean(W_c_trials, axis=0)                     # Shape: (Channels,)

        chs = self.epoch_data.get_channel_names()
        motor_channels = ['C3', 'Cz', 'C4', 'FC3', 'FCz', 'FC4'] # 運動腦區
        
        motor_power = 0
        all_power = np.sum(W_c)  # 這是全腦區在 8-30Hz 的總能量
        
        for idx, ch in enumerate(chs):
            if ch in motor_channels:
                motor_power += W_c[idx]
                
        msfi = (motor_power / all_power) * 100 if all_power > 0 else 0

        # ==========================================
        # 2. 計算 MBSR
        # ==========================================
        # P(f): 將所有 Trial 與 "所有 Channel" 的 PSD 平均，得到單一全腦頻譜 (Freqs,)
        P_f = psd.mean(axis=0).mean(axis=0)

        # 使用 np.trapz 進行數值積分 \int P(f) df
        power_1_40 = np.trapz(P_f[mask_1_40], freqs[mask_1_40])
        power_8_30 = np.trapz(P_f[mask_8_30], freqs[mask_8_30])

        mbsr = (power_8_30 / power_1_40) * 100 if power_1_40 > 0 else 0

        return mbsr, msfi

    # def compute_quantitative_metrics(self, label_idx, method="SmoothGrad"):
    #     """
    #     計算量化指標：
    #     1. MBSR (Motor-Band Saliency Ratio): 8-30Hz 頻段佔 1-40Hz 的比例
    #     2. MSFI (Motor Saliency Focus Index): 運動腦區特徵權重佔全腦區的比例
    #     """
    #     # 取得 Saliency: (Trials, Channels, Time)
    #     saliency = self.get_saliency(method, label_idx)
    #     if len(saliency) == 0:
    #         return None, None

    #     # ==========================================
    #     # 計算 MSFI
    #     # ==========================================
    #     # 計算每個 Channel 隨時間的絕對值總和 W_c = \sum_{t} |M_{c,t}|
    #     # 先對時間軸 (axis=-1) 求和，再對所有 Trial (axis=0) 取平均
    #     W_c_trials = np.sum(np.abs(saliency), axis=-1)
    #     W_c = np.mean(W_c_trials, axis=0)  # Shape: (Channels,)

    #     chs = self.epoch_data.get_channel_names()
    #     motor_channels = ['C3', 'Cz', 'C4', 'FC3', 'FCz', 'FC4']
    #     # motor_channels = ['C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']
        
    #     motor_power = 0
    #     all_power = np.sum(W_c)
        
    #     for idx, ch in enumerate(chs):
    #         if ch in motor_channels:
    #             motor_power += W_c[idx]
                
    #     msfi = (motor_power / all_power) * 100 if all_power > 0 else 0

    #     # ==========================================
    #     # 計算 MBSR
    #     # ==========================================
    #     freqs, psd = self._compute_saliency_psd(method, label_idx)
    #     if psd is None:
    #         return None, msfi

    #     # P(f): 將所有 Trial 與 Channel 的 PSD 平均，得到單一頻譜 (Freqs,)
    #     P_f = psd.mean(axis=0).mean(axis=0)

    #     # 設定頻帶遮罩
    #     mask_1_40 = (freqs >= 1) & (freqs <= 40)
    #     mask_8_30 = (freqs >= 8) & (freqs <= 30)

    #     # 使用 np.trapz 進行數值積分 \int P(f) df
    #     power_1_40 = np.trapz(P_f[mask_1_40], freqs[mask_1_40])
    #     power_8_30 = np.trapz(P_f[mask_8_30], freqs[mask_8_30])

    #     mbsr = (power_8_30 / power_1_40) * 100 if power_1_40 > 0 else 0

    #     return mbsr, msfi

def get_stats(data_list):
    """計算平均值與標準誤 (SEM) 用於繪製誤差陰影區間"""
    arr = np.array(data_list)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)  # 有效樣本數
    sem = std / np.sqrt(n)
    return mean, sem
def plot_trend(ax, data1, label1, color1, data2, label2, color2, title):
    runs_labels = [f'Run {i}' for i in range(1, 8)]
    base_font_size = 15
    m1, se1 = get_stats(data1)
    m2, se2 = get_stats(data2)

    # 繪製主線與陰影誤差區間
    ax.plot(runs_labels, m1, marker='o', color=color1, linewidth=3.5, label=label1, markersize=8)
    ax.fill_between(runs_labels, m1 - se1, m1 + se1, color=color1, alpha=0.15)

    ax.plot(runs_labels, m2, marker='s', color=color2, linewidth=3.5, label=label2, markersize=8)
    ax.fill_between(runs_labels, m2 - se2, m2 + se2, color=color2, alpha=0.15)

    # 圖表美化
    ax.set_title(title, fontsize=base_font_size + 4, fontweight='bold', pad=15)
    ax.set_xlabel('Training Runs', fontsize=base_font_size)
    ax.set_ylabel('Accuracy (%)', fontsize=base_font_size)

    # 鎖定 Y 軸範圍讓三張圖有相同的比較基準
    ax.set_ylim(45, 85)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=base_font_size - 1)

def plot_bar_box(data1, label1, color1, data2, label2, color2, title, ylabel, ylim=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    d1 = np.array(data1)[~np.isnan(data1)]
    d2 = np.array(data2)[~np.isnan(data2)]
    
    is_paired = (len(d1) == len(d2)) 
    
    # 統計檢定
    if is_paired:
        t_stat, p_val = stats.ttest_rel(d2, d1)
    else:
        t_stat, p_val = stats.ttest_ind(d2, d1)

    # Bar & Boxplot
    means, sems = [np.mean(d1), np.mean(d2)], [stats.sem(d1), stats.sem(d2)]
    ax.bar([1, 2], means, yerr=sems, color=[color1, color2], alpha=0.5, capsize=8, width=0.5)
    ax.boxplot([d1, d2], positions=[1, 2], widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='none', color='black', lw=1.5),
               medianprops=dict(color='black', lw=2), showfliers=False)

    # 連線 (Paired)
    if is_paired:
        for i in range(len(d1)):
            ax.plot([1, 2], [d1[i], d2[i]], color='gray', alpha=0.3, lw=1, marker='o', ms=5)

    # 標示顯著性
    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"\n{title}")
    print(f"t = {t_stat:.4f}, p = {p_val:.6f}")

    # y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)

    # ax.plot([1, 1, 2, 2], [y_max - 1, y_max, y_max, y_max - 1], lw=1.5, color='black')

    y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)

    ax.plot([1, 1, 2, 2],
            [y_max - 1, y_max, y_max, y_max - 1],
            lw=1.5, color='black')

    ax.text(1.5,
            y_max + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            f"{sig_str}\n(p={p_val:.4f})",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold')
   

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label1, label2], fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=30)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 判斷要存檔還是顯示
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def plot_metric_visualizations(mbsr_data, msfi_data, ids):
    """將算好的 MBSR 與 MSFI 字典資料抽出，處理向右對齊，並繪製趨勢圖與箱型圖，最後存成圖片"""
    s1_mbsr_runs, s2_mbsr_runs = [], []
    s1_msfi_runs, s2_msfi_runs = [], []
    
    s1_mbsr_avg, s2_mbsr_avg = [], []
    s1_msfi_avg, s2_msfi_avg = [], []

    # 內部輔助函式：處理 6 Run 向右平移的邏輯
    def extract_and_shift(data_dict, session, sub):
        runs_available = sorted(data_dict[session][sub].keys())
        shift = 1 if len(runs_available) == 6 else 0 # 只有 6 run 就向右 shift 1 格
        
        shifted_array = [np.nan] * 7
        for r in runs_available:
            target_idx = (r - 1) + shift
            if target_idx < 7:
                shifted_array[target_idx] = data_dict[session][sub][r]
        return shifted_array

    # 萃取資料
    for sub in ids:
        sub_s1_mbsr = extract_and_shift(mbsr_data, 's1', sub)
        sub_s2_mbsr = extract_and_shift(mbsr_data, 's2', sub)
        s1_mbsr_runs.append(sub_s1_mbsr)
        s2_mbsr_runs.append(sub_s2_mbsr)
        s1_mbsr_avg.append(np.nanmean(sub_s1_mbsr))
        s2_mbsr_avg.append(np.nanmean(sub_s2_mbsr))

        sub_s1_msfi = extract_and_shift(msfi_data, 's1', sub)
        sub_s2_msfi = extract_and_shift(msfi_data, 's2', sub)
        s1_msfi_runs.append(sub_s1_msfi)
        s2_msfi_runs.append(sub_s2_msfi)
        s1_msfi_avg.append(np.nanmean(sub_s1_msfi))
        s2_msfi_avg.append(np.nanmean(sub_s2_msfi))

    # ==========================================
    # 建立輸出資料夾
    # ==========================================
    out_dir = "./metric_plot"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📂 正在將指標圖片儲存至 {out_dir}/ ...")

    # ==========================================
    # 繪製 1：折線圖 (Trend) - S1 vs S2 
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, s1_mbsr_runs, 'Session 1', '#9467bd', s2_mbsr_runs, 'Session 2', '#8c564b', 
               'MBSR Trend (Session 1 vs Session 2)')
    ax.set_ylabel('MBSR (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MBSR_Trend.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, s1_msfi_runs, 'Session 1', '#9467bd', s2_msfi_runs, 'Session 2', '#8c564b', 
               'MSFI Trend (Session 1 vs Session 2)')
    ax.set_ylabel('MSFI (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MSFI_Trend.png'), dpi=200)
    plt.close(fig)

    # ==========================================
    # 繪製 2：柱狀箱型圖 (Bar + Boxplot) - S1 vs S2 
    # ==========================================
    plot_bar_box(s1_mbsr_avg, 'Session 1', '#9467bd', 
                 s2_mbsr_avg, 'Session 2', '#8c564b', 
                 'MBSR (Session 1 vs Session 2)', 'MBSR (%)', ylim=(0, 110),
                 save_path=os.path.join(out_dir, 'MBSR_BarBox.png'))
    
    plot_bar_box(s1_msfi_avg, 'Session 1', '#9467bd', 
                 s2_msfi_avg, 'Session 2', '#8c564b', 
                 'MSFI (Session 1 vs Session 2)', 'MSFI (%)', ylim=(0, 110),
                 save_path=os.path.join(out_dir, 'MSFI_BarBox.png'))

    print("✅ 圖片儲存完畢！")

def calculate_and_print_metrics():
    montage = mne.channels.make_standard_montage('standard_1020')
    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"
    ids = ["35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]
    sessions = ["s1", "s2"]
    runs = [f"run{i}" for i in range(1, 8)]
    is_13 = True

    print("\n計算量化指標中...\n")

    # 建立儲存結構： data[session][subject_id][run_num] = value
    mbsr_data = defaultdict(lambda: defaultdict(dict))
    msfi_data = defaultdict(lambda: defaultdict(dict))

    for subject_id in ids:
        for session in sessions:
            for run in runs:
                run_num = int(run.replace("run", ""))
                data_dir = os.path.join(base_dir, subject_id, session, run)
                if is_13:
                    load_path_eval = os.path.join(data_dir, "13_eval_record.pkl")
                    load_path_xb = os.path.join(data_dir, "13_eval_xb_epochs.pkl")
                else:
                    load_path_eval = os.path.join(data_dir, "22_eval_record.pkl")
                    load_path_xb = os.path.join(data_dir, "22_eval_xb_epochs.pkl")
                if not os.path.exists(load_path_eval) or not os.path.exists(load_path_xb):
                    continue
                
                try:
                    with open(load_path_eval, 'rb') as f:
                        eval_record = pickle.load(f)
                    with open(load_path_xb, 'rb') as f:
                        xb_epochs = pickle.load(f)

                    ch_names = xb_epochs.get_channel_names()
                    pos = [montage.get_positions()['ch_pos'].get(ch, [0, 0, 0]) for ch in ch_names]
                    xb_epochs.set_channels(ch_names, np.array(pos))

                    viz = SaliencyPSDVisualizer(eval_record, xb_epochs)

                    temp_mbsr = []
                    temp_msfi = []
                    
                    # 1. 分別算出 Label 0 和 Label 1
                    for target_class in [0, 1]:
                        mbsr, msfi = viz.compute_quantitative_metrics(label_idx=target_class, method="Gradient")
                        if mbsr is not None and msfi is not None:
                            temp_mbsr.append(mbsr)
                            temp_msfi.append(msfi)
                    
                    # 2. 相加除以 2，當作該 Run 的最終指標
                    if len(temp_mbsr) > 0:
                        mbsr_data[session][subject_id][run_num] = np.mean(temp_mbsr)
                        msfi_data[session][subject_id][run_num] = np.mean(temp_msfi)

                except Exception as e:
                    # 避免單筆錯誤中斷整個流程
                    print(f"Error computing {subject_id} {session} {run}: {e}")
                    continue

    # 3. 重新映射 ID (從小到大: S1 ~ S24)
    sorted_ids = sorted(ids, key=lambda x: int(x))
    id_map = {sub: f"S{i+1}" for i, sub in enumerate(sorted_ids)}

    # ------------------------------------------
    # 輸出 Run 級別的表格 (分 Session 與 指標)
    # ------------------------------------------
    metrics_to_print = [("MBSR", mbsr_data), ("MSFI", msfi_data)]
    
    for metric_name, data_dict in metrics_to_print:
        for session in sessions:
            print(f"\n### 表格: {metric_name} (%) - Session {session.upper()}")
            print("| **id** | **run 1** | **run 2** | **run 3** | **run 4** | **run 5** | **run 6** | **run 7** |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            
            for sub in sorted_ids:
                row_str = f"| {id_map[sub]} |"
                for r in range(1, 8):
                    val = data_dict[session][sub].get(r, None)
                    if val is not None:
                        row_str += f" {val:.2f} |"
                    else:
                        row_str += " ... |"
                print(row_str)

    # ------------------------------------------
    # 輸出 終極統計表格：將整個 Session 的 Run 平均
    # ------------------------------------------
    print("\n### 終極統計表格：各 Session 整體平均指標 (%)")
    print("| **id** | **S1 MBSR** | **S2 MBSR** | **S1 MSFI** | **S2 MSFI** |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    avg_s1_mbsr, avg_s2_mbsr = [], []
    avg_s1_msfi, avg_s2_msfi = [], []

    for sub in sorted_ids:
        # 計算單一受試者在整個 Session 的平均表現
        s1_mbsr_vals = list(mbsr_data['s1'][sub].values())
        s2_mbsr_vals = list(mbsr_data['s2'][sub].values())
        s1_msfi_vals = list(msfi_data['s1'][sub].values())
        s2_msfi_vals = list(msfi_data['s2'][sub].values())

        s1_mbsr = np.mean(s1_mbsr_vals) if s1_mbsr_vals else np.nan
        s2_mbsr = np.mean(s2_mbsr_vals) if s2_mbsr_vals else np.nan
        s1_msfi = np.mean(s1_msfi_vals) if s1_msfi_vals else np.nan
        s2_msfi = np.mean(s2_msfi_vals) if s2_msfi_vals else np.nan

        # 蒐集全體平均用
        if not np.isnan(s1_mbsr): avg_s1_mbsr.append(s1_mbsr)
        if not np.isnan(s2_mbsr): avg_s2_mbsr.append(s2_mbsr)
        if not np.isnan(s1_msfi): avg_s1_msfi.append(s1_msfi)
        if not np.isnan(s2_msfi): avg_s2_msfi.append(s2_msfi)

        # 格式化輸出
        s1_mbsr_str = f"{s1_mbsr:.2f}" if not np.isnan(s1_mbsr) else "..."
        s2_mbsr_str = f"{s2_mbsr:.2f}" if not np.isnan(s2_mbsr) else "..."
        s1_msfi_str = f"{s1_msfi:.2f}" if not np.isnan(s1_msfi) else "..."
        s2_msfi_str = f"{s2_msfi:.2f}" if not np.isnan(s2_msfi) else "..."

        print(f"| {id_map[sub]} | {s1_mbsr_str} | {s2_mbsr_str} | {s1_msfi_str} | {s2_msfi_str} |")

    # 全局總平均
    print(f"| **Mean** | **{np.mean(avg_s1_mbsr):.2f}** | **{np.mean(avg_s2_mbsr):.2f}** | **{np.mean(avg_s1_msfi):.2f}** | **{np.mean(avg_s2_msfi):.2f}** |")
    plot_metric_visualizations(mbsr_data, msfi_data, ids)

@tee_log("compute_saliency_metric.txt")
def main():
    calculate_and_print_metrics()

if __name__ == "__main__":
    main()