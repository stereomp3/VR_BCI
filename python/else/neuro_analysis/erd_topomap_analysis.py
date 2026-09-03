"""
================================================================================
BCI 全局 ERD Topomap 空間地形圖與學習軌跡分析系統 (Session Median 專用版)
(Global BCI ERD Topomap Spatial & Learning Progression Analyzer)
================================================================================

【功能特色】
1. 採用最佳全域基準策略 (Session Median Baseline)：
   - 以全 Session 4 秒 MI Data 之頻段能量中位數作為強健基準 (P_base = Median(P_trials))。
   - 擷取 Active MI 視窗 (預設 1.0s ~ 3.5s) 計算任務能量 (P_task)。
   - 計算標準去同步化百分比：ERD% = (P_task - P_base) / P_base * 100%。

2. 三大類專業論文級 Topomap 獨立輸出 (儲存於 <output_dir>/<id>/)：
   - `1_ERD_Topomap_Left_vs_Right_<ID>_<SESS>.png`  : 左右手 (Left/Right MI) x 頻段 (Mu 8-12Hz / Beta 13-30Hz) 空間對比
   - `2_ERD_Topomap_Run_Evolution_<ID>_<SESS>.png`   : Run 演化學習進程 (支援 6 Run: Run 1, 2, 4, 5, 6, 7，全電極文字標籤)
   - `3_ERD_Topomap_Differential_<ID>_<SESS>.png`    : 差分空間地形圖 (Left MI - Right MI) 展現左右側化偶極分化度

3. 支援 `-all` / `--all` 全受試者 (S1~S24, ID: 35~70) 批次自動運算與獨立資料夾歸檔。
4. 內建 `--demo` 擬真資料測試模式，無實體資料亦可快速驗證圖表生成。

使用範例：
  # 1. 批次生成所有 24 位受試者 (S1~S24)
  python erd_topomap_analysis.py --data_dir /mnt/project/MIEXP/DATA_Cygnus -all

  # 2. 生成單一受試者 (例如 Subject 70, Session 1)
  python erd_topomap_analysis.py --data_dir /mnt/project/MIEXP/DATA_Cygnus --subject 70 --session s1

  # 3. 執行 Demo 擬真測試
  python erd_topomap_analysis.py --demo
================================================================================
"""

import os
import sys
import re
import argparse
import datetime
from datetime import datetime, timezone, timedelta
from functools import wraps

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

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt, welch

# 嘗試引入 MNE
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# 設定 Matplotlib 樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 0. 系統通道定義與全受試者對照表
# ==============================================================================
CH_NAMES_22 = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4',
    'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'
]

CH_INDICES_22 = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33]

CH_NAMES_13 = [
    'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'Pz'
]

CH_INDICES_13 = [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 28]

ALL_SUBJECT_IDS = [
    "35", "37", "38", "40", "41", "42", "43", "44", "45", "47",
    "48", "50", "51", "52", "54", "55", "57", "58", "63", "64",
    "65", "68", "69", "70"
]

SUBJECT_MAP = {
    "35": "S1", "37": "S2", "38": "S3", "40": "S4", "41": "S5",
    "42": "S6", "43": "S7", "44": "S8", "45": "S9", "47": "S10",
    "48": "S11", "50": "S12", "51": "S13", "52": "S14", "54": "S15",
    "55": "S16", "57": "S17", "58": "S18", "63": "S19", "64": "S20",
    "65": "S21", "68": "S22", "69": "S23", "70": "S24"
}


def get_subject_display_name(raw_sub_id):
    """將原始 ID (例如 70, 35) 或代號 (例如 S24) 轉換為 Subject 1 ~ Subject 24 正式名稱"""
    sub_str = str(raw_sub_id).strip()
    if sub_str in SUBJECT_MAP:
        s_code = SUBJECT_MAP[sub_str]  # e.g. "S24"
        num = s_code.replace("S", "")
        return f"Subject {num}"
    elif sub_str.upper().startswith("S") and sub_str[1:].isdigit():
        return f"Subject {int(sub_str[1:])}"
    elif sub_str.isdigit() and 1 <= int(sub_str) <= 24:
        return f"Subject {int(sub_str)}"
    else:
        return f"Subject {sub_str}"


# ==============================================================================
# 1. 輸出日誌工具
# ==============================================================================
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
        log_file = f"erd_topomap_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
            print(f"📄 執行記錄已儲存至: {log_file}")
            return result
        return wrapper
    return decorator


def infer_channel_names(n_channels):
    """根據 channel 數量推測標籤清單"""
    if n_channels == 22:
        return CH_NAMES_22
    elif n_channels == 13:
        return CH_NAMES_13
    else:
        return [f"Ch{i+1}" for i in range(n_channels)]


# ==============================================================================
# 2. 頻譜能量與 Session Median ERD 核心運算模組
# ==============================================================================
def compute_band_power(x_data, fs=500, band=(8, 12)):
    """
    計算訊號在特定頻段的平均 PSD 能量
    x_data: shape (n_trials, n_channels, n_samples) 或 (n_channels, n_samples)
    return: shape (n_trials, n_channels) 或 (n_channels,)
    """
    if x_data.ndim == 2:
        x_data = x_data[np.newaxis, :, :]

    n_trials, n_channels, n_samples = x_data.shape
    nperseg = min(n_samples, int(fs * 0.5))
    if nperseg < 8:
        nperseg = n_samples

    freqs, psd = welch(x_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    mask = (freqs >= band[0]) & (freqs <= band[1])

    if np.sum(mask) == 0:
        band_power = np.mean(psd, axis=-1)
    else:
        band_power = np.mean(psd[:, :, mask], axis=-1)

    return band_power.squeeze()


def extract_session_median_powers(x_trials, fs=500, band=(8, 12), task_range=(1.0, 3.5)):
    """
    使用全 Session 4s MI 能量中位數作為 Baseline，並擷取 Active MI 視窗 (1.0~3.5s) 作為 Task 能量
    x_trials: (n_trials, n_channels, n_samples)
    """
    n_trials, n_channels, n_samples = x_trials.shape
    t_axis = np.arange(n_samples) / fs

    # 1. 擷取 Task 區間 (1.0s ~ 3.5s)
    t_mask = (t_axis >= task_range[0]) & (t_axis <= task_range[1])
    if np.sum(t_mask) == 0:
        task_data = x_trials
    else:
        task_data = x_trials[:, :, t_mask]

    p_task = compute_band_power(task_data, fs=fs, band=band)  # (n_trials, n_channels)
    if p_task.ndim == 1:
        p_task = p_task[np.newaxis, :]

    # 2. 全 Session 4s 中位數基準
    all_powers = compute_band_power(x_trials, fs=fs, band=band)
    if all_powers.ndim == 1:
        all_powers = all_powers[np.newaxis, :]
    med_p = np.median(all_powers, axis=0)  # (n_channels,)
    p_base = np.tile(med_p, (n_trials, 1))  # (n_trials, n_channels)

    return p_base, p_task


def calculate_erd_percentage(p_base, p_task):
    """
    標準 Pfurtscheller 公式計算 ERD 百分比：
    ERD% = (P_task - P_base) / P_base * 100%
    """
    erd = (p_task - p_base) / (p_base + 1e-8) * 100.0
    return erd


# ==============================================================================
# 3. Topomap 繪圖核心工具 (MNE 與備用 2D 內插自適應支援)
# ==============================================================================
def draw_single_topomap_ax(erd_values, ch_names, ax, title="", vmax=60.0,
                           show_names=True, highlight_chs=('C3', 'C4', 'Cz')):
    """
    在指定的 Matplotlib Axes 上繪製單一 ERD Topomap
    """
    n_ch = len(ch_names)
    vmax = float(vmax)
    vmin = -vmax

    if HAS_MNE:
        # 使用 MNE 標準 10-20 座標系統
        info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')

        try:
            # 支援新舊版本 MNE 參數
            im, _ = mne.viz.plot_topomap(
                data=erd_values,
                pos=info,
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(vmin, vmax),
                sensors=True,
                names=ch_names if show_names else None,
                contours=4
            )
            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            return im
        except Exception:
            pass

    # 備用方案 (Fallback)：2D 簡易散點極坐標繪圖
    ax.set_title(title, fontsize=11, fontweight='bold')
    sc = ax.scatter(np.arange(n_ch), erd_values, c=erd_values, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=120)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_names, rotation=45, fontsize=8)
    return sc


# ==============================================================================
# 4. 【獨立圖表 1】左右手 x 頻段 2x2 空間對比 Topomap
# ==============================================================================
def plot_erd_topomap_left_vs_right(all_runs_base, all_runs_task, all_runs_y,
                                   ch_names, subject_label="Subject 24", session_label="Session 1",
                                   save_dir="output", vmax=50.0):
    """
    輸出圖 1：Left MI vs. Right MI 在 Mu (8-12Hz) 與 Beta (13-30Hz) 之空間對比 Topomap
    """
    os.makedirs(save_dir, exist_ok=True)

    # 合併所有 Run
    flat_base_mu = np.concatenate([r['base_mu'] for r in all_runs_base])
    flat_task_mu = np.concatenate([r['task_mu'] for r in all_runs_task])
    flat_base_beta = np.concatenate([r['base_beta'] for r in all_runs_base])
    flat_task_beta = np.concatenate([r['task_beta'] for r in all_runs_task])
    flat_y = np.concatenate(all_runs_y)

    left_mask = (flat_y == 1)
    right_mask = (flat_y == 0)

    # 計算平均 ERD%
    erd_mu_l = np.mean(calculate_erd_percentage(flat_base_mu[left_mask], flat_task_mu[left_mask]), axis=0)
    erd_mu_r = np.mean(calculate_erd_percentage(flat_base_mu[right_mask], flat_task_mu[right_mask]), axis=0)

    erd_beta_l = np.mean(calculate_erd_percentage(flat_base_beta[left_mask], flat_task_beta[left_mask]), axis=0)
    erd_beta_r = np.mean(calculate_erd_percentage(flat_base_beta[right_mask], flat_task_beta[right_mask]), axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.25, wspace=0.15, right=0.88, top=0.88)

    fig.suptitle(f"Motor Imagery ERD/ERS Topomap: {subject_label} | {session_label}",
                 fontsize=14, fontweight='bold', y=0.97)

    # (0, 0) Left MI Mu
    im1 = draw_single_topomap_ax(erd_mu_l, ch_names, axs[0, 0],
                                 title="Left Hand MI - Mu Band (8-12 Hz)", vmax=vmax, show_names=True)
    # (0, 1) Right MI Mu
    im2 = draw_single_topomap_ax(erd_mu_r, ch_names, axs[0, 1],
                                 title="Right Hand MI - Mu Band (8-12 Hz)", vmax=vmax, show_names=True)
    # (1, 0) Left MI Beta
    im3 = draw_single_topomap_ax(erd_beta_l, ch_names, axs[1, 0],
                                 title="Left Hand MI - Beta Band (13-30 Hz)", vmax=vmax, show_names=True)
    # (1, 1) Right MI Beta
    im4 = draw_single_topomap_ax(erd_beta_r, ch_names, axs[1, 1],
                                 title="Right Hand MI - Beta Band (13-30 Hz)", vmax=vmax, show_names=True)

    # 統一右側垂直 Colorbar
    cbar_ax = fig.add_axes([0.91, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('ERD / ERS (%)', fontsize=11, fontweight='bold')

    out_name = f"1_ERD_Topomap_Left_vs_Right_{subject_label.replace(' ', '')}_{session_label.replace(' ', '')}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ [已生成] 左右手對比 Topomap: {out_name}")


# ==============================================================================
# 5. 【獨立圖表 2】Run 1 ~ Run 7 學習進程演化 Topomap (Mu & Beta 頻段並列)
# ==============================================================================
def plot_erd_topomap_run_evolution(all_runs_base, all_runs_task, all_runs_y,
                                    ch_names, run_names=None,
                                    subject_label="Subject 24", session_label="Session 1",
                                    save_dir="output", vmax=50.0):
    """
    輸出圖 2：Run 演化進程 Topomap，支援 6 Run (Run 1, 2, 4, 5, 6, 7) 並同時展示 Mu 與 Beta 頻段 (4 列)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_runs = len(all_runs_base)
    if n_runs < 2:
        return

    # 處理 6 Run 與自訂 Run 名稱順序
    if run_names is None or len(run_names) != n_runs:
        if n_runs == 6:
            run_names = ["Run 1", "Run 2", "Run 4", "Run 5", "Run 6", "Run 7"]
        else:
            run_names = [f"Run {i+1}" for i in range(n_runs)]

    fig, axs = plt.subplots(4, n_runs, figsize=(3.6 * n_runs, 15.0))
    fig.subplots_adjust(hspace=0.28, wspace=0.12, right=0.91, top=0.94, bottom=0.03)

    fig.suptitle(f"Neural Learning Progression (Mu & Beta Band ERD Topomap Evolution): {subject_label} | {session_label}",
                 fontsize=15, fontweight='bold', y=0.98)

    for r in range(n_runs):
        b_mu = all_runs_base[r]['base_mu']
        t_mu = all_runs_task[r]['task_mu']
        b_beta = all_runs_base[r]['base_beta']
        t_beta = all_runs_task[r]['task_beta']
        y_r = all_runs_y[r]

        l_mask = (y_r == 1)
        r_mask = (y_r == 0)

        # Mu 頻段 (8-12 Hz)
        erd_mu_l = np.mean(calculate_erd_percentage(b_mu[l_mask], t_mu[l_mask]), axis=0) if np.sum(l_mask) > 0 else np.zeros(len(ch_names))
        erd_mu_r = np.mean(calculate_erd_percentage(b_mu[r_mask], t_mu[r_mask]), axis=0) if np.sum(r_mask) > 0 else np.zeros(len(ch_names))

        # Beta 頻段 (13-30 Hz)
        erd_beta_l = np.mean(calculate_erd_percentage(b_beta[l_mask], t_beta[l_mask]), axis=0) if np.sum(l_mask) > 0 else np.zeros(len(ch_names))
        erd_beta_r = np.mean(calculate_erd_percentage(b_beta[r_mask], t_beta[r_mask]), axis=0) if np.sum(r_mask) > 0 else np.zeros(len(ch_names))

        r_title_name = run_names[r] if r < len(run_names) else f"Run {r+1}"

        # 第 1 列: Left Hand MI (Mu 8-12 Hz)
        im = draw_single_topomap_ax(erd_mu_l, ch_names, axs[0, r], title=f"{r_title_name} (Left MI - Mu)",
                                    vmax=vmax, show_names=True)

        # 第 2 列: Right Hand MI (Mu 8-12 Hz)
        draw_single_topomap_ax(erd_mu_r, ch_names, axs[1, r], title=f"{r_title_name} (Right MI - Mu)",
                               vmax=vmax, show_names=True)

        # 第 3 列: Left Hand MI (Beta 13-30 Hz)
        draw_single_topomap_ax(erd_beta_l, ch_names, axs[2, r], title=f"{r_title_name} (Left MI - Beta)",
                               vmax=vmax, show_names=True)

        # 第 4 列: Right Hand MI (Beta 13-30 Hz)
        draw_single_topomap_ax(erd_beta_r, ch_names, axs[3, r], title=f"{r_title_name} (Right MI - Beta)",
                               vmax=vmax, show_names=True)

    # 統一右側垂直 Colorbar
    cbar_ax = fig.add_axes([0.93, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('ERD / ERS (%)', fontsize=11, fontweight='bold')

    out_name = f"2_ERD_Topomap_Run_Evolution_{subject_label.replace(' ', '')}_{session_label.replace(' ', '')}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ [已生成] 學習演化 Topomap: {out_name}")


# ==============================================================================
# 6. 【獨立圖表 3】差分空間地形圖 (Differential Topomap: Left - Right)
# ==============================================================================
def plot_erd_topomap_differential(all_runs_base, all_runs_task, all_runs_y,
                                  ch_names, subject_label="Subject 24", session_label="Session 1",
                                  save_dir="output", vmax=60.0):
    """
    輸出圖 3：差分空間地形圖 (ΔERD = Left MI ERD - Right MI ERD)
    """
    os.makedirs(save_dir, exist_ok=True)

    flat_base_mu = np.concatenate([r['base_mu'] for r in all_runs_base])
    flat_task_mu = np.concatenate([r['task_mu'] for r in all_runs_task])
    flat_base_beta = np.concatenate([r['base_beta'] for r in all_runs_base])
    flat_task_beta = np.concatenate([r['task_beta'] for r in all_runs_task])
    flat_y = np.concatenate(all_runs_y)

    left_mask = (flat_y == 1)
    right_mask = (flat_y == 0)

    erd_mu_l = np.mean(calculate_erd_percentage(flat_base_mu[left_mask], flat_task_mu[left_mask]), axis=0)
    erd_mu_r = np.mean(calculate_erd_percentage(flat_base_mu[right_mask], flat_task_mu[right_mask]), axis=0)
    diff_mu = erd_mu_l - erd_mu_r

    erd_beta_l = np.mean(calculate_erd_percentage(flat_base_beta[left_mask], flat_task_beta[left_mask]), axis=0)
    erd_beta_r = np.mean(calculate_erd_percentage(flat_base_beta[right_mask], flat_task_beta[right_mask]), axis=0)
    diff_beta = erd_beta_l - erd_beta_r

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(wspace=0.20, right=0.88, top=0.82)

    fig.suptitle(f"Differential Spatial Topomap (ΔERD = Left Hand MI - Right Hand MI): {subject_label} | {session_label}",
                 fontsize=13.5, fontweight='bold', y=0.96)

    im1 = draw_single_topomap_ax(diff_mu, ch_names, axs[0],
                                 title="Mu Band (8-12 Hz) ΔERD", vmax=vmax, show_names=True)

    im2 = draw_single_topomap_ax(diff_beta, ch_names, axs[1],
                                 title="Beta Band (13-30 Hz) ΔERD", vmax=vmax, show_names=True)

    cbar_ax = fig.add_axes([0.91, 0.22, 0.025, 0.55])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Differential ΔERD (%)', fontsize=11, fontweight='bold')

    out_name = f"3_ERD_Topomap_Differential_{subject_label.replace(' ', '')}_{session_label.replace(' ', '')}.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ [已生成] 差分側化 Topomap: {out_name}")


# ==============================================================================
# 7. 單一受試者單一 Session 處理主流程
# ==============================================================================
def process_subject_session_topomap(data_dir, raw_sub_id, session_str, output_root,
                                    channels="22", is_demo=False, task_range=(1.0, 3.5)):
    """
    處理單一受試者的 ERD Topomap 流程 (採用 Session Median Baseline)
    """
    sub_dir_name = str(raw_sub_id)
    subject_out_dir = os.path.join(output_root, sub_dir_name)
    os.makedirs(subject_out_dir, exist_ok=True)

    subject_label = get_subject_display_name(raw_sub_id)
    sess_num = "1" if "1" in str(session_str).lower() else "2"
    session_label = f"Session {sess_num}"

    all_runs_base = []
    all_runs_task = []
    all_runs_y = []
    run_names = []
    ch_names = None

    if is_demo:
        # 生成擬真資料
        ch_names = CH_NAMES_22
        run_names = [f"Run {r+1}" for r in range(7)]
        for r in range(7):
            n_tr = 40
            n_ch = 22
            n_samp = 2000
            fs = 500

            x_fake = np.random.randn(n_tr, n_ch, n_samp) * 8.0
            y_fake = np.random.choice([0, 1], size=n_tr)

            # 模擬對側 ERD 效果隨 Run 加強
            c3_idx = ch_names.index('C3')
            c4_idx = ch_names.index('C4')
            prog = (r + 1) / 7.0

            time_vec = np.arange(n_samp) / fs
            t_mask = (time_vec >= 1.0) & (time_vec <= 3.5)

            for t in range(n_tr):
                if y_fake[t] == 1:  # Left MI -> C4 ERD (能量下降)
                    x_fake[t, c4_idx, t_mask] *= (1.0 - 0.45 * prog)
                else:               # Right MI -> C3 ERD (能量下降)
                    x_fake[t, c3_idx, t_mask] *= (1.0 - 0.45 * prog)

            p_base_mu, p_task_mu = extract_session_median_powers(
                x_fake, fs=fs, band=(8, 12), task_range=task_range
            )
            p_base_beta, p_task_beta = extract_session_median_powers(
                x_fake, fs=fs, band=(13, 30), task_range=task_range
            )

            all_runs_base.append({'base_mu': p_base_mu, 'base_beta': p_base_beta})
            all_runs_task.append({'task_mu': p_task_mu, 'task_beta': p_task_beta})
            all_runs_y.append(y_fake)

    else:
        subject_dir = os.path.join(data_dir, str(raw_sub_id), session_str)

        for r in range(1, 8):
            run_dir = os.path.join(subject_dir, f"run{r}")
            pt_candidates = [
                os.path.join(run_dir, f"mi_{channels}.pt"),
                os.path.join(run_dir, "mi_22.pt"),
                os.path.join(run_dir, "mi_13.pt"),
                os.path.join(run_dir, "data.pt"),
                os.path.join(subject_dir, f"run_{r}.pt"),
            ]

            for pt_path in pt_candidates:
                if os.path.exists(pt_path):
                    try:
                        data = torch.load(pt_path, map_location='cpu')
                        x_d = data.get('x_data', data.get('x'))
                        y_d = data.get('y_data', data.get('y'))
                        if isinstance(x_d, torch.Tensor): x_d = x_d.numpy()
                        if isinstance(y_d, torch.Tensor): y_d = y_d.numpy()

                        if x_d is not None and len(x_d) > 0:
                            if ch_names is None:
                                ch_names = infer_channel_names(x_d.shape[1])

                            p_base_mu, p_task_mu = extract_session_median_powers(
                                x_d, fs=500, band=(8, 12), task_range=task_range
                            )
                            p_base_beta, p_task_beta = extract_session_median_powers(
                                x_d, fs=500, band=(13, 30), task_range=task_range
                            )

                            all_runs_base.append({'base_mu': p_base_mu, 'base_beta': p_base_beta})
                            all_runs_task.append({'task_mu': p_task_mu, 'task_beta': p_task_beta})
                            all_runs_y.append(y_d)
                            run_names.append(f"Run {r}")
                            break
                    except Exception:
                        pass

    if len(all_runs_base) == 0:
        print(f"  ⚠️ [跳過] 未在受試者 {raw_sub_id} {session_str} 找到有效資料。")
        return False

    # 若剛好讀到 6 個 Run 且命名為 Run 1~Run 6，自動對齊為 Run 2, Run 3, Run 4, Run 5, Run 6, Run 7
    if len(run_names) == 6 and run_names == [f"Run {i}" for i in range(1, 7)]:
        run_names = ["Run 2", "Run 3", "Run 4", "Run 5", "Run 6", "Run 7"]

    print(f"  ▶ 正在為 {subject_label} {session_label} 生成 Topomap...")

    # 1. 輸出左右手 x 頻段 2x2 Topomap
    plot_erd_topomap_left_vs_right(
        all_runs_base, all_runs_task, all_runs_y, ch_names,
        subject_label=subject_label, session_label=session_label,
        save_dir=subject_out_dir
    )

    # 2. 輸出 Run 演化進程 Topomap
    plot_erd_topomap_run_evolution(
        all_runs_base, all_runs_task, all_runs_y, ch_names,
        run_names=run_names,
        subject_label=subject_label, session_label=session_label,
        save_dir=subject_out_dir
    )

    # 3. 輸出差分側化 Topomap
    plot_erd_topomap_differential(
        all_runs_base, all_runs_task, all_runs_y, ch_names,
        subject_label=subject_label, session_label=session_label,
        save_dir=subject_out_dir
    )

    print(f"  ✓ 成功儲存所有 Topomap 至: {os.path.abspath(subject_out_dir)}")
    return True


# ==============================================================================
# 8. 主程式入口
# ==============================================================================
@tee_log()
def main():
    parser = argparse.ArgumentParser(description="BCI 全局 ERD Topomap 空間特徵與學習軌跡分析系統 (Session Median 專用版，支援 -all 全受試者批次輸出)")
    parser.add_argument("-all", "--all", dest="all_subjects", action="store_true",
                        help="批次生成所有 24 位受試者 (S1~S24, ID: 35~70) 的全部 Topomap")
    parser.add_argument("--demo", action="store_true", help="執行 Demo 擬真合成資料模式")
    parser.add_argument("--data_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集根目錄路徑")
    parser.add_argument("--channels", type=str, default="22", choices=["13", "22"],
                        help="優先載入之通道模式 ('13' 或 '22')")
    parser.add_argument("--subject", type=str, default="70",
                        help="單一受試者 ID (例如: 70, 44, 37 或 S24, S8, S2)")
    parser.add_argument("--ids", type=str, default=None,
                        help="指定受試者清單，以逗號分隔 (例如: '35,37,70')")
    parser.add_argument("--session", type=str, default="all", choices=["s1", "s2", "all"],
                        help="指定 Session (s1, s2, 或 all)")
    parser.add_argument("--output_dir", type=str, default="erd_topomap_output",
                        help="圖表儲存根目錄 (內部會自動建立 \\<id>\\ 子資料夾)")

    parser.add_argument("--task_start", type=float, default=1.0,
                        help="Task 任務視窗起始秒數 (預設 1.0s)")
    parser.add_argument("--task_end", type=float, default=3.5,
                        help="Task 任務視窗結束秒數 (預設 3.5s)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 85)
    print(f"🚀 BCI 全局 ERD Topomap 空間地形圖與學習軌跡分析系統 開始 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📌 基準策略: [Session Median (4s MI Data 中位數基準)]")
    print("=" * 85)

    if args.ids:
        subjects_to_process = [x.strip() for x in args.ids.split(",") if x.strip()]
        print(f"🌟 [指定清單模式] 即將處理 {len(subjects_to_process)} 位受試者: {subjects_to_process}")
    elif args.all_subjects:
        subjects_to_process = ALL_SUBJECT_IDS
        print(f"🌟 [批次模式] 即將處理全部 24 位受試者: {subjects_to_process}")
    else:
        subjects_to_process = [args.subject]
        print(f"🎯 [單一模式] 即將處理受試者: {args.subject}")

    if args.session == "all":
        sessions_to_process = ["s1", "s2"]
    else:
        sessions_to_process = [args.session]

    total_tasks = len(subjects_to_process) * len(sessions_to_process)
    completed_count = 0

    task_range = (args.task_start, args.task_end)

    for s_idx, sub_id in enumerate(subjects_to_process):
        print(f"\n{'='*30} [{s_idx+1}/{len(subjects_to_process)}] 受試者 ID: {sub_id} ({SUBJECT_MAP.get(sub_id, sub_id)}) {'='*30}")
        for sess in sessions_to_process:
            success = process_subject_session_topomap(
                data_dir=args.data_dir,
                raw_sub_id=sub_id,
                session_str=sess,
                output_root=args.output_dir,
                channels=args.channels,
                is_demo=args.demo,
                task_range=task_range
            )
            if success:
                completed_count += 1

    print("\n" + "=" * 85)
    print(f"🎉 全部 Topomap 處理完畢！成功完成 {completed_count}/{total_tasks} 個 Session 分析。")
    print(f"📁 輸出圖表已儲存至: {os.path.abspath(args.output_dir)}/<id>/")
    print("=" * 85)


if __name__ == "__main__":
    main()
