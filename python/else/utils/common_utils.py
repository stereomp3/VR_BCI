"""
VR-BCI 共用工具模組 (Common Utilities)
集中管理跨檔案重複的：
1. Logging 與系統工具 (Tee, tee_log)
2. 訊號與資料前處理 (arrange_by_label, bandpass, down_sample, crop_center, prepare_datasets, cat_all_data, pad_run_data)
3. 受試者、Session 與 Condition 實驗設定 (DEFAULT_IDS, S1_COND_MAP, CHANNEL_CONFIGS, reverse_condition, get_subject_alias_map)
4. 統計檢定函式 (safe_wilcoxon, safe_mannwhitneyu, safe_ttest_1samp, safe_ttest_ind, calc_wsi)
5. 繪圖輔助元件 (plot_bar_box, plot_trend, add_significance_labels)
"""

import os
import sys
from datetime import datetime
from functools import wraps

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import numpy as np
import scipy.stats as stats
from scipy.signal import butter, filtfilt, decimate
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# ============================================================
# 1. Logging 與系統工具
# ============================================================

class Tee:
    """同時輸出到 stdout 與檔案"""
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
    """裝飾器：將 print 輸出同時存檔與顯示在終端機"""
    if log_file is None:
        log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

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


# ============================================================
# 2. 實驗設定、受試者與通道對照
# ============================================================

DEFAULT_IDS = [
    "35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50",
    "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"
]

DEFAULT_SESSIONS = ["s1", "s2"]

S1_COND_MAP = {
    "35": "B", "37": "A", "38": "B", "40": "B", "41": "B", "42": "A",
    "43": "A", "44": "B", "45": "B", "47": "B", "48": "B", "50": "A",
    "51": "B", "52": "B", "54": "B", "55": "A", "57": "B", "58": "A",
    "63": "A", "64": "A", "65": "A", "68": "A", "69": "A", "70": "A",
}

CHANNEL_CONFIGS = {
    "13": {
        "channel_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "raw_channel_index": [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 28],
        "ch_names": ['F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'Pz'],
        "mi_filename": "mi.pt",
        "model_filename": "scc_13_model.pth",
        "eval_record_filename": "13_eval_record.pkl",
        "eval_xb_epochs_filename": "13_eval_xb_epochs.pkl",
    },
    "22": {
        "channel_index": list(range(22)),
        "raw_channel_index": [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33],
        "ch_names": ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz',
                     'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'],
        "mi_filename": "mi_22.pt",
        "model_filename": "scc_22_model.pth",
        "eval_record_filename": "22_eval_record.pkl",
        "eval_xb_epochs_filename": "22_eval_xb_epochs.pkl",
    }
}


def reverse_condition(cond):
    """A <-> B 反轉"""
    return "B" if str(cond).upper() == "A" else "A"


def get_subject_alias_map(ids=None):
    """
    將受試者 ID (如 35, 37...) 排序映射為 S1, S2, ... S24
    """
    if ids is None:
        ids = DEFAULT_IDS
    sorted_ids = sorted(ids, key=lambda x: int(x))
    return {sub: f"S{i+1}" for i, sub in enumerate(sorted_ids)}


# ============================================================
# 3. 訊號與資料處理函式
# ============================================================

def arrange_by_label(x, y, f=None):
    """
    根據 y label 按照 0 1 0 1 交錯排列以平衡訓練資料。
    如果資料不平衡，未配對的剩餘資料排在最前面，配對好的排在後面。
    支援帶入或不帶入 f (failures 陣列)。
    """
    label_0_idx = np.where(y == 0)[0]
    label_1_idx = np.where(y == 1)[0]
    label_0_rev = label_0_idx[::-1]
    label_1_rev = label_1_idx[::-1]

    num_pairs = min(len(label_0_rev), len(label_1_rev))

    paired_0 = label_0_rev[:num_pairs][::-1]
    paired_1 = label_1_rev[:num_pairs][::-1]

    interleaved_idx = np.empty(num_pairs * 2, dtype=int)
    interleaved_idx[0::2] = paired_0
    interleaved_idx[1::2] = paired_1

    remaining_0 = label_0_rev[num_pairs:][::-1]
    remaining_1 = label_1_rev[num_pairs:][::-1]
    remaining_idx = np.concatenate((remaining_0, remaining_1))

    final_idx = np.concatenate((remaining_idx, interleaved_idx))

    x_sorted = x[final_idx]
    y_sorted = y[final_idx]

    if f is not None:
        f_sorted = f[final_idx]
        return x_sorted, y_sorted, f_sorted
    return x_sorted, y_sorted


def bandpass(data, fs=500, low=1, high=40):
    """帶通濾波器"""
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)


def down_sample(data, new_fs=125, old_fs=500):
    """降取樣"""
    decimation_factor = old_fs // new_fs
    return decimate(data, decimation_factor, axis=0, zero_phase=True)


def crop_center(data: np.ndarray, target_length: int) -> np.ndarray:
    """中心截取"""
    if target_length > data.shape[-1]:
        raise ValueError(f"target_length ({target_length}) 不能大於資料長度 ({data.shape[-1]})")
    total = data.shape[-1]
    cut = (total - target_length) // 2
    remainder = (total - target_length) % 2
    return data[..., cut: total - cut - remainder]


def prepare_datasets(x_np_data, y_np_data, valid_num=0, segment_len=500, stride=20):
    """
    滑動視窗增強與轉換為 PyTorch TensorDataset。
    輸入 x: (trial, channel, sample)
    輸出 dataset: X 具有維度 (N, 1, channel, sample)
    """
    x_np_data = np.transpose(x_np_data, (0, 2, 1))  # 轉成 (trial, sample, channel)
    augmented_segments_valid = []
    augmented_labels_valid = []
    augmented_segments_train = []
    augmented_labels_train = []

    for i, s in enumerate(x_np_data):
        label = y_np_data[i]
        if s.shape[0] < segment_len:
            continue

        for start in range(0, s.shape[0] - segment_len + 1, stride):
            data = s[start:start + segment_len]
            window = bandpass(data - np.mean(data, axis=1, keepdims=True))
            if i < valid_num:
                augmented_segments_valid.append(window)
                augmented_labels_valid.append(label)
            else:
                augmented_segments_train.append(window)
                augmented_labels_train.append(label)

    def to_dataset(segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))  # (N, channel, sample)
        X = torch.tensor(data_x, dtype=torch.float32).unsqueeze(1)  # (N, 1, channel, sample)
        y = F.one_hot(torch.tensor(label_list).long())
        return TensorDataset(X, y)

    dataset_train = to_dataset(augmented_segments_train, augmented_labels_train)
    if valid_num > 0:
        dataset_valid = to_dataset(augmented_segments_valid, augmented_labels_valid)
        return dataset_train, dataset_valid
    return dataset_train


def cat_all_data(data_list):
    """將多個檔案中的 x_data 與 y_data 合併為單一矩陣"""
    all_x_data = []
    all_y_data = []

    for p in data_list:
        if os.path.exists(p):
            train_data = torch.load(p, map_location='cpu')
            all_x_data.append(train_data['x_data'])
            all_y_data.append(train_data['y_data'])
        else:
            print(f"File not found: {p}")
            continue

    if len(all_x_data) == 0:
        return None, None
    return np.concatenate(all_x_data, axis=0), np.concatenate(all_y_data, axis=0)


def pad_run_data(arr, metric=None, target_len=7):
    """
    處理 Run 數據：
    - 將缺失值 ('...', 'nan', None) 轉為 np.nan
    - 若僅有 6 個有效 Run，對齊為 2, 3, 4, 5, 6, 7 (Run 1 缺失，設為 np.nan)
    - 若 metric 為 'off_run' 且數值小於等於 1.05，轉換為百分比 (x 100)
    """
    clean_arr = []
    for x in arr:
        if x is None or str(x).strip().lower() in ('...', 'nan', 'none', ''):
            clean_arr.append(np.nan)
        else:
            try:
                clean_arr.append(float(x))
            except (ValueError, TypeError):
                clean_arr.append(np.nan)

    valid_vals = [x for x in clean_arr if not np.isnan(x)]
    if len(valid_vals) == 6:
        # 6 個 Run 對齊為 2, 3, 4, 5, 6, 7 (Run 1 為 np.nan)
        clean_arr = [np.nan, valid_vals[0], valid_vals[1], valid_vals[2], valid_vals[3], valid_vals[4], valid_vals[5]]
    elif len(clean_arr) < target_len:
        clean_arr = clean_arr + [np.nan] * (target_len - len(clean_arr))
    elif len(clean_arr) > target_len:
        clean_arr = clean_arr[:target_len]

    clean_arr = np.array(clean_arr, dtype=float)

    if metric == 'off_run':
        if np.nanmax(clean_arr) <= 1.05:
            clean_arr = clean_arr * 100

    return clean_arr


# ============================================================
# 4. 統計輔助函式
# ============================================================

def safe_wilcoxon(x, y=None):
    """安全執行 Wilcoxon Signed-Rank Test，處理 NaN 與全為 0 的情況"""
    if y is not None:
        diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    else:
        diff = np.array(x, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 3 or np.all(diff == 0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(diff)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_mannwhitneyu(x, y):
    """安全執行 Mann-Whitney U test"""
    x_clean = np.array(x, dtype=float)[~np.isnan(x)]
    y_clean = np.array(y, dtype=float)[~np.isnan(y)]
    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan, np.nan
    try:
        res = stats.mannwhitneyu(x_clean, y_clean)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_ttest_1samp(x, popmean=0):
    """安全執行單樣本 t-test"""
    x_clean = np.array(x, dtype=float)[~np.isnan(x)]
    if len(x_clean) < 2:
        return np.nan, np.nan
    try:
        res = stats.ttest_1samp(x_clean, popmean)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_ttest_ind(x, y):
    """安全執行雙獨立樣本 t-test"""
    x_clean = np.array(x, dtype=float)[~np.isnan(x)]
    y_clean = np.array(y, dtype=float)[~np.isnan(y)]
    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan, np.nan
    try:
        res = stats.ttest_ind(x_clean, y_clean)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def calc_wsi(acc_list):
    """
    計算 Within-Session Improvement (WSI)：
    將後段 Run 平均減去前段 Run 平均。
    - 7 個 Run：Run 4-7 平均 - Run 1-3 平均
    - 6 個 Run：Run 3-6 平均 - Run 1-2 平均
    """
    valid_acc = []
    for x in acc_list:
        if x is not None and str(x).strip() not in ('...', 'nan', 'none', ''):
            val = float(x)
            if val <= 1.05:
                val *= 100
            valid_acc.append(val)

    if len(valid_acc) >= 7:
        return float(np.mean(valid_acc[3:7]) - np.mean(valid_acc[0:3]))
    elif len(valid_acc) == 6:
        return float(np.mean(valid_acc[2:6]) - np.mean(valid_acc[0:2]))
    return np.nan


# ============================================================
# 5. 繪圖輔助函式
# ============================================================

def plot_trend(ax, data1, label1, color1, data2, label2, color2, title, ylabel='Accuracy (%)', ylim=(45, 85)):
    """在給定的 ax 上繪製包含標準誤 (SEM) 陰影區間的趨勢折線圖"""
    runs_labels = [f'Run {i}' for i in range(1, 8)]
    base_font_size = 14

    arr1 = np.array(data1, dtype=float)
    arr2 = np.array(data2, dtype=float)

    m1 = np.nanmean(arr1, axis=0)
    se1 = np.nanstd(arr1, axis=0) / np.sqrt(np.sum(~np.isnan(arr1), axis=0))

    m2 = np.nanmean(arr2, axis=0)
    se2 = np.nanstd(arr2, axis=0) / np.sqrt(np.sum(~np.isnan(arr2), axis=0))

    ax.plot(runs_labels, m1, marker='o', color=color1, linewidth=3.0, label=label1, markersize=7)
    ax.fill_between(runs_labels, m1 - se1, m1 + se1, color=color1, alpha=0.15)

    ax.plot(runs_labels, m2, marker='s', color=color2, linewidth=3.0, label=label2, markersize=7)
    ax.fill_between(runs_labels, m2 - se2, m2 + se2, color=color2, alpha=0.15)

    ax.set_title(title, fontsize=base_font_size + 2, fontweight='bold', pad=12)
    ax.set_xlabel('Training Runs', fontsize=base_font_size)
    ax.set_ylabel(ylabel, fontsize=base_font_size)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right' if 'Acc' in ylabel else 'upper left', fontsize=base_font_size - 2)


def plot_bar_box(data1, label1, color1, data2, label2, color2, title, ylabel, ylim=None, p_val=None, save_path=None):
    """繪製 Bar + Boxplot 組合圖並連線標註顯著性"""
    fig, ax = plt.subplots(figsize=(8, 6))

    d1 = np.array(data1, dtype=float)[~np.isnan(data1)]
    d2 = np.array(data2, dtype=float)[~np.isnan(data2)]
    is_paired = (len(d1) == len(d2))

    if p_val is None:
        if is_paired and len(d1) > 1:
            _, p_val = stats.ttest_rel(d2, d1)
        elif len(d1) > 1 and len(d2) > 1:
            _, p_val = stats.ttest_ind(d2, d1)
        else:
            p_val = 1.0

    means = [np.mean(d1), np.mean(d2)]
    sems = [stats.sem(d1), stats.sem(d2)]

    ax.bar([1, 2], means, yerr=sems, color=[color1, color2], alpha=0.5, capsize=8, width=0.5)
    ax.boxplot([d1, d2], positions=[1, 2], widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='none', color='black', lw=1.5),
               medianprops=dict(color='black', lw=2), showfliers=False)

    if is_paired:
        for i in range(len(d1)):
            ax.plot([1, 2], [d1[i], d2[i]], color='gray', alpha=0.3, lw=1, marker='o', ms=5)

    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)

    ax.plot([1, 1, 2, 2], [y_max - 1, y_max, y_max, y_max - 1], lw=1.5, color='black')
    ax.text(1.5, y_max + 0.02 * (y_max - min(np.min(d1), np.min(d2))),
            f"{sig_str}\n(p={p_val:.4f})", ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label1, label2], fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=25)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def add_significance_labels(rects, significances, ax):
    """在長條圖上方添加顯著性標註 (***, **, *, ns)"""
    for rect, sig in zip(rects, significances):
        height = rect.get_height()
        y_offset = 3 if height >= 0 else -6
        va = 'bottom' if height >= 0 else 'top'

        ax.annotate(sig,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=11, fontweight='bold', color='black')
