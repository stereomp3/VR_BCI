"""
VR-BCI 整合指標分析、CSV 持久化與自動繪圖模組 (Analyze Metrics, CSV Export & Plotting)
整合以下 6 個腳本：
1. compute_saliency_metric.py
2. sum_up_all_metric_with_log_and_output_list.py
3. generate_metric_ttest.py
4. generate_metric_plot.py
5. generate_metric_WSI.py
6. generate_metric_plot_each_run.py

功能流程：
- 解析/計算 MBSR、MSFI、Offline 各 Run 準確度、Session 準確度與 Online 準確度。
- 將原本的字典資料結構儲存為乾淨、標準的 metrics_summary.csv。
- 全面基於該 CSV 載入資料並產生：
  - 統計檢定表 Table 1 (組間比較)、Table 2 (LE & AE 檢定)、Table 3 (Two-way Mixed ANOVA & LMM)
  - LE vs AE 動態標註長條圖 (Accuracy, MBSR, MSFI)
  - WSI (Within-Session Improvement) 分析與盒狀柱狀圖
  - 各 Run 趨勢折線圖 (S1 vs S2, Static vs Adaptive)
  - Saliency 指標趨勢圖與盒狀柱狀圖
- 支援 --from_csv 直接讀取 CSV 快速重繪所有圖表與印出統計。
"""

import os
import sys
import re
import argparse
import pickle
from scipy import signal
import scipy.stats as stats

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自動搜尋專案模組
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
search_dirs = [current_dir, parent_dir, os.path.join(parent_dir, "utils"), os.path.join(parent_dir, "test"), os.path.join(parent_dir, "demo")]
for d in search_dirs:
    if d not in sys.path and os.path.exists(d):
        sys.path.insert(0, d)

from common_utils import (
    tee_log, pad_run_data, calc_wsi,
    safe_wilcoxon, safe_mannwhitneyu, safe_ttest_1samp, safe_ttest_ind,
    plot_bar_box, plot_trend, add_significance_labels,
    DEFAULT_IDS, DEFAULT_SESSIONS, S1_COND_MAP, CHANNEL_CONFIGS,
    reverse_condition, get_subject_alias_map
)

# 避免字型與顯示警告
plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'Arial', 'Microsoft JhengHei', 'sans-serif']})


# ============================================================
# 1. 日誌解析與指標載入
# ============================================================

def parse_offline_runs_log(file_path):
    """從 training_log_*_{channels}.txt 解析每個 run 的 avg_val acc"""
    offline_runs = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(file_path):
        print(f"[提示] 找不到 Offline Run Log 檔案: {file_path}")
        return offline_runs

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        curr_id, curr_sess, curr_run = None, None, None
        for line in f:
            match_new_data = re.search(
                r"new data .*?[\\/](\d+)[\\/](s\d+)[\\/]run(\d+)[\\/](?:mi|mi_22)\.pt",
                line, re.IGNORECASE
            )
            if match_new_data:
                curr_id = match_new_data.group(1)
                curr_sess = match_new_data.group(2).lower()
                curr_run = int(match_new_data.group(3))
                continue

            if curr_id is not None and curr_sess is not None and curr_run is not None:
                match_acc = re.search(r"avg_val acc:\s*([0-9.]+)", line, re.IGNORECASE)
                if match_acc:
                    acc = float(match_acc.group(1))
                    offline_runs[curr_id][curr_sess][curr_run] = acc
                    curr_run = None

    print(f"✅ 成功讀取 Offline Run Log: {file_path}")
    return offline_runs


def parse_offline_session_log(file_path):
    """從 training_log_*_{channels}_all.txt 解析整個 session 的 avg_val acc"""
    offline_session = defaultdict(dict)
    if not os.path.exists(file_path):
        print(f"[提示] 找不到 Offline Session Log 檔案: {file_path}")
        return offline_session

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        curr_id, curr_sess = None, None
        for line in f:
            match_subject = re.search(
                r"subject\s+(\d+),\s*session\s+(s\d+)\s+start",
                line, re.IGNORECASE
            )
            if match_subject:
                curr_id = match_subject.group(1)
                curr_sess = match_subject.group(2).lower()
                continue

            if curr_id is not None and curr_sess is not None:
                match_acc = re.search(r"avg_val acc:\s*([0-9.]+)", line, re.IGNORECASE)
                if match_acc:
                    acc = float(match_acc.group(1))
                    offline_session[curr_id][curr_sess] = acc
                    curr_id, curr_sess = None, None

    print(f"✅ 成功讀取 Offline Session Log: {file_path}")
    return offline_session


def parse_online_logs(online_dir, log_name="log.txt"):
    """從 online log 目錄中解析線上受試者各 run 的即時準確度"""
    online_runs = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(online_dir):
        print(f"[提示] 未找到 Online log 目錄: {online_dir}")
        return online_runs

    online_file_count = 0
    for root, dirs, files in os.walk(online_dir):
        if log_name not in files:
            continue

        file_path = os.path.join(root, log_name)
        path_match = re.search(r"[\\/](\d+)[\\/](s\d+)[\\/]log\.txt$", file_path, re.IGNORECASE)
        if not path_match:
            continue

        curr_id = path_match.group(1)
        curr_sess = path_match.group(2).lower()
        online_file_count += 1

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            run_active_id = None
            for line in f:
                run_match = re.search(r"目前\s*Run\s*編號:\s*(\d+)", line, re.IGNORECASE)
                if run_match:
                    run_active_id = int(run_match.group(1))
                    continue

                if run_active_id is not None:
                    acc_match = re.search(r"Received from \('127\.0\.0\.1', .*?\):.*?=\s*([0-9.]+)", line, re.IGNORECASE)
                    if acc_match:
                        acc = float(acc_match.group(1))
                        online_runs[curr_id][curr_sess][run_active_id] = acc
                        run_active_id = None

    print(f"✅ 成功讀取 {online_file_count} 個 Online log.txt")
    return online_runs


def parse_saliency_metric_txt(file_path):
    """解析 compute_saliency_metric.txt 中的 Markdown 表格"""
    metric_data = {
        "Spectral Saliency": {"s1": {}, "s2": {}},
        "Spatial Saliency": {"s1": {}, "s2": {}},
        "MBSR": {"s1": {}, "s2": {}},
        "MSFI": {"s1": {}, "s2": {}}
    }
    if not os.path.exists(file_path):
        print(f"[提示] 找不到 Saliency Metric txt 檔案: {file_path}")
        return metric_data

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_metric, current_session = None, None
    for line in lines:
        stripped = line.strip()
        title_match = re.search(r"表格\s*:\s*(MBSR|MSFI|Spectral\s+Saliency|Spatial\s+Saliency)\s*\(%\)\s*-\s*Session\s+(S1|S2)", stripped, re.IGNORECASE)
        if title_match:
            raw_name = title_match.group(1).upper()
            if "MBSR" in raw_name or "SPECTRAL" in raw_name:
                current_metric = "Spectral Saliency"
            else:
                current_metric = "Spatial Saliency"
            current_session = title_match.group(2).lower()
            continue

        if stripped.startswith("###"):
            if not re.search(r"表格\s*:\s*(MBSR|MSFI|Spectral|Spatial)", stripped, re.IGNORECASE):
                current_metric, current_session = None, None
            continue

        if current_metric is None or current_session is None or not stripped.startswith("|"):
            continue
        if "**id**" in stripped or re.match(r"^\|\s*:?-+", stripped):
            continue

        cols = [item.strip() for item in stripped.strip("|").split("|")]
        if len(cols) < 2:
            continue

        row_id = cols[0].strip().upper()
        if not re.fullmatch(r"S\d+", row_id, re.IGNORECASE):
            continue

        values = []
        for run_idx in range(7):
            col_idx = run_idx + 1
            raw_val = cols[col_idx] if col_idx < len(cols) else "..."
            if raw_val in ('...', '', 'nan', 'None'):
                values.append(np.nan)
            else:
                try:
                    values.append(float(raw_val))
                except ValueError:
                    values.append(np.nan)

        metric_data[current_metric][current_session][row_id] = values
        # 兼容原本簡稱
        alias_key = "MBSR" if current_metric == "Spectral Saliency" else "MSFI"
        metric_data[alias_key][current_session][row_id] = values

    print(f"✅ 成功讀取 Saliency Metric txt: {file_path}")
    return metric_data


def compute_saliency_metrics_from_pkl(base_dir, channels="22", ids=DEFAULT_IDS, sessions=DEFAULT_SESSIONS):
    """
    直接從各 Run 目錄中的 {channels}_eval_record.pkl 與 {channels}_eval_xb_epochs.pkl 計算 Spectral Saliency 與 Spatial Saliency。
    若找不到 compute_saliency_metric.txt，會自動啟用此函式直接計算，無需額外手動執行 compute_saliency_metric.py。
    """
    ch_str = str(channels)
    eval_pkl_name = f"{ch_str}_eval_record.pkl"
    xb_pkl_name = f"{ch_str}_eval_xb_epochs.pkl"

    metric_data = {
        "Spectral Saliency": {"s1": {}, "s2": {}},
        "Spatial Saliency": {"s1": {}, "s2": {}},
        "MBSR": {"s1": {}, "s2": {}},
        "MSFI": {"s1": {}, "s2": {}}
    }
    motor_channels = ['C3', 'Cz', 'C4', 'FC3', 'FCz', 'FC4']

    sorted_ids = sorted(ids, key=lambda x: int(x))
    alias_map = get_subject_alias_map(sorted_ids)

    found_count = 0
    print(f"🔄 正在從 {base_dir} 掃描 {eval_pkl_name} 並計算 Saliency 特徵...")

    for sub in sorted_ids:
        sub_alias = alias_map[sub]
        for sess in sessions:
            spectral_runs = [np.nan] * 7
            spatial_runs = [np.nan] * 7

            for r in range(1, 8):
                run_dir = os.path.join(base_dir, sub, sess, f"run{r}")
                eval_path = os.path.join(run_dir, eval_pkl_name)
                xb_path = os.path.join(run_dir, xb_pkl_name)

                if not (os.path.exists(eval_path) and os.path.exists(xb_path)):
                    continue

                try:
                    with open(eval_path, "rb") as f:
                        eval_record = pickle.load(f)
                    with open(xb_path, "rb") as f:
                        xb_epochs = pickle.load(f)

                    chs = xb_epochs.get_channel_names()

                    temp_spectral, temp_spatial = [], []
                    for target_class in [0, 1]:
                        saliency = eval_record.gradient.get(target_class, None)
                        if saliency is None or len(saliency) == 0:
                            continue

                        sfreq = 500
                        freqs, psd = signal.welch(saliency, fs=sfreq, nperseg=sfreq, axis=-1)
                        mask_1_40 = (freqs >= 1) & (freqs <= 40)
                        mask_8_30 = (freqs >= 8) & (freqs <= 30)

                        # Spatial Saliency: 8-30Hz 運動腦區能量佔比 (原 MSFI)
                        freqs_8_30 = freqs[mask_8_30]
                        psd_8_30 = psd[:, :, mask_8_30]
                        W_c_trials = np.trapz(psd_8_30, freqs_8_30, axis=-1)
                        W_c = np.mean(W_c_trials, axis=0)

                        motor_power = sum(W_c[idx] for idx, ch in enumerate(chs) if ch in motor_channels)
                        all_power = np.sum(W_c)
                        spatial_val = (motor_power / all_power) * 100 if all_power > 0 else 0

                        # Spectral Saliency: 8-30Hz 佔 1-40Hz 頻段能量比 (原 MBSR)
                        P_f = psd.mean(axis=0).mean(axis=0)
                        power_1_40 = np.trapz(P_f[mask_1_40], freqs[mask_1_40])
                        power_8_30 = np.trapz(P_f[mask_8_30], freqs[mask_8_30])
                        spectral_val = (power_8_30 / power_1_40) * 100 if power_1_40 > 0 else 0

                        temp_spectral.append(spectral_val)
                        temp_spatial.append(spatial_val)

                    if temp_spectral:
                        spectral_runs[r - 1] = float(np.mean(temp_spectral))
                        spatial_runs[r - 1] = float(np.mean(temp_spatial))
                        found_count += 1
                except Exception as e:
                    print(f"Error computing saliency for {sub} {sess} run{r}: {e}")
                    continue

            metric_data["Spectral Saliency"][sess][sub_alias] = spectral_runs
            metric_data["Spatial Saliency"][sess][sub_alias] = spatial_runs
            metric_data["MBSR"][sess][sub_alias] = spectral_runs
            metric_data["MSFI"][sess][sub_alias] = spatial_runs

    print(f"✅ Saliency 計算完畢 (共計算 {found_count} 個 Run 的 Spectral 與 Spatial Saliency 指標)！")
    return metric_data


# ============================================================
# 2. 彙整資料結構並輸出為 CSV
# ============================================================

def build_metrics_dataframe(offline_runs, offline_session, online_runs, saliency_metrics,
                            ids=DEFAULT_IDS, sessions=DEFAULT_SESSIONS):
    """
    將所有來源的指標整理成結構化的 Pandas DataFrame (Long-format)
    欄位: subject_id, subject_alias, session, condition, sequence, run,
          online_acc, offline_run_acc, offline_sess_acc, Spectral_Saliency, Spatial_Saliency
    """
    sorted_ids = sorted(ids, key=lambda x: int(x))
    alias_map = get_subject_alias_map(sorted_ids)
    rows = []

    def get_aligned_run_val(run_dict, target_run):
        """若該 session 只有 6 個 run，對齊為 2, 3, 4, 5, 6, 7 (Run 1 為 NaN)"""
        if not run_dict:
            return np.nan
        valid_keys = [k for k, v in run_dict.items() if not pd.isna(v) and str(v).strip() not in ('', 'nan', '...')]
        if len(valid_keys) == 6:
            sorted_keys = sorted(valid_keys)
            if target_run == 1:
                return np.nan
            else:
                src_key = sorted_keys[target_run - 2]
                return run_dict.get(src_key, np.nan)
        return run_dict.get(target_run, np.nan)

    for sub in sorted_ids:
        sub_alias = alias_map[sub]
        s1_cond = S1_COND_MAP.get(sub, "A")
        seq = "Seq_AB" if s1_cond == "A" else "Seq_BA"

        for sess in sessions:
            cond = s1_cond if sess == "s1" else reverse_condition(s1_cond)
            off_sess_val = offline_session.get(sub, {}).get(sess, np.nan)

            spectral_raw = saliency_metrics.get("Spectral Saliency", {}).get(sess, {}).get(sub_alias,
                            saliency_metrics.get("MBSR", {}).get(sess, {}).get(sub_alias, [np.nan] * 7))
            spatial_raw = saliency_metrics.get("Spatial Saliency", {}).get(sess, {}).get(sub_alias,
                           saliency_metrics.get("MSFI", {}).get(sess, {}).get(sub_alias, [np.nan] * 7))

            spectral_list = pad_run_data(spectral_raw)
            spatial_list = pad_run_data(spatial_raw)

            sub_online_runs = online_runs.get(sub, {}).get(sess, {})
            sub_offline_runs = offline_runs.get(sub, {}).get(sess, {})

            for r in range(1, 8):
                on_acc = get_aligned_run_val(sub_online_runs, r)
                off_run_acc = get_aligned_run_val(sub_offline_runs, r)
                spectral_val = spectral_list[r - 1] if len(spectral_list) >= r else np.nan
                spatial_val = spatial_list[r - 1] if len(spatial_list) >= r else np.nan

                rows.append({
                    "subject_id": int(sub),
                    "subject_alias": sub_alias,
                    "session": sess,
                    "condition": cond,
                    "sequence": seq,
                    "run": r,
                    "online_acc": on_acc if not pd.isna(on_acc) else np.nan,
                    "offline_run_acc": off_run_acc if not pd.isna(off_run_acc) else np.nan,
                    "offline_sess_acc": off_sess_val if not pd.isna(off_sess_val) else np.nan,
                    "Spectral_Saliency": spectral_val if not pd.isna(spectral_val) else np.nan,
                    "Spatial_Saliency": spatial_val if not pd.isna(spatial_val) else np.nan
                })

    df = pd.DataFrame(rows)
    return df


def export_metrics_to_csv(df, csv_path):
    """將指標 DataFrame 儲存至 CSV"""
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n💾 [儲存完畢] 已成功將指標資料庫儲存為 CSV: {csv_path} (共 {len(df)} 筆紀錄)")


def load_metrics_from_csv(csv_path):
    """
    從 CSV 讀取指標資料庫，並回傳 (df, raw_data_dict)
    raw_data_dict: 結構如 { sub_int: { 's1': { 'cond': ..., 'on': [...], 'off_run': [...], ... } } }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到指定的 CSV 檔案: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📂 [讀取成功] 已從 CSV 載入資料: {csv_path} (受試者數: {df['subject_id'].nunique()})")

    raw_data = defaultdict(dict)
    for (sub_id, sess), group in df.groupby(["subject_id", "session"]):
        group_sorted = group.sort_values("run")
        cond = group_sorted["condition"].iloc[0]
        off_sess = group_sorted["offline_sess_acc"].iloc[0]

        on_list = group_sorted["online_acc"].tolist()
        off_run_list = group_sorted["offline_run_acc"].tolist()

        spectral_col = "Spectral_Saliency" if "Spectral_Saliency" in group_sorted.columns else "MBSR"
        spatial_col = "Spatial_Saliency" if "Spatial_Saliency" in group_sorted.columns else "MSFI"
        spectral_list = group_sorted[spectral_col].tolist()
        spatial_list = group_sorted[spatial_col].tolist()

        raw_data[int(sub_id)][sess] = {
            "cond": cond,
            "on": on_list,
            "off_run": off_run_list,
            "off_sess": off_sess,
            "Spectral Saliency": spectral_list,
            "Spatial Saliency": spatial_list,
            # 兼容舊別名
            "MBSR": spectral_list,
            "MSFI": spatial_list
        }

    return df, raw_data


# ============================================================
# 3. 統計檢定 (Table 1, Table 2, Table 3 LMM)
# ============================================================

def calculate_and_print_tables(raw_data, df_long=None):
    """計算並輸出 Table 1、Table 2 與 Table 3 (LMM) 檢定結果"""
    ids = sorted(raw_data.keys())
    metrics = ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']

    g1_deltas = {m: [] for m in metrics}
    g2_deltas = {m: [] for m in metrics}
    le_vals = {m: [] for m in metrics}
    ae_vals = {m: [] for m in metrics}

    for sub in ids:
        data = raw_data[sub]
        s1_acc = (data['s1']['off_sess'] * 100) if not pd.isna(data['s1']['off_sess']) else np.nan
        s2_acc = (data['s2']['off_sess'] * 100) if not pd.isna(data['s2']['off_sess']) else np.nan

        s1_spec = np.nanmean([float(x) for x in data['s1']['Spectral Saliency'] if not pd.isna(x)])
        s2_spec = np.nanmean([float(x) for x in data['s2']['Spectral Saliency'] if not pd.isna(x)])

        s1_spat = np.nanmean([float(x) for x in data['s1']['Spatial Saliency'] if not pd.isna(x)])
        s2_spat = np.nanmean([float(x) for x in data['s2']['Spatial Saliency'] if not pd.isna(x)])

        vals_s1 = {'Accuracy': s1_acc, 'Spectral Saliency': s1_spec, 'Spatial Saliency': s1_spat}
        vals_s2 = {'Accuracy': s2_acc, 'Spectral Saliency': s2_spec, 'Spatial Saliency': s2_spat}

        cond_s1 = data['s1']['cond']

        for m in metrics:
            d_s2_s1 = vals_s2[m] - vals_s1[m]
            d_s1_s2 = vals_s1[m] - vals_s2[m]
            le_vals[m].append(d_s2_s1)

            if cond_s1 == 'A':  # Group 1 (A -> B): S2(B) - S1(A) = Adaptive - Static
                g1_deltas[m].append(d_s2_s1)
                ae_vals[m].append(d_s2_s1)
            else:  # Group 2 (B -> A): S1(B) - S2(A) = Adaptive - Static
                g2_deltas[m].append(d_s2_s1)
                ae_vals[m].append(d_s1_s2)

    stats_summary = {}

    print("\n" + "=" * 125)
    print("📊 Table 1: Between-group comparison (Group 1 vs Group 2) for Sequence Effect")
    print("   * W-Test = Mann-Whitney U test (Wilcoxon rank-sum test)")
    print("   * t-Test = Independent Two-Sample t-test")
    print("=" * 125)
    print(f"| {'Metric':<18} | {'Group 1 Mean ± SD':<18} | {'Group 2 Mean ± SD':<18} | {'Mean Diff':<12} | {'U-Stat':<8} | {'W-p-val':<8} | {'t-Stat':<8} | {'t-p-val':<8} |")
    print(f"|{'-' * 20}|{'-' * 20}|{'-' * 20}|{'-' * 14}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|")

    for m in metrics:
        v1 = np.array(g1_deltas[m])[~np.isnan(g1_deltas[m])]
        v2 = np.array(g2_deltas[m])[~np.isnan(g2_deltas[m])]
        m1, std1 = np.mean(v1), np.std(v1, ddof=1) if len(v1) > 1 else 0
        m2, std2 = np.mean(v2), np.std(v2, ddof=1) if len(v2) > 1 else 0
        diff = m1 - m2
        w_stat, w_p = safe_mannwhitneyu(v1, v2)
        t_stat, t_p = safe_ttest_ind(v1, v2)
        print(f"| {m:<18} | {f'{m1:.2f} ± {std1:.2f}':>18} | {f'{m2:.2f} ± {std2:.2f}':>18} | {diff:>12.2f} | {w_stat:>8.2f} | {w_p:>8.4f} | {t_stat:>8.2f} | {t_p:>8.4f} |")
    print("=" * 125)

    print("\n" + "=" * 135)
    print("📊 Table 2: Overall Learning Effect (LE) and Adaptive Effect (AE)")
    print("   * W-Test = Wilcoxon Signed-Rank Test against 0")
    print("   * t-Test = One-Sample t-test against 0")
    print("   * LE Diff = Session 2 - Session 1  |  AE Diff = Adaptive(Cond B) - Static(Cond A)")
    print("=" * 135)
    print(f"| {'Metric':<18} | {'LE Diff':<8} | {'LE W-Stat':<9} | {'LE W-p':<8} | {'LE t-Stat':<9} | {'LE t-p':<8} | {'AE Diff':<8} | {'AE W-Stat':<9} | {'AE W-p':<8} | {'AE t-Stat':<9} | {'AE t-p':<8} |")
    print(f"|{'-' * 20}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 11}|{'-' * 10}|")

    for m in metrics:
        le_arr = np.array(le_vals[m])[~np.isnan(le_vals[m])]
        ae_arr = np.array(ae_vals[m])[~np.isnan(ae_vals[m])]

        le_diff = float(np.mean(le_arr)) if len(le_arr) > 0 else np.nan
        le_w_stat, le_w_p = safe_wilcoxon(le_arr)
        le_t_stat, le_t_p = safe_ttest_1samp(le_arr)

        ae_diff = float(np.mean(ae_arr)) if len(ae_arr) > 0 else np.nan
        ae_w_stat, ae_w_p = safe_wilcoxon(ae_arr)
        ae_t_stat, ae_t_p = safe_ttest_1samp(ae_arr)

        stats_summary[m] = {
            'le': {'diff': le_diff, 'w_stat': le_w_stat, 'w_p': le_w_p, 't_stat': le_t_stat, 't_p': le_t_p},
            'ae': {'diff': ae_diff, 'w_stat': ae_w_stat, 'w_p': ae_w_p, 't_stat': ae_t_stat, 't_p': ae_t_p}
        }

        print(f"| {m:<18} | {le_diff:>8.2f} | {le_w_stat:>9.2f} | {le_w_p:>8.4f} | {le_t_stat:>9.2f} | {le_t_p:>8.4f} | {ae_diff:>8.2f} | {ae_w_stat:>9.2f} | {ae_w_p:>8.4f} | {ae_t_stat:>9.2f} | {ae_t_p:>8.4f} |")
    print("=" * 135)

    # Table 3: Linear Mixed Model (LMM)
    _run_lmm_analysis(df_long if df_long is not None else None, raw_data)

    return stats_summary


def _run_lmm_analysis(df, raw_data):
    """執行 LMM 與 Two-way Mixed ANOVA 分析"""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n[提示] 系統缺少 statsmodels 套件，跳過 LMM 模型計算。")
        return

    # 建立 Session-Level DataFrame 用於 LMM
    rows = []
    for sub_int, data in raw_data.items():
        sub = str(sub_int)
        seq = 'Seq_AB' if data['s1']['cond'] == 'A' else 'Seq_BA'
        for sess in ['s1', 's2']:
            cond = data[sess]['cond']
            acc = data[sess]['off_sess'] * 100 if not pd.isna(data[sess]['off_sess']) else np.nan
            spec_raw = [float(x) for x in data[sess].get('Spectral Saliency', []) if not pd.isna(x)]
            spat_raw = [float(x) for x in data[sess].get('Spatial Saliency', []) if not pd.isna(x)]
            rows.append({
                'Subject': sub,
                'Session': sess.upper(),
                'Condition': cond,
                'Sequence': seq,
                'Accuracy': acc,
                'Spectral_Saliency': np.mean(spec_raw) if spec_raw else np.nan,
                'Spatial_Saliency': np.mean(spat_raw) if spat_raw else np.nan
            })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("📊 Table 3: Linear Mixed Model (LMM) & Two-way Mixed ANOVA")
    print("   * 模型公式: Metric ~ Session * Sequence + (1|Subject)")
    print("   * [Session 主效應] = 學習效應 (Learning Effect, LE)")
    print("   * [Sequence 主效應] = 順序/組別效應 (Carryover Effect)")
    print("   * [Session × Sequence 交互作用] = 系統效應 (Adaptive Effect, AE) - 等價於 Condition 主效應")
    print("=" * 120)
    print(f"| {'Metric':<18} | {'Session (LE) p-val':<20} | {'Sequence p-val':<20} | {'Interaction (AE) p-val':<22} |")
    print(f"|{'-' * 20}|{'-' * 22}|{'-' * 22}|{'-' * 22}|")

    for m in ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']:
        col = m.replace(' ', '_')
        df_clean = df.dropna(subset=[col]).copy()
        if len(df_clean) < 10:
            continue
        try:
            subj_col = 'Subject' if 'Subject' in df_clean.columns else 'subject_id'
            sess_col = 'Session' if 'Session' in df_clean.columns else 'session'
            seq_col = 'Sequence' if 'Sequence' in df_clean.columns else 'sequence'

            md = smf.mixedlm(f"{col} ~ C({sess_col}) * C({seq_col})", df_clean, groups=df_clean[subj_col])
            mdf = md.fit(disp=False)

            p_sess = np.nan
            for k in mdf.pvalues.keys():
                if f"C({sess_col})" in k and ":" not in k:
                    p_sess = mdf.pvalues[k]

            seq_key = [k for k in mdf.pvalues.keys() if f"C({seq_col})" in k and ":" not in k]
            p_seq = mdf.pvalues[seq_key[0]] if seq_key else np.nan

            int_key = [k for k in mdf.pvalues.keys() if ":" in k]
            p_int = mdf.pvalues[int_key[0]] if int_key else np.nan

            def fmt_p(p):
                if pd.isna(p): return "N/A"
                return f"{p:.4f}" + ("*" if p < 0.05 else "")

            print(f"| {m:<18} | {fmt_p(p_sess):<20} | {fmt_p(p_seq):<20} | {fmt_p(p_int):<22} |")
        except Exception as e:
            print(f"| {m:<18} | 模型擬合失敗: {str(e)[:35]:<50} |")
    print("=" * 120)


# ============================================================
# 4. 全圖表自動生成模組
# ============================================================

def get_sig_star(p_val):
    """根據 p-value 轉換為顯著星號"""
    if pd.isna(p_val): return "ns"
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"


def plot_effects_across_metrics_bar(stats_summary, save_path=None):
    """
    動態繪製 Effects Across Metrics (LE vs AE) 柱狀圖 (取代原 generate_metric_plot.py)
    根據 Table 2 的 LE Diff 與 AE Diff 呈現，顯著性依據 t-p 判定 (One-sample t-test against 0)
    """
    metrics = ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']
    keys = ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']

    le_values = [stats_summary[k]['le']['diff'] for k in keys]
    ae_values = [stats_summary[k]['ae']['diff'] for k in keys]

    # 顯著性判定完全依據 Table 2 中的 t-p (One-sample t-test)
    le_sigs = [get_sig_star(stats_summary[k]['le']['t_p']) for k in keys]
    ae_sigs = [get_sig_star(stats_summary[k]['ae']['t_p']) for k in keys]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    rects1 = ax.bar(x - width / 2, le_values, width, label='Learning Effect (LE)', color='#4C72B0', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width / 2, ae_values, width, label='Adaptive Effect (AE)', color='#DD8452', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Effect (%)', fontsize=13, fontweight='bold')
    ax.set_title('Effects Across Metrics (LE vs AE)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=12, loc='upper right')

    add_significance_labels(rects1, le_sigs, ax)
    add_significance_labels(rects2, ae_sigs, ax)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"✅ 成功儲存長條圖: {save_path}")
    else:
        plt.show()


def plot_wsi_charts(raw_data, plot_dir):
    """
    繪製 Session 1 vs Session 2 的 Within-Session Improvement (WSI)
    - 寬版版面 (figsize=(10, 4.5))
    - y-axis 標籤: 'Improvement (%)'
    - 不顯示上方標題
    - 長條為純色 (alpha=0.5, 無黑色外框線, width=0.5)
    - 箱形圖 widths=0.3, lw=1.5, median lw=2, showfliers=False
    - 單 Session 顯著性檢定 (vs 0)，顯著星號置於箱形圖鬚頂端上方 (y ≈ max_whisker + 1.8)
    - 無跨 Session 連線，無 y=0 虛線
    """
    wsi_s1, wsi_s2 = [], []
    for sub, sess_data in raw_data.items():
        w1 = calc_wsi(sess_data['s1']['off_run'])
        w2 = calc_wsi(sess_data['s2']['off_run'])
        wsi_s1.append(w1)
        wsi_s2.append(w2)

    d1 = np.array(wsi_s1)[~np.isnan(wsi_s1)]
    d2 = np.array(wsi_s2)[~np.isnan(wsi_s2)]

    # 單 Session 顯著性檢定 (vs 0)
    res1 = stats.ttest_1samp(d1, 0) if len(d1) > 1 else None
    res2 = stats.ttest_1samp(d2, 0) if len(d2) > 1 else None
    p1 = res1.pvalue if res1 else np.nan
    p2 = res2.pvalue if res2 else np.nan

    def get_wsi_star(p):
        if pd.isna(p): return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    sig1 = get_wsi_star(p1)
    sig2 = get_wsi_star(p2)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    means = [np.mean(d1), np.mean(d2)]
    sems = [stats.sem(d1), stats.sem(d2)]

    bars = ax.bar([1, 2], means, yerr=sems, color=['#ff7f0e', '#1f77b4'], alpha=0.5, capsize=8, width=0.5)
    ax.boxplot([d1, d2], positions=[1, 2], widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='none', color='black', lw=1.5),
               medianprops=dict(color='black', lw=2), showfliers=False)

    # 繪製配對受試者連線
    if len(d1) == len(d2):
        for i in range(len(d1)):
            ax.plot([1, 2], [d1[i], d2[i]], color='gray', alpha=0.3, lw=1, marker='o', ms=5)

    # 星號標註於鬚頂端上方
    q75_1, q25_1 = np.percentile(d1, [75, 25])
    w_top1 = max(x for x in d1 if x <= q75_1 + 1.5 * (q75_1 - q25_1))
    q75_2, q25_2 = np.percentile(d2, [75, 25])
    w_top2 = max(x for x in d2 if x <= q75_2 + 1.5 * (q75_2 - q25_2))
    y_star = max(w_top1, w_top2) + 1.8

    if sig1:
        ax.text(1, y_star, sig1, ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    if sig2:
        ax.text(2, y_star, sig2, ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Session 1', 'Session 2'], fontweight='bold', fontsize=13)
    ax.set_ylabel('Improvement (%)', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    curr_top = ax.get_ylim()[1]
    ax.set_ylim(top=max(curr_top, y_star + 2.5))

    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'wsi_session_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ 成功儲存 WSI 分析圖表: {save_path}")


def plot_each_run_trends_all(raw_data, plot_dir):
    """
    繪製每 Run 的整體趨勢圖 (取代原 generate_metric_plot_each_run.py)
    若只有 6 個 Run，對齊為 2, 3, 4, 5, 6, 7 (Run 1 缺失)
    """
    metrics = ['off_run', 'Spectral Saliency', 'Spatial Saliency']
    metric_titles = ['Offline Accuracy (%)', 'Spectral Saliency (%)', 'Spatial Saliency (%)']
    y_limits = [(40, 110), (0, 100), (0, 100)]
    runs_labels = [f'Run {i}' for i in range(1, 8)]
    base_font_size = 12

    # 圖 1: Session 1 vs Session 2
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle('All Subjects Trend - Session 1 vs Session 2', fontsize=base_font_size + 4, fontweight='bold', y=1.03)

    for i, m in enumerate(metrics):
        ax = axes1[i]
        all_y_s1, all_y_s2 = [], []

        for sub in raw_data.keys():
            y_s1 = pad_run_data(raw_data[sub]['s1'][m], metric=m)
            y_s2 = pad_run_data(raw_data[sub]['s2'][m], metric=m)
            all_y_s1.append(y_s1)
            all_y_s2.append(y_s2)

            ax.plot(runs_labels, y_s1, color='#9467bd', alpha=0.15, linewidth=1.2)
            ax.plot(runs_labels, y_s2, color='#8c564b', alpha=0.15, linewidth=1.2)

        mean_s1 = np.nanmean(all_y_s1, axis=0)
        mean_s2 = np.nanmean(all_y_s2, axis=0)

        ax.plot(runs_labels, mean_s1, color='#9467bd', linewidth=3.5, marker='o', markersize=7, label='Session 1 Avg')
        ax.plot(runs_labels, mean_s2, color='#8c564b', linewidth=3.5, marker='s', markersize=7, label='Session 2 Avg')

        ax.set_title(metric_titles[i], fontsize=base_font_size + 2, fontweight='bold')
        ax.set_xlabel('Training Runs', fontweight='bold')
        ax.set_ylabel(metric_titles[i], fontweight='bold')
        ax.set_ylim(y_limits[i])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right' if m == 'off_run' else 'upper left')

    plt.tight_layout()
    save_path1 = os.path.join(plot_dir, 's1_s2_all_metrics_combined.png')
    plt.savefig(save_path1, dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # 圖 2: Static vs Adaptive Condition
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('All Subjects Trend - Static vs Adaptive Condition', fontsize=base_font_size + 4, fontweight='bold', y=1.03)

    for i, m in enumerate(metrics):
        ax = axes2[i]
        all_y_condA, all_y_condB = [], []

        for sub in raw_data.keys():
            if raw_data[sub]['s1']['cond'] == 'A':
                y_condA = pad_run_data(raw_data[sub]['s1'][m], metric=m)
                y_condB = pad_run_data(raw_data[sub]['s2'][m], metric=m)
            else:
                y_condB = pad_run_data(raw_data[sub]['s1'][m], metric=m)
                y_condA = pad_run_data(raw_data[sub]['s2'][m], metric=m)

            all_y_condA.append(y_condA)
            all_y_condB.append(y_condB)

            ax.plot(runs_labels, y_condA, color='#2ca02c', alpha=0.15, linewidth=1.2)
            ax.plot(runs_labels, y_condB, color='#d62728', alpha=0.15, linewidth=1.2)

        mean_condA = np.nanmean(all_y_condA, axis=0)
        mean_condB = np.nanmean(all_y_condB, axis=0)

        ax.plot(runs_labels, mean_condA, color='#2ca02c', linewidth=3.5, marker='o', markersize=7, label='Static Avg (Cond A)')
        ax.plot(runs_labels, mean_condB, color='#d62728', linewidth=3.5, marker='s', markersize=7, label='Adaptive Avg (Cond B)')

        ax.set_title(metric_titles[i], fontsize=base_font_size + 2, fontweight='bold')
        ax.set_xlabel('Training Runs', fontweight='bold')
        ax.set_ylabel(metric_titles[i], fontweight='bold')
        ax.set_ylim(y_limits[i])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right' if m == 'off_run' else 'upper left')

    plt.tight_layout()
    save_path2 = os.path.join(plot_dir, 'static_adaptive_all_metrics_combined.png')
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    print(f"✅ 成功儲存 Run 趨勢圖: {save_path1} 與 {save_path2}")


# ============================================================
# 5. 主流程與 CLI
# ============================================================

def run_analysis_pipeline(csv_path="metrics_summary.csv", from_csv=False,
                          offline_runs_log=None, offline_sess_log=None,
                          metric_txt=None, online_dir=None, base_dir=r"/mnt/project/MIEXP/DATA_Cygnus",
                          channels="22", plot_dir="./charts", do_plot=True):
    """
    執行分析管線：
    1. 載入或解析數據 (若無 metric_txt 則自動從 base_dir 的 PKL 計算)
    2. 存為 CSV (若非 from_csv)
    3. 計算檢定並繪製全套圖表
    """
    os.makedirs(plot_dir, exist_ok=True)

    if from_csv or (os.path.exists(csv_path) and not offline_runs_log and not metric_txt):
        print(f"⚡ [模式] 直接自現有 CSV 讀取指標並進行分析: {csv_path}")
        df, raw_data = load_metrics_from_csv(csv_path)
    else:
        # 自動尋找預設日誌檔案 (依據 channels)
        ch_str = str(channels)
        if offline_runs_log is None:
            candidate_files = [f"training_log_20260416_{ch_str}.txt", f"demo/training_log_20260416_{ch_str}.txt"]
            for cand in candidate_files:
                if os.path.exists(cand):
                    offline_runs_log = cand
                    break

        if offline_sess_log is None:
            candidate_files = [f"training_log_20260416_{ch_str}_all.txt", f"demo/training_log_20260416_{ch_str}_all.txt"]
            for cand in candidate_files:
                if os.path.exists(cand):
                    offline_sess_log = cand
                    break

        if metric_txt is None:
            candidate_files = ["compute_saliency_metric.txt", f"compute_saliency_metric_{ch_str}.txt", "demo/compute_saliency_metric.txt"]
            for cand in candidate_files:
                if os.path.exists(cand):
                    metric_txt = cand
                    break

        if online_dir is None:
            candidate_dirs = ["log_data", "re_make_txt/log_data", "../re_make_txt/log_data"]
            for cand in candidate_dirs:
                if os.path.exists(cand):
                    online_dir = cand
                    break

        print("🔍 開始解析 Log 與 Saliency 指標檔案...")
        offline_runs = parse_offline_runs_log(offline_runs_log) if offline_runs_log else defaultdict(lambda: defaultdict(dict))
        offline_session = parse_offline_session_log(offline_sess_log) if offline_sess_log else defaultdict(dict)
        online_runs = parse_online_logs(online_dir) if online_dir else defaultdict(lambda: defaultdict(dict))

        # 若有 Saliency TXT 則直接讀取，若無則自動從各 Run 的 PKL 檔案直接計算
        if metric_txt and os.path.exists(metric_txt):
            saliency_metrics = parse_saliency_metric_txt(metric_txt)
        elif base_dir and os.path.exists(base_dir):
            print(f"🔄 未提供或未找到 Saliency txt，正在直接從 {base_dir} 的各 Run PKL 檔案計算 MBSR 與 MSFI...")
            saliency_metrics = compute_saliency_metrics_from_pkl(base_dir=base_dir, channels=channels)
        else:
            saliency_metrics = {"MBSR": {"s1": {}, "s2": {}}, "MSFI": {"s1": {}, "s2": {}}}

        df = build_metrics_dataframe(offline_runs, offline_session, online_runs, saliency_metrics)
        export_metrics_to_csv(df, csv_path)
        df, raw_data = load_metrics_from_csv(csv_path)

    # 執行 Table 1~3 檢定
    stats_summary = calculate_and_print_tables(raw_data, df_long=df)

    if do_plot:
        print("\n🎨 正在產出全套圖表...")
        # 1. LE vs AE 條形圖 (動態數值與顯著星號，依據 Table 2 t-p)
        plot_effects_across_metrics_bar(stats_summary, save_path=os.path.join(plot_dir, 'effects_across_metrics_le_ae.png'))
        # 2. WSI 分析圖表 (Session 1 vs Session 2，單 Session 顯著性標註於長條上方，無跨組黑線)
        plot_wsi_charts(raw_data, plot_dir)
        # 3. 每 Run 趨勢圖 (若僅 6 Run 則對齊為 1, 2, 4, 5, 6, 7)
        plot_each_run_trends_all(raw_data, plot_dir)
        print(f"\n🎉 全部圖表生成完成！所有圖片已保存至: {os.path.abspath(plot_dir)}")


def main():
    parser = argparse.ArgumentParser(description="VR-BCI 整合指標分析、CSV 輸出與自動繪圖")
    parser.add_argument("--csv_path", type=str, default="metrics_summary.csv",
                        help="CSV 儲存或讀取路徑 (預設: metrics_summary.csv)")
    parser.add_argument("--from_csv", action="store_true", default=False,
                        help="直接自現有 CSV 讀取資料進行分析與繪圖 (跳過日誌解析)")
    parser.add_argument("--channels", type=str, default="22", choices=["13", "22"],
                        help="通道設定 ('13' 或 '22')")
    parser.add_argument("--offline_runs_log", type=str, default=None,
                        help="Offline Run Log 檔案路徑")
    parser.add_argument("--offline_sess_log", type=str, default=None,
                        help="Offline Session Log 檔案路徑")
    parser.add_argument("--metric_txt", type=str, default=None,
                        help="Saliency 指標 Markdown 表格路徑 (compute_saliency_metric.txt)")
    parser.add_argument("--online_dir", type=str, default=None,
                        help="Online Log 資料夾路徑")
    parser.add_argument("--base_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集存放目錄 (若無 Saliency txt，將自動由此目錄的 PKL 檔案計算 MBSR/MSFI)")
    parser.add_argument("--plot_dir", type=str, default="./charts",
                        help="圖表儲存目錄 (預設: ./charts)")
    parser.add_argument("--no_plot", action="store_true", default=False,
                        help="僅輸出表格與 CSV，不繪製圖表")

    args = parser.parse_args()

    run_analysis_pipeline(
        csv_path=args.csv_path,
        from_csv=args.from_csv,
        offline_runs_log=args.offline_runs_log,
        offline_sess_log=args.offline_sess_log,
        metric_txt=args.metric_txt,
        online_dir=args.online_dir,
        base_dir=args.base_dir,
        channels=args.channels,
        plot_dir=args.plot_dir,
        do_plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
