import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats  # 新增 scipy 用來做統計檢定

# 1. 控制參數與資料夾設定
SAVE_FIG = False  # 設定變數來控制是否要儲存圖片 (True / False)
OUTPUT_DIR = "nasa_tlx_charts"  # 指定您想要存圖片的資料夾名稱

# 如果要存檔，且該資料夾不存在，就建立資料夾
if SAVE_FIG and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. 準備數據
# NASA-TLX 數據: [Mental, Physical, Temporal, Performance, Effort, Frustration]
data = {
    17: {'Session 1': [40, 30, 30, 65, 65, 50], 'Session 2': [65, 50, 25, 70, 75, 45]},
    21: {'Session 1': [35, 15, 40, 85, 65, 10], 'Session 2': [15, 25, 15, 95, 80, 5]},
    24: {'Session 1': [25, 15, 15, 40, 20, 10], 'Session 2': [20, 15, 20, 30, 5, 10]},
    25: {'Session 1': [80, 95, 5, 80, 85, 15], 'Session 2': [75, 95, 5, 85, 85, 15]},
    26: {'Session 1': [85, 75, 65, 50, 60, 65], 'Session 2': [50, 50, 45, 75, 85, 35]},
    27: {'Session 1': [10, 20, 10, 75, 90, 50], 'Session 2': [45, 10, 55, 85, 55, 5]},
    28: {'Session 1': [25, 10, 30, 50, 40, 30], 'Session 2': [15, 15, 25, 35, 20, 15]},
    29: {'Session 1': [35, 15, 15, 45, 60, 60], 'Session 2': [60, 50, 30, 40, 50, 60]},
    30: {'Session 1': [65, 50, 75, 80, 80, 25], 'Session 2': [50, 50, 65, 60, 75, 50]},
    31: {'Session 1': [70, 35, 20, 85, 80, 15], 'Session 2': [60, 10, 10, 65, 70, 5]},
    32: {'Session 1': [65, 75, 40, 35, 85, 20], 'Session 2': [75, 80, 45, 45, 70, 35]},
    33: {'Session 1': [25, 25, 50, 10, 85, 30], 'Session 2': [35, 20, 40, 10, 35, 20]},
    34: {'Session 1': [60, 85, 35, 50, 75, 30], 'Session 2': [70, 90, 80, 80, 90, 50]},

    35: {'Session 1': [25, 35, 30, 40, 60, 40], 'Session 2': [55, 35, 35, 50, 65, 50]},
    37: {'Session 1': [75, 75, 50, 100, 100, 25], 'Session 2': [100, 75, 75, 100, 100, 0]},
    38: {'Session 1': [85, 15, 35, 55, 85, 75], 'Session 2': [85, 70, 45, 60, 85, 65]},
    40: {'Session 1': [0, 20, 10, 45, 85, 50], 'Session 2': [5, 20, 15, 20, 85, 50]},
    41: {'Session 1': [60, 45, 60, 50, 75, 55], 'Session 2': [60, 45, 55, 50, 70, 60]},
    42: {'Session 1': [75, 55, 75, 35, 90, 75], 'Session 2': [85, 75, 45, 85, 80, 80]},
    43: {'Session 1': [50, 0, 0, 50, 50, 0], 'Session 2': [50, 35, 0, 75, 50, 0]},
    44: {'Session 1': [35, 45, 20, 70, 70, 30], 'Session 2': [35, 25, 15, 90, 85, 10]},
    45: {'Session 1': [80, 75, 85, 50, 80, 75], 'Session 2': [50, 50, 40, 90, 80, 25]},
    47: {'Session 1': [65, 65, 50, 40, 70, 65], 'Session 2': [50, 65, 40, 60, 75, 65]},
    48: {'Session 1': [80, 75, 25, 25, 75, 60], 'Session 2': [65, 35, 50, 50, 75, 50]},
    50: {'Session 1': [80, 85, 60, 70, 85, 65], 'Session 2': [60, 75, 60, 80, 80, 70]},
    51: {'Session 1': [90, 40, 75, 35, 85, 80], 'Session 2': [90, 65, 40, 40, 90, 60]},
    52: {'Session 1': [90, 85, 90, 10, 90, 90], 'Session 2': [85, 90, 85, 20, 85, 80]},
    54: {'Session 1': [75, 60, 30, 30, 85, 65], 'Session 2': [60, 45, 20, 55, 75, 40]},
    55: {'Session 1': [70, 25, 15, 65, 75, 65], 'Session 2': [55, 35, 30, 60, 50, 35]},
    57: {'Session 1': [75, 65, 70, 85, 80, 60], 'Session 2': [75, 80, 65, 75, 75, 60]},
    58: {'Session 1': [50, 15, 10, 50, 75, 65], 'Session 2': [15, 15, 10, 85, 50, 5]},
    63: {'Session 1': [95, 65, 75, 65, 95, 75], 'Session 2': [70, 35, 55, 70, 70, 50]},
    64: {'Session 1': [65, 40, 10, 55, 85, 45], 'Session 2': [80, 35, 5, 65, 90, 55]},
    65: {'Session 1': [20, 65, 50, 25, 65, 75], 'Session 2': [65, 60, 35, 45, 70, 55]},
    68: {'Session 1': [70, 30, 50, 45, 65, 50], 'Session 2': [75, 0, 20, 35, 70, 60]},
    69: {'Session 1': [65, 50, 35, 90, 50, 15], 'Session 2': [65, 75, 65, 90, 75, 25]},
    70: {'Session 1': [75, 50, 60, 60, 70, 60], 'Session 2': [55, 30, 50, 75, 75, 25]},
}

raw_data_22 = {
    35: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    37: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    38: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    40: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    41: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    42: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    43: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    44: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    45: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    47: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    48: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    50: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    51: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    52: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    54: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    55: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    57: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    58: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    63: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    64: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    65: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    68: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    69: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    70: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}}
}

group_tradition_ids = [17, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
group_vr_ids = [35, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 54, 55, 57, 58, 63, 64, 65, 68, 69, 70]

categories = ['Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

session_colors = {
    'Session 1': '#1f77b4',
    'Session 2': '#ff7f0e',
    'Overall Average': '#2ca02c',
    'Stactic': '#4C72B0',
    'Adaptive': '#DD8452'
}


# ----------------------------------------
# 4. 繪圖共用函式
# ----------------------------------------
def plot_radar(data_dict, title, filename):
    plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, size=12)
    ax.tick_params(axis='x', pad=50)
    ax.set_rlabel_position(60)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)

    for label_name, values in data_dict.items():
        val = values + [values[0]]
        color = session_colors.get(label_name, '#d62728')
        ax.plot(angles, val, linewidth=2, linestyle='solid', color=color, label=label_name)
        ax.fill(angles, val, alpha=0.25, color=color)

    plt.title(title, size=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    if SAVE_FIG:
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"✅ 圖表已儲存至: {filepath}")

    plt.close()


# ----------------------------------------
# 5. 處理分組資料繪圖
# ----------------------------------------
def process_group_data(group_name, member_ids):
    group_data = {k: v for k, v in data.items() if k in member_ids}
    session1_all = []
    session2_all = []

    for user_id, sessions in group_data.items():
        plot_radar(sessions, f'[{group_name}] NASA-TLX (ID: {user_id})', f'{group_name}_id_{user_id}.png')
        session1_all.append(sessions['Session 1'])
        session2_all.append(sessions['Session 2'])

    avg_session1 = np.mean(session1_all, axis=0).tolist()
    avg_session2 = np.mean(session2_all, axis=0).tolist()
    avg_overall = np.mean(session1_all + session2_all, axis=0).tolist()

    plot_radar({'Session 1': avg_session1}, f'[{group_name}] Average (S1)', f'{group_name}_avg_s1.png')
    plot_radar({'Session 2': avg_session2}, f'[{group_name}] Average (S2)', f'{group_name}_avg_s2.png')
    plot_radar({'Session 1': avg_session1, 'Session 2': avg_session2}, f'[{group_name}] Comparison (S1 vs S2)',
               f'{group_name}_comparison.png')
    plot_radar({'Overall Average': avg_overall}, f'[{group_name}] Overall Average', f'{group_name}_overall.png')


def plot_condition_comparison(data_dict, cond_data):
    cond_a_all = []
    cond_b_all = []
    for subject_id, sessions in data_dict.items():
        if subject_id in cond_data:
            for session_name, values in sessions.items():
                sess_key = 's1' if session_name == 'Session 1' else 's2'
                cond = cond_data[subject_id].get(sess_key, {}).get('cond')
                if cond == 'A':
                    cond_a_all.append(values)
                elif cond == 'B':
                    cond_b_all.append(values)

    if cond_a_all and cond_b_all:
        avg_cond_a = np.mean(cond_a_all, axis=0).tolist()
        avg_cond_b = np.mean(cond_b_all, axis=0).tolist()
        plot_radar({'Stactic': avg_cond_a, 'Adaptive': avg_cond_b},
                   '[VR] Average Comparison (Stactic vs Adaptive)',
                   'VR_avg_comparison_CondA_vs_CondB.png')


# ----------------------------------------
# 6. 匯出 Excel
# ----------------------------------------
def export_to_excel(data_dict, tradition_ids, vr_ids, cond_data, output_dir):
    records = []
    for subject_id, sessions in data_dict.items():
        group = "Tradition" if subject_id in tradition_ids else ("VR" if subject_id in vr_ids else "Unknown")
        for session_name, values in sessions.items():
            sess_key = 's1' if session_name == 'Session 1' else 's2'
            cond = cond_data.get(subject_id, {}).get(sess_key, {}).get('cond', 'N/A')
            record = {
                "ID": subject_id, "Group": group, "Condition": cond, "Session": session_name,
                "Mental Demand": values[0], "Physical Demand": values[1], "Temporal Demand": values[2],
                "Performance": values[3], "Effort": values[4], "Frustration": values[5]
            }
            records.append(record)
    df = pd.DataFrame(records).sort_values(by=["Group", "Condition", "ID", "Session"])
    excel_path = os.path.join(output_dir, "NASA_TLX_Records.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n📊 數據已匯出至 Excel: {excel_path}")


# ----------------------------------------
# 7. 統計分析函式
# ----------------------------------------
def print_stat_result(dim_name, mean1, name1, mean2, name2, p_val):
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    if p_val < 0.05:
        higher = name1 if mean1 > mean2 else name2
        trend = f"({higher} 顯著較高)"
    else:
        trend = "(無顯著差異)"
    # print(f"{dim_name:18s} | {name1}: {mean1:5.1f} vs {name2}: {mean2:5.1f} | p-value: {p_val:.4f} {sig} {trend}")
    print(f"| {dim_name:18s} | {mean1:5.1f} vs{mean2:5.1f} | {p_val:.4f} {sig} | {trend} |")


def run_statistical_analysis():
    print("\n" + "=" * 60)
    print("📈 統計分析報告 (NASA-TLX 六個維度)")
    print("=" * 60)

    # 1. Condition 分析 (Static(A) vs Adaptive(B)) - 僅限 VR 組，相依樣本 (Paired t-test)
    print("\n[分析一] Condition: Static (A) vs Adaptive (B) - 針對 VR 組 (Paired t-test)")
    print("-" * 60)
    cond_a_data, cond_b_data = [], []

    for uid in group_vr_ids:
        if uid in raw_data_22:
            s1_cond = raw_data_22[uid]['s1']['cond']
            val_a = data[uid]['Session 1'] if s1_cond == 'A' else data[uid]['Session 2']
            val_b = data[uid]['Session 1'] if s1_cond == 'B' else data[uid]['Session 2']
            cond_a_data.append(val_a)
            cond_b_data.append(val_b)

    cond_a_arr = np.array(cond_a_data)
    cond_b_arr = np.array(cond_b_data)

    for i, dim in enumerate(categories):
        arr_a, arr_b = cond_a_arr[:, i], cond_b_arr[:, i]
        t_stat, p_val = stats.ttest_rel(arr_a, arr_b)
        print_stat_result(dim, np.mean(arr_a), "Static(A)", np.mean(arr_b), "Adaptive(B)", p_val)

    # # 2. Session 分析 (Session 1 vs Session 2) - 所有受測者，相依樣本 (Paired t-test)
    # print("\n[分析二] Session: Session 1 vs Session 2 - 所有受測者 (Paired t-test)")
    # print("-" * 60)
    # s1_data = np.array([sessions['Session 1'] for uid, sessions in data.items()])
    # s2_data = np.array([sessions['Session 2'] for uid, sessions in data.items()])
    #
    # for i, dim in enumerate(categories):
    #     arr_s1, arr_s2 = s1_data[:, i], s2_data[:, i]
    #     t_stat, p_val = stats.ttest_rel(arr_s1, arr_s2)
    #     print_stat_result(dim, np.mean(arr_s1), "S1", np.mean(arr_s2), "S2", p_val)
    # 2. Session 分析 (Session 1 vs Session 2) - 僅限 VR 組，相依樣本 (Paired t-test)
    print("\n[分析二] Session: Session 1 vs Session 2 - 針對 VR 組 (Paired t-test)")
    print("-" * 60)

    # 在這裡加上 if uid in group_vr_ids 來篩選出 VR 組的人
    s1_data = np.array([sessions['Session 1'] for uid, sessions in data.items() if uid in group_vr_ids])
    s2_data = np.array([sessions['Session 2'] for uid, sessions in data.items() if uid in group_vr_ids])

    for i, dim in enumerate(categories):
        arr_s1, arr_s2 = s1_data[:, i], s2_data[:, i]
        t_stat, p_val = stats.ttest_rel(arr_s1, arr_s2)
        print_stat_result(dim, np.mean(arr_s1), "S1", np.mean(arr_s2), "S2", p_val)
    # 3. Group 分析 (Tradition vs VR) - 取每個人的整體平均，獨立樣本 (Independent t-test)
    print("\n[分析三] Group: Tradition vs VR - 取受測者 S1+S2 的平均進行比較 (Independent t-test)")
    print("-" * 60)
    trad_data, vr_data = [], []
    for uid, sessions in data.items():
        avg_scores = np.mean([sessions['Session 1'], sessions['Session 2']], axis=0)
        if uid in group_tradition_ids:
            trad_data.append(avg_scores)
        elif uid in group_vr_ids:
            vr_data.append(avg_scores)

    trad_arr = np.array(trad_data)
    vr_arr = np.array(vr_data)

    for i, dim in enumerate(categories):
        arr_trad, arr_vr = trad_arr[:, i], vr_arr[:, i]
        # 使用 welch's t-test (equal_var=False) 來處理兩組樣本數可能不均等、變異數未知的狀況
        t_stat, p_val = stats.ttest_ind(arr_trad, arr_vr, equal_var=False)
        print_stat_result(dim, np.mean(arr_trad), "Tradition", np.mean(arr_vr), "VR", p_val)
    print("=" * 60 + "\n")


# ----------------------------------------
# 8. 執行主程序
# ----------------------------------------
# 1. 執行繪圖
process_group_data("Tradition", group_tradition_ids)
process_group_data("VR", group_vr_ids)
plot_condition_comparison(data, raw_data_22)

# 2. 匯出 Excel
if SAVE_FIG:
    export_to_excel(data, group_tradition_ids, group_vr_ids, raw_data_22, OUTPUT_DIR)

# 3. 執行統計檢定並在終端機輸出報告
run_statistical_analysis()

print("\n🎉 所有資料處理、雷達圖形繪製、Excel 匯出及「統計檢定」皆已完成！")