"""
================================================================================
BCI 受試者分群篩選與神經生理特徵診斷系統 (Subject Stratification & Profiling)
(原 generate_metric_sum01.py 升級改版)
================================================================================

【功能說明】
依據分類準確度 (Accuracy) 與可解釋性神經特徵 (Spectral Saliency, Spatial Saliency)，
對受試者進行多維度分析與群體篩選：
1. 全面優異表現組 (All-Around Top Performers):
   - 全部指標 (Acc, Spectral Saliency, Spatial Saliency) 均落在前 30% (可自訂百分比)
2. 學習突破組 (Top Improved Learners):
   - 跨 Session (S2 - S1) 綜合進步幅度前 N 名
3. 潛在學習組 (Potential Learners: 高特徵但低準確度):
   - Spectral Saliency 或 Spatial Saliency >= 前 30%，但準確度落於後 30% (或後 50%)
   - 代表已成功誘發特定腦波特徵，但模型尚未完全調適或受雜訊影響
4. 替代控制組 (Alternative Controllers: 高準確度但低特徵):
   - 準確度 >= 前 30%，但 Spectral 或 Spatial Saliency 落於後 30%
   - 代表受試者可能採用不同頻段或空間區域之替代腦波模式達成控制

【使用範例】
  # 1. 直接讀取 metrics_summary.csv (預設)
  python subject_stratification.py --csv_path metrics_summary.csv

  # 2. 自訂前 25%、後 25% 門檻並輸出報告至檔案
  python subject_stratification.py --top_pct 25 --bottom_pct 25 --output_txt stratification_report.txt
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

# 自動支援 UTF-8
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

from common_utils import tee_log, get_subject_alias_map, DEFAULT_IDS, DEFAULT_SESSIONS


def load_stratification_data_from_csv(csv_path):
    """自 CSV 載入資料並計算受試者平均指標與進步幅度"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到指定的 CSV 檔案: {csv_path}")

    df = pd.read_csv(csv_path)

    spectral_col = "Spectral_Saliency" if "Spectral_Saliency" in df.columns else "MBSR"
    spatial_col = "Spatial_Saliency" if "Spatial_Saliency" in df.columns else "MSFI"

    stats = {}
    for sub_id, sub_group in df.groupby("subject_id"):
        sub_alias = sub_group["subject_alias"].iloc[0] if "subject_alias" in sub_group.columns else f"S{sub_id}"

        sub_stats = {
            "id": int(sub_id),
            "alias": sub_alias
        }

        # 分別取出 s1 與 s2
        for sess in ["s1", "s2"]:
            sess_rows = sub_group[sub_group["session"] == sess]
            if len(sess_rows) > 0:
                # off_sess_acc 轉為百分比
                off_sess_val = sess_rows["offline_sess_acc"].dropna()
                if len(off_sess_val) > 0:
                    acc_val = float(off_sess_val.iloc[0])
                    if acc_val <= 1.05:
                        acc_val *= 100
                else:
                    # 若無 off_sess 則使用 off_run 平均
                    off_runs = sess_rows["offline_run_acc"].dropna().tolist()
                    acc_val = float(np.mean(off_runs)) * 100 if len(off_runs) > 0 else np.nan

                spec_vals = sess_rows[spectral_col].dropna().tolist()
                spat_vals = sess_rows[spatial_col].dropna().tolist()

                sub_stats[f"{sess}_acc"] = acc_val
                sub_stats[f"{sess}_spectral"] = float(np.mean(spec_vals)) if len(spec_vals) > 0 else np.nan
                sub_stats[f"{sess}_spatial"] = float(np.mean(spat_vals)) if len(spat_vals) > 0 else np.nan
            else:
                sub_stats[f"{sess}_acc"] = np.nan
                sub_stats[f"{sess}_spectral"] = np.nan
                sub_stats[f"{sess}_spatial"] = np.nan

        # 計算全域平均
        sub_stats["mean_acc"] = np.nanmean([sub_stats["s1_acc"], sub_stats["s2_acc"]])
        sub_stats["mean_spectral"] = np.nanmean([sub_stats["s1_spectral"], sub_stats["s2_spectral"]])
        sub_stats["mean_spatial"] = np.nanmean([sub_stats["s1_spatial"], sub_stats["s2_spatial"]])

        # 計算進步幅度 (S2 - S1)
        sub_stats["diff_acc"] = sub_stats["s2_acc"] - sub_stats["s1_acc"]
        sub_stats["diff_spectral"] = sub_stats["s2_spectral"] - sub_stats["s1_spectral"]
        sub_stats["diff_spatial"] = sub_stats["s2_spatial"] - sub_stats["s1_spatial"]

        # 三項指標總進步量
        sub_stats["total_improvement"] = (
            (sub_stats["diff_acc"] if not np.isnan(sub_stats["diff_acc"]) else 0) +
            (sub_stats["diff_spectral"] if not np.isnan(sub_stats["diff_spectral"]) else 0) +
            (sub_stats["diff_spatial"] if not np.isnan(sub_stats["diff_spatial"]) else 0)
        )

        stats[int(sub_id)] = sub_stats

    return stats


def perform_stratification(stats, top_pct=30.0, bottom_pct=30.0, mid_pct=50.0, top_n_improved=3):
    """
    執行分群篩選核心演算法
    """
    acc_list = [s['mean_acc'] for s in stats.values() if not np.isnan(s['mean_acc'])]
    spectral_list = [s['mean_spectral'] for s in stats.values() if not np.isnan(s['mean_spectral'])]
    spatial_list = [s['mean_spatial'] for s in stats.values() if not np.isnan(s['mean_spatial'])]

    acc_top = np.percentile(acc_list, 100.0 - top_pct)
    acc_mid = np.percentile(acc_list, mid_pct)
    acc_bottom = np.percentile(acc_list, bottom_pct)

    spectral_top = np.percentile(spectral_list, 100.0 - top_pct)
    spectral_bottom = np.percentile(spectral_list, bottom_pct)

    spatial_top = np.percentile(spatial_list, 100.0 - top_pct)
    spatial_bottom = np.percentile(spatial_list, bottom_pct)

    thresholds = {
        "acc_top": acc_top, "acc_mid": acc_mid, "acc_bottom": acc_bottom,
        "spectral_top": spectral_top, "spectral_bottom": spectral_bottom,
        "spatial_top": spatial_top, "spatial_bottom": spatial_bottom,
        "top_pct": top_pct, "bottom_pct": bottom_pct, "mid_pct": mid_pct
    }

    # 各指標前 3 名
    top3_acc = sorted(stats.keys(), key=lambda x: stats[x]['mean_acc'], reverse=True)[:3]
    top3_spectral = sorted(stats.keys(), key=lambda x: stats[x]['mean_spectral'], reverse=True)[:3]
    top3_spatial = sorted(stats.keys(), key=lambda x: stats[x]['mean_spatial'], reverse=True)[:3]

    # 分類名單
    all_top = []
    high_spectral_low_acc_bottom = []
    high_spectral_low_acc_mid = []
    high_spatial_low_acc_bottom = []
    high_spatial_low_acc_mid = []
    high_acc_low_saliency = []

    for sub_id, s in stats.items():
        m_acc, m_spec, m_spat = s['mean_acc'], s['mean_spectral'], s['mean_spatial']

        # 1. 全部指標前 top_pct%
        if m_acc >= acc_top and m_spec >= spectral_top and m_spat >= spatial_top:
            all_top.append(sub_id)

        # 2. 高 Spectral 但低 Acc
        if m_spec >= spectral_top:
            if m_acc <= acc_bottom:
                high_spectral_low_acc_bottom.append(sub_id)
            if m_acc <= acc_mid:
                high_spectral_low_acc_mid.append(sub_id)

        # 3. 高 Spatial 但低 Acc
        if m_spat >= spatial_top:
            if m_acc <= acc_bottom:
                high_spatial_low_acc_bottom.append(sub_id)
            if m_acc <= acc_mid:
                high_spatial_low_acc_mid.append(sub_id)

        # 4. 準確度高但 Saliency 低
        if m_acc >= acc_top and (m_spec <= spectral_bottom or m_spat <= spatial_bottom):
            high_acc_low_saliency.append(sub_id)

    # 進步幅度前 N 名
    sorted_by_imp = sorted(stats.keys(), key=lambda x: stats[x]['total_improvement'], reverse=True)
    top_improved = sorted_by_imp[:top_n_improved]

    results = {
        "thresholds": thresholds,
        "top3_acc": top3_acc,
        "top3_spectral": top3_spectral,
        "top3_spatial": top3_spatial,
        "all_top": all_top,
        "top_improved": top_improved,
        "high_spectral_low_acc_bottom": high_spectral_low_acc_bottom,
        "high_spectral_low_acc_mid": high_spectral_low_acc_mid,
        "high_spatial_low_acc_bottom": high_spatial_low_acc_bottom,
        "high_spatial_low_acc_mid": high_spatial_low_acc_mid,
        "high_acc_low_saliency": high_acc_low_saliency,
    }
    return results


def format_report_text(stats, results):
    """格式化報告字串"""
    th = results["thresholds"]
    lines = []
    lines.append("=" * 80)
    lines.append("📊 BCI 受試者分群篩選與特徵診斷分析報告 (Subject Stratification Report)")
    lines.append(f"   產出時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 受試者總數: {len(stats)}")
    lines.append("=" * 80)

    lines.append("\n📌 分位數基準門檻 (Percentile Thresholds):")
    lines.append(f"  • 準確度 (Accuracy)     : 前 {th['top_pct']:.0f}% >= {th['acc_top']:.2f}% | 後 {th['mid_pct']:.0f}% <= {th['acc_mid']:.2f}% | 後 {th['bottom_pct']:.0f}% <= {th['acc_bottom']:.2f}%")
    lines.append(f"  • Spectral Saliency     : 前 {th['top_pct']:.0f}% >= {th['spectral_top']:.2f}% | 後 {th['bottom_pct']:.0f}% <= {th['spectral_bottom']:.2f}%")
    lines.append(f"  • Spatial Saliency      : 前 {th['top_pct']:.0f}% >= {th['spatial_top']:.2f}% | 後 {th['bottom_pct']:.0f}% <= {th['spatial_bottom']:.2f}%")

    def format_subs(sub_ids):
        if not sub_ids:
            return "無"
        return ", ".join([f"{stats[s]['alias']} (ID {s})" for s in sub_ids])

    lines.append("\n" + "-" * 80)
    lines.append("🥇 0. 各指標總體表現前 3 名 (Top 3):")
    lines.append(f"  • 準確度 (Acc)        : " + ", ".join([f"{stats[s]['alias']} ({stats[s]['mean_acc']:.2f}%)" for s in results['top3_acc']]))
    lines.append(f"  • Spectral Saliency   : " + ", ".join([f"{stats[s]['alias']} ({stats[s]['mean_spectral']:.2f}%)" for s in results['top3_spectral']]))
    lines.append(f"  • Spatial Saliency    : " + ", ".join([f"{stats[s]['alias']} ({stats[s]['mean_spatial']:.2f}%)" for s in results['top3_spatial']]))

    lines.append("\n" + "-" * 80)
    lines.append(f"🎯 1. 全面優異組 (Acc, Spectral, Spatial 皆落在前 {th['top_pct']:.0f}%):")
    lines.append(f"  名單: {format_subs(results['all_top'])}")

    lines.append("\n" + "-" * 80)
    lines.append(f"📈 2. 學習突破組 (跨 Session 進步幅度前 {len(results['top_improved'])} 名, S2 - S1 總和):")
    for i, s_id in enumerate(results['top_improved'], 1):
        s = stats[s_id]
        lines.append(
            f"  Top {i} -> {s['alias']} (ID {s_id}): 總進步 {s['total_improvement']:+.2f} "
            f"[Acc提升 {s['diff_acc']:+.2f}%, Spectral提升 {s['diff_spectral']:+.2f}%, Spatial提升 {s['diff_spatial']:+.2f}%]"
        )

    lines.append("\n" + "-" * 80)
    lines.append("🧠 3. 潛在學習組 (高 Saliency 特徵但低 Acc 表現):")
    lines.append(f"  • 高 Spectral Saliency (前{th['top_pct']:.0f}%) 但 準確度落於後 {th['bottom_pct']:.0f}%: {format_subs(results['high_spectral_low_acc_bottom'])}")
    lines.append(f"  • 高 Spectral Saliency (前{th['top_pct']:.0f}%) 但 準確度落於後 {th['mid_pct']:.0f}%:    {format_subs(results['high_spectral_low_acc_mid'])}")
    lines.append(f"  • 高 Spatial Saliency  (前{th['top_pct']:.0f}%) 但 準確度落於後 {th['bottom_pct']:.0f}%: {format_subs(results['high_spatial_low_acc_bottom'])}")
    lines.append(f"  • 高 Spatial Saliency  (前{th['top_pct']:.0f}%) 但 準確度落於後 {th['mid_pct']:.0f}%:    {format_subs(results['high_spatial_low_acc_mid'])}")

    lines.append("\n" + "-" * 80)
    lines.append(f"🚀 4. 替代控制組 (高表現但非典型顯著特徵: 準確度前 {th['top_pct']:.0f}% 但 Spectral 或 Spatial 落於後 {th['bottom_pct']:.0f}%):")
    lines.append(f"  名單: {format_subs(results['high_acc_low_saliency'])}")

    lines.append("=" * 80)
    return "\n".join(lines)


def export_stratification_table(stats, results, output_csv):
    """將每位受試者的詳細統計與分群標籤匯出為 CSV"""
    rows = []
    th = results["thresholds"]
    for sub_id, s in stats.items():
        is_all_top = sub_id in results["all_top"]
        is_top_improved = sub_id in results["top_improved"]
        is_potential = (
            sub_id in results["high_spectral_low_acc_mid"] or
            sub_id in results["high_spatial_low_acc_mid"]
        )
        is_alt = sub_id in results["high_acc_low_saliency"]

        cohort = []
        if is_all_top: cohort.append("All_Top")
        if is_top_improved: cohort.append("Top_Improved")
        if is_potential: cohort.append("Potential_Learner")
        if is_alt: cohort.append("Alt_Controller")
        if not cohort: cohort.append("Standard")

        rows.append({
            "subject_id": sub_id,
            "subject_alias": s["alias"],
            "s1_acc": round(s["s1_acc"], 2),
            "s2_acc": round(s["s2_acc"], 2),
            "mean_acc": round(s["mean_acc"], 2),
            "s1_spectral": round(s["s1_spectral"], 2),
            "s2_spectral": round(s["s2_spectral"], 2),
            "mean_spectral": round(s["mean_spectral"], 2),
            "s1_spatial": round(s["s1_spatial"], 2),
            "s2_spatial": round(s["s2_spatial"], 2),
            "mean_spatial": round(s["mean_spatial"], 2),
            "diff_acc": round(s["diff_acc"], 2),
            "diff_spectral": round(s["diff_spectral"], 2),
            "diff_spatial": round(s["diff_spatial"], 2),
            "total_improvement": round(s["total_improvement"], 2),
            "cohort": ";".join(cohort)
        })

    df_out = pd.DataFrame(rows).sort_values("subject_id")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"💾 受試者分群資料庫已儲存至: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="BCI 受試者分群篩選與神經特徵診斷系統")
    parser.add_argument("--csv_path", type=str, default="metrics_summary.csv",
                        help="輸入的指標 CSV 檔案路徑 (預設: metrics_summary.csv)")
    parser.add_argument("--top_pct", type=float, default=30.0,
                        help="頂尖表現百分位門檻 (預設 30.0，代表前 30%%)")
    parser.add_argument("--bottom_pct", type=float, default=30.0,
                        help="低表現百分位門檻 (預設 30.0，代表後 30%%)")
    parser.add_argument("--mid_pct", type=float, default=50.0,
                        help="中度低表現百分位門檻 (預設 50.0，代表後 50%%)")
    parser.add_argument("--top_n_improved", type=int, default=3,
                        help="進步幅度前 N 名榜單人數 (預設 3)")
    parser.add_argument("--output_txt", type=str, default=None,
                        help="報告文字輸出檔案路徑 (留空則僅印至控制台)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="受試者分群彙整表格 CSV 輸出路徑 (可選)")

    args = parser.parse_args()

    stats = load_stratification_data_from_csv(args.csv_path)
    results = perform_stratification(
        stats,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        mid_pct=args.mid_pct,
        top_n_improved=args.top_n_improved
    )

    report_str = format_report_text(stats, results)
    print(report_str)

    if args.output_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)), exist_ok=True)
        with open(args.output_txt, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"📄 報告已保存至: {args.output_txt}")

    if args.output_csv:
        export_stratification_table(stats, results, args.output_csv)


if __name__ == "__main__":
    main()
