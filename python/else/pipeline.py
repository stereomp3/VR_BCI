"""
VR-BCI 一鍵全流程總管主程式 (VR-BCI End-to-End Master Pipeline)

支援以單一指令執行全流程，或指定單獨執行某個階段：
1. create_np : 將原始 EEG CSV 與 Log 轉換為 .pt (可指定 base_dir, output_dir, arrange_by_label)
2. train     : 執行 4-fold Cross Validation 訓練並儲存 Saliency Map
3. analyze   : 解析 Log，將指標結構匯出為 CSV，計算統計檢定 (Table 1-3) 並自動生成全套圖表
4. plot      : 直接自現有 CSV 載入資料並重新產生圖表 (秒級出圖)
5. all       : 依序一鍵執行上述所有階段

範例指令：
  python pipeline.py --step all --channels 22 --base_dir /path/to/data
  python pipeline.py --step analyze --channels 22
  python pipeline.py --step plot --csv_path metrics_summary.csv
"""

import os
import sys
import time
import argparse

# 設定編碼
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sub_dirs = ["preprocessing", "training", "analysis", "neuro_analysis", "utils"]
for sub in [current_dir] + [os.path.join(current_dir, d) for d in sub_dirs]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

from preprocessing.create_MInp import process_data_dir
from training.train_cv import train_cross_validation
from analysis.analyze_metrics_and_plot import run_analysis_pipeline
from analysis.subject_stratification import load_stratification_data_from_csv, perform_stratification, format_report_text


def run_pipeline():
    parser = argparse.ArgumentParser(description="VR-BCI 一鍵全自動處理管線")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "create_np", "train", "analyze", "plot", "stratify"],
                        help="執行階段: all (全流程), create_np (資料轉換), train (模型訓練), analyze (指標統計與出圖), plot (僅從 CSV 繪圖), stratify (受試者分群篩選)")
    parser.add_argument("--channels", type=str, default="22", choices=["13", "22"],
                        help="通道設定 ('13' 或 '22')")
    parser.add_argument("--base_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集根目錄")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="資料轉換 (.pt) 輸出資料夾 (若為 None 則與原始 CSV 同層)")
    parser.add_argument("--arrange_by_label", action="store_true", default=True,
                        help="資料轉換時是否依照標籤進行 0,1,0,1 平衡交錯排序 (預設: True)")
    parser.add_argument("--no_arrange_by_label", dest="arrange_by_label", action="store_false",
                        help="關閉標籤排序")
    parser.add_argument("--overwrite_np", action="store_true", default=False,
                        help="是否覆蓋已存在的 .pt 檔案")

    # 訓練相關參數
    parser.add_argument("--train_mode", type=str, default="both", choices=["per_run", "all_runs", "both"],
                        help="訓練模式: per_run (單 run 獨立訓練), all_runs (全 session 合併訓練), both (兩者皆跑)")
    parser.add_argument("--batch_size", type=int, default=8, help="訓練批次大小 (預設 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="訓練學習率 (預設 1e-3)")
    parser.add_argument("--epochs", type=int, default=100, help="訓練 Epoch 數 (預設 100)")
    parser.add_argument("--k_folds", type=int, default=4, help="交叉驗證 K 數 (預設 4)")
    parser.add_argument("--ids", type=str, default=None, help="指定受試者清單，逗號分隔 (留空為預設 24 人)")
    parser.add_argument("--sessions", type=str, default="s1,s2", help="指定 Session 清單 (預設 s1,s2)")

    # 分析與出圖相關參數
    parser.add_argument("--csv_path", type=str, default="metrics_summary.csv",
                        help="輸出或讀取的 CSV 檔案路徑 (預設: metrics_summary.csv)")
    parser.add_argument("--plot_dir", type=str, default="./charts",
                        help="圖表儲存目錄 (預設: ./charts)")
    parser.add_argument("--offline_runs_log", type=str, default=None,
                        help="Offline Run Log 檔案路徑 (留空自動搜尋)")
    parser.add_argument("--offline_sess_log", type=str, default=None,
                        help="Offline Session Log 檔案路徑 (留空自動搜尋)")
    parser.add_argument("--metric_txt", type=str, default=None,
                        help="Saliency 指標 Markdown 表格 (留空自動搜尋)")
    parser.add_argument("--online_dir", type=str, default=None,
                        help="Online Log 資料夾路徑")

    args = parser.parse_args()

    # 智慧路徑適配：若給定的路徑不存在於目前工作目錄，自動檢查是否在腳本同層目錄
    if not os.path.isabs(args.csv_path):
        if not os.path.exists(args.csv_path):
            script_csv = os.path.join(current_dir, args.csv_path)
            if os.path.exists(script_csv):
                args.csv_path = script_csv

    if not os.path.isabs(args.plot_dir) and args.plot_dir in ["./charts", "charts"]:
        args.plot_dir = os.path.join(current_dir, "charts")

    start_total_time = time.time()
    print("=" * 80)
    print("🚀 VR-BCI Master Pipeline 啟動")
    print(f"  ├── 執行目標: {args.step.upper()}")
    print(f"  ├── 通道模式: {args.channels} Channels")
    print(f"  ├── 資料目錄: {args.base_dir}")
    print(f"  ├── CSV 檔名: {args.csv_path}")
    print(f"  └── 圖表目錄: {args.plot_dir}")
    print("=" * 80)

    # 階段 1: 資料轉換 (create_np)
    if args.step in ["all", "create_np"]:
        print("\n" + "#" * 80)
        print("【階段 1】原始 EEG 資料轉換為 .pt (create_MInp)")
        print("#" * 80)
        if not os.path.exists(args.base_dir):
            print(f"⚠️ [警告] 找不到資料集目錄: {args.base_dir}")
            print(f"👉 請透過 --base_dir <路徑> 指定您的 EEG 資料夾位置 (例如: --base_dir D:/data/DATA_Cygnus)")
            if args.step == "create_np":
                return
            print("跳過資料轉換，繼續後續流程...\n")
        else:
            t0 = time.time()
            process_data_dir(
                base_dir=args.base_dir,
                output_dir=args.output_dir,
                channels=args.channels,
                do_arrange_by_label=args.arrange_by_label,
                overwrite=args.overwrite_np
            )
            print(f"⏱️ 階段 1 完成，耗時: {time.time() - t0:.2f} 秒\n")

    # 階段 2: 4-Fold 交叉驗證訓練 (train)
    if args.step in ["all", "train"]:
        print("\n" + "#" * 80)
        print("【階段 2】4-Fold 交叉驗證訓練與 Saliency 運算 (train_cv)")
        print("#" * 80)
        t0 = time.time()
        modes = ["per_run", "all_runs"] if args.train_mode == "both" else [args.train_mode]
        for m in modes:
            print(f"\n>>> 執行訓練模式: {m} ...")
            train_cross_validation(
                channels=args.channels,
                mode=m,
                base_dir=args.base_dir,
                ids=args.ids,
                sessions=args.sessions,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                k_folds=args.k_folds
            )
        print(f"⏱️ 階段 2 完成，耗時: {time.time() - t0:.2f} 秒\n")

    # 階段 3: 指標整合、儲存 CSV 與產出全部統計/圖表 (analyze / plot)
    if args.step in ["all", "analyze", "plot"]:
        print("\n" + "#" * 80)
        print("【階段 3】指標分析、CSV 資料庫持久化與全套圖表繪製")
        print("#" * 80)
        t0 = time.time()
        from_csv_flag = (args.step == "plot")
        run_analysis_pipeline(
            csv_path=args.csv_path,
            from_csv=from_csv_flag,
            offline_runs_log=args.offline_runs_log,
            offline_sess_log=args.offline_sess_log,
            metric_txt=args.metric_txt,
            online_dir=args.online_dir,
            base_dir=args.base_dir,
            channels=args.channels,
            plot_dir=args.plot_dir,
            do_plot=True
        )
        print(f"⏱️ 階段 3 完成，耗時: {time.time() - t0:.2f} 秒\n")

    # 階段 4: 受試者分群篩選與特徵診斷分析 (stratify)
    if args.step in ["all", "stratify"]:
        print("\n" + "#" * 80)
        print("【階段 4】受試者多維度分群篩選與特徵診斷分析 (subject_stratification)")
        print("#" * 80)
        t0 = time.time()
        csv_file = args.csv_path if os.path.exists(args.csv_path) else "metrics_summary.csv"
        if os.path.exists(csv_file):
            stats = load_stratification_data_from_csv(csv_file)
            res = perform_stratification(stats)
            print(format_report_text(stats, res))
        else:
            print(f"⚠️ 找不到 CSV 資料檔: {csv_file}，略過分群分析。")
        print(f"⏱️ 階段 4 完成，耗時: {time.time() - t0:.2f} 秒\n")

    print("=" * 80)
    print(f"🎉 全部指定流程執行完成！總耗時: {time.time() - start_total_time:.2f} 秒")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
