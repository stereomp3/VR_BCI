"""
13 Channel 全 Session 合併交叉驗證訓練 (4-Fold Cross Validation for All Runs Combined)
此檔案為相容外殼包裝，核心實作調用 train_cv.py。
支援直接執行或透過 CLI 傳入自訂參數覆蓋預設值。
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train_cv import train_cross_validation, parse_train_args

def main():
    args = parse_train_args(default_channels="13", default_mode="all_runs")
    train_cross_validation(
        channels=args.channels,
        mode=args.mode,
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        ids=args.ids,
        sessions=args.sessions,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        k_folds=args.k_folds,
        strides=args.strides,
        seed=args.seed,
        log_file=args.log_file
    )

if __name__ == "__main__":
    main()