"""
快速模型訓練與驗證工具 (向前相容轉接外殼)
完整實作與工具已移至 python/tools/quick_trainer.py
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(current_dir, "tools", "quick_trainer.py")
    cmd = [sys.executable, target_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
