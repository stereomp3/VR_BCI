"""
線上即時 LSL 腦波串流至 Unity 快速測試工具 (向前相容轉接外殼)
完整實作與工具已移至 python/tools/online_test_lsl_to_unity.py
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(current_dir, "tools", "online_test_lsl_to_unity.py")
    cmd = [sys.executable, target_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
