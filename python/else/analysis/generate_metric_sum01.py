"""
相容外殼包裝 (Backward Compatibility Wrapper)
此檔案保留原檔名以支援舊有流程，核心實作調用 subject_stratification.py。
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from subject_stratification import main

if __name__ == "__main__":
    main()