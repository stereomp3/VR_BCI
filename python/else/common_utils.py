"""
Common utilities root facade.
Redirects to utils.common_utils for modular organization and backward compatibility.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")
for d in [current_dir, utils_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from utils.common_utils import *
