"""
VR-BCI 系統全域設定模組 (System Configuration)
支援自動讀取共用設定檔 config.json (Single Source of Truth)，
並動態適配 8, 13, 22, 32 通道設定與模型參數。
"""

import os
import sys
import json
from enum import Enum
from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, ATCNet
from main.EEG.models import SCCNet

# --- 1. 路徑基礎配置 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.dirname(_current_dir)  # python/main
PYTHON_DIR = os.path.dirname(BASE_FILE)    # python
REPO_ROOT = os.path.dirname(PYTHON_DIR)    # VR_BCI repo root

# --- 2. 搜尋並載入 config.json (Single Source of Truth) ---
CONFIG_CANDIDATES = [
    os.path.join(PYTHON_DIR, "config.json"),
    os.path.join(REPO_ROOT, "unity", "Assets", "StreamingAssets", "config.json"),
    os.path.join(REPO_ROOT, "unity_noVR", "Assets", "StreamingAssets", "config.json"),
    os.path.join(os.getcwd(), "config.json"),
]

JSON_CONFIG = {}
CONFIG_FILE_LOADED = None
for candidate in CONFIG_CANDIDATES:
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                JSON_CONFIG = json.load(f)
                CONFIG_FILE_LOADED = candidate
                break
        except Exception as e:
            print(f"[WARNING] 讀取設定檔 {candidate} 失敗: {e}")

# --- 3. 通道定義設定 (8, 13, 22, 32 通道) ---
DEFAULT_CHANNEL_DEFINITIONS = {
    "8": list(range(2, 10)),  # [2, 3, 4, 5, 6, 7, 8, 9]
    "13": [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 28],
    "22": [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33],
    "32": list(range(2, 34))  # 2~33
}

CHANNEL_DEFINITIONS = JSON_CONFIG.get("channel_definitions", DEFAULT_CHANNEL_DEFINITIONS)
ACTIVE_CHANNEL_MODE = str(JSON_CONFIG.get("active_channels", "32"))

if ACTIVE_CHANNEL_MODE not in CHANNEL_DEFINITIONS:
    print(f"[WARNING] 未知 active_channels '{ACTIVE_CHANNEL_MODE}', 自動切換為預設 32 通道")
    ACTIVE_CHANNEL_MODE = "32"

channel_index = CHANNEL_DEFINITIONS[ACTIVE_CHANNEL_MODE]
N_CHANNELS = len(channel_index)
EEG_CHANNELS = len(channel_index)

# --- 4. 網路與 LSL 設定 ---
tcp_net = JSON_CONFIG.get("tcp_network", {})
TCP_PORT = int(tcp_net.get("port", 50007))
TCP_HOST = str(tcp_net.get("python_bind_host", "0.0.0.0"))
SEPARATE_STR = str(tcp_net.get("separate_str", "@@@"))

lsl_cfg = JSON_CONFIG.get("lsl_settings", {})
RECEIVE_CYGNUS_LSL_STREAM = str(lsl_cfg.get("receive_cygnus_lsl_stream", "Cygnus-329018-RawEEG"))
TO_UNITY_LSL_STREAM = str(lsl_cfg.get("to_unity_lsl_stream", "MarkerStream"))

# --- 5. 訊號與腦波前處理參數 ---
eeg_cfg = JSON_CONFIG.get("eeg_settings", {})
SAMPLE_RATE = int(eeg_cfg.get("sample_rate", 500))
WINDOW_SECONDS = float(eeg_cfg.get("window_seconds", 1.0))
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
PREDICTION_INTERVAL = 0.01

band_pass_low = float(eeg_cfg.get("band_pass_low", 1.0))
band_pass_high = float(eeg_cfg.get("band_pass_high", 40.0))
is_simulated_eeg = bool(eeg_cfg.get("is_simulated_eeg", False))
is_simulated_unity = bool(eeg_cfg.get("is_simulated_unity", False))
SAVE_CSV = bool(eeg_cfg.get("save_csv", True))
REPLAY_BUFFER_LIMIT = int(eeg_cfg.get("replay_buffer_limit", 320))

# --- 6. 模型架構與參數 ---
N_Class = 2  # left 1, right 0
USE_MODEL = SCCNet
LOAD_MODEL_PARAM = dict(samples=SAMPLE_RATE, channels=N_CHANNELS, n_classes=N_Class, sfreq=SAMPLE_RATE)
verbose = False

# --- 7. 遊戲設定 ---
game_cfg = JSON_CONFIG.get("game_settings", {})
group_note_num = int(game_cfg.get("group_note_num", 5))
cube_space_time = float(game_cfg.get("cube_space_time", 0.7))
trial_train_interval = int(game_cfg.get("trial_train_interval", 4))
adaptive_model = bool(game_cfg.get("adaptive_model", False))

# --- 8. 線上自適應訓練參數 ---
adapt_cfg = JSON_CONFIG.get("online_adaptation", {})
adaption_batch_size = int(adapt_cfg.get("batch_size", 8))
adaption_learning_rate = float(adapt_cfg.get("learning_rate", 1e-3))
adaption_epochs = int(adapt_cfg.get("epochs", 4))
adaption_use_val = bool(adapt_cfg.get("use_val", True))

# --- 9. 檔案目錄設定 (跨平台標準寫法) ---
__REALTIME_BASE_FILE = os.path.join(BASE_FILE, "real_time_data")
CSV_FILENAME = os.path.join(__REALTIME_BASE_FILE, "eeg_record.csv")
LOG_FILENAME = os.path.join(__REALTIME_BASE_FILE, "log.txt")
PT_DATA_FILENAME = os.path.join(__REALTIME_BASE_FILE, "data.pt")

EEG_CHECKPOINT_MAIN_BASE_FILE = os.path.join(BASE_FILE, "EEG", "checkpoint_main") + os.sep
EEG_CHECKPOINT_TMP_BASE_FILE = os.path.join(BASE_FILE, "EEG", "checkpoints") + os.sep

os.makedirs(EEG_CHECKPOINT_MAIN_BASE_FILE, exist_ok=True)
os.makedirs(EEG_CHECKPOINT_TMP_BASE_FILE, exist_ok=True)
os.makedirs(__REALTIME_BASE_FILE, exist_ok=True)

MAIN_CHECKPOINT = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, "model.pth")
TRAINED_CHECKPOINT = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, "model_trained.pth")

TRAINING_FINISH_STR = "training done"
SENT_UNITY_MODEL_STR = "SENT_UNITY_MODEL_STR"
RECEIVE_UNITY_MODEL_STR = "send_python_tcp_model_str"
RECEIVE_UNITY_SELECT_MODEL_STR = "send_python_tcp_select_model_str"

RECEIVE_UNITY_CALIBRATION_START_STR = "send_python_tcp_calibration_start"
SENT_UNITY_CALIBRATION_DONE_STR = "SENT_UNITY_CALIBRATION_DONE_STR"
CALIBRATION_FINISH_STR = "calibration done"


def getRunDataDir():
    """依據當前 global_value.runCount 回傳對應 run 資料夾路徑"""
    try:
        import main.Utils.global_value as global_value
        run_count = global_value.runCount
    except Exception:
        run_count = 0
    runDir = os.path.join(__REALTIME_BASE_FILE, f"run{run_count}")
    os.makedirs(runDir, exist_ok=True)
    return runDir


def getRunCsvFilename():
    return os.path.join(getRunDataDir(), "eeg_record.csv")


def getRunLogFilename():
    return os.path.join(getRunDataDir(), "log.txt")


def getRunPtFilename():
    return os.path.join(getRunDataDir(), "data.pt")


# --- 10. 列舉型別 ---
class GameSTATE(Enum):
    Calibration = "Calibration"
    BeatSaber = "BeatSaber"
    MI = "MI"
    LOBBY = "Lobby"
    TRAIN = "Training"


class MIClass(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class TAGS(Enum):
    INFO = "[INFO]"
    MARKER = "[MARKER]"
    WARNING = "[WARNING]"
    ERROR = "[ERROR]"
