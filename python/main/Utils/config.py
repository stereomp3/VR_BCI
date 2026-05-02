"""
不會變動的變數存放位置，global
"""
from enum import Enum
import os

RECEIVE_CYGNUS_LSL_STREAM = "Cygnus-329018-RawEEG"
# RECEIVE_UNITY_LSL_STREAM = "UnityMarkerStream"
TO_UNITY_LSL_STREAM = "MarkerStream"
# TO_UNITY_TRAIN_LSL_STREAM = "Train_MarkerStream"
TCP_PORT = 50007
TCP_HOST = "0.0.0.0"  # all
SAMPLE_RATE = 500
# 主要針對資料讀取的 channel，但是在 CygnusEEGReader.py read_eeg 裡面要對應 channel 做額外處裡
# channel_index = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 27, 28, 29] # 15
channel_index = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 28]  # 13
N_CHANNELS = 13  # use n channel_index to train, prediction and read buffer data (CygnusEEGReader.py)
EEG_CHANNELS = 32  # use 22 channel, (32 channel eeg cap
N_Class = 2  # left 1, right 0
WINDOW_SECONDS = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
PREDICTION_INTERVAL = 0.01  # seconds between predictions # 目前暫時沒用 # EEG_Prediction.py 裡面 # 實際會比這個慢，大概 1 s 70 個
band_pass_low = 1
band_pass_high = 40

is_simulated_unity = False  # use in EEG_Calibration, UnityMarkerReader # 改成 TCP 後用到較少 # 目前不要動
is_simulated_eeg = True  # use in CygnusEEGReader True, 真正測試要改成 False

SAVE_CSV = True
# --- 路徑設定開始 (使用 os 自動偵測) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.dirname(_current_dir) # 取得上一層 main 資料夾

__REALTIME_BASE_FILE = os.path.join(BASE_FILE, "real_time_data")
CSV_FILENAME = os.path.join(__REALTIME_BASE_FILE, "eeg_record.csv")
LOG_FILENAME = os.path.join(__REALTIME_BASE_FILE, "log.txt")
PT_DATA_FILENAME = os.path.join(__REALTIME_BASE_FILE, "data.pt")


# ---- Run 子資料夾相關函式 ----
def getRunDataDir():
    """根據目前 global_value.runCount 回傳對應的 run 資料夾路徑，並自動建立資料夾"""
    import main.Utils.global_value as global_value
    runDir = os.path.join(__REALTIME_BASE_FILE, f"run{global_value.runCount}")
    os.makedirs(runDir, exist_ok=True)
    print(f"[DEBUG] getRunDataDir: {runDir}")
    return runDir + os.sep


def getRunCsvFilename():
    """回傳目前 run 資料夾下的 CSV 檔案路徑"""
    return f"{getRunDataDir()}eeg_record.csv"


def getRunLogFilename():
    """回傳目前 run 資料夾下的 LOG (TXT) 檔案路徑"""
    return f"{getRunDataDir()}log.txt"


def getRunPtFilename():
    """回傳目前 run 資料夾下的 PT 檔案路徑"""
    return f"{getRunDataDir()}data.pt"

EEG_CHECKPOINT_MAIN_BASE_FILE = os.path.join(BASE_FILE, "EEG", "checkpoint_main\\")
EEG_CHECKPOINT_TMP_BASE_FILE = os.path.join(BASE_FILE, "EEG", "checkpoints\\")

MAIN_CHECKPOINT = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, "model.pth")  # 用於 calibration 的模型 # 用在 EEG_Train.py
TRAINED_CHECKPOINT = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, "model_trained.pth")  # 訓練過後的模型 # 改到 global # 這個目前沒有使用了

TRAINING_FINISH_STR = "training done"  # 用於 MI_train.py
SENT_UNITY_MODEL_STR = "SENT_UNITY_MODEL_STR"  # 用於 UnityMarkerReader.py，get model list
RECEIVE_UNITY_MODEL_STR = "send_python_tcp_model_str"  # 用於 UnityMarkerReader.py，get model list
RECEIVE_UNITY_SELECT_MODEL_STR = "send_python_tcp_select_model_str"  # 用於 UnityMarkerReader.py，接收切換模型
SEPARATE_STR = "@@@"  # 用於分開字串的符號，python unity 都使用這個

# 用於 Calibration
RECEIVE_UNITY_CALIBRATION_START_STR = "send_python_tcp_calibration_start"
SENT_UNITY_CALIBRATION_DONE_STR = "SENT_UNITY_CALIBRATION_DONE_STR"
group_note_num = 5  # 一個 group 幾個 note，和 unity 那邊一樣
CALIBRATION_FINISH_STR = "calibration done"  # 用於 EEG_Train.py，MI_train.py 訓練完成後回到 EEG_Train 並觸發
REPLAY_BUFFER_LIMIT = 320  # 單一類別最大容量，20 trial × 16 windows = 320，兩類共 40 trial
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
