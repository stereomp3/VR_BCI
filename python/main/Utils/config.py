"""
不會變動的變數存放位置，global
"""
from enum import Enum

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
BASE_FILE = "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\"

__REALTIME_BASE_FILE = "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\"
CSV_FILENAME = f"{__REALTIME_BASE_FILE}eeg_record.csv"
LOG_FILENAME = f"{__REALTIME_BASE_FILE}log.txt"
PT_DATA_FILENAME = f"{__REALTIME_BASE_FILE}data.pt"

__EEG_DATA_BASE_FILE = "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\EEG\\val_data\\"

FT_CSV_FILENAME = f"{__EEG_DATA_BASE_FILE}7_8_9_10_11_12.csv"  # 暫時設定，用於測試
FT_LOG_FILENAME = f"{__EEG_DATA_BASE_FILE}7.txt"  # 暫時設定，用於測試

EEG_CHECKPOINT_MAIN_BASE_FILE = "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\EEG\\checkpoint_main\\"
EEG_CHECKPOINT_TMP_BASE_FILE = "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\EEG\\checkpoints\\"

MAIN_CHECKPOINT = f"{EEG_CHECKPOINT_MAIN_BASE_FILE}model.pth"  # 用於 calibration 的模型 # 用在 EEG_Train.py
TRAINED_CHECKPOINT = f"{EEG_CHECKPOINT_MAIN_BASE_FILE}model_trained.pth"  # 訓練過後的模型 # 改到 global # 這個目前沒有使用了

TRAINING_FINISH_STR = "training done"  # 用於 MI_train.py
SENT_UNITY_MODEL_STR = "SENT_UNITY_MODEL_STR"  # 用於 UnityMarkerReader.py，get model list
RECEIVE_UNITY_MODEL_STR = "send_python_tcp_model_str"   # 用於 UnityMarkerReader.py，get model list
RECEIVE_UNITY_SELECT_MODEL_STR = "send_python_tcp_select_model_str"   # 用於 UnityMarkerReader.py，接收切換模型
SEPARATE_STR = "@@@"  # 用於分開字串的符號，python unity 都使用這個


class GameSTATE(Enum):
    Calibration = "EEG_Calibration"
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
