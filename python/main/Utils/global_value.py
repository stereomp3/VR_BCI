"""
會變動的變數存放位置，global
"""
import numpy as np
import main.Utils.config as config
import threading

unity_marker_string_stage = config.GameSTATE.LOBBY.value  # unity 傳送 marker 到這個變數，儲存 config GameSTATE 對應的 str，一開始再 Lobby
unity_marker_string_log = ""  # unity 傳送 marker 到這個變數，儲存 Start End CUT 的時間點，在 Utils UnityMarkerReader 改，game state 使用
unity_marker_string_change_model = "20251101_13c_shollownet_g_train-epoch10"  # 一樣在 Utils UnityMarkerReader，預設為標準模型
# ----------------------------
# Global buffer & concurrency
# ----------------------------
eeg_buffer = np.empty((config.N_CHANNELS, 500), dtype=float)  # 用在 CygnusEEGReader 和 EEGPrediction
data = np.empty((config.N_CHANNELS, 1), dtype=float)  # 用在 CygnusEEGReader 和 EEGPrediction
ts = 0  # 用在 CygnusEEGReader 和 EEGPrediction timestamp
buffer_lock = threading.Lock()  # 使用 lock 保護共享資源 eeg_buffer

data_lookup_table = {  # 暫時不使用
    config.GameSTATE.LOBBY.value: [],  # save tuple in list, (csv, log)
    config.GameSTATE.MI.value: [],
    config.GameSTATE.Calibration.value: [],  # use in EEG_Calibration.py
    config.GameSTATE.BeatSaber.value: [],
}

train_np_data = []  # 取代 data_lookup_table
# train_np_data = [
#     "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\data_20251111_041320.pt",
#     "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\data_20251111_041814.pt",
#     "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\data_20251111_045219.pt",]

# model_trained.pth 會用在最開始的模型，model.pth 用於當作 base model ft 的模型
models_name = []  # 顯示在 unity 的模型 # 會讀取 config.EEG_CHECKPOINT_MAIN_BASE_FILE 底下所有模型名稱，然後在切換場景重新載入 (game_state)
update_model = False  # 在 unity 傳輸字串後，會更新這個 (UnityMarkerReader.py)，然後在 game_state.py 使用，使用過後還原成 False (update model)
unity_update_model_str = None  # 跟上面的那個為一組的，都從 unity 那邊接收，接收後，根據模型名稱替換模型
NOW_TRAINED_CHECKPOINT = f"{config.EEG_CHECKPOINT_MAIN_BASE_FILE}c_000.pth"
