import time
import main.Utils.config as config
import main.Utils.global_value as global_value
from main.Utils.UnityMarkerReader import UnityTCPReader  # UnityLSLReader
from main.EEG.CygnusEEGReader import EEGReader
from main.EEG.EEG_Train import EEGFineTunePipeline, EEGTrainingPipeline, EEGSelfDataLoader
from main.EEG.EEGPrediction import EEGPredictor
from abc import ABC, abstractmethod
# import main.Utils.LSL as LSL
from main.Utils.TCPServer import TCPServer
from main.Utils.file_pointer_reader import readCsvFromPointer, readLogFromPointer, extractRecordEpoch
from main.EEG.data_process_np import EEGDataLoader
from main.EEG.EEG_Train import arrange_by_label
from main.Utils.some_functions import rename_file_with_time
import torch
import numpy as np
import os


class GameState(ABC):
    """所有遊戲狀態的基類"""

    def __init__(self):  # def __init__(self, name: str)
        pass

    @property
    @abstractmethod  # 強制子類要實作，這個等同於設定 self.name，比上面那種還清楚
    def name(self):
        pass

    def start(self, game):
        """進入 state 開始動作"""
        pass

    def update(self, game):
        """此方法用來檢查 global_value.unity_marker_string 並決定是否要進行狀態變更"""
        pass

    def go_to_training(self, game):
        # game.print_and_record_csv_log_table(self.name)
        game.saveDataAsPtFromPointer()  # game.save_data_as_pt()

        # 註解下面，因為後續加入睜眼閉眼任務各 10 秒，然後同步訓練任務，這邊註解，訓練需要加入 go_to_lobby
        game.eeg_predictor.end_predict_eeg_thread()
        # game.eeg_reader.end_read_eeg_thread()
        # game.unity_reader.stop_and_save_log()

        print(f"\n{config.TAGS.INFO.value} 切換 Training")
        game.change_state(Train())

    def go_to_lobby(self, game, stop_predict=True):
        # game.print_and_record_csv_log_table(self.name)
        if stop_predict:
            game.saveDataAsPtFromPointer()  # game.save_data_as_pt()
            game.eeg_predictor.end_predict_eeg_thread()

        game.eeg_reader.end_read_eeg_thread()
        game.unity_reader.stop_and_save_log()

        print(f"\n{config.TAGS.INFO.value} 現在返回 Lobby。")
        game.change_state(LobbyState())


class LobbyState(GameState):
    def __init__(self):
        super().__init__()  # super().__init__("Lobby")

    @property
    def name(self):
        return config.GameSTATE.LOBBY.value

    def start(self, game):
        global_value.runCount += 1
        print(f"\n{config.TAGS.INFO.value} [DEBUG] 目前 Run 編號: {global_value.runCount}")
        game.resetFilePointers()  # 新 run 重設 pointer
        game.eeg_predictor.update_check_point(global_value.NOW_TRAINED_CHECKPOINT)  # default c_000
        game.eeg_predictor.update_model()
        game.eeg_reader.start_read_eeg_thread()
        game.eeg_predictor.start_predict_eeg_thread()
        game.unity_reader.start_write_log()
        print(f"\n{config.TAGS.INFO.value} 歡迎來到遊戲大廳！")

    def update(self, game):
        # select model
        if global_value.update_model:
            game.eeg_predictor.update_check_point(global_value.unity_update_model_str)
            game.eeg_predictor.update_model()
            global_value.NOW_TRAINED_CHECKPOINT = global_value.unity_update_model_str
            global_value.update_model = False
        if global_value.unity_marker_string_stage == config.GameSTATE.Calibration.value:
            game.change_state(CalibrationState())
        elif global_value.unity_marker_string_stage == config.GameSTATE.BeatSaber.value:
            game.change_state(BeatSaberState())
        elif global_value.unity_marker_string_stage == config.GameSTATE.MI.value:
            game.change_state(MIState())


class Train(GameState):  # 目前沒用
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.TRAIN.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 開始 Training ")
        if game.pre_state == config.GameSTATE.Calibration.value:
            # game.ft_pipeline.run_calibration()  # 這個不是 thread，所以需要卡一下 unity 那邊，unity 那邊自己卡
            game.train_pipeline.run_training()  # 20260207 新的邏輯與 training 一樣
        if game.pre_state == config.GameSTATE.MI.value:
            game.train_pipeline.run_training()

    def update(self, game):  # 需要加入一些文字 list
        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            print(f"\n{config.TAGS.INFO.value} 現在返回 Lobby。")
            self.go_to_lobby(game, stop_predict=False)  # 因為已經停止訓練了


class CalibrationState(GameState):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.Calibration.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 你進入了 CalibrationState")
        game.ft_pipeline.init_pipeline()

    def update(self, game):  # 需要加入一些文字 list
        # 20260226 改用 pointer-based 讀取，不再關閉/重建檔案，檔案只在回到 Lobby 時才關閉
        if global_value.unity_marker_string_calibration == config.RECEIVE_UNITY_CALIBRATION_START_STR:
            print(f"\n{config.TAGS.INFO.value} 開始 online Calibration (pointer-based)")
            global_value.unity_marker_string_calibration = ""  # 清空字串

            # 使用 pointer-based 讀取，只讀取新增的 CSV/LOG 資料
            game.saveDataAsPtFromPointer()

            game.ft_pipeline.run_calibration()

            # game.eeg_predictor.end_predict_eeg_thread()
            game.eeg_predictor.update_check_point(global_value.NOW_TRAINED_CHECKPOINT)  # default c_000
            game.eeg_predictor.update_model()

        if global_value.unity_marker_string_stage == config.GameSTATE.TRAIN.value:
            self.go_to_training(game)

        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            self.go_to_lobby(game)


class BeatSaberState(GameState):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.BeatSaber.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 你進入了 BeatSaberState")

    def update(self, game):
        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            self.go_to_lobby(game)


class MIState(GameState):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.MI.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 你進入了 MIState")
        game.ft_pipeline.init_pipeline()

    def update(self, game):
        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            self.go_to_lobby(game)
        if global_value.unity_marker_string_stage == config.GameSTATE.TRAIN.value:  # 訓練的部分改掉，改成下面這種方式， trial 式的訓練
            self.go_to_training(game)
        if global_value.unity_marker_string_calibration == config.RECEIVE_UNITY_CALIBRATION_START_STR:
            print(f"\n{config.TAGS.INFO.value} 開始 online Calibration (pointer-based)")
            global_value.unity_marker_string_calibration = ""  # 清空字串

            # 使用 pointer-based 讀取，只讀取新增的 CSV/LOG 資料
            game.saveDataAsPtFromPointer()

            game.ft_pipeline.run_calibration()

            # game.eeg_predictor.end_predict_eeg_thread()
            game.eeg_predictor.update_check_point(global_value.NOW_TRAINED_CHECKPOINT)  # default c_000
            game.eeg_predictor.update_model()


class Game:
    def __init__(self):
        # lsl_train_outlet = LSL.setup_lsl_outlet(config.TO_UNITY_TRAIN_LSL_STREAM, stream_type="string")  # log
        # lsl_predict_outlet = LSL.setup_lsl_outlet(config.TO_UNITY_LSL_STREAM)  # stream type = int
        self.eeg_reader = EEGReader()  # cygnus use lsl
        self.unity_reader = UnityTCPReader()  # log use TCP
        # setup the message event and TCP Server
        tcp_server = TCPServer(host=config.TCP_HOST, port=config.TCP_PORT, on_message=self.unity_reader.process_message)
        tcp_server.start()

        self.unity_reader.setup_tcp_server(tcp_server)

        self.eeg_predictor = EEGPredictor(tcp_server=tcp_server)  # csv

        self.ft_pipeline = EEGFineTunePipeline(tcp_server=tcp_server)
        self.train_pipeline = EEGTrainingPipeline(tcp_server=tcp_server)
        self.pre_state = ""
        # ---- Pointer-based 檔案讀取指標 ----
        self.csvLinePointer = 0  # CSV 資料行已讀行數（不含 header）
        self.logLinePointer = 0  # LOG 已讀行數
        self.set_global_model_name()
        self.state = LobbyState()

    def change_state(self, new_state):
        self.set_global_model_name()
        self.pre_state = self.state.name
        self.state = new_state
        self.state.start(self)

    def start(self):
        self.state.start(self)
        while True:
            self.state.update(self)  # 檢查是否要更新狀態
            time.sleep(1)  # update 更新頻率

    def print_and_record_csv_log_table(self, name):  # 紀錄對應 csv 和 log 的資料  (csv, log)
        global_value.data_lookup_table[name].append(
            (self.eeg_reader.filename, self.unity_reader.filename))
        print(f"\n{config.TAGS.INFO.value} lookup tables: ", end="")
        for k, v in global_value.data_lookup_table.items():
            print(f"{k}: {v}", end="; ")
        print()

    def save_data_as_pt(self):  # save data to np，把  csv 和 log 轉 讀取比較快的 Data，並存在 global value train_np_data 裡面
        count = 0
        with open(self.unity_reader.filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                count += 1
        if count > 7:  # 至少 7 行才給進去
            data = EEGSelfDataLoader(
                file_paths=[self.eeg_reader.filename],  # cygnus csv # only one
                log_paths=[self.unity_reader.filename],  # unity log # only one
                channel_index=config.channel_index
            )
            data.save_as_pt()

    def resetFilePointers(self):
        """重設 CSV/LOG pointer 為 0，在進入新 run (Lobby) 時呼叫"""
        self.csvLinePointer = 0
        self.logLinePointer = 0
        print(f"{config.TAGS.INFO.value} [DEBUG] resetFilePointers: pointer 已重設為 0")

    def saveDataAsPtFromPointer(self):
        """
        使用 pointer-based 方式讀取 CSV/LOG 的新增資料，
        儲存為 .pt 檔案。pointer 會在讀取後更新，
        下次只讀取上次之後的資料。
        """
        csvPath = self.eeg_reader.filename
        logPath = self.unity_reader.filename

        print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
              f"csvPointer={self.csvLinePointer}, logPointer={self.logLinePointer}")

        # 1. Flush 確保 buffer 已寫入磁碟
        self.eeg_reader.flushCsv()
        self.unity_reader.flushLog()

        # 2. 使用 pointer 讀取 LOG 新資料，檢查是否足夠
        trials, newLogPointer, newLogLineCount = readLogFromPointer(
            logPath, self.logLinePointer
        )
        if newLogLineCount < 7:
            print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
                  f"新 LOG 行數 {newLogLineCount} < 7，略過儲存")
            return

        # 3. 使用 pointer 讀取 CSV 新資料
        timestamps, eeg, newCsvPointer = readCsvFromPointer(
            csvPath, config.channel_index, self.csvLinePointer
        )
        if timestamps.size == 0:
            print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
                  f"無新的 CSV 資料，略過儲存")
            return

        # 4. 提取 record epoch（從 CSV header，每次都讀取）
        recEpoch = extractRecordEpoch(csvPath)
        # [DEBUG] 印出用於 searchsorted 的關鍵數值
        # print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: recEpoch={recEpoch}")
        # print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
        #       f"timestamps range=[{timestamps[0]:.3f} ~ {timestamps[-1]:.3f}], "
        #       f"len={len(timestamps)}")
        # for trialIdx in sorted(trials):
        #     t = trials[trialIdx]
        #     if 'start' in t and 'end' in t:
        #         startRel = t['start'] - recEpoch
        #         endRel = t['end'] - recEpoch
        #         print(f"{config.TAGS.INFO.value} [DEBUG] Trial {trialIdx}: "
        #               f"start_abs={t['start']:.3f}, end_abs={t['end']:.3f}, "
        #               f"start_rel={startRel:.3f}, end_rel={endRel:.3f}")
        # 5. 利用 EEGDataLoader 提取 segments
        loader = EEGDataLoader(
            file_paths=[], log_paths=[],
            channel_index=config.channel_index
        )
        loader.rec_epoch = recEpoch
        loader._extract_segments(eeg, timestamps, trials)

        if not loader.segments:
            print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
                  f"無有效 segments，略過儲存")
            return

        # 6. 轉換格式並存為 .pt
        xData, yData, failuresData = loader.get_eeg_trial_channel_sample_np()
        xData, yData, failuresData = arrange_by_label(xData, yData, failuresData)

        dataName = rename_file_with_time(config.getRunPtFilename())
        global_value.train_np_data.append(dataName)
        torch.save({'x_data': xData, 'y_data': yData, 'failures': failuresData}, dataName)

        print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
              f"已儲存 {dataName} (segments={len(loader.segments)})")

        # 7. 更新 pointer
        # LOG pointer: 直接推進到讀取結束位置
        self.logLinePointer = newLogPointer
        # CSV pointer: 基於最後一個 trial 的 end 時間來決定下次讀取起點
        #   這樣可以確保下次的新 trial 的 start_rel 一定落在 timestamps 範圍內
        latestEndRel = max(
            t['end'] - recEpoch
            for t in trials.values()
            if 'end' in t
        )
        safetyMarginSeconds = 0.0  # 往前多讀 N 秒，避免時間精度問題
        safeEndRel = latestEndRel - safetyMarginSeconds
        csvIndexForEnd = int(np.searchsorted(timestamps, safeEndRel, side='left'))
        self.csvLinePointer = self.csvLinePointer + csvIndexForEnd
        print(f"{config.TAGS.INFO.value} [DEBUG] saveDataAsPtFromPointer: "
              f"latestEndRel={latestEndRel:.3f}, safeEndRel={safeEndRel:.3f}, "
              f"更新 pointer: csv={self.csvLinePointer}, log={self.logLinePointer}")

    def set_global_model_name(self):  # 為了把資料夾下面的檔案內容傳送到 unity，所以需要抓取內容然後放入到 global 變數
        folder_path = config.EEG_CHECKPOINT_MAIN_BASE_FILE
        # 定義黑名單
        # exclude_files = {"main_model.pth", "c_000.pth"}
        exclude_files = {"model.pth"}

        global_value.models_name = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f not in exclude_files
        ]
        print(global_value.models_name)


if __name__ == "__main__":
    game = Game()


    # 模擬外部字串更新的行為
    def simulate_external_input():
        inputs = ['stage1', 'correct', 'stage2', 'yes', 'stage3', 'ready', 'invalid']
        for i in inputs:
            time.sleep(3)  # 每隔 3 秒模擬一次外部輸入
            global_value.unity_marker_string_stage = i
            print(f"\n{config.TAGS.INFO.value} 外部輸入: {global_value.unity_marker_string_stage}")


    # 啟動模擬輸入
    import threading

    input_thread = threading.Thread(target=simulate_external_input)
    input_thread.daemon = True
    input_thread.start()

    game.start()
