import time
import main.Utils.config as config
import main.Utils.global_value as global_value
from main.Utils.UnityMarkerReader import UnityLSLReader, UnityTCPReader
from main.EEG.CygnusEEGReader import EEGReader
from main.EEG.EEG_Train import EEGFineTunePipeline, EEGTrainingPipeline, EEGSelfDataLoader
from main.EEG.EEGPrediction import EEGPredictor
from abc import ABC, abstractmethod
# import main.Utils.LSL as LSL
from main.Utils.TCPServer import TCPServer
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
        game.save_data_as_pt()

        game.eeg_predictor.end_predict_eeg_thread()
        game.eeg_reader.end_read_eeg_thread()
        game.unity_reader.stop_and_save_log()

        print(f"\n{config.TAGS.INFO.value} 切換 Training")
        game.change_state(Train())

    def go_to_lobby(self, game):
        # game.print_and_record_csv_log_table(self.name)
        game.save_data_as_pt()

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
            global_value.update_model = False
        if global_value.unity_marker_string_stage == config.GameSTATE.Calibration.value:
            game.change_state(CalibrationState())
        elif global_value.unity_marker_string_stage == config.GameSTATE.BeatSaber.value:
            game.change_state(BeatSaberState())
        elif global_value.unity_marker_string_stage == config.GameSTATE.MI.value:
            game.change_state(MIState())


class Train(GameState):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.TRAIN.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 開始 Training ")
        if game.pre_state == config.GameSTATE.Calibration.value:
            game.ft_pipeline.run_calibration()  # 這個不是 thread，所以需要卡一下 unity 那邊，unity 那邊自己卡
        if game.pre_state == config.GameSTATE.MI.value:
            game.train_pipeline.run_training()

    def update(self, game):  # 需要加入一些文字 list
        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            print(f"\n{config.TAGS.INFO.value} 現在返回 Lobby。")
            game.change_state(LobbyState())


class CalibrationState(GameState):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return config.GameSTATE.Calibration.value

    def start(self, game):
        print(f"\n{config.TAGS.INFO.value} 你進入了 CalibrationState")

    def update(self, game):  # 需要加入一些文字 list
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

    def update(self, game):
        if global_value.unity_marker_string_stage == config.GameSTATE.TRAIN.value:
            self.go_to_training(game)
        if global_value.unity_marker_string_stage == config.GameSTATE.LOBBY.value:
            self.go_to_lobby(game)


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
        if count > 20:  # 至少 20 行才給進去
            data = EEGSelfDataLoader(
                file_paths=[self.eeg_reader.filename],  # cygnus csv # only one
                log_paths=[self.unity_reader.filename],  # unity log # only one
                channel_index=config.channel_index
            )
            data.save_as_pt()

    def set_global_model_name(self):  # 為了把資料夾下面的檔案內容傳送到 unity，所以需要抓取內容然後放入到 global 變數
        folder_path = config.EEG_CHECKPOINT_MAIN_BASE_FILE
        # 定義黑名單
        # exclude_files = {"main_model.pth", "c_000.pth"}
        exclude_files = {"main_model.pth"}

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
