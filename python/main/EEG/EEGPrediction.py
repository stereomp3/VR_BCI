import time
import torch.nn as nn
import numpy as np
import torch
import threading
from braindecode.models import ShallowFBCSPNet
from MI_train import load_shallowfbcsp_params
from torch.utils.data import TensorDataset
import main.Utils.config as config
import main.Utils.preprocess as preprocess
import main.Utils.global_value as global_value


class EEGPredictor:  # 調整模型需要改 self.model，和 self.load_fun 方法
    def __init__(self, tcp_server):  # old: lsl_outlet=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.load_check_point_path = config.TRAINED_CHECKPOINT
        self.load_check_point_path = global_value.NOW_TRAINED_CHECKPOINT
        self.model = ShallowFBCSPNet
        self.load_fun = load_shallowfbcsp_params  # load shallow net parameter
        self.model_arg = self.set_model_arg()
        self.model = self.init_model()
        self.predict_eeg_stop_event = threading.Event()
        self.predict_eeg_thread = None

        self.is_updating = False  # 用來卡更新模型，更新模型不預測 # 使用 bool 判斷如果 unity 那邊按太快會出錯，所以有在 True 的時候會卡不能再次更新
        # self.outlet = lsl_outlet
        # TCP Server
        self.tcp_server = tcp_server

    def start_predict_eeg_thread(self):
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return  # 不要重啟

        print(f"{config.TAGS.INFO.value} Start predict loop...")
        self.predict_eeg_stop_event.clear()  # 重置 stop_event，允許重新啟動

        self.predict_eeg_thread = threading.Thread(target=self.predict_loop, daemon=True)
        self.predict_eeg_thread.start()

    def end_predict_eeg_thread(self):
        print(f"{config.TAGS.INFO.value} End predict loop...")
        self.predict_eeg_stop_event.set()  # 設置停止事件
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():
            self.predict_eeg_thread.join()
            self.predict_eeg_thread = None

    def init_model(self):
        # ======== Model Setup ============
        model = self.model(**self.model_arg).to(self.device)  # n channel, sample, n class
        checkpoint = torch.load(self.load_check_point_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def update_check_point(self, path):
        self.load_check_point_path = path

    def update_model(self):
        if self.is_updating:  # 防止一直進來，出現 crash 的問題
            return
        self.is_updating = True
        self.model = ShallowFBCSPNet
        self.model = self.init_model()
        self.is_updating = False

    def to_dataset(self, segment_list, label_list):
        data_x = np.array(segment_list)  # (T, C, N) # (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)  # (T, 1, C, N)
        y = torch.tensor(label_list).long()
        return TensorDataset(X, y)

    def set_model_arg(self):
        simulate_x = np.zeros((1, config.N_CHANNELS, config.SAMPLE_RATE))  # (sample, channel, trial)
        simulate_y = np.zeros((1, config.N_Class))  # (trail, n class)
        dataset = self.to_dataset(simulate_x, simulate_y)  # 這邊是為了要對應 load_conformer_params 才使用這個 (懶得改 XD
        return self.load_fun(dataset)  # load_shallowfbcsp_params

    def predict_loop(self):
        while not self.predict_eeg_stop_event.is_set():
            if global_value.eeg_buffer.shape[1] >= config.BUFFER_SIZE:
                if self.is_updating:
                    print(f"{config.TAGS.INFO.value} Waiting for model update ...")
                    time.sleep(0.5)
                    continue

                # shape: (channel, BUFFER_SIZE) -> (channel, sample)
                # input_data = preprocess.down_sample(preprocess.bandpass(eeg_buffer[:, -BUFFER_SIZE:].copy()))
                # indices = [5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 20, 21, 25, 26, 27]  # 需要選擇的 channel
                # data = global_value.eeg_buffer[indices, :].copy()  # shape (15, N)
                data = preprocess.bandpass(global_value.eeg_buffer[:, -config.BUFFER_SIZE:], axis=-1)  # shape (15, 500)
                input_data = data - np.mean(data, axis=0, keepdims=True)
                x_batch = torch.from_numpy(input_data.copy()).float().unsqueeze(0).to(self.device)
                # print(f"shape: {x_batch.shape}") # (1,15,500) (trial, channel, sample)
                with torch.no_grad():
                    if not self.is_updating:
                        outputs = self.model(x_batch)  # (2,) [-1, 2]
                    # print(outputs)
                    prediction = int(torch.argmax(outputs).cpu().item())
                    # print(f"Prediction: {prediction}")
                    # print(f"outputs: {outputs}")
                    # self.outlet.push_sample([prediction])
                    # 推送到 TCP
                    if self.tcp_server:
                        self.tcp_server.broadcast(f"{prediction}")
                    time.sleep(config.PREDICTION_INTERVAL)


# ======== Main ============
def main():
    pass
    # 建立 TCP Server
    # tcp_server = TCPServer(host="0.0.0.0", port=50007)
    # tcp_server.start()

    # 建立 EEGPredictor，傳入 TCPServer
    # predictor = EEGPredictor(tcp_server=tcp_server)
    # predictor.start_predict_eeg_thread()

    # 停止
    # predictor.end_predict_eeg_thread()
    # tcp_server.stop()
    # while True:
    #    pass


if __name__ == "__main__":
    main()
