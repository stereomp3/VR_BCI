"""
20250828 測試使用 Conformer 的 checkpoint，sample rate 125，band pass 1~40，channel 15 個 [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 27, 28, 29]
後續會製作把 EEG BUFFER 裡面的東西存成檔案，或是直接把他變成 np (透過 np 和 string list 去切割，內容會和 data process np 差不多，用那個改)，然後丟到模型訓練，訓練完成後再預測，把它跟 unity 做連動，寫在另外一份文件裡面。
訓練部分，可以使用 20250922 的內容訓練 (update)
20250924 加入 demean np.mean(data, axis=1, keepdims=True)
記得要把 is_simulated 開成 False 測試 !!!
"""
import numpy as np
import torch
import threading
import time
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from scipy.signal import butter, filtfilt, decimate
from pylsl import StreamInfo, StreamOutlet
from braindecode.models import ShallowFBCSPNet
from main.EEG.models import SCCNet
from main.EEG.MI_train import load_shallowfbcsp_params, load_sccnet_params
from torch.utils.data import TensorDataset
import main.Utils.config as config

# 這份文件需要先開，unity 才能開
# sample = 125  # 根據 windows size # 會進行 down sample 的數值
sample = 500  # 根據 windows size # 會進行 down sample 的數值
# ======== Globals for Threading ============
SAMPLE_RATE = 500  # Hz (adjust this based on your Cygnus device)
BUFFER_SIZE = SAMPLE_RATE  # samples needed per prediction # 這邊可以調成跟訓練一樣的 sample
CHANNEL_COUNT = 13  # number of EEG channels # max 22 # 需要改下面 read_eeg channnel

PREDICTION_INTERVAL = 0.05  # seconds between predictions
N_Class = 2
eeg_buffer = np.zeros((CHANNEL_COUNT, 0))  # initialize empty buffer
band_pass_low = 1
band_pass_high = 40
is_simulated = False # 這個設定為 True 就不會連接 LSL，而是使用 np rand 產生 data

# main 裡面的 model_arg 需要對應模型更改 # channel(CHANNEL_COUNT) 也要對應更改
# 20250914 使用 ShallowFBCSPNet 出現 26 個 float element，不是輸出 0 或是 1 不知為啥 ...，20250922 因為沒有 downsample..
# 最後一層出錯就是因為 sample 設錯導致
use_model = ShallowFBCSPNet  # ShallowFBCSPNet # 使用 SCCNet 下面在 prediction 的時候，需要把維度改成 4 維

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_class, model_kwargs):  # 這邊需要根據需求調整 epoch
    # ======== Model Setup ============
    model = model_class(**model_kwargs).to(device)  # n channel, sample, n class
    # conformer model: ft-epoch1_eegconfomer_test, train-epoch1_eegconfomer_test
    # ShallowNet: ft-epoch1_shollownet_test, train-epoch1_shollownet_test
    # checkpoint = torch.load("checkpoints/20251111_13c_shollownet_g_all_train-epoch650.pth", map_location=device)
    checkpoint = torch.load("EEG/checkpoint_main/13c_s_g_all.pth", map_location=device)
    # checkpoint = torch.load("best_model_Train.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print(checkpoint.keys())
    # for k, v in checkpoint['model_state_dict'].items():
    #     print(k, v.shape)
    return model


def to_dataset(segment_list, label_list):
    data_x = np.array(segment_list)  # (T, C, N) # (trial, channel, sample)
    X = torch.tensor(data_x).unsqueeze(1)  # (T, 1, C, N)
    y = torch.tensor(label_list).long()
    print(X.shape)
    print(y.shape)
    return TensorDataset(X, y)


# ======== LSL Inlet Setup ============
def setup_lsl_inlet(stream_name="Cygnus-329018-RawEEG"):
    # Option 1: get all streams
    streams = resolve_streams()
    for stream in streams:
        print(stream.name())
    print("Resolving EEG stream...")
    streams = resolve_byprop('name', stream_name)
    inlet = StreamInlet(streams[0])
    print("Stream resolved.")
    return inlet


# ======== EEG Reading Thread ============
def read_eeg(inlet):
    global eeg_buffer
    end = time.time()
    while True:
        sample, _ = inlet.pull_sample()
        # sample = np.array(sample[:CHANNEL_COUNT]).reshape(-1, 1)  # shape: (CHANNEL_COUNT,1)
        # print(f"sample[0:2] + sample[4:6]: {sample[0:2] + sample[4:6]}")
        # print(f"sample...: {sample[0:4] + sample[5:8] + sample[10:13] + sample[15:18] + sample[20:23] + sample[25:28] + sample[29:32]}")
        # Fp1,Fp2,AF3,AF4,F7(6),F3,Fz,F4,F8,FT7(11),FC3,FCz,FC4,FT8,T7(16),C3,Cz,C4,T8,TP7(21),CP3,CPz,CP4,TP8,P7(26),P3,Pz,P4,P8,O1(31),Oz,O2(33)
        # channel_index = [2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 31, 32, 33]
        # sample = np.array(sample[15:18]).reshape(-1, 1)  # shape: (3,1) # 3 # 15 16 17
        # sample = np.array(sample[0:4] + sample[5:8] + sample[10:13] +
        #                   sample[15:18] + sample[20:23] + sample[25:28] + sample[29:32]).reshape(-1, 1)  # shape: (22,1)
        # sample = np.array(sample[15:18]).reshape(-1, 1)  # shape: (3,1) # 3 # 15 16 17
        # sample = np.array(sample[5:8] + sample[10:13] + sample[15:18] +
        #                   sample[20:23] + sample[25:28] + sample[29:32]).reshape(-1,1)  # shape: (18,1) # 3 # 15 16 17
        # sample = np.array(sample[5:8] + sample[10:13] + sample[15:18] +
        #                   sample[20:23] + sample[25:28]).reshape(-1, 1)  # shape: (15,1) # 3 # 15 16 17
        sample = np.array(sample[5:8] + sample[10:13] + sample[15:18] +
                          sample[20:23] + sample[26:27]).reshape(-1, 1)  # shape: (13,1) # 3 # 15 16 17
        eeg_buffer = np.hstack((eeg_buffer, sample))
        # start = time.time()
        # print(f"eeg_buffer.shape[1]: {eeg_buffer.shape[1]}")
        if eeg_buffer.shape[1] > BUFFER_SIZE:
            eeg_buffer = eeg_buffer[:, -BUFFER_SIZE:]  # keep last BUFFER_SIZE samples
            # print(f"last eeg_buffer data: {eeg_buffer[:, BUFFER_SIZE - 1]}")  # equal to sample[0:2] + sample[4:6]
            # print(f"eeg_buffer read time: {start-end}")
        # time.sleep(1.0 / SAMPLE_RATE)


def simulated_read_eeg():
    global eeg_buffer
    end = time.time()
    while True:
        # sample = np.random.rand(32, 1)  # shape: (32, 1)
        sample = np.random.rand(CHANNEL_COUNT, 1)  # shape: (CHANNEL_COUNT, 1)
        eeg_buffer = np.hstack((eeg_buffer, sample))
        # start = time.time()
        # print(f"eeg_buffer.shape[1]: {eeg_buffer.shape[1]}")
        if eeg_buffer.shape[1] > BUFFER_SIZE:
            eeg_buffer = eeg_buffer[:, -BUFFER_SIZE:]  # keep last BUFFER_SIZE samples
            # print(f"last eeg_buffer data: {eeg_buffer[:, BUFFER_SIZE - 1]}")  # equal to sample[0:2] + sample[4:6]
            # print(f"eeg_buffer read time: {start-end}")
        # time.sleep(1.0 / SAMPLE_RATE)


def bandpass(data, fs=SAMPLE_RATE, low=band_pass_low, high=band_pass_high):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def down_sample(data, new_fs=sample):  # data: (n_samples, n_channels)
    decimation_factor = SAMPLE_RATE // new_fs  # 500/125 = 4
    return decimate(data, decimation_factor, axis=-1, zero_phase=True)


# ======== Prediction Thread ============
def predict_loop(model):
    info = StreamInfo(name=config.TO_UNITY_LSL_STREAM, type='Markers', channel_count=1,
                      nominal_srate=10, channel_format='int32', source_id='user_input_1234')
    outlet = StreamOutlet(info)

    global eeg_buffer
    is_start = False
    count = 0
    while True:
        if eeg_buffer.shape[1] >= BUFFER_SIZE:
            # shape: (channel, BUFFER_SIZE) -> (channel, sample)
            # input_data = down_sample(bandpass(eeg_buffer[:, -BUFFER_SIZE:].copy()))
            data = bandpass(eeg_buffer[:, -BUFFER_SIZE:].copy())
            input_data = data - np.mean(data, axis=0, keepdims=True)
            # input_data = bandpass(eeg_buffer[:, -BUFFER_SIZE:].copy())
            # print(input_data.shape)
            # print(input_data)
            # for sccnet (1,1,4,1249)
            # x_batch = torch.from_numpy(input_data.copy()).float().unsqueeze(0).unsqueeze(0).to(device)  # 4 維
            x_batch = torch.from_numpy(input_data.copy()).float().unsqueeze(0).to(device)
            # print(f"shape: {x_batch.shape}") # (1,15,125) (trial, channel, sample)
            with torch.no_grad():
                # if not is_start:
                #     start = time.time()
                #     is_start = True
                # count += 1
                # if count == 100:
                #     end = time.time()
                #     print(f"Time: {end - start}")
                #     is_start = False
                #     count = 0
                # model prediction
                outputs = model(x_batch)
                # print(outputs)
                prediction = int(torch.argmax(outputs).cpu().item())
                print(f"Prediction: {prediction}")
                # print(f"outputs: {outputs}")
                outlet.push_sample([prediction])

        time.sleep(PREDICTION_INTERVAL)


# ======== Main ============
def main():
    simulate_x = np.zeros((1, CHANNEL_COUNT, sample))  # (sample, channel, trial)
    simulate_y = np.zeros((1, N_Class))  # (trail, n class)
    dataset = to_dataset(simulate_x, simulate_y)  # 這邊是為了要對應 load_conformer_params 才使用這個 (懶得改 XD
    model_arg = load_shallowfbcsp_params(dataset)  # load_conformer_params, load_shallowfbcsp_params, load_sccnet_params
    model = init_model(use_model, model_arg)

    if is_simulated:
        threading.Thread(target=simulated_read_eeg, daemon=True).start()
    else:
        inlet = setup_lsl_inlet()
        thread1 = threading.Thread(target=read_eeg, args=(inlet,), daemon=True)
        thread1.start()

    predict_loop(model)


if __name__ == "__main__":
    main()
