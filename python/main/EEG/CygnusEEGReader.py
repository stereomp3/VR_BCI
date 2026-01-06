import time

import main.Utils.config as config
import main.Utils.global_value as global_value
import main.Utils.LSL as LSL
import csv
import threading
import numpy as np
from datetime import datetime
from main.Utils.some_functions import rename_file_with_time
import os


class EEGReader():
    def __init__(self):
        self.save_csv = config.SAVE_CSV
        self.filename = config.CSV_FILENAME
        self.is_simulated = config.is_simulated_eeg
        self.csv_writer = None
        self.csv_file = None
        self.read_eeg_stop_event = threading.Event()
        self.read_eeg_thread = None

    def start_read_eeg_thread(self):
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return  # 不要重啟

        print(f"{config.TAGS.INFO.value} Start read cygnus eeg...")
        self.filename = rename_file_with_time(config.CSV_FILENAME)
        self.read_eeg_stop_event.clear()  # 重置 stop_event，允許重新啟動

        self.read_eeg_thread = threading.Thread(target=self.read_eeg, daemon=True)  # 每次都重設定 thread
        self.read_eeg_thread.start()

    def end_read_eeg_thread(self):
        print(f"{config.TAGS.INFO.value} End read cygnus eeg...")
        self.read_eeg_stop_event.set()  # 設置停止事件
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            self.read_eeg_thread.join()
            self.read_eeg_thread = None

    def read_eeg(self):
        """
        Continuously pull samples from inlet and append to eeg_buffer.
        Optionally write each sample + timestamp + marker to CSV.
        """
        self.build_csv_file()
        first_ts_lsl, ts, ts_lsl = None, None, None
        count = 0.000
        inlet = None
        if not self.is_simulated:
            inlet = LSL.setup_lsl_inlet(config.RECEIVE_CYGNUS_LSL_STREAM)
        while not self.read_eeg_stop_event.is_set():
            if self.is_simulated:
                data = np.random.randint(-10001, 10001, size=(config.EEG_CHANNELS, 1)).reshape(-1)
                sample = np.random.randint(-10001, 10001, size=(config.N_CHANNELS, 1))  # shape: (15, 1)
                # global_value.eeg_buffer = np.hstack((global_value.eeg_buffer, sample))
                time.sleep(0.1 / config.SAMPLE_RATE)  # 模擬 sample rate 500, 0.002 穩一點所以 0.0002，讓他一定有資料讀取
            else:
                sample, ts = inlet.pull_sample(timeout=0.0)  # 可以在執行這個之前先清空 # 這裡拿到 sample 為 list
                if first_ts_lsl is None:
                    first_ts_lsl = ts
                if sample is None:
                    continue
                data = np.array(sample).reshape(-1)  # 不加入這個會顯示 [float]，而不是 float，用這個寫入文件
                # sample = np.array(sample[5:8] + sample[10:13] + sample[15:18] +
                #                   sample[20:23] + sample[25:28]).reshape(-1, 1)  # shape: (15,1) # 3 # 15 16 17
                sample = np.array(sample[5:8] + sample[10:13] + sample[15:18] +  # shape: (13,1)
                                  sample[20:23] + sample[26:27]).reshape(-1, 1)  # 3 # 15 16 17 拿掉 P3 P4
                # sample = np.array([sample[i - 2] for i in config.channel_index])  # 根據設定 channel 做讀取
                # sample = data.reshape(config.N_CHANNELS, 1)
                # arr = np.array(sample[0:4] + sample[5:8] + sample[10:13] + sample[15:18] +
                #                sample[20:23] + sample[25:28] + sample[29:32]).reshape(-1)  # shape: (22,1)
                # append to shared buffer under lock
            with global_value.buffer_lock:
                global_value.eeg_buffer = np.hstack((global_value.eeg_buffer, sample))
                if global_value.eeg_buffer.shape[1] > config.BUFFER_SIZE:
                    global_value.eeg_buffer = global_value.eeg_buffer[:, -config.BUFFER_SIZE:]

            # write CSV row if enabled
            if self.save_csv and self.csv_writer is not None:
                try:
                    if self.is_simulated:
                        ts_lsl = f"{count:.3f}"
                        count += 1 / config.SAMPLE_RATE
                    else:
                        ts_lsl = f"{ts - first_ts_lsl:.3f}"
                    row = [ts_lsl] + [999] + data.tolist()
                    self.csv_writer.writerow(row)
                except Exception as e:
                    print("[CSV] write error:", e)
        self.close_csv_file()  # end to close csv file

    def build_csv_file(self):
        if self.save_csv:
            self.csv_file = open(self.filename, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([f"Simulating Cygnus version: 0.28.0.7,File version: 2021.11"])
            self.csv_writer.writerow([f"Operative system: Windows"])
            self.csv_writer.writerow([f"Record datetime: {datetime.now()}"])
            self.csv_writer.writerow([f"Simulating Device ID: STEEG_DG329018"])
            self.csv_writer.writerow([f"Device verison: "])
            self.csv_writer.writerow([f"Device bandwidth: DC to 131 Hz"])
            self.csv_writer.writerow([f"Device sampling rate: 500 samples/second"])
            self.csv_writer.writerow([f"Data type / unit: EEG / micro-volt (uV)"])
            self.csv_writer.writerow([f"Bandpass filter: None"])
            self.csv_writer.writerow([f"Notch filter: None"])
            header = ["Timestamp"] + ["Serial Number"] + [
                "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
                "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8",
                "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8",
                "O1", "Oz", "O2"
            ]
            self.csv_writer.writerow(header)
            print("[CSV] Start write EEG data to", self.filename)

    def close_csv_file(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print("[CSV] Saved EEG data to", self.filename)
            except Exception:
                print("[CSV] Saved fail")
