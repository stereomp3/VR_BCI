"""
ERD + Riemannian online pipeline (full flow) + CSV logging + Marker stream
Author: ChatGPT
Date: 2025-09-13
Notes:
 - Requires pylsl, numpy, scipy, sklearn, pyriemann, mne (optional for ICA)
 - Set N_CHANNELS to actual number of channels your LSL stream provides
"""
import csv
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, resolve_streams
import sys
from datetime import datetime
from functools import wraps
import main.Utils.config as config
import main.Utils.global_value as global_value
import main.Utils.LSL as LSL
import re
from main.Utils.some_functions import rename_file_with_time
from main.Utils.TCPServer import TCPServer


class UnityLSLReader:
    def __init__(self):
        self.save_csv = config.SAVE_CSV
        self.filename = config.LOG_FILENAME
        self.is_simulated = config.is_simulated_unity
        self.csv_writer = None
        self.csv_file = None
        self.read_marker_stop_event = threading.Event()
        self.read_marker_thread = None

    def start_read_marker_thread(self):
        if self.read_marker_thread and self.read_marker_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return  # 不要重啟

        print(f"{config.TAGS.INFO.value} Start read lsl unity marker...")
        self.filename = rename_file_with_time(config.LOG_FILENAME)
        self.read_marker_stop_event.clear()  # 重置 stop_event，允許重新啟動

        self.read_marker_thread = threading.Thread(target=self.read_unity_marker, daemon=True)
        self.read_marker_thread.start()

    def end_read_marker_thread(self):
        print(f"{config.TAGS.INFO.value} End read lsl unity marker...")
        self.read_marker_stop_event.set()  # 設置停止事件
        if self.read_marker_thread and self.read_marker_thread.is_alive():
            self.read_marker_thread.join()
            self.read_marker_thread = None

    def read_unity_marker(self):
        unity_inlet = LSL.setup_lsl_inlet(config.RECEIVE_UNITY_LSL_STREAM)
        if unity_inlet is None:
            print(f"{config.TAGS.ERROR.value}  Unity_inlet is not found!")
            return
        with open(self.filename, "w", newline="") as f:  # 初始化文件
            pass

        while not self.read_marker_stop_event.is_set():
            data, ts = unity_inlet.pull_sample(timeout=0.0)
            if data is not None:
                print(f"{config.TAGS.MARKER.value} {data[0]}")
                for state in config.GameSTATE:  # 比對字串
                    if state.value == data[0]:
                        print(f"{config.TAGS.INFO.value} set unity_marker_string_stage")
                        global_value.unity_marker_string_stage = data[0]
                # 比對 log 的格式，用 data_process_np.py 裡面的內容
                # Trial 0 START: 1760276373.814 LABEL: 1
                # Trial 0 CUT: 1760276374.228
                # Trial 0 END: 1760276378.466 LABEL: 1
                pat = re.compile(r'Trial\s+(\d+)\s+(START|CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
                m = pat.match(data[0])
                if m:
                    print(f"{config.TAGS.INFO.value} set unity_marker_string_log")
                    global_value.unity_marker_string_log = data[0]
                    # 會傳送的東西:
                    with open(self.filename, "a", encoding="utf-8") as f:
                        f.write(f"{data[0]}\n")


class UnityTCPReader:
    def __init__(self):
        self.save_csv = config.SAVE_CSV
        self.filename = config.LOG_FILENAME
        self.csv_file = None  # CSV 檔案物件
        self.tcp_server = None  # TCPServer(host, port, on_message=self.process_message)

    def setup_tcp_server(self, tcp_server):
        self.tcp_server = tcp_server

    def start_write_log(self):
        self.filename = rename_file_with_time(config.LOG_FILENAME)
        if self.save_csv:
            try:
                self.csv_file = open(self.filename, "w", encoding="utf-8", newline="")
            except Exception as e:
                print(f"[CSV] Failed to open file: {e}")

    def stop_and_save_log(self):
        # self.tcp_server.stop()
        self.close_csv_file()

    def close_csv_file(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print("[CSV] Saved EEG data to", self.filename)
            except Exception:
                print("[CSV] Saved fail")
            finally:
                self.csv_file = None

    def process_message(self, msg: str, tcp_server: TCPServer):  # 在 game_state 裡面加入到 TCPServer 的 on message
        # 處理 stage
        for state in config.GameSTATE:
            if state.value == msg:
                global_value.unity_marker_string_stage = msg
                print(f"{config.TAGS.INFO.value} set unity_marker_string_stage")
        if msg == config.RECEIVE_UNITY_MODEL_STR:
            name = config.SENT_UNITY_MODEL_STR
            for i in global_value.models_name:
                name += config.SEPARATE_STR
                name += i
            tcp_server.broadcast(name)
            print(f"{config.TAGS.INFO.value} SENT_UNITY_MODEL_STR {name}")
        text = msg.split("@@@")
        if text[0] == config.RECEIVE_UNITY_SELECT_MODEL_STR: # send_python_tcp_select_model_str@@@model_name
            if len(text) == 2:
                global_value.update_model = True
                global_value.unity_update_model_str = f"{config.EEG_CHECKPOINT_MAIN_BASE_FILE}{text[1]}"
                print(f"{config.TAGS.INFO.value} Model selected {global_value.unity_update_model_str}")
        # 處理 log
        pat = re.compile(r'Trial\s+(\d+)\s+(START|CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
        m = pat.match(msg)
        if m:
            global_value.unity_marker_string_log = msg
            print(f"{config.TAGS.INFO.value} set unity_marker_string_log")
            if self.save_csv and self.csv_file:
                try:
                    self.csv_file.write(f"{msg}\n")
                    self.csv_file.flush()  # 即時寫入
                except Exception as e:
                    print(f"[CSV] Write failed: {e}")


def send_marker(label: int, marker_outlet):
    """Send integer marker through LSL, update latest_marker, and log to CSV."""
    try:
        marker_outlet.push_sample([int(label)])
    except Exception as e:
        print(f"{config.TAGS.MARKER.value} Push failed:", e)

    print(f"{config.TAGS.MARKER.value} Sent: {label}")


def main_flow():
    unity_reader = UnityLSLReader()
    unity_reader.start_read_marker_thread()
    while True:
        pass


if __name__ == "__main__":
    try:
        main_flow()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
