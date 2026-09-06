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
import time


class UnityTCPReader:
    def __init__(self):
        self.save_csv = config.SAVE_CSV
        self.filename = config.LOG_FILENAME
        self.csv_file = None  # CSV 檔案物件
        self.tcp_server = None  # TCPServer(host, port, on_message=self.process_message)

    def setup_tcp_server(self, tcp_server):
        self.tcp_server = tcp_server

    def start_write_log(self):
        self.filename = rename_file_with_time(config.getRunLogFilename())  # config.LOG_FILENAME
        if self.save_csv:
            try:
                self.csv_file = open(self.filename, "w", encoding="utf-8", newline="")
            except Exception as e:
                print(f"[CSV] Failed to open file: {e}")

    def stop_and_save_log(self):
        # self.tcp_server.stop()
        self.close_csv_file()

    def flushLog(self):
        """強制將 LOG 檔案 buffer 寫入磁碟，供 pointer-based 讀取前呼叫"""
        if self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.flush()
                print(f"{config.TAGS.INFO.value} [DEBUG] flushLog: LOG buffer 已寫入磁碟")
            except Exception as e:
                print(f"{config.TAGS.WARNING.value} flushLog failed: {e}")

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
        if msg == config.RECEIVE_UNITY_MODEL_STR:  # send exist model
            name = config.SENT_UNITY_MODEL_STR
            for i in global_value.models_name:
                name += config.SEPARATE_STR
                name += i
            tcp_server.broadcast(name)
            print(f"{config.TAGS.INFO.value} SENT_UNITY_MODEL_STR {name}")
        text = msg.split("@@@")
        if text[0] == config.RECEIVE_UNITY_SELECT_MODEL_STR:  # send_python_tcp_select_model_str@@@model_name # 選擇使用的模型
            if len(text) == 2:
                global_value.update_model = True
                global_value.unity_update_model_str = f"{config.EEG_CHECKPOINT_MAIN_BASE_FILE}{text[1]}"
                print(f"{config.TAGS.INFO.value} Model selected {global_value.unity_update_model_str}")

        # 處裡 Calibration 字串
        if msg == config.RECEIVE_UNITY_CALIBRATION_START_STR:
            global_value.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR

        # 處理 log
        pat = re.compile(r'Trial\s+(\d+)\s+(START|CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
        pat_eyes = re.compile(r'(Close eyes|Open eyes):\s*([\d\.]+)', re.IGNORECASE)
        m = pat.match(msg)

        if not m:  # 沒有比對到 MI，判斷 eyes 邏輯
            m = pat_eyes.match(msg)
        if m:
            global_value.unity_marker_string_log = msg

            print(f"{config.TAGS.INFO.value} set unity_marker_string_log")
            if config.verbose:
                try:
                    if m.group(3):
                        tms = (float(m.group(3)) - time.time()) * 1000  # ms = 1000s
                        print(f"{config.TAGS.INFO.value} [TCP Latency] {tms} ms")
                except IndexError:
                    pass

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

# def main_flow():
#     unity_reader = UnityLSLReader()
#     unity_reader.start_read_marker_thread()
#     while True:
#         pass
#
#
# if __name__ == "__main__":
#     try:
#         main_flow()
#     except KeyboardInterrupt:
#         print("Interrupted by user. Exiting.")
