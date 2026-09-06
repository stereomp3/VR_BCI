"""
線上即時 LSL 腦波串流至 Unity 預測快速測試工具 (Online LSL-to-Unity Quick Tester)
用於在不啟動完整遊戲狀態機的情況下，快速驗證：
「Cygnus 腦波帽/模擬訊號 -> 帶通濾波/去均值 -> 深度學習模型即時推論 -> LSL MarkerStream 發送至 Unity」
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
import torch
from scipy.signal import butter, filtfilt, decimate
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams, resolve_byprop

# 確保路徑
_tools_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(_tools_dir)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import main.Utils.config as config
from main.EEG.models import SCCNet
from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, ATCNet
from main.EEG.generate_random_models import create_matching_random_checkpoint

MODEL_DICT = {
    "SCCNet": SCCNet,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "EEGNetv4": EEGNetv4,
    "EEGConformer": EEGConformer,
    "ATCNet": ATCNet
}


class OnlineTester:
    def __init__(self, channels=32, model_name="SCCNet", checkpoint_path=None,
                 simulated=False, stream_name=config.RECEIVE_CYGNUS_LSL_STREAM,
                 outlet_name=config.TO_UNITY_LSL_STREAM, sample_rate=500,
                 prediction_interval=0.05):
        self.channels = channels
        self.channel_indices = config.CHANNEL_DEFINITIONS.get(str(channels), list(range(channels)))
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path or os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, "c_000.pth")
        self.simulated = simulated
        self.stream_name = stream_name
        self.outlet_name = outlet_name
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate  # 1 秒視窗
        self.prediction_interval = prediction_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eeg_buffer = np.zeros((self.channels, 0))
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.model = self._setup_model()
        self.outlet = self._setup_outlet()

    def _setup_model(self):
        model_cls = MODEL_DICT.get(self.model_name, SCCNet)
        print(f"📦 初始化模型架構: {self.model_name} (通道數={self.channels}, 採樣率={self.sample_rate}, 裝置={self.device})")

        if self.model_name == "SCCNet":
            model = model_cls(samples=self.sample_rate, channels=self.channels, n_classes=config.N_Class, sfreq=self.sample_rate)
        else:
            model = model_cls(n_chans=self.channels, n_outputs=config.N_Class, n_times=self.sample_rate)

        model = model.to(self.device)

        # 嘗試載入 Checkpoint，若不匹配則自動修復
        try:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"找不到權重檔: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功載入模型權重: {self.checkpoint_path}")
        except Exception as e:
            print(f"⚠️ [警告] 權重載入失敗 ({e})，自動生成符合 {self.channels} 通道之隨機初始權重...")
            checkpoint = create_matching_random_checkpoint(
                target_path=self.checkpoint_path,
                model_class=model_cls,
                n_chans=self.channels,
                n_outputs=config.N_Class,
                n_times=self.sample_rate
            )
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        return model

    def _setup_outlet(self):
        print(f"📡 建立 Unity LSL 輸出串流: '{self.outlet_name}' (type=Markers)")
        info = StreamInfo(name=self.outlet_name, type='Markers', channel_count=1,
                          nominal_srate=10, channel_format='int32', source_id='online_tester_id')
        return StreamOutlet(info)

    def _resolve_inlet(self):
        print(f"🔍 正在搜尋 LSL 串流: '{self.stream_name}' ...")
        streams = resolve_byprop('name', self.stream_name, timeout=3.0)
        if len(streams) == 0:
            print(f"⚠️ 未找到 LSL 串流 '{self.stream_name}'！")
            print("👉 自動切換為「模擬隨機腦波 (Simulated EEG)」模式運行...")
            self.simulated = True
            return None
        inlet = StreamInlet(streams[0], recover=True)
        print(f"✅ 成功連線至腦波串流: {streams[0].name()}")
        return inlet

    def _filter_eeg(self, data):
        # 帶通濾波 1 ~ 40 Hz
        b, a = butter(4, [1.0 / (0.5 * self.sample_rate), 40.0 / (0.5 * self.sample_rate)], btype='band')
        filtered = filtfilt(b, a, data, axis=-1)
        # 去均值 (Demean)
        filtered = filtered - np.mean(filtered, axis=-1, keepdims=True)
        return filtered

    def eeg_read_loop(self, inlet):
        while not self.stop_event.is_set():
            if self.simulated or inlet is None:
                # 模擬產生隨機腦波
                sample = np.random.randn(self.channels, 1) * 20.0
                time.sleep(1.0 / self.sample_rate)
            else:
                try:
                    raw_sample, _ = inlet.pull_sample(timeout=0.5)
                    if raw_sample is None:
                        continue
                    # 擷取對應通道
                    if len(raw_sample) >= max(self.channel_indices) + 1:
                        selected = [raw_sample[idx] for idx in self.channel_indices]
                    else:
                        selected = raw_sample[:self.channels]
                    sample = np.array(selected).reshape(-1, 1)
                except Exception:
                    continue

            with self.buffer_lock:
                self.eeg_buffer = np.hstack((self.eeg_buffer, sample))
                if self.eeg_buffer.shape[1] > self.buffer_size:
                    self.eeg_buffer = self.eeg_buffer[:, -self.buffer_size:]

    def predict_loop(self):
        print("🚀 即時預測循環已啟動 (按 Ctrl+C 可停止)...")
        pred_count = 0
        while not self.stop_event.is_set():
            has_enough = False
            with self.buffer_lock:
                if self.eeg_buffer.shape[1] >= self.buffer_size:
                    data_slice = self.eeg_buffer[:, -self.buffer_size:].copy()
                    has_enough = True

            if has_enough:
                try:
                    filtered_data = self._filter_eeg(data_slice)
                    # 轉換為 (1, 1, C, T) 給 SCCNet，或 (1, C, T) 給 Braindecode
                    if self.model_name == "SCCNet":
                        x_batch = torch.from_numpy(filtered_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    else:
                        x_batch = torch.from_numpy(filtered_data).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(x_batch)
                        pred = int(torch.argmax(outputs).cpu().item())
                        prob = torch.softmax(outputs, dim=-1).cpu().numpy().flatten()

                    pred_count += 1
                    self.outlet.push_sample([pred])

                    label_str = "左手 (Left)" if pred == 1 else "右手 (Right)"
                    if pred_count % 10 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] 預測 #{pred_count:<5} -> {label_str:<12} (機率: L={prob[1]:.2f}, R={prob[0]:.2f})")
                except Exception as e:
                    print(f"⚠️ 預測異常: {e}")

            time.sleep(self.prediction_interval)

    def run(self):
        inlet = None if self.simulated else self._resolve_inlet()
        read_thread = threading.Thread(target=self.eeg_read_loop, args=(inlet,), daemon=True)
        read_thread.start()

        try:
            self.predict_loop()
        except KeyboardInterrupt:
            print("\n🛑 接收到中斷訊號，正在安全停止線上測試...")
        finally:
            self.stop_event.set()
            read_thread.join(timeout=1.0)
            print("✅ 測試已安全結束。")


def main():
    parser = argparse.ArgumentParser(description="VR-BCI 即時 LSL 至 Unity 預測快速測試工具")
    parser.add_argument("--channels", type=int, default=int(config.ACTIVE_CHANNEL_MODE),
                        choices=[8, 13, 22, 32], help="通道數設定 (預設讀取 config.json)")
    parser.add_argument("--model", type=str, default="SCCNet",
                        choices=list(MODEL_DICT.keys()), help="欲測試之模型架構")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="欲載入之模型權重檔路徑 (預設為 c_000.pth)")
    parser.add_argument("--simulated", action="store_true",
                        help="強制使用隨機模擬腦波訊號 (無實體腦波帽時使用)")
    parser.add_argument("--stream_name", type=str, default=config.RECEIVE_CYGNUS_LSL_STREAM,
                        help="Cygnus 腦波 LSL 串流名稱")
    parser.add_argument("--outlet_name", type=str, default=config.TO_UNITY_LSL_STREAM,
                        help="發送給 Unity 的 Marker 串流名稱")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="預測間隔秒數 (預設 0.05s)")

    args = parser.parse_args()

    tester = OnlineTester(
        channels=args.channels,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        simulated=args.simulated,
        stream_name=args.stream_name,
        outlet_name=args.outlet_name,
        prediction_interval=args.interval
    )
    tester.run()


if __name__ == "__main__":
    main()
