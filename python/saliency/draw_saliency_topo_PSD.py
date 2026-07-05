"""
20260504 內容，修改寬的地方可以參考 (XBrainlab 的)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
from scipy import signal

# XBrainLab 基礎類別引用
from XBrainLab.visualization.base import Visualizer

class SaliencyPSDVisualizer(Visualizer):
    """
    客製化視覺化器：
    提供 plot_combined_for_label() 繪製特定 Label 的複合圖表
    (包含 Alpha/Beta Topomap 與 Saliency PSD)
    """

    def _compute_saliency_psd(self, method, label_idx, n_fft=None):
        """計算指定類別 Saliency 的 PSD"""
        saliency = self.get_saliency(method, label_idx)
        sfreq = self.epoch_data.sfreq

        if len(saliency) == 0:
            return None, None

        if n_fft is None:
            n_fft = int(sfreq)  # 1Hz 解析度

        # 使用 Welch 方法計算 PSD
        freqs, psd = signal.welch(saliency, fs=sfreq, nperseg=n_fft, axis=-1)
        return freqs, psd

    def plot_combined_for_label(self, label_idx, method="SmoothGrad", fmin=1, fmax=40,
                                save_path=None, use_abs=True, font_size=16,
                                show_y_axis=True, normalize=False):
        """
        繪製單一 Label 的綜合圖表
        上方：Alpha (8-13Hz) 與 Beta (13-30Hz) 的 Saliency Topomap
        下方：Saliency PSD
        """
        plt.rcParams.update({'font.size': font_size})

        try:
            class_name = self.epoch_data.label_map[label_idx]
        except (KeyError, IndexError):
            return

        positions = self.epoch_data.get_montage_position()
        chs = self.epoch_data.get_channel_names()

        # 計算 PSD
        freqs, psd = self._compute_saliency_psd(method, label_idx)
        if psd is None:
            print(f"警告：Label {class_name} 沒有 Saliency 資料可以繪製。")
            return

        # ==========================================
        # 設定版面配置 (精算比例以放大 Topomap 並容納大字體)
        # ==========================================
        fig = plt.figure(figsize=(12, 10))

        # 1. 下方長方形 PSD 
        # [left, bottom, width, height]
        ax_psd = fig.add_axes([0.12, 0.12, 0.84, 0.25])     
        
        # 2. 左上 Alpha (X 軸範圍：0.12 ~ 0.54，左側與 PSD 切齊)
        # ax_alpha = fig.add_axes([0.12, 0.38, 0.42, 0.60])
        ax_alpha = fig.add_axes([0.00, 0.42, 0.50, 0.60])  # 13, x, y, w, h
        # ax_alpha = fig.add_axes([0.04, 0.42, 0.42, 0.60])  # 22, x, y, w, h

        # 3. 右上 Beta (X 軸範圍：0.54 ~ 0.96，右側與 PSD 切齊)
        # ax_beta = fig.add_axes([0.54, 0.38, 0.42, 0.60])
        ax_beta = fig.add_axes([0.50, 0.42, 0.50, 0.60]) # 13
        # ax_beta = fig.add_axes([0.54, 0.42, 0.42, 0.60])  # 22, x, y, w, h

        cmap = 'Reds' if use_abs else 'RdBu_r'

        # ------------------------------------------
        # 1. 繪製左上角：Alpha Band (8-13Hz) Topomap
        # ------------------------------------------
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = psd[:, :, alpha_mask].mean(axis=-1).mean(axis=0)

        if normalize:
            a_min, a_max = np.min(alpha_power), np.max(alpha_power)
            if a_max > a_min:
                alpha_power = (alpha_power - a_min) / (a_max - a_min)
            else:
                alpha_power = np.zeros_like(alpha_power)

        im_alpha, _ = mne.viz.plot_topomap(
            alpha_power, pos=positions[:, :2], axes=ax_alpha,
            show=False, cmap=cmap, names=chs
        )

        # ------------------------------------------
        # 2. 繪製右上角：Beta Band (13-30Hz) Topomap
        # ------------------------------------------
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = psd[:, :, beta_mask].mean(axis=-1).mean(axis=0)

        if normalize:
            b_min, b_max = np.min(beta_power), np.max(beta_power)
            if b_max > b_min:
                beta_power = (beta_power - b_min) / (b_max - b_min)
            else:
                beta_power = np.zeros_like(beta_power)

        im_beta, _ = mne.viz.plot_topomap(
            beta_power, pos=positions[:, :2], axes=ax_beta,
            show=False, cmap=cmap, names=chs
        )

        # ------------------------------------------
        # 3. 繪製下方長方形：Saliency PSD
        # ------------------------------------------
        avg_psd = psd.mean(axis=0)

        if normalize:
            p_min, p_max = np.min(avg_psd), np.max(avg_psd)
            if p_max > p_min:
                avg_psd = (avg_psd - p_min) / (p_max - p_min)
            else:
                avg_psd = np.zeros_like(avg_psd)

        mask = (freqs >= fmin) & (freqs <= fmax)

        # 畫所有 Channel 的灰色細線
        ax_psd.plot(freqs[mask], avg_psd[:, mask].T, color='gray', alpha=0.3, linewidth=0.5)
        # 畫平均紅色粗線
        ax_psd.plot(freqs[mask], avg_psd[:, mask].mean(axis=0), color='red', linewidth=2.5, label='Mean Saliency')

        # 美化與背景色塊
        ax_psd.axvspan(8, 13, color='skyblue', alpha=0.15, label='Alpha (8-13)')
        ax_psd.axvspan(13, 30, color='salmon', alpha=0.1, label='Beta (13-30)')

        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_xlim(fmin, fmax)

        if normalize:
            ax_psd.set_ylim(-0.05, 1.05)
            y_label = "Power"
        else:
            y_label = "Power"

        if show_y_axis:
            ax_psd.set_ylabel(y_label)
        else:
            ax_psd.set_ylabel("")
            ax_psd.set_yticks([])

        ax_psd.legend(loc='upper right')
        ax_psd.grid(True, alpha=0.3)

        # ------------------------------------------
        # 4. 輸出與儲存
        # ------------------------------------------
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

# ==========================================
# 伺服器主程式 (批次執行版本)
# ==========================================
def main():
    # ================= 參數 =================
    FONT_SIZE = 30
    USE_ABS = True
    SHOW_Y_AXIS = True
    NORMALIZE = True  # 設定為 True，Y 軸及顏色範圍將變成 0~1
    is_13 = True # 生成圖片是否為 13 channel False 則是 22 channel (plot_combined_for_label 裡面也要改!!)
    # =======================================

    montage = mne.channels.make_standard_montage('standard_1020')

    # 資料庫基礎路徑設定
    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"
    ids = ["35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]
    sessions = ["s1", "s2"]
    runs = [f"run{i}" for i in range(1, 8)]  # run1 ~ run7

    for TARGET_CLASS in range(2):
        
        # 迴圈讀取所有 Subject, Session
        for subject_id in ids:
            for session in sessions:
                # 建立分層的儲存路徑 Saliency/id/session/combined_output_class
                if is_13:
                    output_dir = os.path.join("Saliency_13", subject_id, session, f"combined_output_{TARGET_CLASS}")
                else:
                    output_dir = os.path.join("Saliency", subject_id, session, f"combined_output_{TARGET_CLASS}")
                
                os.makedirs(output_dir, exist_ok=True)
                
                # 準備要檢查的資料夾清單
                dirs_to_check = []
                
                # 1. 包含 Session 層級 (e.g., .../DATA_Cygnus/35/s1)
                session_dir = os.path.join(base_dir, subject_id, session)
                dirs_to_check.append(("session", session_dir))
                
                # 2. 包含 Run 層級 (e.g., .../DATA_Cygnus/35/s1/run1 ~ run7)
                for run in runs:
                    dirs_to_check.append((run, os.path.join(session_dir, run)))
                    
                # 遍歷這些路徑進行繪圖
                for level_name, data_dir in dirs_to_check:
                    if is_13:
                        load_path_eval = os.path.join(data_dir, "13_eval_record.pkl")
                        load_path_xb = os.path.join(data_dir, "13_eval_xb_epochs.pkl")
                    else:
                        load_path_eval = os.path.join(data_dir, "22_eval_record.pkl")
                        load_path_xb = os.path.join(data_dir, "22_eval_xb_epochs.pkl")
                    # 若檔案不存在則跳過
                    if not os.path.exists(load_path_eval) or not os.path.exists(load_path_xb):
                        continue

                    # 根據是 Session 層級還是 Run 層級來命名圖片
                    if level_name == "session":
                        file_name_safe = f"Sub{subject_id}_{session}"
                    else:
                        file_name_safe = f"Sub{subject_id}_{session}_{level_name}"
                    
                    try:
                        print(f"處理中: {file_name_safe} (Class {TARGET_CLASS})...")
                        with open(load_path_eval, 'rb') as f:
                            eval_record = pickle.load(f)
                        with open(load_path_xb, 'rb') as f:
                            xb_epochs = pickle.load(f)

                        # 套用 Montage 座表
                        ch_names = xb_epochs.get_channel_names()
                        pos = [montage.get_positions()['ch_pos'].get(ch, [0, 0, 0]) for ch in ch_names]
                        xb_epochs.set_channels(ch_names, np.array(pos))

                        viz = SaliencyPSDVisualizer(eval_record, xb_epochs)

                        # 儲存路徑
                        save_file = os.path.join(output_dir, f"{file_name_safe}_c{TARGET_CLASS}_combined.png")

                        # 一次產生並儲存綜合圖表
                        viz.plot_combined_for_label(
                            label_idx=TARGET_CLASS,
                            method="Gradient",
                            fmin=0, fmax=40,
                            save_path=save_file,
                            use_abs=USE_ABS,
                            font_size=FONT_SIZE,
                            show_y_axis=SHOW_Y_AXIS,
                            normalize=NORMALIZE
                        )

                    except Exception as e:
                        print(f"處理 {file_name_safe} 時發生錯誤: {e}")
                        continue

        print(f"完成 Class={TARGET_CLASS}, Norm={NORMALIZE}")

if __name__ == "__main__":
    main()