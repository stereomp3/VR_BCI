"""
繪製 Saliency Map 與 PSD 綜合圖 (Mu / Beta 獨立色階無 Colorbar 版)
- 移除右側 Colorbar，版面自動水平居中放大
- Mu 與 Beta 各自擁有獨立動態範圍 (vlim)，確保雙頻帶對比度清晰
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
from scipy import signal

# 防止終端編碼問題
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# XBrainLab 基礎類別引用
from XBrainLab.visualization.base import Visualizer

class SaliencyPSDVisualizer(Visualizer):
    """
    客製化視覺化器：
    繪製特定 Label 的複合圖表 (Mu/Beta Topomap 與 Saliency PSD)
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
        上方：Mu (8-13Hz) 與 Beta (13-30Hz) 的 Saliency Topomap (獨立色階，無 Colorbar)
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
        # 1. 頻帶能量計算與【獨立色彩尺度 (vlim)】設定
        # ==========================================
        # Mu 頻帶 (8-13 Hz)
        mu_mask = (freqs >= 8) & (freqs <= 13)
        mu_power = psd[:, :, mu_mask].mean(axis=-1).mean(axis=0)

        # Beta 頻帶 (13-30 Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = psd[:, :, beta_mask].mean(axis=-1).mean(axis=0)

        cmap = 'Reds' if use_abs else 'RdBu_r'

        if normalize:
            m_min, m_max = np.min(mu_power), np.max(mu_power)
            mu_power = (mu_power - m_min) / (m_max - m_min) if m_max > m_min else np.zeros_like(mu_power)
            vlim_mu = (0.0, 1.0)

            b_min, b_max = np.min(beta_power), np.max(beta_power)
            beta_power = (beta_power - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(beta_power)
            vlim_beta = (0.0, 1.0)
        else:
            # Mu 與 Beta 各自使用獨立的極值，確保雙頻帶細節完整展開
            if use_abs:
                vmax_mu = np.max(mu_power)
                vmin_mu = 0.0
                vmax_beta = np.max(beta_power)
                vmin_beta = 0.0
            else:
                max_val_mu = np.max(np.abs(mu_power))
                vmin_mu, vmax_mu = -max_val_mu, max_val_mu
                max_val_beta = np.max(np.abs(beta_power))
                vmin_beta, vmax_beta = -max_val_beta, max_val_beta

            if vmax_mu == vmin_mu:
                vmax_mu = vmin_mu + 1e-6
            if vmax_beta == vmin_beta:
                vmax_beta = vmin_beta + 1e-6

            vlim_mu = (vmin_mu, vmax_mu)
            vlim_beta = (vmin_beta, vmax_beta)

        # ==========================================
        # 2. 設定版面配置 (移除 Colorbar 後水平對稱展開)
        # ==========================================
        fig = plt.figure(figsize=(12, 10))

        # 1. 下方長方形 PSD [left, bottom, width, height]
        ax_psd = fig.add_axes([0.10, 0.10, 0.82, 0.26])   
        
        # 2. 左上 Mu Topomap (水平居中對稱，寬度擴展至 0.42)
        ax_mu = fig.add_axes([0.05, 0.42, 0.42, 0.52])

        # 3. 右上 Beta Topomap (水平居中對稱，寬度擴展至 0.42)
        ax_beta = fig.add_axes([0.53, 0.42, 0.42, 0.52])

        # ------------------------------------------
        # 3. 繪製 Topomap (套用獨立 vlim，不產生 Colorbar)
        # ------------------------------------------
        im_mu, _ = mne.viz.plot_topomap(
            mu_power, pos=positions[:, :2], axes=ax_mu,
            vlim=vlim_mu, show=False, cmap=cmap, names=chs
        )
        ax_mu.set_title("Mu (8-13 Hz)", pad=10)

        im_beta, _ = mne.viz.plot_topomap(
            beta_power, pos=positions[:, :2], axes=ax_beta,
            vlim=vlim_beta, show=False, cmap=cmap, names=chs
        )
        ax_beta.set_title("Beta (13-30 Hz)", pad=10)

        # ------------------------------------------
        # 4. 繪製下方長方形：Saliency PSD
        # ------------------------------------------
        avg_psd = psd.mean(axis=0)
        if normalize:
            p_min, p_max = np.min(avg_psd), np.max(avg_psd)
            avg_psd = (avg_psd - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(avg_psd)

        mask = (freqs >= fmin) & (freqs <= fmax)
        # 畫所有 Channel 灰色細線與平均紅色粗線
        ax_psd.plot(freqs[mask], avg_psd[:, mask].T, color='gray', alpha=0.3, linewidth=0.5)
        ax_psd.plot(freqs[mask], avg_psd[:, mask].mean(axis=0), color='red', linewidth=2.5, label='Mean Saliency')

        # 標註頻帶色塊
        ax_psd.axvspan(8, 13, color='skyblue', alpha=0.15, label='Mu (8-13)')
        ax_psd.axvspan(13, 30, color='salmon', alpha=0.1, label='Beta (13-30)')
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_xlim(fmin, fmax)

        if normalize:
            ax_psd.set_ylim(-0.05, 1.05)
            y_label = "Normalized Power"
        else:
            y_label = "Saliency Power"

        if show_y_axis:
            ax_psd.set_ylabel(y_label)
        else:
            ax_psd.set_ylabel("")
            ax_psd.set_yticks([])

        ax_psd.legend(loc='upper right')
        ax_psd.grid(True, alpha=0.3)

        # ------------------------------------------
        # 5. 輸出與儲存
        # ------------------------------------------
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

# ==========================================
# 伺服器主程式 (批次執行版本)
# ==========================================
def main():
    # ===== 參數設定 =====
    FONT_SIZE = 24
    USE_ABS = True
    SHOW_Y_AXIS = True
    NORMALIZE = False   # 設為 False 呈現真實能量
    is_13 = False       # True: 13 channel; False: 22 channel
    # ===================

    montage = mne.channels.make_standard_montage('standard_1020')
    base_dir = r"/mnt/project/MIEXP/DATA_Cygnus"
    ids = ["35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]
    sessions = ["s1", "s2"]
    runs = [f"run{i}" for i in range(1, 8)]

    for TARGET_CLASS in range(2):
        for subject_id in ids:
            for session in sessions:
                if is_13:
                    output_dir = os.path.join("Saliency_13_Raw", subject_id, session, f"combined_output_{TARGET_CLASS}")
                else:
                    output_dir = os.path.join("Saliency_22_Raw", subject_id, session, f"combined_output_{TARGET_CLASS}")
                
                os.makedirs(output_dir, exist_ok=True)
                
                dirs_to_check = []
                session_dir = os.path.join(base_dir, subject_id, session)
                dirs_to_check.append(("session", session_dir))
                
                for run in runs:
                    dirs_to_check.append((run, os.path.join(session_dir, run)))
                    
                for level_name, data_dir in dirs_to_check:
                    if is_13:
                        load_path_eval = os.path.join(data_dir, "13_eval_record.pkl")
                        load_path_xb = os.path.join(data_dir, "13_eval_xb_epochs.pkl")
                    else:
                        load_path_eval = os.path.join(data_dir, "22_eval_record.pkl")
                        load_path_xb = os.path.join(data_dir, "22_eval_xb_epochs.pkl")
                    
                    if not os.path.exists(load_path_eval) or not os.path.exists(load_path_xb):
                        continue

                    if level_name == "session":
                        file_name_safe = f"Sub{subject_id}_{session}"
                    else:
                        file_name_safe = f"Sub{subject_id}_{session}_{level_name}"
                    
                    try:
                        print(f"處理中: {file_name_safe} (Class {TARGET_CLASS}, Norm={NORMALIZE})...")
                        with open(load_path_eval, 'rb') as f:
                            eval_record = pickle.load(f)
                        with open(load_path_xb, 'rb') as f:
                            xb_epochs = pickle.load(f)

                        # 套用 Montage 座標
                        ch_names = xb_epochs.get_channel_names()
                        pos = [montage.get_positions()['ch_pos'].get(ch, [0, 0, 0]) for ch in ch_names]
                        xb_epochs.set_channels(ch_names, np.array(pos))

                        viz = SaliencyPSDVisualizer(eval_record, xb_epochs)
                        save_file = os.path.join(output_dir, f"{file_name_safe}_c{TARGET_CLASS}_combined.png")

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