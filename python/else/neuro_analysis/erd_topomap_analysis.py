"""
================================================================================
BCI 全局 ERD Topomap 空間地形圖與學習軌跡分析系統 (Session 獨立總覽 + Left/Right MI 版)
(Global BCI ERD Topomap Spatial & Learning Progression Analyzer)
================================================================================
【核心升級與視覺樣式】
1. 學習軌跡總覽大圖 (2_ERD_Topomap_Run_Evolution) 改以 Session 獨立劃分：
   - 每個 Session (Session 1 與 Session 2) 各自輸出一張獨立總覽大圖
   - 左側縱軸兩列標籤：第一列為 Left MI (換行顯示)、第二列為 Right MI (換行顯示)
   - 頂部橫軸欄位：Run 1 ~ Run 7，下方對稱標註 Mu (8-13 Hz) 與 Beta (13-30 Hz)
   - 左上角標註：兩位數受試者序號 (01 ~ 24) 與 Session 標籤 (S1 / S2)
   - 最頂層半透明遮罩：奇數 Run 欄位 (Run 1, 3, 5, 7) 覆蓋 0.05 深灰半透明斑馬紋
   - 水平灰色虛線：位於最頂層，清晰分隔 Header 與各動作類別列
   - 缺圖對齊邏輯：若 Run 數量為 6，自動將 Run 3 (第 3 格) 留白對齊
2. 保留三大類 Topomap 獨立分析：
   - 1_ERD_Topomap_Left_vs_Right_<ID>_<SESS>.png : 左右手 x 頻段空間對比圖
   - 2_ERD_Topomap_Run_Evolution_Sub<ID>_Session<X>_Grid.png : 專業總覽 Grid 地形圖
   - 3_ERD_Topomap_Differential_<ID>_<SESS>.png  : 差分空間地形圖 (Left MI - Right MI)
3. 支援 -all 批次處理 24 位受試者，並內建 --demo 擬真測試模式。
================================================================================
"""
import os
import sys
import io
import argparse
from datetime import datetime
from functools import wraps

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt, welch
from PIL import Image, ImageDraw, ImageFont

# 嘗試引入 MNE
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# 設定 Matplotlib 樣式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# 0. 系統通道定義與全受試者對照表
# ==========================================
CH_NAMES_22 = [
    'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4',
    'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'
]
CH_NAMES_13 = [
    'F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'Pz'
]

ALL_SUBJECT_IDS = [
    "35", "37", "38", "40", "41", "42", "43", "44", "45", "47",
    "48", "50", "51", "52", "54", "55", "57", "58", "63", "64",
    "65", "68", "69", "70"
]

SUBJECT_MAP = {
    "35": "S1", "37": "S2", "38": "S3", "40": "S4", "41": "S5",
    "42": "S6", "43": "S7", "44": "S8", "45": "S9", "47": "S10",
    "48": "S11", "50": "S12", "51": "S13", "52": "S14", "54": "S15",
    "55": "S16", "57": "S17", "58": "S18", "63": "S19", "64": "S20",
    "65": "S21", "68": "S22", "69": "S23", "70": "S24"
}

def get_subject_display_name(raw_sub_id):
    """將原始 ID (例如 70, 44) 轉換為 Subject X 正式名稱"""
    sub_str = str(raw_sub_id).strip()
    if sub_str in SUBJECT_MAP:
        s_code = SUBJECT_MAP[sub_str]
        num = s_code.replace("S", "")
        return f"Subject {num}"
    elif sub_str.upper().startswith("S") and sub_str[1:].isdigit():
        return f"Subject {int(sub_str[1:])}"
    elif sub_str.isdigit() and 1 <= int(sub_str) <= 24:
        return f"Subject {int(sub_str)}"
    else:
        return f"Subject {sub_str}"

def get_subject_two_digit_order(raw_sub_id):
    """取得受試者 01~24 兩位數字串"""
    sub_str = str(raw_sub_id).strip()
    if sub_str in ALL_SUBJECT_IDS:
        return f"{ALL_SUBJECT_IDS.index(sub_str) + 1:02d}"
    elif sub_str in SUBJECT_MAP:
        num = int(SUBJECT_MAP[sub_str].replace("S", ""))
        return f"{num:02d}"
    elif sub_str.isdigit():
        return f"{int(sub_str):02d}"
    return "01"

# ==========================================
# 1. 跨平台字體載入與繪圖工具函式
# ==========================================
def get_serif_font(size):
    """載入襯線體 (Times New Roman / DejaVuSerif)"""
    font_candidates = [
        "times.ttf", "timesbd.ttf", "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/System/Library/Fonts/Times.ttc",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def draw_horizontal_dashed_line(draw, y, total_width, dash_len=24, gap_len=14, line_width=4, color=(195, 195, 195)):
    """繪製水平灰色虛線"""
    x = 0
    while x < total_width:
        x_end = min(x + dash_len, total_width)
        draw.line([(x, y), (x_end, y)], fill=color, width=line_width)
        x += dash_len + gap_len

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def tee_log(log_file=None):
    if log_file is None:
        log_file = f"erd_topomap_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(log_file, "w", encoding="utf-8") as f:
                sys.stdout = Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout
            print(f"📄 執行記錄已儲存至: {log_file}")
            return result
        return wrapper
    return decorator

def infer_channel_names(n_channels):
    if n_channels == 22:
        return CH_NAMES_22
    elif n_channels == 13:
        return CH_NAMES_13
    else:
        return [f"Ch{i+1}" for i in range(n_channels)]

# ==========================================
# 2. 頻譜能量與 Session Median ERD 核心運算
# ==========================================
def compute_band_power(x_data, fs=500, band=(8, 13)):
    if x_data.ndim == 2:
        x_data = x_data[np.newaxis, :, :]
    n_trials, n_channels, n_samples = x_data.shape
    nperseg = min(n_samples, int(fs * 0.5))
    if nperseg < 8:
        nperseg = n_samples
    freqs, psd = welch(x_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if np.sum(mask) == 0:
        band_power = np.mean(psd, axis=-1)
    else:
        band_power = np.mean(psd[:, :, mask], axis=-1)
    return band_power.squeeze()

def extract_session_median_powers(x_trials, fs=500, band=(8, 13), task_range=(1.0, 3.5)):
    n_trials, n_channels, n_samples = x_trials.shape
    t_axis = np.arange(n_samples) / fs
    t_mask = (t_axis >= task_range[0]) & (t_axis <= task_range[1])
    task_data = x_trials if np.sum(t_mask) == 0 else x_trials[:, :, t_mask]
    
    p_task = compute_band_power(task_data, fs=fs, band=band)
    if p_task.ndim == 1:
        p_task = p_task[np.newaxis, :]
        
    all_powers = compute_band_power(x_trials, fs=fs, band=band)
    if all_powers.ndim == 1:
        all_powers = all_powers[np.newaxis, :]
    med_p = np.median(all_powers, axis=0)
    p_base = np.tile(med_p, (n_trials, 1))
    return p_base, p_task

def calculate_erd_percentage(p_base, p_task):
    return (p_task - p_base) / (p_base + 1e-8) * 100.0

# ==========================================
# 3. Topomap 繪製核心模組
# ==========================================
def draw_single_topomap_ax(erd_values, ch_names, ax, title="", vmax=60.0,
                           show_names=True, cmap='RdBu_r'):
    n_ch = len(ch_names)
    vmax = float(vmax)
    vmin = -vmax
    if HAS_MNE:
        info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')
        try:
            im, _ = mne.viz.plot_topomap(
                data=erd_values,
                pos=info,
                axes=ax,
                show=False,
                cmap=cmap,
                vlim=(vmin, vmax),
                sensors=True,
                names=ch_names if show_names else None,
                contours=4
            )
            if title:
                ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            return im
        except Exception:
            pass

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    sc = ax.scatter(np.arange(n_ch), erd_values, c=erd_values, cmap=cmap, vmin=vmin, vmax=vmax, s=120)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_names, rotation=45, fontsize=8)
    return sc

def render_run_cell_image(erd_mu, erd_beta, ch_names, tw=600, th=340, vmax=50.0):
    """將單一 Run 的 Mu 與 Beta Topomap 繪製為一張子圖圖片 (二者水平並列)"""
    fig, axs = plt.subplots(1, 2, figsize=(tw / 100.0, th / 100.0), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.96, wspace=0.06)

    draw_single_topomap_ax(erd_mu, ch_names, axs[0], title="", vmax=vmax, show_names=True)
    draw_single_topomap_ax(erd_beta, ch_names, axs[1], title="", vmax=vmax, show_names=True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')

def render_vertical_colorbar(cbar_w=160, th=340, vmax=50.0, label="ERD / ERS (%)"):
    """繪製垂直獨立 Colorbar 供 Grid 右側掛載"""
    fig = plt.figure(figsize=(cbar_w / 100.0, th / 100.0), dpi=100)
    ax = fig.add_axes([0.18, 0.15, 0.22, 0.70])
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=ax)
    cb.set_label(label, fontsize=13, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=11)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')

# ==========================================
# 4. 【獨立圖表 1】左右手 x 頻段 2x2 空間對比 Topomap
# ==========================================
def plot_erd_topomap_left_vs_right(all_runs_base, all_runs_task, all_runs_y,
                                   ch_names, subject_label="Subject 24", session_label="Session 1",
                                   save_dir="output", vmax=50.0):
    os.makedirs(save_dir, exist_ok=True)
    flat_base_mu = np.concatenate([r['base_mu'] for r in all_runs_base])
    flat_task_mu = np.concatenate([r['task_mu'] for r in all_runs_task])
    flat_base_beta = np.concatenate([r['base_beta'] for r in all_runs_base])
    flat_task_beta = np.concatenate([r['task_beta'] for r in all_runs_task])
    flat_y = np.concatenate(all_runs_y)
    
    left_mask = (flat_y == 1)
    right_mask = (flat_y == 0)

    erd_mu_l = np.mean(calculate_erd_percentage(flat_base_mu[left_mask], flat_task_mu[left_mask]), axis=0)
    erd_mu_r = np.mean(calculate_erd_percentage(flat_base_mu[right_mask], flat_task_mu[right_mask]), axis=0)
    erd_beta_l = np.mean(calculate_erd_percentage(flat_base_beta[left_mask], flat_task_beta[left_mask]), axis=0)
    erd_beta_r = np.mean(calculate_erd_percentage(flat_base_beta[right_mask], flat_task_beta[right_mask]), axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.25, wspace=0.15, right=0.88, top=0.88)
    fig.suptitle(f"Motor Imagery ERD/ERS Topomap: {subject_label} | {session_label}",
                 fontsize=14, fontweight='bold', y=0.97)

    im1 = draw_single_topomap_ax(erd_mu_l, ch_names, axs[0, 0], title="Left Hand MI - Mu Band (8-13 Hz)", vmax=vmax)
    draw_single_topomap_ax(erd_mu_r, ch_names, axs[0, 1], title="Right Hand MI - Mu Band (8-13 Hz)", vmax=vmax)
    draw_single_topomap_ax(erd_beta_l, ch_names, axs[1, 0], title="Left Hand MI - Beta Band (13-30 Hz)", vmax=vmax)
    draw_single_topomap_ax(erd_beta_r, ch_names, axs[1, 1], title="Right Hand MI - Beta Band (13-30 Hz)", vmax=vmax)

    cbar_ax = fig.add_axes([0.91, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('ERD / ERS (%)', fontsize=11, fontweight='bold')

    out_name = f"1_ERD_Topomap_Left_vs_Right_{subject_label.replace(' ', '')}_{session_label.replace(' ', '')}.png"
    plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ [已生成] 左右手對比 Topomap: {out_name}")

# ==========================================
# 5. 【獨立圖表 2】專業 Paper 級 Run 演化進程總覽 Grid (Session 獨立版)
# ==========================================
def generate_erd_session_run_grid(session_data, ch_names, raw_sub_id, session_str="s1",
                                 save_dir="output", vmax=50.0):
    """
    輸出圖 2：單一 Session 專用總覽圖 (7 Run 欄位 x 2 動作類別列)
    【排版架構】
    - 畫布規格：以單一 Session (Session 1 或 Session 2) 獨立繪製
    - 第一列 (Row 0)：Left MI (Label 1)
    - 第二列 (Row 1)：Right MI (Label 0)
    - 左側縱軸文字：分別標註 'Left\\nMI' 與 'Right\\nMI'
    - 左上角：標示受試者序號 (01~24) 與當前 Session 代碼 (S1/S2)
    - 頂部橫軸：Run 1 ~ Run 7，下方對應 Mu 與 Beta 頻帶
    - 斑馬紋：奇數 Run 欄位覆蓋 0.05 半透明深灰遮罩，灰色水平虛線置頂
    """
    os.makedirs(save_dir, exist_ok=True)
    order_str = get_subject_two_digit_order(raw_sub_id)
    sess_num = "1" if "1" in str(session_str).lower() else "2"
    sess_tag = f"Session {sess_num}"

    # 定義兩列的動作類別 (Row 0: Left MI, Row 1: Right MI)
    rows_info = [
        {"class_id": 1, "label": "Left\n\nMI"},
        {"class_id": 0, "label": "Right\n\nMI"}
    ]

    runs_base = session_data['base']
    runs_task = session_data['task']
    runs_y = session_data['y']
    n_runs = len(runs_base)

    if n_runs == 0:
        return

    tw = 560
    th = 320
    header_left_w = int(tw * 0.45)
    header_top_h = int(th * 0.42)
    cbar_w = int(tw * 0.28)

    grid_w = header_left_w + 7 * tw + cbar_w
    grid_h = header_top_h + len(rows_info) * th

    font_id = get_serif_font(int(th * 0.15))        # 左上角序號字體
    font_sess_sub = get_serif_font(int(th * 0.12))  # 左上角 Session 標籤字體
    font_run = get_serif_font(int(th * 0.13))       # Run 1~7 字體
    font_band = get_serif_font(int(th * 0.12))      # Mu / Beta 字體
    font_row = get_serif_font(int(th * 0.13))       # Left/Right MI 類別字體

    COLOR_WHITE = (255, 255, 255, 255)
    COLOR_TEXT = (0, 0, 0)
    COLOR_LINE = (195, 195, 195)
    OVERLAY_ALPHA = int(255 * 0.05)
    OVERLAY_COLOR = (0, 0, 0, OVERLAY_ALPHA)

    canvas = Image.new('RGBA', (grid_w, grid_h), COLOR_WHITE)

    # 1. 依序繪製並貼上各 Run 的 Topomap 子圖
    for row_idx, r_info in enumerate(rows_info):
        target_class = r_info["class_id"]

        for r_idx in range(n_runs):
            # 6 Run 空缺處理邏輯：強制空出 Run 3 (index 2)
            if n_runs == 6:
                target_col = r_idx if r_idx < 2 else r_idx + 1
            else:
                target_col = r_idx

            if target_col >= 7:
                continue

            y_r = runs_y[r_idx]
            cls_mask = (y_r == target_class)

            if np.sum(cls_mask) == 0:
                erd_mu = np.zeros(len(ch_names))
                erd_beta = np.zeros(len(ch_names))
            else:
                b_mu = runs_base[r_idx]['base_mu'][cls_mask]
                t_mu = runs_task[r_idx]['task_mu'][cls_mask]
                b_beta = runs_base[r_idx]['base_beta'][cls_mask]
                t_beta = runs_task[r_idx]['task_beta'][cls_mask]
                erd_mu = np.mean(calculate_erd_percentage(b_mu, t_mu), axis=0)
                erd_beta = np.mean(calculate_erd_percentage(b_beta, t_beta), axis=0)

            cell_img = render_run_cell_image(erd_mu, erd_beta, ch_names, tw=tw, th=th, vmax=vmax)
            paste_x = header_left_w + target_col * tw
            paste_y = header_top_h + row_idx * th
            canvas.paste(cell_img, (paste_x, paste_y), mask=cell_img.split()[3])

        # 在每列右側貼上獨立垂直 Colorbar
        cbar_img = render_vertical_colorbar(cbar_w=cbar_w, th=th, vmax=vmax)
        cbar_x = header_left_w + 7 * tw
        cbar_y = header_top_h + row_idx * th
        canvas.paste(cbar_img, (cbar_x, cbar_y), mask=cbar_img.split()[3])

    # 2. 最上層半透明斑馬紋遮罩 (奇數欄位 Run 1, 3, 5, 7 覆蓋 0.05 深灰)
    overlay = Image.new('RGBA', (grid_w, grid_h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for col in range(7):
        if col % 2 == 0:
            col_x = header_left_w + col * tw
            overlay_draw.rectangle([col_x, 0, col_x + tw, grid_h], fill=OVERLAY_COLOR)

    canvas = Image.alpha_composite(canvas, overlay)
    canvas = canvas.convert('RGB')
    draw = ImageDraw.Draw(canvas)

    # 3. 繪製最頂層文字標籤
    # 左上角受試者序號 (01~24) 與 Session 標籤 (S1/S2)
    draw.text((header_left_w // 2, header_top_h * 0.28), order_str, fill=COLOR_TEXT, font=font_id, anchor="mm")
    draw.text((header_left_w // 2, header_top_h * 0.72), sess_tag, fill=COLOR_TEXT, font=font_sess_sub, anchor="mm")

    # 頂部 Run 1~7 與 Mu / Beta 標籤
    for col in range(7):
        col_x = header_left_w + col * tw
        run_cx = col_x + tw // 2
        draw.text((run_cx, header_top_h * 0.28), f"Run {col + 1}", fill=COLOR_TEXT, font=font_run, anchor="mm")

        mu_cx = col_x + int(tw * 0.25)
        draw.text((mu_cx, header_top_h * 0.72), "Mu", fill=COLOR_TEXT, font=font_band, anchor="mm")

        beta_cx = col_x + int(tw * 0.75)
        draw.text((beta_cx, header_top_h * 0.72), "Beta", fill=COLOR_TEXT, font=font_band, anchor="mm")

    # 左側縱軸標籤：分別標註 'Left\nMI' 與 'Right\nMI'
    for row_idx, r_info in enumerate(rows_info):
        row_y = header_top_h + row_idx * th
        row_cy = row_y + th // 2
        draw.text(
            (header_left_w // 2, row_cy),
            r_info["label"],
            fill=COLOR_TEXT,
            font=font_row,
            anchor="mm",
            align="center"
        )

    # 4. 繪製頂層水平灰色虛線
    draw_horizontal_dashed_line(draw, header_top_h, grid_w - cbar_w, line_width=4, color=COLOR_LINE)
    draw_horizontal_dashed_line(draw, header_top_h + th, grid_w - cbar_w, line_width=4, color=COLOR_LINE)

    # 5. 輸出儲存
    out_name = f"2_ERD_Topomap_Run_Evolution_Sub{raw_sub_id}_Session{sess_num}_Grid.png"
    out_path = os.path.join(save_dir, out_name)
    canvas.save(out_path, quality=95)
    print(f"  ✓ [已生成] 學習演化總覽 Grid (Session {sess_num}): {out_name}")

# ==========================================
# 6. 【獨立圖表 3】差分空間地形圖 (Differential Topomap)
# ==========================================
def plot_erd_topomap_differential(all_runs_base, all_runs_task, all_runs_y,
                                  ch_names, subject_label="Subject 24", session_label="Session 1",
                                  save_dir="output", vmax=60.0):
    os.makedirs(save_dir, exist_ok=True)
    flat_base_mu = np.concatenate([r['base_mu'] for r in all_runs_base])
    flat_task_mu = np.concatenate([r['task_mu'] for r in all_runs_task])
    flat_base_beta = np.concatenate([r['base_beta'] for r in all_runs_base])
    flat_task_beta = np.concatenate([r['task_beta'] for r in all_runs_task])
    flat_y = np.concatenate(all_runs_y)
    
    left_mask = (flat_y == 1)
    right_mask = (flat_y == 0)

    erd_mu_l = np.mean(calculate_erd_percentage(flat_base_mu[left_mask], flat_task_mu[left_mask]), axis=0)
    erd_mu_r = np.mean(calculate_erd_percentage(flat_base_mu[right_mask], flat_task_mu[right_mask]), axis=0)
    diff_mu = erd_mu_l - erd_mu_r

    erd_beta_l = np.mean(calculate_erd_percentage(flat_base_beta[left_mask], flat_task_beta[left_mask]), axis=0)
    erd_beta_r = np.mean(calculate_erd_percentage(flat_base_beta[right_mask], flat_task_beta[right_mask]), axis=0)
    diff_beta = erd_beta_l - erd_beta_r

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(wspace=0.20, right=0.88, top=0.82)
    fig.suptitle(f"Differential Spatial Topomap (ΔERD = Left Hand MI - Right Hand MI): {subject_label} | {session_label}",
                 fontsize=13.5, fontweight='bold', y=0.96)

    im1 = draw_single_topomap_ax(diff_mu, ch_names, axs[0], title="Mu Band (8-13 Hz) ΔERD", vmax=vmax)
    draw_single_topomap_ax(diff_beta, ch_names, axs[1], title="Beta Band (13-30 Hz) ΔERD", vmax=vmax)

    cbar_ax = fig.add_axes([0.91, 0.22, 0.025, 0.55])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Differential ΔERD (%)', fontsize=11, fontweight='bold')

    out_name = f"3_ERD_Topomap_Differential_{subject_label.replace(' ', '')}_{session_label.replace(' ', '')}.png"
    plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ [已生成] 差分側化 Topomap: {out_name}")

# ==========================================
# 7. 單一受試者跨 Session 整合主流程
# ==========================================
def process_single_subject_pipeline(data_dir, raw_sub_id, output_root,
                                    target_sessions=("s1", "s2"), is_demo=False,
                                    task_range=(1.0, 3.5)):
    sub_dir_name = str(raw_sub_id)
    subject_out_dir = os.path.join(output_root, sub_dir_name)
    os.makedirs(subject_out_dir, exist_ok=True)
    subject_label = get_subject_display_name(raw_sub_id)

    subject_session_data = {}
    ch_names = None

    for session_str in target_sessions:
        all_runs_base = []
        all_runs_task = []
        all_runs_y = []
        sess_num = "1" if "1" in str(session_str).lower() else "2"
        session_label = f"Session {sess_num}"

        if is_demo:
            ch_names = CH_NAMES_22
            n_runs = 6 if session_str == "s2" else 7  # 模擬 S2 缺 1 Run
            for r in range(n_runs):
                n_tr = 40
                n_ch = 22
                n_samp = 2000
                fs = 500
                x_fake = np.random.randn(n_tr, n_ch, n_samp) * 8.0
                y_fake = np.random.choice([0, 1], size=n_tr)
                
                c3_idx = ch_names.index('C3')
                c4_idx = ch_names.index('C4')
                prog = (r + 1) / 7.0
                time_vec = np.arange(n_samp) / fs
                t_mask = (time_vec >= 1.0) & (time_vec <= 3.5)
                for t in range(n_tr):
                    if y_fake[t] == 1:
                        x_fake[t, c4_idx, t_mask] *= (1.0 - 0.45 * prog)
                    else:
                        x_fake[t, c3_idx, t_mask] *= (1.0 - 0.45 * prog)

                p_base_mu, p_task_mu = extract_session_median_powers(x_fake, fs=fs, band=(8, 13), task_range=task_range)
                p_base_beta, p_task_beta = extract_session_median_powers(x_fake, fs=fs, band=(13, 30), task_range=task_range)
                all_runs_base.append({'base_mu': p_base_mu, 'base_beta': p_base_beta})
                all_runs_task.append({'task_mu': p_task_mu, 'task_beta': p_task_beta})
                all_runs_y.append(y_fake)
        else:
            subject_dir = os.path.join(data_dir, str(raw_sub_id), session_str)
            for r in range(1, 8):
                run_dir = os.path.join(subject_dir, f"run{r}")
                pt_candidates = [
                    os.path.join(run_dir, "mi_22.pt"),
                    os.path.join(run_dir, "mi_13.pt"),
                    os.path.join(run_dir, "data.pt"),
                    os.path.join(subject_dir, f"run_{r}.pt"),
                ]
                for pt_path in pt_candidates:
                    if os.path.exists(pt_path):
                        try:
                            data = torch.load(pt_path, map_location='cpu')
                            x_d = data.get('x_data', data.get('x'))
                            y_d = data.get('y_data', data.get('y'))
                            if isinstance(x_d, torch.Tensor): x_d = x_d.numpy()
                            if isinstance(y_d, torch.Tensor): y_d = y_d.numpy()
                            if x_d is not None and len(x_d) > 0:
                                if ch_names is None:
                                    ch_names = infer_channel_names(x_d.shape[1])
                                p_base_mu, p_task_mu = extract_session_median_powers(x_d, fs=500, band=(8, 13), task_range=task_range)
                                p_base_beta, p_task_beta = extract_session_median_powers(x_d, fs=500, band=(13, 30), task_range=task_range)
                                all_runs_base.append({'base_mu': p_base_mu, 'base_beta': p_base_beta})
                                all_runs_task.append({'task_mu': p_task_mu, 'task_beta': p_task_beta})
                                all_runs_y.append(y_d)
                                break
                        except Exception:
                            pass

        if len(all_runs_base) > 0:
            subject_session_data[session_str] = {
                'base': all_runs_base,
                'task': all_runs_task,
                'y': all_runs_y
            }
            # 輸出單一 Session 的獨立對比圖與差分圖
            plot_erd_topomap_left_vs_right(
                all_runs_base, all_runs_task, all_runs_y, ch_names,
                subject_label=subject_label, session_label=session_label,
                save_dir=subject_out_dir
            )
            plot_erd_topomap_differential(
                all_runs_base, all_runs_task, all_runs_y, ch_names,
                subject_label=subject_label, session_label=session_label,
                save_dir=subject_out_dir
            )

    if not subject_session_data:
        print(f"  ⚠️ [跳過] 未在受試者 {raw_sub_id} 找到任何有效 Session 資料。")
        return False

    # ★ 依序為各 Session 合成獨立總覽 Grid (左欄顯示 Left MI 與 Right MI)
    print(f"  ▶ 正在為 {subject_label} 合成各 Session 專業 Paper Grid 學習軌跡圖...")
    for sess_key in target_sessions:
        if sess_key in subject_session_data:
            generate_erd_session_run_grid(
                session_data=subject_session_data[sess_key],
                ch_names=ch_names,
                raw_sub_id=raw_sub_id,
                session_str=sess_key,
                save_dir=subject_out_dir,
                vmax=50.0
            )

    return True

# ==========================================
# 8. 主程式入口
# ==========================================
@tee_log()
def main():
    parser = argparse.ArgumentParser(description="BCI 全局 ERD Topomap 空間特徵與學習軌跡分析系統 (Session 獨立總覽 + Left/Right MI 版)")
    parser.add_argument("-all", "--all", dest="all_subjects", action="store_true",
                        help="批次生成所有 24 位受試者 (S1~S24, ID: 35~70) 的全部 Topomap 與 Grid")
    parser.add_argument("--demo", action="store_true", help="執行 Demo 擬真合成資料模式")
    parser.add_argument("--data_dir", type=str, default=r"/mnt/project/MIEXP/DATA_Cygnus",
                        help="資料集根目錄路徑")
    parser.add_argument("--subject", type=str, default="44",
                        help="單一受試者 ID (例如: 44, 70, 35 或 S8, S24)")
    parser.add_argument("--session", type=str, default="all", choices=["s1", "s2", "all"],
                        help="指定 Session (s1, s2, 或 all)")
    parser.add_argument("--output_dir", type=str, default="erd_topomap_output",
                        help="圖表儲存根目錄 (內部會自動建立 <id>/ 子資料夾)")
    parser.add_argument("--task_start", type=float, default=1.0,
                        help="Task 任務視窗起始秒數 (預設 1.0s)")
    parser.add_argument("--task_end", type=float, default=3.5,
                        help="Task 任務視窗結束秒數 (預設 3.5s)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 85)
    print(f"🚀 BCI 全局 ERD Topomap 分析系統 開始 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📌 樣式規範: [Session 獨立總覽 | Left\\nMI 與 Right\\nMI 縱軸 | 0.05 頂層半透明斑馬紋 | 灰色虛線]")
    print("=" * 85)

    subjects_to_process = ALL_SUBJECT_IDS if args.all_subjects else [args.subject]
    target_sessions = ("s1", "s2") if args.session == "all" else (args.session,)
    task_range = (args.task_start, args.task_end)

    success_count = 0
    for s_idx, sub_id in enumerate(subjects_to_process):
        disp_name = get_subject_display_name(sub_id)
        order_code = get_subject_two_digit_order(sub_id)
        print(f"\n{'='*30} [{s_idx+1}/{len(subjects_to_process)}] 受試者 ID: {sub_id} ({disp_name}, 序號: {order_code}) {'='*30}")
        
        ok = process_single_subject_pipeline(
            data_dir=args.data_dir,
            raw_sub_id=sub_id,
            output_root=args.output_dir,
            target_sessions=target_sessions,
            is_demo=args.demo,
            task_range=task_range
        )
        if ok:
            success_count += 1

    print("\n" + "=" * 85)
    print(f"🎉 全部 Topomap 處理完畢！成功完成 {success_count}/{len(subjects_to_process)} 位受試者之分析。")
    print(f"📁 圖表與總覽 Grid 已儲存至: {os.path.abspath(args.output_dir)}/<id>/")
    print("=" * 85)

if __name__ == "__main__":
    main()