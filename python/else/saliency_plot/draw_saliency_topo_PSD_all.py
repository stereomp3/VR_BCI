"""
自動合併全部 24 位受試者為單一總覽大圖 (AllSubjects Grid)
設定：
1. 畫布規格：7 欄 (Run 1~7) x 24 列 (Subject 01~24)
2. 頂部標註：Run 1~7 與 Mu / Beta 頻帶標籤
3. 左側標註：Subject 01~24 兩位數序號
4. 視覺層次：奇數 Run 欄位覆蓋 0.05 深灰半透明遮罩，文字與水平虛線位於最頂層
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 跨平台字體載入工具函式
# ==========================================
def get_serif_font(size):
    """嘗試載入襯線體 (Times New Roman / DejaVuSerif)，失敗則退回預設字型"""
    font_candidates = [
        # Windows
        "times.ttf", "timesbd.ttf", "arial.ttf",
        # Linux / Ubuntu / Docker
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        # macOS
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

# ==========================================
# 繪製橫向虛線工具函式
# ==========================================
def draw_horizontal_dashed_line(draw, y, total_width, dash_len=24, gap_len=14, line_width=4, color=(195, 195, 195)):
    x = 0
    while x < total_width:
        x_end = min(x + dash_len, total_width)
        draw.line([(x, y), (x_end, y)], fill=color, width=line_width)
        x += dash_len + gap_len

# ==========================================
# 主合併排版程式 (全部受試者彙整版)
# ==========================================
def merge_all_subjects_to_grid():
    ids = [
        "35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", 
        "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"
    ]
    sessions = ["s1", "s2"]
    is_13 = False  # False: 22 通道, True: 13 通道
    runs = [f"run{i}" for i in range(1, 8)]
    classes = [0, 1]

    # 設定縮放比例以節省記憶體並加快合成
    scale_factor = 0.25 
    Image.MAX_IMAGE_PIXELS = None 
    os.makedirs("Grid_Outputs", exist_ok=True)

    # 1. 自動偵測來源資料夾
    possible_folders = [
        "Saliency_13_Raw" if is_13 else "Saliency_22_Raw",
        "Saliency_13" if is_13 else "Saliency"
    ]
    folder = next((f for f in possible_folders if os.path.exists(f)), possible_folders[0])

    for target_class in classes:
        for session in sessions:
            print(f"\n🖼️ 正在合併大圖：Session {session.upper()}, Label {target_class} ...")

            # 2. 動態抓取範例圖尺寸
            sample_w, sample_h = None, None
            for sub in ids:
                output_dir = os.path.join(folder, sub, session, f"combined_output_{target_class}")
                for fname in [
                    f"Sub{sub}_{session}_run1_c{target_class}_MEAN_combined.png",
                    f"Sub{sub}_{session}_run1_c{target_class}_combined.png"
                ]:
                    sample_path = os.path.join(output_dir, fname)
                    if os.path.exists(sample_path):
                        with Image.open(sample_path) as img:
                            sample_w, sample_h = img.size
                        break
                if sample_w:
                    break

            if sample_w is None:
                print(f"⚠️ 找不到任何 Session {session} Label {target_class} 的圖片，跳過此組合。")
                continue

            # 計算單格長寬
            tw = int(sample_w * scale_factor)
            th = int(sample_h * scale_factor)

            # 定義標題邊界與畫布整體尺寸 (7 欄 x 24 列)
            header_left_w = int(tw * 0.45)
            header_top_h = int(th * 0.42)

            grid_w = header_left_w + 7 * tw
            grid_h = header_top_h + len(ids) * th

            # 載入字體
            font_id = get_serif_font(int(th * 0.16))    # 左側序號字體
            font_run = get_serif_font(int(th * 0.13))   # Run 1~7 字體
            font_band = get_serif_font(int(th * 0.12))  # Mu / Beta 字體
            font_sess = get_serif_font(int(th * 0.18))  # 左上角 Session 標誌字體

            # 色彩與遮罩設定
            COLOR_WHITE = (255, 255, 255, 255)
            COLOR_TEXT = (0, 0, 0)
            COLOR_LINE = (195, 195, 195)

            # 頂層半透明遮罩：透明度設為 0.05 (約 13/255)
            OVERLAY_ALPHA = int(255 * 0.05)
            OVERLAY_COLOR = (0, 0, 0, OVERLAY_ALPHA)

            # 建立底層畫布 (RGBA)
            canvas = Image.new('RGBA', (grid_w, grid_h), COLOR_WHITE)

            # -------------------------------------------------------------
            # A. 依序貼上各受試者的子圖內容 (排在最下層)
            # -------------------------------------------------------------
            for row_idx, subject_id in enumerate(ids):
                output_dir = os.path.join(folder, subject_id, session, f"combined_output_{target_class}")
                
                # 收集該受試者在該 session 實際擁有的圖片
                available_imgs = []
                for run in runs:
                    candidates = [
                        os.path.join(output_dir, f"Sub{subject_id}_{session}_{run}_c{target_class}_MEAN_combined.png"),
                        os.path.join(output_dir, f"Sub{subject_id}_{session}_{run}_c{target_class}_combined.png")
                    ]
                    found = next((p for p in candidates if os.path.exists(p)), None)
                    if found:
                        available_imgs.append(found)

                if not available_imgs:
                    continue

                for idx, img_path in enumerate(available_imgs):
                    # 6 張圖時將空缺強制排在 Run 3 (index 2)
                    if len(available_imgs) == 6:
                        target_col = idx if idx < 2 else idx + 1
                    else:
                        target_col = idx

                    if target_col >= 7:
                        continue

                    paste_x = header_left_w + target_col * tw
                    paste_y = header_top_h + row_idx * th

                    try:
                        with Image.open(img_path) as img:
                            img_resized = img.resize((tw, th), Image.Resampling.LANCZOS)
                            canvas.paste(img_resized, (paste_x, paste_y))
                    except Exception as e:
                        print(f"載入圖片失敗 {img_path}: {e}")

            # -------------------------------------------------------------
            # B. 覆蓋最上層半透明遮罩 (奇數欄位覆蓋 0.05 深灰)
            # -------------------------------------------------------------
            overlay = Image.new('RGBA', (grid_w, grid_h), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            for col in range(7):
                if col % 2 == 0:  # 奇數欄位 (Run 1, 3, 5, 7)
                    col_x = header_left_w + col * tw
                    overlay_draw.rectangle([col_x, 0, col_x + tw, grid_h], fill=OVERLAY_COLOR)

            # 將半透明遮罩圖層覆蓋在已貼好的底圖之上
            canvas = Image.alpha_composite(canvas, overlay)
            canvas = canvas.convert('RGB')
            draw = ImageDraw.Draw(canvas)

            # -------------------------------------------------------------
            # C. 繪製最頂層文字標籤
            # -------------------------------------------------------------
            # 1. 左上角標註 Session (S1 或 S2)
            draw.text((header_left_w // 2, header_top_h * 0.35), session.upper(), fill=COLOR_TEXT, font=font_sess, anchor="mm")

            # 2. 頂部 Run 1~7 與 Mu / Beta 標籤
            for col in range(7):
                col_x = header_left_w + col * tw
                run_cx = col_x + tw // 2

                # Run X 標籤
                draw.text((run_cx, header_top_h * 0.28), f"Run {col + 1}", fill=COLOR_TEXT, font=font_run, anchor="mm")

                # Mu 與 Beta 標籤
                mu_cx = col_x + int(tw * 0.25)
                draw.text((mu_cx, header_top_h * 0.72), "Mu", fill=COLOR_TEXT, font=font_band, anchor="mm")

                beta_cx = col_x + int(tw * 0.75)
                draw.text((beta_cx, header_top_h * 0.72), "Beta", fill=COLOR_TEXT, font=font_band, anchor="mm")

            # 3. 左側 Subject 序號 (01 ~ 24)
            for row_idx in range(len(ids)):
                order_str = f"{row_idx + 1:02d}"
                row_y = header_top_h + row_idx * th
                sub_cy = row_y + th // 2
                draw.text((header_left_w // 2, sub_cy), order_str, fill=COLOR_TEXT, font=font_id, anchor="mm")

            # -------------------------------------------------------------
            # D. 繪製水平灰色虛線 (位於最頂層)
            # -------------------------------------------------------------
            # Header 與第 1 位受試者之間
            draw_horizontal_dashed_line(draw, header_top_h, grid_w, line_width=4, color=COLOR_LINE)

            # 各受試者之間的橫向虛線分隔
            for row_idx in range(1, len(ids) + 1):
                y_pos = header_top_h + row_idx * th
                draw_horizontal_dashed_line(draw, y_pos, grid_w, line_width=4, color=COLOR_LINE)

            # 3. 儲存大圖
            prefix = "13_" if is_13 else "22_"
            out_name = os.path.join("Grid_Outputs", f"{prefix}AllSubjects_{session}_Label{target_class}_Grid.png")
            canvas.save(out_name, quality=95)
            print(f"✅ 成功儲存：{out_name}")

def main():
    merge_all_subjects_to_grid()

if __name__ == "__main__":
    main()