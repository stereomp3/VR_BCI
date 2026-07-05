import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
from scipy import signal
from PIL import Image
# XBrainLab 基礎類別引用
from XBrainLab.visualization.base import Visualizer

def merge_all_subjects_to_grid():
    ids = ["35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]
    sessions = ["s1", "s2"]
    is_13 = True
    runs = [f"run{i}" for i in range(1, 8)]  # run1 ~ run7
    classes = [0, 1]

    # 設定縮放比例避免大圖導致記憶體不足 (例如 0.2 代表長寬縮小為原本的 20%)
    scale_factor = 0.25 
    
    # 關閉 PIL 圖片尺寸限制警告
    Image.MAX_IMAGE_PIXELS = None 

    os.makedirs("Grid_Outputs", exist_ok=True)

    for target_class in classes:
        for session in sessions:
            print(f"🖼️ 正在合併大圖：Session {session}, Label {target_class} ...")
            
            # 1. 先隨便找一張存在的圖來獲取基準尺寸
            sample_w, sample_h = None, None
            for sub in ids:
                if is_13:
                    sample_path = os.path.join(f"Saliency_13", sub, session, f"combined_output_{target_class}", f"Sub{sub}_{session}_run1_c{target_class}_combined.png")
                else:
                    sample_path = os.path.join(f"Saliency", sub, session, f"combined_output_{target_class}", f"Sub{sub}_{session}_run1_c{target_class}_combined.png")
                if os.path.exists(sample_path):
                    with Image.open(sample_path) as img:
                        sample_w, sample_h = img.size
                    break
            
            if sample_w is None:
                print(f"⚠️ 找不到任何 Session {session} Label {target_class} 的圖片，跳過。")
                continue

            # 計算縮放後的單格長寬
            tw = int(sample_w * scale_factor)
            th = int(sample_h * scale_factor)

            # 2. 建立 7 (col) x 24 (row) 的白色大畫布
            grid_w = 7 * tw
            grid_h = 24 * th
            canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

            # 3. 將圖片依序貼上
            for row_idx, subject_id in enumerate(ids):
                if is_13:
                    output_dir = os.path.join(f"Saliency_13", subject_id, session, f"combined_output_{target_class}")
                else:
                    output_dir = os.path.join(f"Saliency", subject_id, session, f"combined_output_{target_class}")
                
                # 找出該 subject 在這個 session 實際擁有的 run 圖片
                available_imgs = []
                for run in runs:
                    img_path = os.path.join(output_dir, f"Sub{subject_id}_{session}_{run}_c{target_class}_combined.png")
                    if os.path.exists(img_path):
                        available_imgs.append(img_path)
                
                 # 【關鍵邏輯】決定貼上畫布的欄位位置 (target_col)
                for idx, img_path in enumerate(available_imgs):
                    if len(available_imgs) == 6:
                        # 只有 6 張圖的情況：將空缺強制安排在 index 2 (第3格)
                        # idx 0, 1 -> 欄位 0, 1
                        # idx 2, 3, 4, 5 -> 欄位 3, 4, 5, 6
                        target_col = idx if idx < 2 else idx + 1
                    else:
                        # 7 張圖或其他數量，直接依序排列
                        target_col = idx 
                        
                    try:
                        with Image.open(img_path) as img:
                            img_resized = img.resize((tw, th), Image.Resampling.LANCZOS)
                            canvas.paste(img_resized, (target_col * tw, row_idx * th))
                    except Exception as e:
                        print(f"載入圖片失敗 {img_path}: {e}")
            # 4. 儲存該 Session/Label 的最終大圖
            if is_13:
                out_name = os.path.join("Grid_Outputs", f"AllSubjects_{session}_Label{target_class}.png")
            else:
                out_name = os.path.join("Grid_Outputs", f"22_AllSubjects_{session}_Label{target_class}.png")
            canvas.save(out_name)
            print(f"✅ 成功儲存：{out_name}")


def main():
    merge_all_subjects_to_grid()

if __name__ == "__main__":
    main()