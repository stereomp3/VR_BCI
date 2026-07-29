import os
from PIL import Image

# ==========================================
# 修改後的合併圖片功能：以 Subject 為單位，包含 s1 與 s2，特定空缺邏輯
# ==========================================
def merge_subjects_to_grid():
    ids = [
        "35", "37", "38", "40", "41", "42", "43", "44", "45", "47", "48", "50", 
        "51", "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"
    ]
    sessions = ["s1", "s2"]
    is_13 = True
    runs = [f"run{i}" for i in range(1, 8)]  # run1 ~ run7
    classes = [0, 1]

    # 設定縮放比例避免大圖導致記憶體不足
    scale_factor = 0.25 
    Image.MAX_IMAGE_PIXELS = None 

    # 1. 取得基準尺寸 (從任何存在的檔案中抓取一次即可)
    sample_w, sample_h = None, None
    for sub in ids:
        for sess in sessions:
            for cls in classes:
                folder = f"Saliency_13" if is_13 else f"Saliency"
                sample_path = os.path.join(folder, sub, sess, f"combined_output_{cls}", f"Sub{sub}_{sess}_run1_c{cls}_combined.png")
                if os.path.exists(sample_path):
                    with Image.open(sample_path) as img:
                        sample_w, sample_h = img.size
                    break
            if sample_w: break
        if sample_w: break
        
    if not sample_w:
        print("⚠️ 找不到任何圖片來決定畫布尺寸，請確認路徑與圖片是否存在！")
        return
        
    # 計算縮放後的單格長寬
    tw = int(sample_w * scale_factor)
    th = int(sample_h * scale_factor)

    # 2. 針對每個 Subject 建立專屬資料夾並處理
    for subject_id in ids:
        # 建立 Grid_Outputs/{id}/ 資料夾
        out_dir = os.path.join("Grid_Outputs", subject_id)
        os.makedirs(out_dir, exist_ok=True)
        
        for target_class in classes:
            print(f"🖼️ 正在處理：Subject {subject_id}, Label {target_class} ...")
            
            # 建立 7 (欄位/run) x 2 (列/session) 的白色大畫布
            grid_w = 7 * tw
            grid_h = len(sessions) * th
            canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
            
            has_images = False  # 紀錄這個畫布是否有貼上任何圖片
            
            # 依序處理 Session 1 (row_idx=0) 與 Session 2 (row_idx=1)
            for row_idx, session in enumerate(sessions):
                folder = f"Saliency_13" if is_13 else f"Saliency"
                output_dir = os.path.join(folder, subject_id, session, f"combined_output_{target_class}")
                
                # 找出該 Session 實際擁有的 run 圖片
                available_imgs = []
                for run in runs:
                    img_path = os.path.join(output_dir, f"Sub{subject_id}_{session}_{run}_c{target_class}_combined.png")
                    if os.path.exists(img_path):
                        available_imgs.append(img_path)
                
                if not available_imgs:
                    continue  # 這個 session 沒圖，跳過此列，留白
                    
                has_images = True
                
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
                        
            # 3. 如果畫布有內容，則儲存到該 Subject 資料夾下
            if has_images:
                prefix = "13_" if is_13 else "22_"
                out_name = os.path.join(out_dir, f"{prefix}Sub{subject_id}_Label{target_class}_Grid.png")
                canvas.save(out_name)
                print(f"✅ 成功儲存：{out_name}")
            else:
                print(f"⚠️ Subject {subject_id} Label {target_class} 沒有任何圖片可合併。")

def main():
    merge_subjects_to_grid()

if __name__ == "__main__":
    main()