import os
import requests

def download_file(url, save_path):
    print(f"正在下載: {url} ...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 下載成功: {save_path}")
    except Exception as e:
        print(f"❌ 下載失敗: {e}")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    # 這是完整的骨幹模型 (Backbone)，可以吃 Spectrogram
    base_url = "https://essentia.upf.edu/models/feature-extractors/musicnn/"
    files = [
        "msd-musicnn-1.pb",   # 模型檔
        "msd-musicnn-1.json"  # 標籤檔 (包含 50 個風格標籤，如 rock, jazz, but also happy, fast 等)
    ]

    for file_name in files:
        url = base_url + file_name
        save_path = os.path.join("models", file_name)
        if not os.path.exists(save_path):
            download_file(url, save_path)
        else:
            print(f"檔案已存在: {save_path}")