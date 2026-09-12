"""
MusiCNN (Essentia paper 提供) 將整首歌曲轉換成 embedding，然後計算各首歌曲在空間維度的距離
音訊 -> 梅爾頻譜 -> CNN 模型 -> 高維特徵向量
"""
import json
import os
import numpy as np
import essentia.standard as es
import tensorflow as tf
from scipy.spatial.distance import cosine

# ==========================================
# 核心分析類別 (保持不變)
# ==========================================
class MusicStyleAnalyzer:
    def __init__(self, model_pb, metadata_json):
        # 1. 載入標籤
        with open(metadata_json, 'r') as f:
            self.metadata = json.load(f)
            self.classes = self.metadata['classes']
        
        # 2. 載入模型
        self.sess = self._load_pb_model(model_pb)
        
        # 3. 定義輸入輸出節點
        try:
            self.input_tensor = self.sess.graph.get_tensor_by_name("model/Placeholder:0")
            self.output_embedding = self.sess.graph.get_tensor_by_name("model/dense/BiasAdd:0")
            try:
                self.output_tags = self.sess.graph.get_tensor_by_name("model/Sigmoid:0")
            except KeyError:
                self.output_tags = self.sess.graph.get_tensor_by_name("model/Identity:0")

        except KeyError as e:
            print("❌ 節點名稱錯誤")
            raise e

        # 4. 初始化特徵提取器
        self.feature_extractor = es.TensorflowInputMusiCNN()

    def _load_pb_model(self, pb_path):
        tf.compat.v1.reset_default_graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        with tf.compat.v1.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
            sess = tf.compat.v1.Session(graph=graph)
            return sess

    def analyze_track(self, audio_path):
        print(f"audio_path: {audio_path}")
        # A. 載入音訊
        loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
        audio = loader()
        
        # B. 轉為 Mel Spectrogram
        pool_mels = []
        for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=True):
            pool_mels.append(self.feature_extractor(frame))
        spectrogram = np.array(pool_mels, dtype=np.float32)
        
        # C. 製作 Patch
        PATCH_SIZE = 187
        total_frames = spectrogram.shape[0]
        
        if total_frames < PATCH_SIZE:
             pad_width = PATCH_SIZE - total_frames
             spectrogram = np.pad(spectrogram, ((0, pad_width), (0, 0)), mode='constant')
             total_frames = PATCH_SIZE

        all_activations = []
        all_embeddings = []

        # D. 逐一 Patch 預測
        for i in range(0, total_frames - PATCH_SIZE + 1, PATCH_SIZE):
            patch = spectrogram[i : i + PATCH_SIZE, :]
            patch_input = patch[np.newaxis, :, :]

            try:
                act, emb = self.sess.run(
                    [self.output_tags, self.output_embedding],
                    feed_dict={self.input_tensor: patch_input}
                )
                all_activations.append(act.flatten())
                all_embeddings.append(emb.flatten())
            except Exception:
                continue

        if not all_activations:
            raise ValueError(f"處理失敗: {audio_path}")

        # E. 聚合結果
        mean_activations = np.mean(all_activations, axis=0)
        global_embedding = np.mean(all_embeddings, axis=0)
        
        top_indices = np.argsort(mean_activations)[::-1][:3]
        top_tags = [f"{self.classes[i]}" for i in top_indices] # 簡化顯示，只取標籤名
        
        return {
            # "filename": os.path.basename(audio_path),
            "filename": os.path.basename(os.path.dirname(audio_path)), # 取倒數第二資料夾名稱
            "style_desc": ", ".join(top_tags),
            "embedding": global_embedding,
        }

    def compute_similarity(self, vec1, vec2):
        return 1.0 - cosine(vec1, vec2)

# ==========================================
# 新增：批次分析功能
# ==========================================
def analyze_playlist_consistency(analyzer, file_list, threshold=0.75):
    """
    分析整個列表的風格一致性，並找出離群值
    """
    print(f"🔄 開始分析 {len(file_list)} 首歌曲...")
    results = []
    
    # 1. 提取所有歌曲特徵
    for filepath in file_list:
        try:
            print(f"  -> Processing: {filepath} ...", end="")
            res = analyzer.analyze_track(filepath)
            results.append(res)
            print(f" [OK] ({res['style_desc']})")
        except Exception as e:
            print(f" [Failed] {e}")

    if len(results) < 2:
        print("❌ 有效歌曲少於 2 首，無法進行比較。")
        return

    # 2. 建立相似度矩陣 (Similarity Matrix)
    n = len(results)
    matrix = np.zeros((n, n))
    
    print("\n📊 計算成對相似度矩陣 (Pairwise Similarity):")
    print(f"{'Song Name':<20} | ", end="")
    for i in range(n):
        print(f"S{i+1:<5}", end="") # S1, S2...
    print("\n" + "-" * (20 + 5 * n))

    consistency_scores = [] # 儲存每首歌與其他的平均相似度

    for i in range(n):
        row_sims = [] # 用於計算這首歌的平均分數
        print(f"{results[i]['filename'][:20]:<20} | ", end="")
        match = 0
        for j in range(n):
            if i == j:
                sim = 1.00
            else:
                sim = analyzer.compute_similarity(results[i]['embedding'], results[j]['embedding'])
                row_sims.append(sim)
            
            matrix[i, j] = sim
            
            # 視覺化輸出：低於門檻標紅字 (如果你在支援 ANSI 的終端機)
            if sim < threshold and i != j:
                print(f"\033[91m{sim:5.2f}\033[0m ", end="") # 紅色
            else:
                print(f"{sim:5.2f} ", end="")
                match += 1
        
        # 計算這首歌與其他歌的「相容性」
        avg_consistency = np.mean(row_sims) if row_sims else 0
        consistency_scores.append(avg_consistency)
        print(f"| M: {match:.2f} ", end="")
        print(f"| Avg: {avg_consistency:.2f}")

    # 3. 總結報告
    print("\n" + "="*40)
    print("📋 風格一致性報告 (Report)")
    print("="*40)
    
    global_avg = np.mean([matrix[i, j] for i in range(n) for j in range(i+1, n)])
    print(f"🔹 整體歌單平均相似度: {global_avg:.4f}")
    
    if global_avg > threshold:
        print(f"✅ 整體判定: [通過] - 實驗材料風格一致")
    else:
        print(f"⚠️ 整體判定: [警告] - 風格變異較大")

    # 4. 抓出離群值 (Outliers)
    # 如果某首歌的平均相似度比整體平均低太多 (例如低 10%)
    print("\n🔍 離群值檢測 (Outlier Detection):")
    outliers_found = False
    for i in range(n):
        score = consistency_scores[i]
        # 判定標準：比閾值低，或者比群體平均低 0.05
        if score < threshold: 
            print(f"  ❌ Outlier 警告: [{results[i]['filename']}]")
            print(f"     -> 平均相似度僅 {score:.2f} (主要標籤: {results[i]['style_desc']})")
            print(f"     -> 建議從實驗清單中移除。")
            outliers_found = True
            
    if not outliers_found:
        print("  ✅ 沒有發現明顯的風格離群值。")

# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    MODEL_PATH = "models/msd-musicnn-1.pb"
    JSON_PATH = "models/msd-musicnn-1.json"
    
    # 【在此設定您的實驗歌單】
    song_list = [
        # "songs/old/ChristmasOfAWanderingGhost/song.ogg",
        # "songs/old/GateOfSteiner/song.ogg",
        "songs/new2/ShriekAndOri/song.ogg",
        "songs/new2/ClimbingTheGinsoTree/song.ogg",
        "songs/new2/NowUseTheLight/song.ogg",
        "songs/new2/LumaPools/song.ogg",
        "songs/old/OriAndTheBlindForest/song.ogg",
        # "songs/new2/LightOfNibel/song.ogg",
        # "songs/new2/DashingAndBashing/song.ogg",
        
       
        
        
        # "songs/new2/OriTheme/song.ogg",
        

        # "songs/old/Crystallize/song.ogg",
        # "songs/old/PeaceOfMind/song.ogg",
        # "songs/new/BalatroMainTheme/song.ogg",
        
        # "songs/new/Ember/song.ogg",
        # "songs/new/HeroesOfMightAndMagic4BattleI/song.ogg",
        # "songs/old/KyokouNoHaikyo/song.ogg",

        # "songs/new/Apache/song.ogg",
        # "songs/new/Backpfeifengesicht/song.ogg",
        
        # "songs/new/BeforeHeDies/song.ogg",
        # "songs/new/Exoplanet/song.ogg",
        # "songs/new/ForYourSake/song.ogg",
        
        # "songs/new/IWonder/song.ogg",
        # "songs/new/Juparo/song.ogg",
        # "songs/new/NIGHTBLAZE/song.ogg",
        # "songs/new/Nigredo/song.ogg",
        # "songs/new/Storyteller/song.ogg",
        # "songs/new/Summit/song.ogg",
        # "songs/new/ULT!MATEEND/song.ogg",
        # "songs/new/whatIsThis/song.ogg",
        # "songs/new/YoukaiMountain/song.ogg",
    ]
    
    try:
        # 初始化模型
        analyzer = MusicStyleAnalyzer(MODEL_PATH, JSON_PATH)
        
        # 執行批次分析
        # threshold=0.80 是較嚴格的標準，若不同曲風建議設 0.75
        analyze_playlist_consistency(analyzer, song_list, threshold=0.75)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n發生錯誤: {e}")