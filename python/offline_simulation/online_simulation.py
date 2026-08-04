"""
lr 10^-3
測試，條件如下
* 把模型 online fine tune 上面設定，用 IRB bci data 測試 fine tune 的參數: 
    * buffer size (10, 20, 40)
    * update 頻率 (4, 8)
    * epoch (4, 8, 16, 32)
    * batch (8, 16, 32)
    * lr: 10^{-4}
* 可以測試看看有和沒有 validation set，等測試完上面比較好的組合後，可以試試看比較好的參數設計，MI Train OnlineCalibrationTrainer 修改。
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.stats import mode
import random
import copy
from functools import wraps
from datetime import datetime
import sys
# 引入您環境中既有的模組 (請確保路徑正確)
from MI_train import OnlineCalibrationTrainer
from torch.utils.data import TensorDataset
from Models import SCCNet
from MI_train import load_sccnet_params
# 引入全域變數與設定，以便管理 Buffer
import global_value as global_value
import config as config
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    print(f"[{config.TAGS.INFO if hasattr(config, 'TAGS') else 'INFO'}] Setting random seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多 GPU
    cudnn.deterministic = True       # 確保卷積運算具備確定性
    cudnn.benchmark = False          # 關閉自動尋找最優卷積演算法 (這會引入隨機性)
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
        log_file = f"joint_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
            print(f"✅ 輸出已保存到 {log_file}")
            return result
        return wrapper
    return decorator

# ==========================================
# 1. 定義與原本一致的前處理工具
# ==========================================
def bandpass(data, fs=500, low=1, high=40):
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0) 

def prepare_buffer_dataset(x_buffer, y_buffer, correctness_buffer, segment_len=500, stride=100):
    """
    將最新收集到的 Trial 轉換為 TensorDataset (包含 failures 權重判斷)
    """
    augmented_segments = []
    augmented_labels = []
    augmented_failures = [] 
    
    for i in range(len(x_buffer)):
        x_trial = x_buffer[i] # (channels, samples)
        label = y_buffer[i]
        is_fail = not correctness_buffer[i] # 如果這把預測錯誤，就是 failure
        
        x_trial_t = x_trial.T 
        
        if x_trial_t.shape[0] < segment_len:
            continue
            
        for start in range(0, x_trial_t.shape[0] - segment_len + 1, stride):
            data_window = x_trial_t[start:start + segment_len, :]
            window_filtered = bandpass(data_window - np.mean(data_window, axis=0, keepdims=True))
            augmented_segments.append(window_filtered)
            augmented_labels.append(label)
            # 寫入 failure 以配合 OnlineCalibrationTrainer 的加權機制
            augmented_failures.append(1.0 if is_fail else 0.0)
            
    data_x = np.transpose(np.stack(augmented_segments), (0, 2, 1))
    X = torch.tensor(data_x).unsqueeze(1).float() 
    y = F.one_hot(torch.tensor(augmented_labels).long(), num_classes=2).float() 
    f = torch.tensor(augmented_failures).float() 
    
    return TensorDataset(X, y, f)

# ==========================================
# 2. 核心模擬函數
# ==========================================
def simulate_online_experiment(subject_id, session_name, condition, base_dir, pretrain_model_path, params, device='cuda', update_freq=4, num_epochs=4, batch_size=8, lr=1e-4, use_val=True):
    print(f"\n[{subject_id} - {session_name}] 開始模擬 | Condition: {condition}")
    
    # 【關鍵】重置全域 Replay Buffer 並設定上限，避免跨受試者污染
    global_value.replay_buffer = {0: [], 1: []}
    config.REPLAY_BUFFER_LIMIT = 320
    
    # 載入 Pretrain Model
    model = SCCNet(**params).to(device)
    model.load_state_dict(torch.load(pretrain_model_path, map_location=device)['model_state_dict'])
    
    # 初始化 Online Trainer
    trainer = OnlineCalibrationTrainer(
        model_class=SCCNet,
        model_kwargs=params,
        batch_size=batch_size,   
        num_epochs=num_epochs,  
        lr=lr,                   
        ft=True,
        use_val=use_val      
    )
    trainer.model = model 
    
    # 僅存放「最近 4 個 trial」的新資料
    recent_x_buffer = []
    recent_y_buffer = []
    recent_correctness_buffer = []
    
    run_accuracies = []
    
    for run_idx in range(1, 8):
        run_path = os.path.join(base_dir, str(subject_id), session_name, f"run{run_idx}", "mi.pt")
        if not os.path.exists(run_path):
            print(f"  [Warning] 找不到檔案: {run_path}")
            continue
            
        data = torch.load(run_path)
        x_run = data['x_data'] 
        y_run = data['y_data'] 
        
        is_adaptive_run = True if condition == 'B' else (run_idx <= 3)
        run_correct_count = 0
        
        recent_N_preds = []
        recent_N_labels = []
        
        for trial_idx in range(len(y_run)):
            x_trial = x_run[trial_idx] 
            y_trial = y_run[trial_idx]
            
            segment_len = 500
            stride = 100
            x_trial_t = x_trial.T 
            
            windows = []
            for start in range(0, x_trial_t.shape[0] - segment_len + 1, stride):
                data_window = x_trial_t[start:start + segment_len, :]
                window_filtered = bandpass(data_window - np.mean(data_window, axis=0, keepdims=True))
                windows.append(window_filtered)
                
            windows_np = np.stack(windows) 
            windows_np = np.transpose(windows_np, (0, 2, 1)) 
            windows_tensor = torch.tensor(windows_np).float().to(device) 
            
            trainer.model.eval()
            with torch.no_grad():
                outputs = trainer.model(windows_tensor)
                preds = outputs.argmax(dim=1).cpu().numpy()
            
            final_pred = mode(preds, keepdims=False)[0]
            
            is_correct = (final_pred == y_trial)
            if is_correct:
                run_correct_count += 1
                
            # 收集這一次的資料
            recent_N_preds.append(final_pred)
            recent_N_labels.append(y_trial)
            
            # 加入 Recent Buffer (只收集這一次的 4 把)
            recent_x_buffer.append(x_trial)
            recent_y_buffer.append(y_trial)
            recent_correctness_buffer.append(is_correct)
            
            # 每 4 個 Trial 判斷是否需要更新
            if (trial_idx + 1) % update_freq == 0:
                # 1. 計算近 N 次的準確率
                recent_acc = np.mean(np.array(recent_N_preds) == np.array(recent_N_labels))
                
                # 2. 計算目前這個 Run 累積到現在的整體準確率
                current_run_acc = run_correct_count / (trial_idx + 1)
                
                # 3. 判斷是否更新 (目前這 Run < 0.75 且 近 4 次 < 0.75 才會更新)
                if is_adaptive_run and recent_acc < 0.75 and current_run_acc < 0.75:
                    
                    # 將新資料包裝成 Dataset
                    update_dataset = prepare_buffer_dataset(recent_x_buffer, recent_y_buffer, recent_correctness_buffer)
                    # 進行更新 (內部會自動加入 global_value.replay_buffer 並限制數量)
                    loss = trainer.online_train(update_dataset)
                    
                    # 接續上一次的模型：載入剛剛儲存下來的最新 epoch 權重 (ft-epoch3.pth)
                    # latest_checkpoint = f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}ft-epoch{trainer.num_epochs-1}.pth"
                    # 找最好的模型
                    latest_checkpoint = f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}ft-best.pth"
                    if os.path.exists(latest_checkpoint):
                        trainer.model.load_state_dict(torch.load(latest_checkpoint, map_location=device)['model_state_dict'])
                    
                    print(f"    [Update] Run {run_idx} Trial {trial_idx+1} | 近4次 Acc: {recent_acc:.2f} | 目前 Run Acc: {current_run_acc:.2f} | Loss: {loss:.4f}")
                
                # 如果沒有更新
                # elif is_adaptive_run:
                #     print(f"    [Skip] Run {run_idx} Trial {trial_idx+1} | 近4次 Acc: {recent_acc:.2f} | 目前 Run Acc: {current_run_acc:.2f} (大於等於 75% 不更新)")

                # 清空 4 個 trial 的暫存器，準備收集下一輪
                recent_x_buffer.clear()
                recent_y_buffer.clear()
                recent_correctness_buffer.clear()
                recent_N_preds.clear()
                recent_N_labels.clear()
                
        run_acc = run_correct_count / len(y_run)
        run_accuracies.append(run_acc)
        print(f"  --> Run {run_idx} 結束 | 模式: {'Adaptive' if is_adaptive_run else 'Static'} | 準確率: {run_acc:.3f}")
        
    return run_accuracies
# ==========================================
# 3. 主程式執行迴圈
# ==========================================
@tee_log(f"online_simulation_24_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
# ==========================================
# 3. 主程式執行迴圈與表格輸出
# ==========================================
def main():
    # 替換為您的 Pretrain Model 檔案路徑
    # best_model_24_pilot_Train, best_model_24_Train, best_model_noise_Train
    pretrain_model_path = "checkpoints/best_model_24_Train.pth"
    set_seed(42)
    # 完整的受試者名單與 condition
    # raw_data_13 = {
    #     35: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     37: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    # }
    # raw_data_13 = {
    #     35: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     37: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     38: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     40: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     41: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     42: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     43: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     44: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     45: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     47: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     48: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     50: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     51: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     52: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     54: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     55: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     57: {'s1': {'cond': 'B'}, 's2': {'cond': 'A'}},
    #     58: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     63: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     64: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     65: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     68: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     69: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}},
    #     70: {'s1': {'cond': 'A'}, 's2': {'cond': 'B'}}
    # }
    raw_data_13 = {
        35: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        37: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        38: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        40: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        41: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        42: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        43: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        44: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        45: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        47: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        48: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        50: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        51: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        52: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        54: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        55: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        57: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        58: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        63: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        64: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        65: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        68: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        69: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}},
        70: {'s1': {'cond': 'B'}, 's2': {'cond': 'B'}}
    }
    
    base_dir = r"/mnt/middle/tmp/2025/MIEXP/DATA_Cygnus"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    params = dict(
        samples=500,
        channels=len(channel_index),
        n_classes=2,
        sfreq=500,  # 根據設備調整
    )
    test_update_freqs = [4, 8] #  [4, 8] 
    test_num_epochs = [4, 8, 16, 32] # [4, 8, 16, 32]
    test_batch_sizes = [8, 16, 32] #  [8, 16, 32] 
    test_use_vals = [True, False]
    for update_freq in test_update_freqs:
        for num_epoch in test_num_epochs:
            for batch_size in test_batch_sizes:
                for use_val in test_use_vals:
                    results = {}
                    
                    for sub_id, sessions in raw_data_13.items():
                        results[sub_id] = {}
                        for session_name, session_info in sessions.items():
                            condition = session_info['cond']
                            
                            run_accs = simulate_online_experiment(
                                subject_id=sub_id,
                                session_name=session_name,
                                condition=condition,
                                base_dir=base_dir,
                                pretrain_model_path=pretrain_model_path,
                                params=params,
                                device=device,
                                update_freq=update_freq,      # 測試 (4, 8)
                                num_epochs=num_epoch,      # 測試 (4, 8, 16, 32)
                                batch_size=batch_size,      # 測試 (8, 16, 32)
                                lr=1e-3,            # 測試 lr
                                use_val=use_val        # 切換 True / False 測試 Val Set 差異
                            )
                            
                            results[sub_id][session_name] = {
                                'cond': condition,
                                'simulated_run_acc': run_accs
                            }
                            
                    print(f"\n========== 模擬完成 update_freq:{update_freq}, num_epoch: {num_epoch}, batch_size: {batch_size}, use_val: {use_val}==========\n")
                    
                    # 輸出 Markdown 表格
                    print("| **id** | **online acc (run)** | **condition** |")
                    print("| ------ | -------------------- | ------------- |")
                    
                    for sub, sess_data in results.items():
                        # s1, s2 依序輸出
                        for sess in ['s1', 's2']:
                            if sess in sess_data:
                                data = sess_data[sess]
                                sess_num = "1" if sess == 's1' else "2"
                                display_id = f"{sub} ({sess_num})"
                                cond = data['cond']
                                
                                # 將所有 run 的 accuracy 轉成小數點後三位的字串列表
                                accs = data['simulated_run_acc']
                                acc_str = "[" + ", ".join([f"{a:.3f}" for a in accs]) + "]"
                                
                                print(f"| {display_id} | {acc_str} | {cond} |")


if __name__ == "__main__":
    main()