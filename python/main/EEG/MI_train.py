"""
20251015 從之前的 MI_train 改過來，把多餘內容刪除
"""
import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import ShallowFBCSPNet
from main.EEG.models import SCCNet
import torch.optim as optim
import main.Utils.config as config
import main.Utils.global_value as global_value
import torch.nn.functional as F


class BraindecodeTrainer:
    def __init__(self, dataset, val_dataset, model_class=ShallowFBCSPNet, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"device: {self.device}")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.ft = ft
        self._load_data()
        self._init_model()

    def _load_data(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # data_shape = [self.dataset.__getitem__(0)[0].shape, self.dataset.__getitem__(0)[1].shape]
        # # model input info
        # self.n_channels = data_shape[0][1]
        # self.input_window_samples = data_shape[0][2]
        # self.n_classes = data_shape[1][0]

    def _init_model(self):
        """初始化可替換的 Braindecode 模型"""
        self.model = self.model_class(
            **self.model_kwargs
        ).to(self.device)
        # summary(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"\n{config.TAGS.INFO.value} Loaded checkpoint from {checkpoint_path}")

    def _evaluate(self, loader):
        self.model.eval()
        total_correct, total_samples, losses = 0, 0, []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float()
                output = self.model(x_batch.squeeze(1))  # 因為輸入的維度是 4 維 (之前自己寫的 class 有多一維)，所以這邊要減 1 維
                # output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                losses.append(loss.item())
                preds = output.argmax(dim=1)
                total_correct += (preds == y_batch.argmax(dim=1)).sum().item()
                total_samples += y_batch.size(0)

        return np.mean(losses), total_correct / total_samples

    def train(self, freeze_layer=False, use_batch_norm=False, tcp_server=None, patience=30):  # outlet=None
        os.makedirs("checkpoints", exist_ok=True)
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        if self.ft:
            tag = "FT"
        else:
            tag = "Train"
        # --- Early Stopping 初始化 ---
        best_val_loss = float('inf')  # 初始最佳 loss 為無限大
        early_stop_counter = 0  # 計數器
        # ---------------------------
        if freeze_layer:  # 只訓練 全連接層 fine tune
            if hasattr(self.model, 'classifier'):
                final_layer = self.model.classifier
            elif hasattr(self.model, 'final_layer'):
                final_layer = self.model.final_layer
            else:
                raise AttributeError("Model does not have 'classifier' or 'final_layer' attribute.")
            # for name, param in self.model.named_parameters():
            #     if "patch_embedding" in name or "transformer" in name:
            #         param.requires_grad = False
            #         print(f"Froze layer: {name}")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in final_layer.final_layer.parameters():
                param.requires_grad = True
            # filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.model.final_layer.parameters(), lr=self.lr, betas=(0.5, 0.999))

        for epoch in range(self.num_epochs):
            self.model.train()

            # --- 如果是 Fine-tuning 與 use_batch_norm，強制鎖定 BatchNorm ---
            if freeze_layer and use_batch_norm:
                # 即使在 model.train() 下，也要讓 BN 保持在 eval 模式，running_mean/var 才不會被新資料洗掉
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                    else:
                        print(f"{config.TAGS.WARNING} model not found")

            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))
                # outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()
            train_loss = running_loss / len(self.dataset)
            train_acc = correct / total
            val_loss, val_acc = self._evaluate(self.val_loader)

            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
            }, f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}{tag.lower()}-epoch{epoch}.pth")

            if tcp_server:
                tcp_server.broadcast(f"Epoch {epoch}/{self.num_epochs} - "
                                     f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                                     f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            # --- Early Stopping 邏輯 ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # 重置計數器
            else:
                early_stop_counter += 1
            # 觸發 Early Stopping
            if early_stop_counter >= patience:
                stop_msg = f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}"
                print(stop_msg)
                if tcp_server:
                    tcp_server.broadcast(stop_msg)
                break

            # print(f"[{tag}] Epoch {epoch}/{self.num_epochs} - "
            #       f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            #       f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        if tcp_server:
            time.sleep(0.5)  # 讓系統可以判斷
            tcp_server.broadcast(config.TRAINING_FINISH_STR)
        return history


def load_shallowfbcsp_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]

    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    params = dict(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        # n_filters_time=40,
        # filter_time_length=25,
        # n_filters_spat=40,
        # pool_time_length=75,
        # pool_time_stride=15,
        # final_conv_length="auto",
        # conv_nonlin=torch.square,  # 常用 square nonlinearity
        # pool_mode="mean",
        # activation_pool_nonlin=torch.log,  # 常用 log nonlinearity
        # split_first_layer=True,
        # batch_norm=True,
        # batch_norm_alpha=0.1,
        # drop_prob=0.5,
        # chs_info=info["chs"],
        # input_window_seconds=input_window_sec,
        # sfreq=sfreq
    )
    return params


def load_sccnet_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    # params = dict(
    #     N=input_window_samples,
    #     C=n_channels,
    #     nb_classes=n_classes,
    # )
    params = dict(
        samples=input_window_samples,
        channels=n_channels,
        n_classes=n_classes,
        sfreq=500,  # 根據設備調整
    )
    return params


class OnlineCalibrationTrainer(BraindecodeTrainer):
    def __init__(self, dataset=None, val_dataset=None, model_class=SCCNet, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False):
        super().__init__(dataset, val_dataset, model_class, model_kwargs,
                         batch_size, num_epochs, lr, device, ft)

        # --- Online Learning 初始化 ---
        # Replay Buffer 改為使用 global_value.replay_buffer（跨 calibration 持久保存）
        # buffer_limit 改為使用 config.REPLAY_BUFFER_LIMIT

        # 線上學習的參數
        self.online_lr = lr
        self.fail_weight = 2.0  # 失敗數據的懲罰權重 2.0，目前權重 1 代表與一般訓練一樣
        # ----------------------------
        print(f"{config.TAGS.INFO.value} [DEBUG] OnlineCalibrationTrainer init: "
              f"buffer_limit={config.REPLAY_BUFFER_LIMIT}, "
              f"buffer_class0={len(global_value.replay_buffer[0])}, "
              f"buffer_class1={len(global_value.replay_buffer[1])}")

    def _load_data(self):
        pass

    def _init_model(self):
        super()._init_model()
        # 必須初始化這個 reduction='none' 的 loss，否則後面會報錯
        self.criterion_none = nn.CrossEntropyLoss(reduction='none')

    def add_to_buffer(self, x, y, is_fail):
        """
        將數據加入全域 Replay Buffer，並執行 FIFO 移除策略
        """
        # 判斷類別
        label_idx = torch.argmax(y).item()

        # 設定權重: 失敗的 trial 權重較高
        weight = self.fail_weight if is_fail else 1.0

        # 初始化 key (如果尚未存在)
        if label_idx not in global_value.replay_buffer:
            global_value.replay_buffer[label_idx] = []

        # 存入全域 Buffer (轉 CPU 以節省 VRAM)
        global_value.replay_buffer[label_idx].append((x.cpu(), y.cpu(), weight))

        # [關鍵邏輯] FIFO: 如果該類別數量超過上限，移除該類別「最舊」的一筆
        if len(global_value.replay_buffer[label_idx]) > config.REPLAY_BUFFER_LIMIT:
            global_value.replay_buffer[label_idx].pop(0)

    def online_train(self, dataset):
        """
        接收 Dataset，unpack 後進行 online training
        dataset: TensorDataset (X, Y_OneHot, Failures)
        """
        # 防呆：如果 dataset 是空的 (例如沒有切出任何 window)
        if dataset is None or len(dataset) == 0:
            print(f"{config.TAGS.WARNING} No data in dataset for online training.")
            return 0.0

        if self.ft:
            tag = "FT"
        else:
            tag = "Train"

        self.model.train()

        # ==========================================
        # 1. 從 Dataset 解包數據
        # ==========================================
        # TensorDataset.tensors 會回傳一個 tuple (tensors[0], tensors[1], tensors[2])
        # 對應到我們剛剛存的 (X, Y, Failures)
        x_new, y_new, failures = dataset.tensors

        # 移動到 Device
        x_new = x_new.to(self.device)
        y_new = y_new.to(self.device)
        # failures 轉成 bool list 或保持 tensor 都可以，這裡配合 add_to_buffer 邏輯
        failures = failures.to(self.device)

        # [除錯] 檢查數據
        print(f"DEBUG: Online Train Input Shape: {x_new.shape}")
        print(
            f"DEBUG: Labels (Class 0): {(torch.argmax(y_new, dim=1) == 0).sum().item()}, Labels (Class 1): {(torch.argmax(y_new, dim=1) == 1).sum().item()}")

        # ==========================================
        # 2. 更新 Buffer (逐筆加入)
        # ==========================================
        # 由於 x_new 已經是 Batch Tensor，我們遍歷它
        count_0, count_1, fail_0, fail_1 = 0, 0, 0, 0
        for i in range(len(x_new)):
            # failures[i] 可能是 tensor(1.) 或 tensor(0.)，轉成 bool 判斷
            label = torch.argmax(y_new[i]).item()
            is_fail = (failures[i].item() > 0.5)
            if label == 0:
                count_0 += 1
                if is_fail:
                    fail_0 += 1
            elif label == 1:
                count_1 += 1
                if is_fail:
                    fail_1 += 1
            self.add_to_buffer(x_new[i], y_new[i], is_fail)
        # ==========================================
        # 3. 準備訓練數據 (全域 Buffer 混合)
        # ==========================================
        train_samples = []
        for label_idx in global_value.replay_buffer:
            train_samples.extend(global_value.replay_buffer[label_idx])

        print(f"{config.TAGS.INFO.value} [DEBUG] online_train: "
              f"buffer_class0={len(global_value.replay_buffer.get(0, []))}, "
              f"buffer_class1={len(global_value.replay_buffer.get(1, []))}, "
              f"total_train_samples={len(train_samples)}")
        if not train_samples: return 0.0

        # Stack 起來
        batch_x = torch.stack([item[0] for item in train_samples])
        batch_y = torch.stack([item[1] for item in train_samples])
        batch_w = torch.tensor([item[2] for item in train_samples], dtype=torch.float32)

        # 建立 DataLoader
        online_dataset = TensorDataset(batch_x, batch_y, batch_w)
        online_loader = DataLoader(online_dataset, batch_size=self.batch_size, shuffle=True)

        # [建議] 針對頑固模型，這裡把 LR 加大
        optimizer = optim.Adam(self.model.parameters(), lr=self.online_lr * 2.0)

        # ==========================================
        # 4. 訓練迴圈
        # ==========================================
        avg_loss, total_loss = 0, 0
        for epoch in range(self.num_epochs):
            for inputs, labels, weights in online_loader:
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).float()
                weights = weights.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))

                # 轉成 Index 再算 Loss
                target_indices = torch.argmax(labels, dim=1)

                loss_per_sample = self.criterion_none(outputs, target_indices)

                # 加權 Loss
                weighted_loss = (loss_per_sample * weights).mean()

                weighted_loss.backward()
                optimizer.step()

                total_loss += weighted_loss.item()

            avg_loss = total_loss / (self.num_epochs * len(online_loader))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{config.EEG_CHECKPOINT_TMP_BASE_FILE}{tag.lower()}-epoch{epoch}.pth")

        print(
            f"{config.TAGS.INFO} Online update finished. Avg Loss: {avg_loss / (self.num_epochs * len(online_loader)):.4f}")

        return total_loss
