"""
20251015 從之前的 MI_train 改過來，把多餘內容刪除
"""
import os
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from braindecode.models import ShallowFBCSPNet
import torch.optim as optim
import main.Utils.config as config


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

    def train(self, freeze_layer=False, tcp_server=None, patience=10):  # outlet=None
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

            # --- 如果是 Fine-tuning，強制鎖定 BatchNorm ---
            if freeze_layer:
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
