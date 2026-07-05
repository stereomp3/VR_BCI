import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


# from torchinfo import summary


# 自己的模型訓練方法 # 跑在輸入為 (N, 1, C, T) 的資料
class EEGTrainerFineTune:  # train without K-fold
    def __init__(self, model_class, ft_data, savepath="checkpoints", device=None, ft_epochs=100,
                 batch_size=16, lr=1e-4, freeze_layers=False, vl_data=None, ft=False):
        self.model_class = model_class
        self.ft_data = ft_data  # subject-specific fine-tuning dataset
        self.vl_data = vl_data  # subject-specific fine-tuning dataset
        self.savepath = savepath
        os.makedirs(savepath, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lr = lr
        self.ft_epochs = ft_epochs
        self.freeze_layers = freeze_layers
        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()
        self._init_model()
        self.ft = ft

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def _init_model(self):  # if don't use the checkpoint
        data_shape = [self.ft_data.__getitem__(0)[0].shape, self.ft_data.__getitem__(0)[1].shape]
        # channel, samples, class
        self.model = self.model_class(data_shape[0][1], data_shape[0][2], data_shape[1][0]).to(
            self.device)
        # summary(self.model)

    def eval_epoch(self, loader):
        self.model.eval()
        total_correct, total_samples = 0, 0
        epoch_loss = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device, dtype=torch.float), y_batch.to(self.device,
                                                                                          dtype=torch.float)  # float
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                # print(f"output: {output}")
                epoch_loss.append(loss.item())
                total_samples += y_batch.size(0)
                total_correct += (output.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
                # total_correct += (output.argmax(dim=1) == y_batch).sum().item()

        return np.mean(epoch_loss), total_correct / total_samples

    def train(self):
        if self.ft:
            tag = "FT"
        else:
            tag = "Train"
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        loader = DataLoader(self.ft_data, batch_size=self.batch_size, shuffle=True)
        vl_loader = None
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        if self.vl_data is None:
            history = {'loss': [], 'acc': []}
        else:
            vl_loader = DataLoader(self.vl_data, batch_size=self.batch_size, shuffle=True)
            print("#################load vl data#########################")

        if self.freeze_layers:  # 只訓練 全連接層
            # print("[INFO] Freezing conv layers...")
            # for name, param in self.model.named_parameters():
            #     if "conv1" in name or "conv2" in name:
            #         param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.final_layer.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(self.model.final_layer.parameters(), lr=self.lr)
        for ep in range(self.ft_epochs):
            self.model.train()
            correct, total, losses = 0, 0, []

            for x, y in loader:
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)  # float

                # print(f"x.shape: {x.shape}")  # batch size, 1, 4, 1249 (如果資料 < batch size，會直接餵剩下資料數量)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                correct += (out.argmax(dim=1) == y.argmax(dim=1)).sum().item()
                # correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            avg_loss = np.mean(losses)
            acc = correct / total

            history["loss"].append(avg_loss)
            history["acc"].append(acc)

            if self.vl_data is None:
                print(f"[{tag}] Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}")
            else:
                val_loss, val_acc = self.eval_epoch(vl_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(
                    f"[{tag}] Epoch {ep}: loss={avg_loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            torch.save({
                'epoch': ep,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join("checkpoints", f"{tag.lower()}-epoch{ep}.pth"))

        return history


from braindecode.models import EEGConformer
import torch.optim as optim


class BraindecodeTrainer:
    def __init__(self, dataset, val_dataset, model_class=EEGConformer, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self._load_data()
        self.ft = ft
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
        print(f"Loaded checkpoint from {checkpoint_path}")

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
            for param in final_layer.parameters():
                param.requires_grad = True
            # filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        for epoch in range(self.num_epochs):
            self.model.train()

            # --- 如果是 Fine-tuning 與 use_batch_norm，強制鎖定 BatchNorm ---
            if freeze_layer and use_batch_norm:
                # 即使在 model.train() 下，也要讓 BN 保持在 eval 模式，running_mean/var 才不會被新資料洗掉
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

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
            }, os.path.join("checkpoints", f"{tag.lower()}-epoch{epoch}.pth"))

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

            print(f"[{tag}] Epoch {epoch}/{self.num_epochs} - "
                  f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        if tcp_server:
            time.sleep(0.5)  # 讓系統可以判斷
            # tcp_server.broadcast(config.TRAINING_FINISH_STR)
        return history


from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler


class BraindecodeTrainerCV:  # cross validation
    def __init__(self, dataset, val_dataset, model_class=EEGConformer, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False, k_folds=5):
        self.dataset = dataset
        self.val_dataset = val_dataset  # cross validation val dataset is none
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.seed = 42
        self.k_folds = KFold(n_splits=k_folds, shuffle=False)

        # self.k_folds = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)

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
        print(f"Loaded checkpoint from {checkpoint_path}")

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

    def _set_ft(self, freeze_layer=False):
        if freeze_layer:  # 只訓練 全連接層 fine tune
            # for name, param in self.model.named_parameters():
            #     if "patch_embedding" in name or "transformer" in name:
            #         param.requires_grad = False
            #         print(f"Froze layer: {name}")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.final_layer.parameters():
                param.requires_grad = True
            self.optimizer = torch.optim.Adam(self.model.final_layer.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train_kfold(self, freeze_layer=False):
        os.makedirs("checkpoints", exist_ok=True)
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'avg_val_acc': []}
        if freeze_layer:
            tag = "FT"
        else:
            tag = "Train"
        step = 0
        base_num = 0
        for fold, (train_ids, val_ids) in enumerate(self.k_folds.split(self.dataset)):
            print(f"\n=== Fold {fold + 1}/{self.k_folds} ===")
            self._set_seed()
            self._init_model()  # reset the model
            self._set_ft(freeze_layer)  # set fine tune
            # load data
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                           sampler=SubsetRandomSampler(train_ids))
            self.val_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                         sampler=SubsetRandomSampler(val_ids))
            tmp_val_acc = []
            for epoch in range(self.num_epochs):
                self.model.train()
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
                tmp_val_acc.append(val_acc)

                torch.save({
                    'epoch': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss,
                }, os.path.join("checkpoints", f"{tag.lower()}-epoch{step}.pth"))

                print(f"[{tag}] Epoch {epoch}/{self.num_epochs} - "
                      f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                      f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
                step += 1
            best_epoch = np.argmax(tmp_val_acc) + base_num
            base_num += self.num_epochs
            print(f"best_acc: {best_epoch}, acc: {history['acc'][best_epoch]:.4f}, "
                  f"loss: {history['loss'][best_epoch]}, "
                  f"val acc: {history['val_acc'][best_epoch]:.4f}, val loss: {history['val_loss'][best_epoch]}")
            history['avg_val_acc'].append(history['val_acc'][best_epoch])
        print("==================================================================")
        return history


def load_conformer_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    print(n_classes, input_window_samples, n_classes)

    # 可選模型參數，這裡是 EEGConformer 的最佳參數範例
    params = dict(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=input_window_samples,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=2,
        att_heads=5,
        att_drop_prob=0.5,
        final_fc_length='auto'
    )
    return params


#
# def load_sccnet_params(dataset):  # 需要輸入 sample rate 和輸入時長
#     data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
#     # model input info
#     n_channels = data_shape[0][1]
#     input_window_samples = data_shape[0][2]
#     n_classes = data_shape[1][0]
#     params = dict(
#         n_chans=n_channels,
#         n_outputs=n_classes,
#         n_times=input_window_samples,
#         # chs_info=info["chs"],  # 從 mne.Info 取
#         # input_window_seconds=1,
#         # sfreq=500,  # sample rate
#         # n_spatial_filters=22,  # 預設
#         # n_spatial_filters_smooth=20,  # 預設
#         # drop_prob=0.5,
#         # activation=torch.log,  # 預設 Log activation
#         # batch_norm_momentum=0.1  # 常見 momentum
#     )
#     return params


def load_eegnetv4_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    params = dict(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        # chs_info=info["chs"],
        # input_window_seconds=input_window_sec,
        # sfreq=sfreq,
        final_conv_length="auto",
        # pool_mode="mean",
        # F1=8,
        # D=2,
        # F2=None,  # 自動設為 F1 * D
        # kernel_length=64,
        # depthwise_kernel_length=16,
        # pool1_kernel_size=4,
        # pool2_kernel_size=8,
        # conv_spatial_max_norm=1.0,
        # #activation=nn.ELU(),
        # batch_norm_momentum=0.01,
        # batch_norm_affine=True,
        # batch_norm_eps=1e-3,
        # drop_prob=0.25,
        # final_layer_with_constraint=False,
        # norm_rate=0.25
    )
    return params


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


def load_ctnet_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    params = dict(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=input_window_samples,
        # chs_info=chs_info,
        # input_window_seconds=input_window_seconds,
        # sfreq=sfreq,
        # activation_patch=nn.ELU(),
        # activation_transformer=nn.GELU(),
        # drop_prob_cnn=0.3,
        # drop_prob_posi=0.1,
        # drop_prob_final=0.5,
        # heads=4,
        # emb_size=128,  # 常用值，可依資料調整
        # depth=6,
        # n_filters_time=20,
        # kernel_size=64,
        # depth_multiplier=2,
        # pool_size_1=8,
        # pool_size_2=8
    )
    return params


def load_eegnex_params(dataset):
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    # model input info
    n_channels = data_shape[0][1]
    input_window_samples = data_shape[0][2]
    n_classes = data_shape[1][0]
    params = dict(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=input_window_samples,
        # chs_info=chs_info,
        # input_window_seconds=input_window_seconds,
        # sfreq=sfreq,
        # activation=nn.ELU,
        # depth_multiplier=2,
        # filter_1=8,
        # filter_2=32,
        # drop_prob=0.5,
        # kernel_block_1_2=64,  # 需依資料調整，這裡假設和 EEGNet 類似
        # kernel_block_4=(1, 16),
        # dilation_block_4=(1, 2),
        # avg_pool_block4=(1, 4),
        # kernel_block_5=(1, 16),
        # dilation_block_5=(1, 4),
        # avg_pool_block5=(1, 8),
        # max_norm_conv=2.0,  # 常見 max-norm 設定值
        # max_norm_linear=0.5  # 常見 max-norm 設定值
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
