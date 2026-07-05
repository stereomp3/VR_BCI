import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np


class SimpleEEGNet(nn.Module):
    def __init__(self, channel, input_samples, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.final_layer = nn.Linear(16, num_classes)

    def forward(self, x):
        # con1d, kernel size 設定為單個，輸入資料會為 3 維，所以遇到 4 維的資料要還原成三維
        if x.dim() > 3:
            x = x.squeeze(1)
        x = self.net(x)
        x = self.final_layer(x)
        return x


# class SCCNet(nn.Module):  # 透過 braindecode 裡面的 SCCNet 修改的，braindecode 程式碼來源為 CECEL 的 XBrainLab (原本的很慢) # 後來沒用
#     def __init__(self, n_chans, n_outputs, n_times,
#                  n_spatial_filters=22,
#                  n_spatial_filters_smooth=20,
#                  drop_prob=0.5):
#         super().__init__()
#
#         # 1. 空間卷積 (Conv1d, 沿 channel 維度學習空間濾波器)
#         self.spatial_conv = nn.Conv1d(
#             in_channels=n_chans,
#             out_channels=n_spatial_filters,
#             kernel_size=1
#         )
#         self.spatial_bn = nn.BatchNorm1d(n_spatial_filters)
#
#         # 2. 時間卷積 (Conv1d, 沿時間維度)
#         self.temporal_conv = nn.Conv1d(
#             in_channels=n_spatial_filters,
#             out_channels=n_spatial_filters_smooth,
#             kernel_size=25,
#             stride=1,
#             padding=12  # same padding
#         )
#         self.temporal_bn = nn.BatchNorm1d(n_spatial_filters_smooth)
#
#         # 3. Dropout
#         self.dropout = nn.Dropout(drop_prob)
#
#         # 4. Global Average Pooling (代替逐步 avgpool)
#         self.global_avgpool = nn.AdaptiveAvgPool1d(1)
#
#         # 5. 全連接層
#         self.final_layer = nn.Linear(n_spatial_filters_smooth, n_outputs)
#
#     def forward(self, x):
#         # 輸入 x: (batch, n_chans, n_times)
#
#         # 1. 空間卷積
#         x = self.spatial_conv(x)  # (batch, n_spatial_filters, n_times)
#         x = self.spatial_bn(x)
#         x = F.relu(x)
#
#         # 2. 時間卷積
#         x = self.temporal_conv(x)  # (batch, n_spatial_filters_smooth, n_times)
#         x = self.temporal_bn(x)
#         x = F.relu(x)
#
#         # 3. Dropout
#         x = self.dropout(x)
#
#         # 4. Global AvgPool
#         x = self.global_avgpool(x)  # (batch, n_spatial_filters_smooth, 1)
#
#         # 5. Flatten + FC
#         x = x.squeeze(-1)  # (batch, n_spatial_filters_smooth)
#         x = self.final_layer(x)  # (batch, n_outputs)
#
#         return x


# class SCCNet(nn.Module):
#     """SCCNet model from Wei et al. 2019. 之前佳城學長寫的
#
#     Parameters
#     ----------
#     C : int
#         Number of EEG input channels.
#     N : int
#         Number of EEG input time samples.
#     nb_classes : int
#         Number of classes to predict.
#     Nu : int, optional
#         Number of spatial kernel (default: C).
#     Nt : int, optional
#         Length of spatial kernel (default: 1).
#     Nc : int, optional
#         Number of spatial-temporal kernel (default: 20).
#     fs : float, optional
#         Sampling frequency of EEG input (default: 1000.0).
#     dropoutRate : float, optional
#         Dropout ratio (default: 0.5).
#     """
#
#     def __init__(self, C, N, nb_classes, Nu=None, Nt=1, Nc=20, fs=1000.0, dropoutRate=0.5):
#         super(SCCNet, self).__init__()
#         self.Nu = Nu if Nu is not None else C
#         self.Nc = Nc
#
#         # Spatial Convolution (across channels)
#         self.conv1 = nn.Conv2d(  # (batch_size, in_channels, height, width)
#             in_channels=1,  # 輸入圖像通道數
#             out_channels=self.Nu,  # 捲基產生通道數量
#             kernel_size=(C, Nt),  # 捲基 kernel
#             padding=0,
#         )
#         self.per = Permute2d(shape=(0, 2, 1, 3))
#         self.bn1 = nn.BatchNorm2d(1)
#
#         # Temporal Convolution (depthwise)
#         self.conv2 = nn.Conv2d(
#             in_channels=1,
#             out_channels=self.Nc,
#             kernel_size=(self.Nu, 12),  # kernel length from paper
#             padding=0,  # 'same' padding for length 12
#         )
#         self.bn2 = nn.BatchNorm2d(self.Nc)
#
#         # Dropout
#         self.dp = nn.Dropout(dropoutRate)
#
#         # Pooling (Average)
#         self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
#
#         # We determine the FC layer input size dynamically
#         dummy_input = torch.zeros(1, 1, C, N)
#
#         with torch.no_grad():
#             dummy_out = self._forward_features(dummy_input)
#             self.feature_dim = dummy_out.shape[1]
#
#         self.final_layer = nn.Linear(self.feature_dim, nb_classes)
#
#     def _forward_features(self, x):
#         # print(f"self.conv1 {x.shape}")
#         x = self.conv1(x)  # Spatial conv # input shape: ([16, 1, 22, 437]), output shape: ([16, 22, 1, 437])
#         # print(x.shape)
#         x = self.per(x)  # permute layer # output shape: ([16, 1, 22, 437])
#         # print(x.shape)
#         x = self.bn1(x)  # output shape: ([16, 1, 22, 437])
#         # print(f"self.conv2 {x.shape}")
#         x = self.conv2(x)  # Temporal conv # input shape: ([16, 22, 1, 437]), output shape: ([16, 20, 1, 426])
#         # print(x.shape)
#         x = self.bn2(x)  # output shape: ([16, 20, 1, 426])
#         # print(x.shape)
#         # x = F.relu(x)  # output shape: ([16, 20, 1, 426])
#         x = x ** 2  # amplify important features and enhance nonlinearity
#         x = self.dp(x)
#         # print(f"pool {x.shape}")
#         x = self.pool(x)  # output shape: ([16, 20, 1, 31])
#         x = torch.log(x)
#         # print(f"out {x.shape}")
#         return x.view(x.size(0), -1)  # x.size(0)
#
#     def forward(self, x):
#         if len(x.shape) < 4:
#             x = x.unsqueeze(1)
#         x = self._forward_features(x)
#         x = self.final_layer(x)
#         return x
#
#
# class Permute2d(nn.Module):
#     def __init__(self, shape):
#         super(Permute2d, self).__init__()
#         self.shape = shape
#
#     def forward(self, x):
#         return torch.permute(x, self.shape)

class SCCNet(nn.Module):
    """Implementation of SCCNet XBrainLab
    https://ieeexplore.ieee.org/document/8716937

    Parameters:
        n_classes: Number of classes.
        channels: Number of channels.
        samples: Number of samples.
        sfreq: Sampling frequency.
        Ns: Number of spatial filters.
    """

    def __init__(self, n_classes, channels, samples, sfreq, Ns=22):
        super().__init__()  # input:bs, 1, channel, sample

        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.octsf = int(math.floor(self.sf * 0.1))

        # (1, n_ch, kernelsize=(n_ch,1))
        self.conv1 = nn.Conv2d(1, Ns, (self.ch, 1))
        self.Bn1 = nn.BatchNorm2d(Ns)  # (n_ch)
        # kernelsize=(1, floor(sf*0.1)) padding= (0, floor(sf*0.1)/2)
        self.conv2 = nn.Conv2d(
            Ns, 20, (1, self.octsf), padding=(0, int(np.ceil((self.octsf - 1) / 2)))
        )
        self.Bn2 = nn.BatchNorm2d(20)

        self.Drop1 = nn.Dropout(0.5)
        # kernelsize=(1, sf/2) revise to 128/2?  stride=(1, floor(sf*0.1))
        self.AvgPool1 = nn.AvgPool2d(
            (1, int(self.sf / 2)), stride=(1, int(self.octsf))
        )
        # (20* ceiling((timepoint-sf/2)/floor(sf*0.1)), n_class)
        self.classifier = nn.Linear(
            (
                    20 *
                    int(
                        (
                                self.tp + (
                                int(np.ceil((self.octsf - 1) / 2)) * 2 - self.octsf + 1
                        ) - int(self.sf / 2)
                        ) / int(self.octsf) + 1
                    )
            ),
            self.n_class, bias=True
        )

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        spX = self.conv1(x)  # (128,22,1,562)

        x = self.Bn1(spX)
        tpX = self.conv2(x)  # (128,20,1,563)

        x = self.Bn2(tpX)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)  # (128,20,1,42)
        x = torch.log(x)
        x = x.view(-1,
                   20 * int((
                                    self.tp + (
                                    int(np.ceil((self.octsf - 1) / 2)) * 2 - self.octsf + 1
                            ) - int(self.sf / 2)) / int(self.octsf) + 1)
                   )
        x = self.classifier(x)

        return x


class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out


# from https://github.com/YuDongPan/SSVEPNet/blob/master/Model/SSVEPNet.py
class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                          max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                       self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes))

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.squeeze(2)
        r_out = self.rnn(out)
        out = self.dense_layers(r_out)
        return out


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)


class EEGNet_SSVEP(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=0),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, n_channels)),
            nn.Dropout(0.5)
        )

        # 利用 dummy input 自動計算 linear 輸入維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            flatten_dim = x.view(1, -1).size(1)

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, n_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.classify(x)
        return x


# ============================FNN======================================
# Legendre basis function generator
def legendre_basis(n, x):
    """Return Legendre basis function up to degree n evaluated at x (torch tensor)."""
    basis = [torch.ones_like(x), x]
    if n == 0:
        return basis[:1]
    elif n == 1:
        return basis
    for k in range(2, n + 1):
        Pk = ((2 * k - 1) * x * basis[-1] - (k - 1) * basis[-2]) / k
        basis.append(Pk)
    return basis


# Functional Convolution Layer
class FunctionalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, basis_dim=5, sample_points=500):
        super().__init__()
        self.basis_dim = basis_dim
        self.sample_points = sample_points
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Legendre basis
        t_grid = torch.linspace(-1, 1, self.sample_points)
        self.register_buffer("t_grid", t_grid)
        self.basis = legendre_basis(self.basis_dim - 1, t_grid)
        self.basis = torch.stack(self.basis)  # [basis_dim, T]
        self.register_buffer("basis_buffer", self.basis)

        # Learnable projection weights
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, basis_dim))  # [O, I, B]

        # Kernel smoothing (simulate integral with convolution kernel)
        self.smooth_kernel = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels,
                                       bias=False)
        with torch.no_grad():
            for p in self.smooth_kernel.parameters():
                p.data.fill_(1.0 / 21.0)

        # Normalization (across T)
        self.norm = nn.LayerNorm(self.sample_points)

    def forward(self, x):
        assert x.ndim == 3, f"Expected input shape [B, C, T], got {x.shape}"
        B, C, T = x.shape

        # Smooth input in time (simulate local linear estimator)
        x = self.smooth_kernel(x)  # [B, C, T]

        # Normalize over time
        x = self.norm(x)  # [B, C, T]

        # Project basis functions using learned weights
        basis = self.basis_buffer.to(x.device)  # [B_dim, T]
        filters = torch.einsum('oib,bt->oit', self.weights, basis)  # [O, I, T]

        # Functional convolution (integral kernel form)
        out = torch.einsum('bct,oit->bot', x, filters)  # [B, C, T] x [O, I, T] -> [B, O, T]

        return F.elu(out)


class FunctionalDenseLayer(nn.Module):
    def __init__(self, in_channels, num_classes, basis_dim=5, sample_points=500):
        super().__init__()
        self.basis_dim = basis_dim
        self.sample_points = sample_points

        t_grid = torch.linspace(-1, 1, self.sample_points)
        self.register_buffer("t_grid", t_grid)
        self.basis = legendre_basis(self.basis_dim - 1, t_grid)
        self.basis = torch.stack(self.basis)  # [basis_dim, T]
        self.register_buffer("basis_buffer", self.basis)

        self.weights = nn.Parameter(torch.randn(num_classes, in_channels, basis_dim))  # [K, C, B]

        # Optional smoothing kernel (simulate integral)
        self.smooth_kernel = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels,
                                       bias=False)
        with torch.no_grad():
            for p in self.smooth_kernel.parameters():
                p.data.fill_(1.0 / 21.0)

        self.norm = nn.LayerNorm(self.sample_points)

    def forward(self, x):
        B, C, T = x.shape
        x = self.smooth_kernel(x)
        x = self.norm(x)

        basis = self.basis_buffer.to(x.device)  # [B_dim, T]
        proj = torch.einsum('cib,bt->cit', self.weights, basis)  # [K, C, T]

        out = torch.einsum('bct,kct->bk', x, proj)  # [B, K]
        return out


# Functional Neural Network (FNN)
class FNN(nn.Module):
    def __init__(self, channel, input_samples, num_classes, basis_dim=5):
        super().__init__()
        self.sample_points = input_samples
        self.basis_dim = basis_dim
        print(f"channel {channel}")
        self.func_conv1 = FunctionalConvLayer(channel, 16, basis_dim, input_samples)
        # self.func_conv1 = FunctionalConvLayer(channel, 20, basis_dim, input_samples)
        # self.func_conv2 = FunctionalConvLayer(20, 10, basis_dim, input_samples)
        self.func_dense = FunctionalDenseLayer(16, num_classes, basis_dim, input_samples)

    def forward(self, x):
        # con1d, kernel size 設定為單個，輸入資料會為 3 維，所以遇到 4 維的資料要還原成三維
        if x.dim() > 3:
            x = x.squeeze(1)  # torch.Size([16, 3, 500]) # batch, channel, sample
        # x: [B, C, T]
        x = self.func_conv1(x)  # -> [B, 20, T]
        # x = self.func_conv2(x)  # -> [B, 10, T]
        x = self.func_dense(x)  # -> [B, num_classes]
        return F.log_softmax(x, dim=1)
