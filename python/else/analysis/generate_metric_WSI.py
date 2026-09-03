"""
ttest_rel: 比較前後的差異 (像是兩天的差異), 配對樣本 t 檢定, 用於 2 session 比較 (paired-samples t-test)
* session 1 vs 2
ttest_ind: 比較兩組的差異 (假設 VAR 一樣), 獨立樣本 t 檢定, 用於 2 condition 比較 (independent-samples t-test)
* condition a vs b: 看 delta a 與 delta b 的差異，也就是兩組別 session 2 - session 1 的差異，透過差異比較。
ttest_1samp: 比較一組 VS 某個值 (像是和隨機猜測機率比較), 單一樣本 t 檢定 (one-sample t-test)
* session 1 vs 2 有沒有大於 0，基本和 rel 差不多

20260518
在 between group 裡面加入 Group 1 Δ Mean (%)Group 2 Δ Mean (%) Mean Difference (%) 的數值
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 參數與資料庫建立
# ==========================================
base_font_size = 15
plt.rcParams.update({'font.size': base_font_size, 'font.family': 'sans-serif'})

# # 整理 24 位受試者的資料
# 13 channel LE 有顯著
# wsi s1 avg: 4.082986111111112, t 3.284683454882381, p 0.0032477157666818903 (ttest_1samp, 跟 0 比)
# wsi s2 avg: 4.591319444444445, t 3.326308989766641, p 0.0029379456244278458 (ttest_1samp, 跟 0 比)
# Learning Effect (LE): 4.87%, t = 2.522, p = 0.0190
# Adaptive Effect (AE): 3.56%, t = 1.950, p = 0.0640
raw_data_13 = {
    35: {'s1': {'cond': 'B', 'on': [0.525, 0.525, 0.6, 0.65, 0.6, 0.6, 0.575],
                'off_run': [0.606, 0.597, 0.602, 0.659, 0.606, 0.605, 0.652], 'off_sess': 0.559},
         's2': {'cond': 'A', 'on': [0.55, 0.5, 0.525, 0.55, 0.575, 0.525, 0.5],
                'off_run': [0.547, 0.547, 0.62, 0.588, 0.573, 0.648, 0.648], 'off_sess': 0.582}},
    37: {'s1': {'cond': 'A', 'on': [0.45, 0.5, 0.6, 0.6, 0.625, 0.95, 0.897],
                'off_run': [0.613, 0.614, 0.83, 0.731, 0.7, 0.994, 0.998], 'off_sess': 0.735},
         's2': {'cond': 'B', 'on': [0.55, 0.9, 0.975, 0.925, 0.95, 0.975],
                'off_run': [0.855, 0.986, 0.994, 0.995, 1.0, 0.998], 'off_sess': 0.965}},
    38: {'s1': {'cond': 'B', 'on': [0.475, 0.5, 0.5, 0.525, 0.5, 0.575, 0.525],
                'off_run': [0.531, 0.634, 0.658, 0.658, 0.566, 0.697, 0.7], 'off_sess': 0.583},
         's2': {'cond': 'A', 'on': [0.6, 0.5, 0.564, 0.425, 0.525, 0.6, 0.575],
                'off_run': [0.559, 0.606, 0.705, 0.681, 0.734, 0.611, 0.567], 'off_sess': 0.575}},
    40: {'s1': {'cond': 'B', 'on': [0.575, 0.525, 0.5, 0.525, 0.6, 0.5, 0.525],
                'off_run': [0.647, 0.655, 0.589, 0.602, 0.58, 0.591, 0.594], 'off_sess': 0.542},
         's2': {'cond': 'A', 'on': [0.55, 0.55, 0.475, 0.55, 0.5, 0.525, 0.59],
                'off_run': [0.58, 0.597, 0.602, 0.569, 0.562, 0.631, 0.634], 'off_sess': 0.569}},
    41: {'s1': {'cond': 'B', 'on': [0.45, 0.575, 0.6, 0.8, 0.675, 0.7, 0.625],
                'off_run': [0.6, 0.686, 0.636, 0.731, 0.728, 0.767, 0.67], 'off_sess': 0.682},
         's2': {'cond': 'A', 'on': [0.6, 0.725, 0.725, 0.65, 0.775, 0.575, 0.625],
                'off_run': [0.681, 0.614, 0.686, 0.697, 0.686, 0.623, 0.628], 'off_sess': 0.639}},
    42: {'s1': {'cond': 'A', 'on': [0.6, 0.55, 0.5, 0.425, 0.5, 0.55, 0.525],
                'off_run': [0.577, 0.609, 0.583, 0.584, 0.608, 0.592, 0.569], 'off_sess': 0.559},
         's2': {'cond': 'B', 'on': [0.575, 0.55, 0.5, 0.525, 0.625, 0.5, 0.5],
                'off_run': [0.598, 0.556, 0.636, 0.573, 0.614, 0.645, 0.609], 'off_sess': 0.542}},
    43: {'s1': {'cond': 'A', 'on': [0.5, 0.5, 0.525, 0.425, 0.525, 0.625, 0.475],
                'off_run': [0.569, 0.575, 0.617, 0.614, 0.661, 0.752, 0.733], 'off_sess': 0.661},
         's2': {'cond': 'B', 'on': [0.45, 0.675, 0.525, 0.55, 0.675, 0.65, 0.625],
                'off_run': [0.656, 0.636, 0.617, 0.697, 0.611, 0.725, 0.683], 'off_sess': 0.658}},
    44: {'s1': {'cond': 'B', 'on': [0.675, 0.725, 0.8, 0.975, 0.975, 0.9, 0.875],
                'off_run': [0.766, 0.869, 0.784, 0.841, 0.855, 0.83, 0.833], 'off_sess': 0.847},
         's2': {'cond': 'A', 'on': [0.475, 0.825, 0.925, 0.875, 0.825, 0.875],
                'off_run': [0.809, 0.87, 0.892, 0.87, 0.928, 0.895], 'off_sess': 0.874}},
    45: {'s1': {'cond': 'B', 'on': [0.425, 0.425, 0.575, 0.525, 0.5, 0.625, 0.45],
                'off_run': [0.597, 0.675, 0.708, 0.658, 0.68, 0.616, 0.72], 'off_sess': 0.615},
         's2': {'cond': 'A', 'on': [0.6, 0.8, 0.85, 0.75, 0.925, 0.9],
                'off_run': [0.637, 0.841, 0.947, 0.881, 0.98, 0.978], 'off_sess': 0.83}},
    47: {'s1': {'cond': 'B', 'on': [0.675, 0.65, 0.6, 0.625, 0.5, 0.475, 0.575],
                'off_run': [0.681, 0.614, 0.689, 0.652, 0.619, 0.591, 0.656], 'off_sess': 0.65},
         's2': {'cond': 'A', 'on': [0.675, 0.525, 0.5, 0.575, 0.55, 0.55, 0.525],
                'off_run': [0.598, 0.584, 0.592, 0.577, 0.586, 0.672, 0.647], 'off_sess': 0.572}},
    48: {'s1': {'cond': 'B', 'on': [0.525, 0.475, 0.45, 0.45, 0.475, 0.475, 0.575],
                'off_run': [0.656, 0.656, 0.631, 0.637, 0.667, 0.598, 0.561], 'off_sess': 0.589},
         's2': {'cond': 'A', 'on': [0.55, 0.525, 0.5, 0.65, 0.525, 0.5, 0.45],
                'off_run': [0.614, 0.581, 0.595, 0.611, 0.641, 0.589, 0.597], 'off_sess': 0.535}},
    50: {'s1': {'cond': 'A', 'on': [0.5, 0.65, 0.5, 0.625, 0.775, 0.65, 0.575],
                'off_run': [0.7, 0.583, 0.636, 0.752, 0.889, 0.798, 0.8], 'off_sess': 0.715},
         's2': {'cond': 'B', 'on': [0.7, 0.7, 0.525, 0.55, 0.6, 0.75, 0.825],
                'off_run': [0.692, 0.759, 0.614, 0.686, 0.811, 0.92, 0.945], 'off_sess': 0.718}},
    51: {'s1': {'cond': 'B', 'on': [0.425, 0.35, 0.5, 0.55, 0.45, 0.575, 0.5],
                'off_run': [0.547, 0.684, 0.714, 0.616, 0.655, 0.669, 0.664], 'off_sess': 0.61},
         's2': {'cond': 'A', 'on': [0.775, 0.475, 0.575, 0.625, 0.55, 0.666],  # 0.666 tmp
                'off_run': [0.658, 0.68, 0.587, 0.642, 0.594, 0.666], 'off_sess': 0.613}},
    52: {'s1': {'cond': 'B', 'on': [0.5, 0.475, 0.425, 0.5, 0.5, 0.45, 0.475],
                'off_run': [0.586, 0.645, 0.583, 0.614, 0.608, 0.619, 0.514], 'off_sess': 0.547},
         's2': {'cond': 'A', 'on': [0.575, 0.6, 0.475, 0.575, 0.625, 0.6, 0.45],
                'off_run': [0.559, 0.566, 0.68, 0.609, 0.63, 0.562, 0.609], 'off_sess': 0.565}},
    54: {'s1': {'cond': 'B', 'on': [0.55, 0.525, 0.525, 0.55, 0.625, 0.55, 0.525],
                'off_run': [0.623, 0.662, 0.63, 0.63, 0.678, 0.681, 0.7], 'off_sess': 0.632},
         's2': {'cond': 'A', 'on': [0.475, 0.75, 0.65, 0.5, 0.7, 0.575],
                'off_run': [0.641, 0.683, 0.731, 0.658, 0.78, 0.811], 'off_sess': 0.708}},
    55: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.55, 0.55, 0.65, 0.55, 0.55],
                'off_run': [0.581, 0.583, 0.592, 0.688, 0.614, 0.664, 0.63], 'off_sess': 0.589},
         's2': {'cond': 'B', 'on': [0.45, 0.45, 0.475, 0.475, 0.55, 0.475, 0.5],
                'off_run': [0.519, 0.566, 0.586, 0.602, 0.672, 0.609, 0.577], 'off_sess': 0.572}},
    57: {'s1': {'cond': 'B', 'on': [0.525, 0.65, 0.55, 0.5, 0.625, 0.55, 0.525],
                'off_run': [0.572, 0.623, 0.588, 0.595, 0.577, 0.641, 0.623], 'off_sess': 0.598},
         's2': {'cond': 'A', 'on': [0.625, 0.5, 0.525, 0.5, 0.525, 0.525, 0.525],
                'off_run': [0.655, 0.58, 0.631, 0.673, 0.588, 0.583, 0.602], 'off_sess': 0.55}},
    58: {'s1': {'cond': 'A', 'on': [0.65, 0.475, 0.575, 0.59, 0.725, 0.5, 0.6],
                'off_run': [0.669, 0.689, 0.597, 0.72, 0.795, 0.661, 0.627], 'off_sess': 0.624},
         's2': {'cond': 'B', 'on': [0.6, 0.65, 0.525, 0.725, 0.825, 0.675, 0.8],
                'off_run': [0.75, 0.75, 0.728, 0.794, 0.792, 0.894, 0.936], 'off_sess': 0.803}},
    63: {'s1': {'cond': 'A', 'on': [0.625, 0.7, 0.55, 0.525, 0.5, 0.65, 0.675],
                'off_run': [0.592, 0.627, 0.633, 0.73, 0.628, 0.559, 0.664], 'off_sess': 0.578},
         's2': {'cond': 'B', 'on': [0.5, 0.775, 0.725, 0.625, 0.7, 0.7],
                'off_run': [0.608, 0.762, 0.798, 0.914, 0.883, 0.814], 'off_sess': 0.764}},
    64: {'s1': {'cond': 'A', 'on': [0.55, 0.55, 0.5, 0.45, 0.575, 0.615, 0.675],
                'off_run': [0.586, 0.684, 0.731, 0.777, 0.764, 0.728, 0.802], 'off_sess': 0.724},
         's2': {'cond': 'B', 'on': [0.5, 0.675, 0.725, 0.575, 0.75, 0.45, 0.6],
                'off_run': [0.741, 0.731, 0.681, 0.664, 0.702, 0.661, 0.636], 'off_sess': 0.713}},
    65: {'s1': {'cond': 'A', 'on': [0.5, 0.625, 0.575, 0.5, 0.5, 0.475, 0.475],
                'off_run': [0.65, 0.616, 0.613, 0.656, 0.58, 0.636, 0.611], 'off_sess': 0.582},
         's2': {'cond': 'B', 'on': [0.575, 0.525, 0.475, 0.525, 0.525, 0.4, 0.475],
                'off_run': [0.689, 0.638, 0.567, 0.636, 0.678, 0.625, 0.6], 'off_sess': 0.629}},
    68: {'s1': {'cond': 'A', 'on': [0.55, 0.475, 0.6, 0.675, 0.45, 0.525, 0.55],
                'off_run': [0.561, 0.583, 0.637, 0.661, 0.547, 0.653, 0.694], 'off_sess': 0.557},
         's2': {'cond': 'B', 'on': [0.525, 0.575, 0.6, 0.575, 0.575, 0.45, 0.55],
                'off_run': [0.642, 0.652, 0.616, 0.605, 0.623, 0.589, 0.594], 'off_sess': 0.597}},
    69: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.775, 0.95, 0.95, 0.9, 1.0],
                'off_run': [0.739, 0.759, 0.837, 0.895, 0.945, 0.891, 0.956], 'off_sess': 0.748},
         's2': {'cond': 'B', 'on': [0.675, 0.85, 0.95, 0.925, 0.95, 0.95],
                'off_run': [0.756, 0.905, 0.969, 0.947, 0.984, 0.952], 'off_sess': 0.91}},
    70: {'s1': {'cond': 'A', 'on': [0.475, 0.55, 0.575, 0.725, 0.625, 0.5, 0.625],
                'off_run': [0.728, 0.622, 0.622, 0.689, 0.616, 0.58, 0.661], 'off_sess': 0.618},
         's2': {'cond': 'B', 'on': [0.475, 0.55, 0.775, 0.75, 0.675, 0.675, 0.825],
                'off_run': [0.686, 0.802, 0.822, 0.836, 0.797, 0.822, 0.728], 'off_sess': 0.831}}
}

# 22 channel 沒有顯著
# wsi s1 avg: 4.3423611111111144
# wsi s2 avg: 4.033680555555557
# Learning Effect (LE): 3.43%, t = 1.850, p = 0.0773
# Adaptive Effect (AE): 2.82%, t = 1.569, p = 0.1309
raw_data_22 = {
    35: {'s1': {'cond': 'B', 'on': [0.525, 0.525, 0.6, 0.65, 0.6, 0.6, 0.575],
                'off_run': [0.609, 0.614, 0.614, 0.659, 0.597, 0.548, 0.603], 'off_sess': 0.567},
         's2': {'cond': 'A', 'on': [0.55, 0.5, 0.525, 0.55, 0.575, 0.525, 0.5],
                'off_run': [0.609, 0.591, 0.683, 0.659, 0.561, 0.589, 0.642], 'off_sess': 0.581}},
    37: {'s1': {'cond': 'A', 'on': [0.45, 0.5, 0.6, 0.6, 0.625, 0.95, 0.897],
                'off_run': [0.688, 0.567, 0.858, 0.786, 0.722, 1.0, 1.0], 'off_sess': 0.772},
         's2': {'cond': 'B', 'on': [0.55, 0.9, 0.975, 0.925, 0.95, 0.975],
                'off_run': [0.87, 0.991, 0.997, 1.0, 1.0, 1.0], 'off_sess': 0.974}},
    38: {'s1': {'cond': 'B', 'on': [0.475, 0.5, 0.5, 0.525, 0.5, 0.575, 0.525],
                'off_run': [0.57, 0.702, 0.573, 0.645, 0.627, 0.741, 0.705], 'off_sess': 0.598},
         's2': {'cond': 'A', 'on': [0.6, 0.5, 0.564, 0.425, 0.525, 0.6, 0.575],
                'off_run': [0.547, 0.6, 0.694, 0.739, 0.741, 0.597, 0.594], 'off_sess': 0.595}},
    40: {'s1': {'cond': 'B', 'on': [0.575, 0.525, 0.5, 0.525, 0.6, 0.5, 0.525],
                'off_run': [0.564, 0.655, 0.589, 0.594, 0.641, 0.528, 0.639], 'off_sess': 0.574},
         's2': {'cond': 'A', 'on': [0.55, 0.55, 0.475, 0.55, 0.5, 0.525, 0.59],
                'off_run': [0.586, 0.652, 0.641, 0.617, 0.658, 0.641, 0.625], 'off_sess': 0.569}},
    41: {'s1': {'cond': 'B', 'on': [0.45, 0.575, 0.6, 0.8, 0.675, 0.7, 0.625],
                'off_run': [0.642, 0.75, 0.705, 0.767, 0.75, 0.786, 0.775], 'off_sess': 0.708},
         's2': {'cond': 'A', 'on': [0.6, 0.725, 0.725, 0.65, 0.775, 0.575, 0.625],
                'off_run': [0.588, 0.6, 0.692, 0.683, 0.667, 0.616, 0.684], 'off_sess': 0.686}},
    42: {'s1': {'cond': 'A', 'on': [0.6, 0.55, 0.5, 0.425, 0.5, 0.55, 0.525],
                'off_run': [0.586, 0.564, 0.606, 0.609, 0.605, 0.589, 0.636], 'off_sess': 0.57},
         's2': {'cond': 'B', 'on': [0.575, 0.55, 0.5, 0.525, 0.625, 0.5, 0.5],
                'off_run': [0.645, 0.628, 0.6, 0.619, 0.588, 0.667, 0.55], 'off_sess': 0.569}},
    43: {'s1': {'cond': 'A', 'on': [0.5, 0.5, 0.525, 0.425, 0.525, 0.625, 0.475],
                'off_run': [0.566, 0.655, 0.641, 0.641, 0.672, 0.83, 0.784], 'off_sess': 0.67},
         's2': {'cond': 'B', 'on': [0.45, 0.675, 0.525, 0.55, 0.675, 0.65, 0.625],
                'off_run': [0.631, 0.695, 0.602, 0.703, 0.609, 0.755, 0.692], 'off_sess': 0.67}},
    44: {'s1': {'cond': 'B', 'on': [0.675, 0.725, 0.8, 0.975, 0.975, 0.9, 0.875],
                'off_run': [0.864, 0.897, 0.805, 0.823, 0.87, 0.827, 0.858], 'off_sess': 0.843},
         's2': {'cond': 'A', 'on': [0.475, 0.825, 0.925, 0.875, 0.825, 0.875],
                'off_run': [0.836, 0.863, 0.897, 0.873, 0.941, 0.891], 'off_sess': 0.872}},
    45: {'s1': {'cond': 'B', 'on': [0.425, 0.425, 0.575, 0.525, 0.5, 0.625, 0.45],
                'off_run': [0.606, 0.694, 0.77, 0.645, 0.698, 0.637, 0.747], 'off_sess': 0.615},
         's2': {'cond': 'A', 'on': [0.6, 0.8, 0.85, 0.75, 0.925, 0.9],
                'off_run': [0.661, 0.898, 0.986, 0.941, 0.98, 0.986], 'off_sess': 0.853}},
    47: {'s1': {'cond': 'B', 'on': [0.675, 0.65, 0.6, 0.625, 0.5, 0.475, 0.575],
                'off_run': [0.623, 0.648, 0.68, 0.648, 0.631, 0.583, 0.637], 'off_sess': 0.659},
         's2': {'cond': 'A', 'on': [0.675, 0.525, 0.5, 0.575, 0.55, 0.55, 0.525],
                'off_run': [0.648, 0.573, 0.561, 0.664, 0.627, 0.686, 0.664], 'off_sess': 0.584}},
    48: {'s1': {'cond': 'B', 'on': [0.525, 0.475, 0.45, 0.45, 0.475, 0.475, 0.575],
                'off_run': [0.781, 0.792, 0.753, 0.727, 0.656, 0.633, 0.625], 'off_sess': 0.646},
         's2': {'cond': 'A', 'on': [0.55, 0.525, 0.5, 0.65, 0.525, 0.5, 0.45],
                'off_run': [0.561, 0.598, 0.609, 0.662, 0.642, 0.573, 0.67], 'off_sess': 0.548}},
    50: {'s1': {'cond': 'A', 'on': [0.5, 0.65, 0.5, 0.625, 0.775, 0.65, 0.575],
                'off_run': [0.703, 0.659, 0.742, 0.809, 0.967, 0.947, 0.867], 'off_sess': 0.776},
         's2': {'cond': 'B', 'on': [0.7, 0.7, 0.525, 0.55, 0.6, 0.75, 0.825],
                'off_run': [0.747, 0.911, 0.697, 0.83, 0.817, 0.956, 0.958], 'off_sess': 0.765}},
    51: {'s1': {'cond': 'B', 'on': [0.425, 0.35, 0.5, 0.55, 0.45, 0.575, 0.5],
                'off_run': [0.602, 0.691, 0.738, 0.619, 0.703, 0.805, 0.677], 'off_sess': 0.691},
         's2': {'cond': 'A', 'on': [0.775, 0.475, 0.575, 0.625, 0.55, 0.666],  # 0.666 tmp
                'off_run': [0.669, 0.725, 0.614, 0.623, 0.711, 0.655], 'off_sess': 0.631}},
    52: {'s1': {'cond': 'B', 'on': [0.5, 0.475, 0.425, 0.5, 0.5, 0.45, 0.475],
                'off_run': [0.686, 0.647, 0.594, 0.633, 0.559, 0.661, 0.542], 'off_sess': 0.555},
         's2': {'cond': 'A', 'on': [0.575, 0.6, 0.475, 0.575, 0.625, 0.6, 0.45],
                'off_run': [0.566, 0.58, 0.63, 0.622, 0.686, 0.595, 0.57], 'off_sess': 0.573}},
    54: {'s1': {'cond': 'B', 'on': [0.55, 0.525, 0.525, 0.55, 0.625, 0.55, 0.525],
                'off_run': [0.595, 0.709, 0.591, 0.698, 0.75, 0.73, 0.695], 'off_sess': 0.642},
         's2': {'cond': 'A', 'on': [0.475, 0.75, 0.65, 0.5, 0.7, 0.575],
                'off_run': [0.702, 0.727, 0.803, 0.72, 0.794, 0.814], 'off_sess': 0.73}},
    55: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.55, 0.55, 0.65, 0.55, 0.55],
                'off_run': [0.578, 0.597, 0.578, 0.65, 0.614, 0.634, 0.669], 'off_sess': 0.609},
         's2': {'cond': 'B', 'on': [0.45, 0.45, 0.475, 0.475, 0.55, 0.475, 0.5],
                'off_run': [0.592, 0.616, 0.589, 0.559, 0.634, 0.6, 0.631], 'off_sess': 0.569}},
    57: {'s1': {'cond': 'B', 'on': [0.525, 0.65, 0.55, 0.5, 0.625, 0.55, 0.525],
                'off_run': [0.608, 0.719, 0.63, 0.642, 0.705, 0.756, 0.719], 'off_sess': 0.669},
         's2': {'cond': 'A', 'on': [0.625, 0.5, 0.525, 0.5, 0.525, 0.525, 0.525],
                'off_run': [0.717, 0.645, 0.692, 0.684, 0.617, 0.627, 0.683], 'off_sess': 0.618}},
    58: {'s1': {'cond': 'A', 'on': [0.65, 0.475, 0.575, 0.59, 0.725, 0.5, 0.6],
                'off_run': [0.681, 0.669, 0.639, 0.736, 0.842, 0.706, 0.761], 'off_sess': 0.706},
         's2': {'cond': 'B', 'on': [0.6, 0.65, 0.525, 0.725, 0.825, 0.675, 0.8],
                'off_run': [0.797, 0.836, 0.783, 0.827, 0.908, 0.914, 0.964], 'off_sess': 0.841}},
    63: {'s1': {'cond': 'A', 'on': [0.625, 0.7, 0.55, 0.525, 0.5, 0.65, 0.675],
                'off_run': [0.617, 0.698, 0.614, 0.717, 0.628, 0.642, 0.678], 'off_sess': 0.586},
         's2': {'cond': 'B', 'on': [0.5, 0.775, 0.725, 0.625, 0.7, 0.7],
                'off_run': [0.634, 0.73, 0.767, 0.914, 0.889, 0.791], 'off_sess': 0.774}},
    64: {'s1': {'cond': 'A', 'on': [0.55, 0.55, 0.5, 0.45, 0.575, 0.615, 0.675],
                'off_run': [0.598, 0.766, 0.766, 0.764, 0.767, 0.734, 0.805], 'off_sess': 0.765},
         's2': {'cond': 'B', 'on': [0.5, 0.675, 0.725, 0.575, 0.75, 0.45, 0.6],
                'off_run': [0.791, 0.755, 0.773, 0.698, 0.761, 0.711, 0.656], 'off_sess': 0.763}},
    65: {'s1': {'cond': 'A', 'on': [0.5, 0.625, 0.575, 0.5, 0.5, 0.475, 0.475],
                'off_run': [0.623, 0.655, 0.603, 0.673, 0.627, 0.627, 0.666], 'off_sess': 0.618},
         's2': {'cond': 'B', 'on': [0.575, 0.525, 0.475, 0.525, 0.525, 0.4, 0.475],
                'off_run': [0.709, 0.698, 0.667, 0.648, 0.616, 0.675, 0.581], 'off_sess': 0.663}},
    68: {'s1': {'cond': 'A', 'on': [0.55, 0.475, 0.6, 0.675, 0.45, 0.525, 0.55],
                'off_run': [0.55, 0.597, 0.614, 0.647, 0.502, 0.703, 0.714], 'off_sess': 0.605},
         's2': {'cond': 'B', 'on': [0.525, 0.575, 0.6, 0.575, 0.575, 0.45, 0.55],
                'off_run': [0.583, 0.694, 0.622, 0.614, 0.631, 0.595, 0.589], 'off_sess': 0.587}},
    69: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.775, 0.95, 0.95, 0.9, 1.0],
                'off_run': [0.744, 0.739, 0.812, 0.939, 0.966, 0.905, 0.983], 'off_sess': 0.79},
         's2': {'cond': 'B', 'on': [0.675, 0.85, 0.95, 0.925, 0.95, 0.95],
                'off_run': [0.748, 0.898, 0.967, 0.973, 0.98, 0.955], 'off_sess': 0.921}},
    70: {'s1': {'cond': 'A', 'on': [0.475, 0.55, 0.575, 0.725, 0.625, 0.5, 0.625],
                'off_run': [0.709, 0.652, 0.652, 0.628, 0.645, 0.6, 0.65], 'off_sess': 0.712},
         's2': {'cond': 'B', 'on': [0.475, 0.55, 0.775, 0.75, 0.675, 0.675, 0.825],
                'off_run': [0.722, 0.75, 0.822, 0.841, 0.806, 0.861, 0.756], 'off_sess': 0.833}}
}

raw_data = raw_data_13  # raw_data_22


# ==========================================
# 2. 統計輔助函式 (Cohen's d 與繪圖)
# ==========================================
def calc_wsi(acc_list):
    valid_acc = [float(x) * 100 for x in acc_list if str(x) != '...']
    if len(valid_acc) >= 7:
        return np.mean(valid_acc[3:7]) - np.mean(valid_acc[0:3])
    elif len(valid_acc) == 6:
        return np.mean(valid_acc[2:6]) - np.mean(valid_acc[0:2])
    else:
        print(f"error {valid_acc}")
    return np.nan


def sum_up_acc(acc_list):  # 全部加
    acc_sum = 0
    for i in acc_list:
        acc_sum += float(i)
    acc_sum /= len(acc_list)
    return acc_sum


def sum_up_acc_with_exc(acc_list):  # 不包含 calibration，取後面四個 run
    acc_sum = 0
    exc_run_num = 4
    counter = exc_run_num
    for i in range(len(acc_list) - 1, -1, -1):
        acc_sum += acc_list[i]
        counter -= 1
        if counter == 0:
            break
    return acc_sum / exc_run_num


sum_up_acc_with_exc([1, 2, 3, 4, 5, 6, 7])


# def cohens_d_paired(d1, d2):
#     """計算 Paired T-test 的效應量 (Cohen's d_z)"""
#     diff = d1 - d2
#     return np.mean(diff) / np.std(diff, ddof=1)


# def cohens_d_ind(d1, d2):
#     """計算 Independent T-test 的效應量 (Cohen's d)"""
#     n1, n2 = len(d1), len(d2)
#     v1, v2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
#     s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
#     return (np.mean(d1) - np.mean(d2)) / s_pooled


def plot_bar_box(data1, label1, color1, data2, label2, color2, title, ylabel, ylim=None, p_val=0):
    fig, ax = plt.subplots(figsize=(8, 6))

    d1 = np.array(data1)[~np.isnan(data1)]
    d2 = np.array(data2)[~np.isnan(data2)]
    # print(d1)
    # print(d2)
    is_paired = (len(d1) == len(d2))  # True

    # Bar & Boxplot
    means, sems = [np.mean(d1), np.mean(d2)], [stats.sem(d1), stats.sem(d2)]
    bars = ax.bar([1, 2], means, yerr=sems, color=[color1, color2], alpha=0.5, capsize=8, width=0.5)
    ax.boxplot([d1, d2], positions=[1, 2], widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='none', color='black', lw=1.5),
               medianprops=dict(color='black', lw=2), showfliers=False)
    # tmp_count = 0 # 手動加入顯著
    # for rect in bars:
    #     height = rect.get_height()
    #     if tmp_count == 0:
    #         y_offset = 210  # 固定往上
    #     else:
    #         y_offset = 200  # 固定往上
    #     tmp_count += 1
    #     ax.annotate("**",
    #                 xy=(rect.get_x() + rect.get_width() / 2, height),
    #                 xytext=(0, y_offset),
    #                 textcoords="offset points",
    #                 ha='center',
    #                 va='bottom',
    #                 fontsize=12,
    #                 fontweight='bold',
    #                 color='black')

    # 連線 (Paired)
    if is_paired:
        for i in range(len(d1)):
            ax.plot([1, 2], [d1[i], d2[i]], color='gray', alpha=0.3, lw=1, marker='o', ms=5)


    # 標示顯著性與效應量
    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    if p_val == 0:
        y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)

        # ax.plot([1, 1, 2, 2], [y_max - 1, y_max, y_max, y_max - 1], lw=1.5, color='black')
    else:
        y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)
        down_space = 5
        ax.plot([1, 1, 2, 2],
                [y_max - 1 - down_space, y_max - down_space, y_max - down_space, y_max - 1 - down_space],
                lw=1.5, color='black')
        ax.text(1.5,
                y_max - down_space + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{sig_str}\n(p={p_val:.4f})",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold')

    # 將 p-value 與 Cohen's d 標在線上
    # stat_text = f'{sig_str}\n$p={p_val:.3f}$\n$d={abs(d_val):.2f}$'
    # ax.text(1.5, y_max, stat_text, ha='center', va='bottom', fontsize=base_font_size - 2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label1, label2], fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=30)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 數據萃取與計算
# ==========================================
wsi_online, wsi_offline_s1, wsi_offline_s2 = [], [], []
wsi_offline_a, wsi_offline_b = [], []
sess1_acc, sess2_acc = [], []
condA_acc, condB_acc = [], []
delta_A, delta_B = [], []  # condition 差值 session 1 - session 2，也就是使用 adaptive -> static, static -> adaptive 有沒有顯著影響
delta_A_WSI, delta_B_WSI = [], []

for sub, sessions in raw_data.items():
    # WSI
    # wsi_online.append(calc_wsi(sessions['s1']['on']))
    # wsi_online.append(calc_wsi(sessions['s2']['on']))
    wsi_offline_s1.append(calc_wsi(sessions['s1']['off_run']))
    wsi_offline_s2.append(calc_wsi(sessions['s2']['off_run']))

    # Session
    s1_acc = sessions['s1']['off_sess'] * 100
    # s1_acc = sum_up_acc(sessions['s1']['on']) * 100
    # s1_acc = sum_up_acc(sessions['s1']['off_run']) * 100
    # s1_acc = sum_up_acc_with_exc(sessions['s1']['off_run']) * 100
    s2_acc = sessions['s2']['off_sess'] * 100
    # s2_acc = sum_up_acc(sessions['s2']['on']) * 100
    # s2_acc = sum_up_acc(sessions['s2']['off_run']) * 100
    # s2_acc = sum_up_acc_with_exc(sessions['s2']['off_run']) * 100

    sess1_acc.append(s1_acc)
    sess2_acc.append(s2_acc)
    # print(sum_up_acc(sessions['s1']['on']))

    # Condition & Delta
    if sessions['s1']['cond'] == 'A':
        condA_acc.append(s1_acc)
        condB_acc.append(s2_acc)
        wsi_offline_a.append(calc_wsi(sessions['s1']['off_run']))
        wsi_offline_b.append(calc_wsi(sessions['s2']['off_run']))
        delta_A.append(s2_acc - s1_acc)  # Seq A 差值
        delta_A_WSI.append(calc_wsi(sessions['s2']['off_run']) - calc_wsi(sessions['s1']['off_run']))  # Seq A 差值
    else:
        condA_acc.append(s2_acc)
        condB_acc.append(s1_acc)
        wsi_offline_a.append(calc_wsi(sessions['s2']['off_run']))
        wsi_offline_b.append(calc_wsi(sessions['s1']['off_run']))
        delta_B.append(s2_acc - s1_acc)  # Seq B 差值
        delta_B_WSI.append(calc_wsi(sessions['s2']['off_run']) - calc_wsi(sessions['s1']['off_run']))  # Seq B 差值
# print(f"s1: {sess1_acc}")
# print(f"s2: {sess2_acc}")

# ==========================================
# 4. LE / AE 計算與統計輸出
# ==========================================
LE = (np.mean(delta_A) + np.mean(delta_B)) / 2
AE = (np.mean(delta_A) - np.mean(delta_B)) / 2

# Learning Effect: 整體 (S2 - S1) 是否大於 0 -> 等同於 Paired t-test S2 vs S1
t_le, p_le = stats.ttest_1samp(delta_A + delta_B, 0)

# d_le = np.mean(delta_A + delta_B) / np.std(delta_A + delta_B, ddof=1)

# # Delta A 是否不同於 Delta B -> 等同於 Independent t-test，在 MBSR 與 MSFI 的部分，目前在 20260514-2 裡面
# t_ae, p_ae = stats.ttest_ind(delta_A, delta_B)
# print(f"acc condition group compare: t: {t_ae:.4f} p: {p_ae:.4f}")
# t_ae, p_ae = stats.ttest_ind(delta_A_WSI, delta_B_WSI)
# print(f"WSI condition group compare: t: {t_ae:.4f} p: {p_ae:.4f}")

# ------------------------------------------
# 輸出 Table: Between-group comparison (Accuracy & WSI)
# ------------------------------------------
# 檢驗 Seq A (A -> B, Group 1) 與 Seq B (B -> A, Group 2) 的進步幅度是否有顯著差異
t_acc, p_acc = stats.ttest_ind(delta_A, delta_B, nan_policy='omit')
t_wsi_ind, p_wsi_ind = stats.ttest_ind(delta_A_WSI, delta_B_WSI, nan_policy='omit')

# 計算 Group 1 與 Group 2 的 Delta 平均值 (Mean) 與標準差 (SD)
# 使用 ddof=1 來計算樣本標準差
mean_delta_A_acc = np.nanmean(delta_A)
std_delta_A_acc = np.nanstd(delta_A, ddof=1)

mean_delta_B_acc = np.nanmean(delta_B)
std_delta_B_acc = np.nanstd(delta_B, ddof=1)

diff_acc = mean_delta_A_acc - mean_delta_B_acc

mean_delta_A_wsi = np.nanmean(delta_A_WSI)
std_delta_A_wsi = np.nanstd(delta_A_WSI, ddof=1)

mean_delta_B_wsi = np.nanmean(delta_B_WSI)
std_delta_B_wsi = np.nanstd(delta_B_WSI, ddof=1)

diff_wsi = mean_delta_A_wsi - mean_delta_B_wsi

# 輸出整理好的 Table
print("\n" + "=" * 105)
print("📊 Table: Between-group comparison (Group 1 vs Group 2)")
print("=" * 105)
print(
    f"| {'Metric':<10} | {'Group 1 Mean ± SD':<20} | {'Group 2 Mean ± SD':<20} | {'Mean Diff (G1−G2)':<20} | {'t':<8} | {'p':<8} |")
print(f"|{'-' * 12}|{'-' * 22}|{'-' * 22}|{'-' * 22}|{'-' * 10}|{'-' * 10}|")

# 預先格式化 Mean ± SD 字串
g1_acc_str = f"{mean_delta_A_acc:.2f} ± {std_delta_A_acc:.2f}"
g2_acc_str = f"{mean_delta_B_acc:.2f} ± {std_delta_B_acc:.2f}"

g1_wsi_str = f"{mean_delta_A_wsi:.2f} ± {std_delta_A_wsi:.2f}"
g2_wsi_str = f"{mean_delta_B_wsi:.2f} ± {std_delta_B_wsi:.2f}"

print(f"| {'Accuracy':<10} | {g1_acc_str:>20} | {g2_acc_str:>20} | {diff_acc:>20.2f} | {t_acc:>8.4f} | {p_acc:>8.4f} |")
print(
    f"| {'WSI':<10} | {g1_wsi_str:>20} | {g2_wsi_str:>20} | {diff_wsi:>20.2f} | {t_wsi_ind:>8.4f} | {p_wsi_ind:>8.4f} |")
print("=" * 105)
def safe_wilcoxon(x, y=None):
    """計算 Wilcoxon signed-rank test (處理 Paired 或 1-samp against 0)"""
    if y is not None:
        diff = np.array(x) - np.array(y)
    else:
        diff = np.array(x)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.all(diff == 0):
        return np.nan, np.nan
    res = stats.wilcoxon(diff)
    return res.statistic, res.pvalue
# Adaptive Effect: adaptive - static - (static - adaptive)
t_ae, p_ae = stats.ttest_1samp(np.array(delta_A) - np.array(delta_B), 0)
# t_ae, p_ae = safe_wilcoxon(np.array(delta_A) - np.array(delta_B))
print("\n" + "=" * 50)

wsi_avg_s1 = 0
for i in wsi_offline_s1:
    wsi_avg_s1 += i
wsi_avg_s1 = wsi_avg_s1 / len(wsi_offline_s1)

wsi_avg_s2 = 0
for i in wsi_offline_s2:
    wsi_avg_s2 += i
wsi_avg_s2 = wsi_avg_s2 / len(wsi_offline_s2)

wsi_avg_a = 0
for i in wsi_offline_a:
    wsi_avg_a += i
wsi_avg_a = wsi_avg_a / len(wsi_offline_a)

wsi_avg_b = 0
for i in wsi_offline_b:
    wsi_avg_b += i
wsi_avg_b = wsi_avg_b / len(wsi_offline_b)

# t_wsi, p_wsi = stats.ttest_rel(wsi_offline_s1, wsi_offline_s2)
t_wsi, p_wsi = stats.ttest_1samp(wsi_offline_s1, 0)
print(f"session 1 wsi: avg: {wsi_avg_s1:.4f} t {t_wsi:.4f}, p {p_wsi:.4f}")
t_wsi, p_wsi = stats.ttest_1samp(wsi_offline_s2, 0)
print(f"session 2 wsi: avg: {wsi_avg_s2:.4f}, t {t_wsi:.4f}, p {p_wsi:.4f}")

t_wsi, p_wsi = stats.ttest_1samp(wsi_offline_a, 0)
print(f"static wsi: avg: {wsi_avg_a:.4f}, t {t_wsi:.4f}, p {p_wsi:.4f}")
t_wsi, p_wsi = stats.ttest_1samp(wsi_offline_b, 0)
print(f"adaptive wsi: avg: {wsi_avg_b:.4f}, t {t_wsi:.4f}, p {p_wsi:.4f}")

print("\n" + "=" * 50)

# session 2 vs session 1 的 WSI 有沒有顯著
t_wsi, p_wsi = stats.ttest_1samp(np.array(wsi_offline_s2) - np.array(wsi_offline_s1), 0)
print(f"session 1 vs 2 wsi: avg:, {wsi_avg_s2 - wsi_avg_s1:.4f} t: {t_wsi:.4f}, p: {p_wsi:.4f}, ")

# static(a) vs adaptive(b) 的 WSI 有沒有顯著
t_wsi, p_wsi = stats.ttest_1samp(np.array(wsi_offline_b) - np.array(wsi_offline_a), 0)
print(f"session 1 vs 2 wsi: avg:, {wsi_avg_b - wsi_avg_a:.4f} t: {t_wsi:.4f}, p: {p_wsi:.4f}, ")


# ==========================================
# 6. 分組次級分析 (Subgroup Analysis)
# ==========================================


def analyze_subgroups_by_sequence(raw_data):
    group_A_first = []  # Session 1 是 Cond A 的受試者
    group_B_first = []  # Session 1 是 Cond B 的受試者

    # 1. 根據 Session 1 的條件進行分流
    for sub, sessions in raw_data.items():
        if sessions['s1']['cond'] == 'A':
            group_A_first.append(sub)
        else:
            group_B_first.append(sub)

    # 2. 定義子群組分析與製表函數
    def process_group(subs, group_name, cond_order):
        acc_A = []
        acc_B = []

        print(f"\n{'=' * 60}")
        print(f"📊 {group_name} (共 {len(subs)} 人) - {cond_order}")
        print(f"{'=' * 60}")
        print(f"| {'Sub ID':<6} | {'Cond A Acc (%)':<15} | {'Cond B Acc (%)':<15} | {'Diff (B - A)':<12} |")
        print(f"|{'-' * 8}|{'-' * 17}|{'-' * 17}|{'-' * 14}|")

        for sub in subs:
            # 這裡以 off_sess 作為主要比較指標 (你可以根據需求改為 sum_up_acc)
            s1_acc = raw_data[sub]['s1']['off_sess'] * 100
            s2_acc = raw_data[sub]['s2']['off_sess'] * 100

            # 確認哪個 session 對應哪個 condition
            if raw_data[sub]['s1']['cond'] == 'A':
                a_acc, b_acc = s1_acc, s2_acc
            else:
                a_acc, b_acc = s2_acc, s1_acc

            acc_A.append(a_acc)
            acc_B.append(b_acc)
            diff = b_acc - a_acc

            print(f"| {sub:<6} | {a_acc:>15.2f} | {b_acc:>15.2f} | {diff:>12.2f} |")

        # 計算平均與統計檢定
        mean_A = np.mean(acc_A)
        mean_B = np.mean(acc_B)
        mean_diff = np.mean(np.array(acc_B) - np.array(acc_A))

        print(f"|{'-' * 8}|{'-' * 17}|{'-' * 17}|{'-' * 14}|")
        print(f"| {'Mean':<6} | {mean_A:>15.2f} | {mean_B:>15.2f} | {mean_diff:>12.2f} |")

        t_stat, p_val = stats.ttest_rel(acc_B, acc_A)
        print(f"\n✅ 統計檢定 (Paired t-test: Cond B vs Cond A):")
        print(f"   [結  果] t = {t_stat:.3f}, p = {p_val:.4f}")
        t, p = stats.ttest_rel(acc_A, acc_B)
        # t, p = stats.ttest_1samp(np.array(acc_B) - np.array(acc_A), 0)
        # 呼叫原本寫好的繪圖函數，直接秀出 12 人的 Bar + Boxplot
        plot_bar_box(acc_A, 'Static (A)', '#2ca02c',
                     acc_B, 'Adaptive (B)', '#d62728',
                     f'{group_name} Condition Effect', 'Offline Accuracy (%)', ylim=(40, 110), p_val=p)

    # 3. 執行分析
    process_group(group_A_first, "Group 1: Static -> Adaptive", "S1=A, S2=B")
    process_group(group_B_first, "Group 2: Adaptive -> Static", "S1=B, S2=A")


# ==========================================
# 7. 13 vs 22 Channels 比較分析
# ==========================================
def compare_channel_performance(data_13, data_22):
    print(f"\n{'=' * 60}")
    print(f"📊 13 Channels vs 22 Channels 準確度比較 (基於 S1 & S2 平均)")
    print(f"{'=' * 60}")
    print(f"| {'Sub ID':<6} | {'13-Ch Acc (%)':<15} | {'22-Ch Acc (%)':<15} | {'Diff (22-13)':<12} |")
    print(f"|{'-' * 8}|{'-' * 17}|{'-' * 17}|{'-' * 14}|")

    acc_13_list = []
    acc_22_list = []

    # 確保兩份資料的受試者順序一致
    for sub in sorted(data_13.keys()):
        # 計算每位受試者 13 channel 的平均 (S1 + S2) / 2
        mean_13 = (data_13[sub]['s1']['off_sess'] + data_13[sub]['s2']['off_sess']) / 2 * 100
        acc_13_list.append(mean_13)

        # 計算每位受試者 22 channel 的平均 (S1 + S2) / 2
        mean_22 = (data_22[sub]['s1']['off_sess'] + data_22[sub]['s2']['off_sess']) / 2 * 100
        acc_22_list.append(mean_22)

        diff = mean_22 - mean_13
        print(f"| {sub:<6} | {mean_13:>15.2f} | {mean_22:>15.2f} | {diff:>12.2f} |")

    # 總平均與統計
    mean_all_13 = np.mean(acc_13_list)
    mean_all_22 = np.mean(acc_22_list)
    mean_diff = np.mean(np.array(acc_22_list) - np.array(acc_13_list))

    print(f"|{'-' * 8}|{'-' * 17}|{'-' * 17}|{'-' * 14}|")
    print(f"| {'Mean':<6} | {mean_all_13:>15.2f} | {mean_all_22:>15.2f} | {mean_diff:>12.2f} |")

    # # 進行 Paired t-test
    # t_stat, p_val = stats.ttest_rel(acc_22_list, acc_13_list)
    # print(f"\n✅ 統計檢定 (Paired t-test: 22-Ch vs 13-Ch):")
    # print(f"   [結  果] t = {t_stat:.3f}, p = {p_val:.4f}")

    # 重用原本的繪圖函數，直接秀出 Bar + Boxplot
    plot_bar_box(acc_13_list, '13 Channels', '#1f77b4',
                 acc_22_list, '22 Channels', '#ff7f0e',
                 '13 vs 22 Channels Overall Performance', 'Offline Accuracy (%)', ylim=(40, 110))


print("\n" + "=" * 50)
print("🎯 Cross-over Design 深入統計結果 (基於 Offline Session Acc)")
print("=" * 50)

print(f"✅ Learning Effect (LE): {LE:.2f}%")
print(f"   [結  果] t = {t_le:.3f}, p = {p_le:.4f}")
# print(f"   [效應量] Cohen's d_z = {abs(d_le):.2f}")

print("-" * 50)

print(f"✅ Adaptive Effect (AE): {AE:.2f}%")
print(f"   [結  果] t = {t_ae:.3f}, p = {p_ae:.4f}")
# print(f"   [效應量] Cohen's d = {abs(d_ae):.2f}")

print("=" * 50)
print("💡 解讀指南：")
print("  - d ≈ 0.2 (微弱/Small), 0.5 (中度/Medium), 0.8 (強烈/Large)")
print("  - 如果 p > 0.05 但 d > 0.5，代表可能遭遇 Type II Error (樣本數 24 不足以支撐該顯著性)。")
print("  - 如果 p > 0.05 且 d < 0.2，則該指標在兩組之間可能真的沒有實質差異。")
print("=" * 50)

# ==========================================
# 5. 繪製並排單圖與輸出統計
# ==========================================
# print("📊 正在產生圖表與統計分析...")

# [圖 1] WSI 比較
plot_bar_box(wsi_offline_s1, 'Session 1', '#ff7f0e',
             wsi_offline_s2, 'Session 2', '#1f77b4',
             'Within-Session Improvement (WSI)', 'Improvement (%)')
# plot_bar_box(wsi_offline_a, 'Static Condition', '#2ca02c',
#              wsi_offline_b, 'Adaptive Condition', '#d62728',
#              'Within-Session Improvement (WSI)', 'Improvement (%)')
# # [圖 2] Session 1 vs Session 2
# plot_bar_box(sess1_acc, 'Session 1', '#9467bd',
#              sess2_acc, 'Session 2', '#8c564b',
#              'Learning Effect', 'Offline Accuracy (%)', ylim=(40, 110), p_val=p_le)
#
# # [圖 3] Condition A vs Condition B
# plot_bar_box(condA_acc, 'Static Condition', '#2ca02c',
#              condB_acc, 'Adaptive Condition', '#d62728',
#              'Adaptive Effect', 'Offline Accuracy (%)', ylim=(40, 110), p_val=p_ae)


# # d_ae = cohens_d_ind(delta_A, delta_B)

# # 執行分組分析
# analyze_subgroups_by_sequence(raw_data)
#
# # 執行 13 vs 22 通道分析。 13 session 2 與 13 session 1 準確度差比較多
# compare_channel_performance(raw_data_13, raw_data_22)
