"""
condition and session of MBSR and MSFI for 13 channel
20260518
在 between group 裡面加入 Group 1 Δ Mean (%)Group 2 Δ Mean (%) Mean Difference (%) 的數值
20260610
加入名次 print 出來的功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from functools import wraps
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')

raw_data_13 = {
    35: {'s1': {'cond': 'B', 'on': [0.525, 0.525, 0.6, 0.65, 0.6, 0.6, 0.575],
                'off_run': [0.606, 0.597, 0.602, 0.659, 0.606, 0.605, 0.652], 'off_sess': 0.559,
                'MBSR': [28.64, 39.03, 32.36, 37.11, 21.53, 23.80, 18.87],
                'MSFI': [38.39, 30.06, 42.14, 43.19, 39.66, 41.63, 18.11]},
         's2': {'cond': 'A', 'on': [0.55, 0.5, 0.525, 0.55, 0.575, 0.525, 0.5],
                'off_run': [0.547, 0.547, 0.62, 0.588, 0.573, 0.648, 0.648], 'off_sess': 0.582,
                'MBSR': [15.28, 30.54, 4.23, 40.70, 18.88, 22.71, 59.61],
                'MSFI': [37.13, 48.17, 62.53, 26.55, 29.21, 43.57, 33.80]}},
    37: {'s1': {'cond': 'A', 'on': [0.45, 0.5, 0.6, 0.6, 0.625, 0.95, 0.897],
                'off_run': [0.613, 0.614, 0.83, 0.731, 0.7, 0.994, 0.998], 'off_sess': 0.735,
                'MBSR': [25.97, 15.76, 37.83, 55.35, 29.42, 78.49, 79.90],
                'MSFI': [29.90, 44.40, 29.50, 16.74, 37.61, 53.53, 23.67]},
         's2': {'cond': 'B', 'on': [0.55, 0.9, 0.975, 0.925, 0.95, 0.975],
                'off_run': [0.855, 0.986, 0.994, 0.995, 1.0, 0.998], 'off_sess': 0.965,
                'MBSR': [37.90, 68.53, '...', 75.41, 70.44, 76.43, 80.45],
                'MSFI': [51.86, 39.81, '...', 43.70, 35.95, 44.14, 29.08]}},
    38: {'s1': {'cond': 'B', 'on': [0.475, 0.5, 0.5, 0.525, 0.5, 0.575, 0.525],
                'off_run': [0.531, 0.634, 0.658, 0.658, 0.566, 0.697, 0.7], 'off_sess': 0.583,
                'MBSR': [23.94, 39.76, 50.62, 52.75, 21.85, 34.84, 27.44],
                'MSFI': [34.16, 50.55, 37.72, 40.16, 34.74, 50.61, 47.02]},
         's2': {'cond': 'A', 'on': [0.6, 0.5, 0.564, 0.425, 0.525, 0.6, 0.575],
                'off_run': [0.559, 0.606, 0.705, 0.681, 0.734, 0.611, 0.567], 'off_sess': 0.575,
                'MBSR': [32.48, 39.71, 52.45, 30.27, 53.29, 79.14, 24.21],
                'MSFI': [19.24, 33.99, 57.73, 40.59, 33.07, 57.87, 32.47]}},
    40: {'s1': {'cond': 'B', 'on': [0.575, 0.525, 0.5, 0.525, 0.6, 0.5, 0.525],
                'off_run': [0.647, 0.655, 0.589, 0.602, 0.58, 0.591, 0.594], 'off_sess': 0.542,
                'MBSR': [28.13, 31.71, 23.72, 37.66, 16.40, 19.65, 17.69],
                'MSFI': [38.91, 48.65, 36.28, 36.13, 29.33, 32.14, 32.68]},
         's2': {'cond': 'A', 'on': [0.55, 0.55, 0.475, 0.55, 0.5, 0.525, 0.59],
                'off_run': [0.58, 0.597, 0.602, 0.569, 0.562, 0.631, 0.634], 'off_sess': 0.569,
                'MBSR': [21.06, 30.49, 35.60, 10.70, 22.73, 30.27, 45.96],
                'MSFI': [32.16, 29.64, 30.85, 46.63, 27.87, 46.88, 32.20]}},
    41: {'s1': {'cond': 'B', 'on': [0.45, 0.575, 0.6, 0.8, 0.675, 0.7, 0.625],
                'off_run': [0.6, 0.686, 0.636, 0.731, 0.728, 0.767, 0.67], 'off_sess': 0.682,
                'MBSR': [18.12, 45.01, 24.69, 84.04, 73.99, 73.67, 25.57],
                'MSFI': [31.01, 41.15, 32.24, 45.07, 51.02, 50.85, 32.77]},
         's2': {'cond': 'A', 'on': [0.6, 0.725, 0.725, 0.65, 0.775, 0.575, 0.625],
                'off_run': [0.681, 0.614, 0.686, 0.697, 0.686, 0.623, 0.628], 'off_sess': 0.639,
                'MBSR': [26.39, 21.25, 77.83, 82.15, 30.42, 17.29, 59.44],
                'MSFI': [24.34, 29.40, 70.30, 67.28, 57.00, 35.02, 53.54]}},
    42: {'s1': {'cond': 'A', 'on': [0.6, 0.55, 0.5, 0.425, 0.5, 0.55, 0.525],
                'off_run': [0.577, 0.609, 0.583, 0.584, 0.608, 0.592, 0.569], 'off_sess': 0.559,
                'MBSR': [9.45, 12.97, 24.96, 9.94, 33.80, 48.03, 10.79],
                'MSFI': [33.21, 40.14, 34.99, 40.58, 38.85, 29.35, 36.99]},
         's2': {'cond': 'B', 'on': [0.575, 0.55, 0.5, 0.525, 0.625, 0.5, 0.5],
                'off_run': [0.598, 0.556, 0.636, 0.573, 0.614, 0.645, 0.609], 'off_sess': 0.542,
                'MBSR': [30.40, 25.96, 41.24, 28.77, 29.41, 26.42, 41.95],
                'MSFI': [51.95, 22.67, 31.54, 37.35, 22.74, 32.97, 38.97]}},
    43: {'s1': {'cond': 'A', 'on': [0.5, 0.5, 0.525, 0.425, 0.525, 0.625, 0.475],
                'off_run': [0.569, 0.575, 0.617, 0.614, 0.661, 0.752, 0.733], 'off_sess': 0.661,
                'MBSR': [18.74, 15.01, 65.75, 13.59, 16.80, 63.97, 82.18],
                'MSFI': [22.43, 36.45, 78.27, 25.71, 47.46, 39.01, 40.44]},
         's2': {'cond': 'B', 'on': [0.45, 0.675, 0.525, 0.55, 0.675, 0.65, 0.625],
                'off_run': [0.656, 0.636, 0.617, 0.697, 0.611, 0.725, 0.683], 'off_sess': 0.658,
                'MBSR': [47.48, 65.82, 46.74, 46.34, 45.92, 62.68, 72.77],
                'MSFI': [51.38, 54.29, 39.27, 45.77, 35.25, 51.29, 37.49]}},
    44: {'s1': {'cond': 'B', 'on': [0.675, 0.725, 0.8, 0.975, 0.975, 0.9, 0.875],
                'off_run': [0.766, 0.869, 0.784, 0.841, 0.855, 0.83, 0.833], 'off_sess': 0.847,
                'MBSR': [73.05, 83.39, 79.59, 78.25, 74.77, 77.86, 83.02],
                'MSFI': [62.59, 58.23, 43.20, 55.13, 60.80, 57.36, 57.58]},
         's2': {'cond': 'A', 'on': [0.475, 0.825, 0.925, 0.875, 0.825, 0.875],
                'off_run': [0.809, 0.87, 0.892, 0.87, 0.928, 0.895], 'off_sess': 0.874,
                'MBSR': [79.58, 78.22, '...', 82.10, 74.38, 79.04, 83.23],
                'MSFI': [52.56, 56.68, '...', 72.24, 62.84, 78.99, 69.10]}},
    45: {'s1': {'cond': 'B', 'on': [0.425, 0.425, 0.575, 0.525, 0.5, 0.625, 0.45],
                'off_run': [0.597, 0.675, 0.708, 0.658, 0.68, 0.616, 0.72], 'off_sess': 0.615,
                'MBSR': [43.24, 51.22, 50.71, 18.65, 69.62, 13.87, 62.72],
                'MSFI': [34.21, 56.77, 45.68, 27.42, 44.17, 44.92, 55.21]},
         's2': {'cond': 'A', 'on': [0.6, 0.8, 0.85, 0.75, 0.925, 0.9],
                'off_run': [0.637, 0.841, 0.947, 0.881, 0.98, 0.978], 'off_sess': 0.83,
                'MBSR': [31.16, 66.86, '...', 75.17, 74.48, 59.48, 40.48],
                'MSFI': [33.58, 49.55, '...', 51.59, 35.15, 41.26, 34.92]}},
    47: {'s1': {'cond': 'B', 'on': [0.675, 0.65, 0.6, 0.625, 0.5, 0.475, 0.575],
                'off_run': [0.681, 0.614, 0.689, 0.652, 0.619, 0.591, 0.656], 'off_sess': 0.65,
                'MBSR': [23.34, 31.12, 50.75, 45.05, 43.70, 66.89, 67.88],
                'MSFI': [41.59, 45.05, 45.91, 47.93, 50.13, 47.36, 31.09]},
         's2': {'cond': 'A', 'on': [0.675, 0.525, 0.5, 0.575, 0.55, 0.55, 0.525],
                'off_run': [0.598, 0.584, 0.592, 0.577, 0.586, 0.672, 0.647], 'off_sess': 0.572,
                'MBSR': [13.59, 36.95, 53.80, 65.51, 35.49, 52.88, 15.92],
                'MSFI': [32.99, 44.52, 19.36, 22.88, 23.12, 40.27, 42.40]}},
    48: {'s1': {'cond': 'B', 'on': [0.525, 0.475, 0.45, 0.45, 0.475, 0.475, 0.575],
                'off_run': [0.656, 0.656, 0.631, 0.637, 0.667, 0.598, 0.561], 'off_sess': 0.589,
                'MBSR': [46.07, 19.55, 13.46, 43.12, 56.70, 33.10, 14.39],
                'MSFI': [25.20, 32.26, 35.02, 32.82, 53.59, 45.10, 35.55]},
         's2': {'cond': 'A', 'on': [0.55, 0.525, 0.5, 0.65, 0.525, 0.5, 0.45],
                'off_run': [0.614, 0.581, 0.595, 0.611, 0.641, 0.589, 0.597], 'off_sess': 0.535,
                'MBSR': [22.59, 17.28, 26.20, 42.14, 31.66, 33.14, 33.96],
                'MSFI': [33.79, 33.79, 38.43, 47.80, 32.53, 55.47, 46.74]}},
    50: {'s1': {'cond': 'A', 'on': [0.5, 0.65, 0.5, 0.625, 0.775, 0.65, 0.575],
                'off_run': [0.7, 0.583, 0.636, 0.752, 0.889, 0.798, 0.8], 'off_sess': 0.715,
                'MBSR': [19.55, 9.28, 28.54, 46.46, 79.07, 76.13, 74.65],
                'MSFI': [40.51, 33.02, 16.16, 43.18, 25.55, 25.96, 38.18]},
         's2': {'cond': 'B', 'on': [0.7, 0.7, 0.525, 0.55, 0.6, 0.75, 0.825],
                'off_run': [0.692, 0.759, 0.614, 0.686, 0.811, 0.92, 0.945], 'off_sess': 0.718,
                'MBSR': [70.10, 71.21, 15.98, 48.24, 75.58, 75.92, 79.53],
                'MSFI': [29.92, 38.12, 31.63, 41.51, 30.11, 41.72, 17.83]}},
    51: {'s1': {'cond': 'B', 'on': [0.425, 0.35, 0.5, 0.55, 0.45, 0.575, 0.5],
                'off_run': [0.547, 0.684, 0.714, 0.616, 0.655, 0.669, 0.664], 'off_sess': 0.61,
                'MBSR': [16.18, 14.02, 22.20, 17.28, 64.88, 41.47, 22.01],
                'MSFI': [36.70, 73.84, 44.75, 36.98, 39.28, 35.05, 30.33]},
         's2': {'cond': 'A', 'on': [0.775, 0.475, 0.575, 0.625, 0.55, 0.666],
                'off_run': [0.658, 0.68, 0.587, 0.642, 0.594, 0.666], 'off_sess': 0.613,
                'MBSR': [51.09, 66.56, '...', 40.86, 25.46, 29.50, 34.83],
                'MSFI': [47.16, 59.96, 59.76, 37.74, 22.68, 58.19]}},
    52: {'s1': {'cond': 'B', 'on': [0.5, 0.475, 0.425, 0.5, 0.5, 0.45, 0.475],
                'off_run': [0.586, 0.645, 0.583, 0.614, 0.608, 0.619, 0.514], 'off_sess': 0.547,
                'MBSR': [14.20, 45.59, 53.14, 41.57, 33.81, 30.93, 32.80],
                'MSFI': [31.10, 37.09, 15.87, 61.14, 32.68, 28.44, 32.66]},
         's2': {'cond': 'A', 'on': [0.575, 0.6, 0.475, 0.575, 0.625, 0.6, 0.45],
                'off_run': [0.559, 0.566, 0.68, 0.609, 0.63, 0.562, 0.609], 'off_sess': 0.565,
                'MBSR': [47.10, 22.90, 30.96, 27.99, 43.07, 47.74, 40.63],
                'MSFI': [57.39, 40.32, '...', 34.77, 34.68, 47.73, 49.44]}},
    54: {'s1': {'cond': 'B', 'on': [0.55, 0.525, 0.525, 0.55, 0.625, 0.55, 0.525],
                'off_run': [0.623, 0.662, 0.63, 0.63, 0.678, 0.681, 0.7], 'off_sess': 0.632,
                'MBSR': [20.79, 13.20, 22.70, 22.79, 7.80, 35.83, 12.57],
                'MSFI': [46.32, 27.46, 29.67, 31.03, 40.35, 47.22, 38.73]},
         's2': {'cond': 'A', 'on': [0.475, 0.75, 0.65, 0.5, 0.7, 0.575],
                'off_run': [0.641, 0.683, 0.731, 0.658, 0.78, 0.811], 'off_sess': 0.708,
                'MBSR': [22.85, 68.37, '...', 52.01, 63.02, 71.85, 78.39],
                'MSFI': [38.99, 44.08, '...', 50.09, 26.85, 43.62, 33.05]}},
    55: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.55, 0.55, 0.65, 0.55, 0.55],
                'off_run': [0.581, 0.583, 0.592, 0.688, 0.614, 0.664, 0.63], 'off_sess': 0.589,
                'MBSR': [3.52, 7.50, 5.05, 12.01, 14.43, 65.19, 21.69],
                'MSFI': [37.28, 36.34, 26.79, 57.04, 36.73, 46.10, 42.62]},
         's2': {'cond': 'B', 'on': [0.45, 0.45, 0.475, 0.475, 0.55, 0.475, 0.5],
                'off_run': [0.519, 0.566, 0.586, 0.602, 0.672, 0.609, 0.577], 'off_sess': 0.572,
                'MBSR': [23.29, 14.15, 24.87, 16.62, 35.60, 4.72, 15.65],
                'MSFI': [40.40, 34.66, 32.96, 21.60, 49.31, 40.07, 30.43]}},
    57: {'s1': {'cond': 'B', 'on': [0.525, 0.65, 0.55, 0.5, 0.625, 0.55, 0.525],
                'off_run': [0.572, 0.623, 0.588, 0.595, 0.577, 0.641, 0.623], 'off_sess': 0.598,
                'MBSR': [11.89, 15.09, 13.33, 25.88, 26.12, 18.72, 13.92],
                'MSFI': [37.90, 35.16, 34.24, 28.22, 42.08, 23.24, 32.23]},
         's2': {'cond': 'A', 'on': [0.625, 0.5, 0.525, 0.5, 0.525, 0.525, 0.525],
                'off_run': [0.655, 0.58, 0.631, 0.673, 0.588, 0.583, 0.602], 'off_sess': 0.55,
                'MBSR': [26.74, 58.78, 35.27, 36.79, 25.47, 30.59, 21.24],
                'MSFI': [52.12, 38.35, 38.45, 53.74, 40.25, 30.00, 40.24]}},
    58: {'s1': {'cond': 'A', 'on': [0.65, 0.475, 0.575, 0.59, 0.725, 0.5, 0.6],
                'off_run': [0.669, 0.689, 0.597, 0.72, 0.795, 0.661, 0.627], 'off_sess': 0.624,
                'MBSR': [30.65, 34.80, 23.45, 43.63, 45.45, 41.52, 37.35],
                'MSFI': [28.40, 41.75, 29.47, 57.19, 63.08, 38.39, 38.34]},
         's2': {'cond': 'B', 'on': [0.6, 0.65, 0.525, 0.725, 0.825, 0.675, 0.8],
                'off_run': [0.75, 0.75, 0.728, 0.794, 0.792, 0.894, 0.936], 'off_sess': 0.803,
                'MBSR': [63.47, 45.13, 51.87, 73.51, 42.78, 50.77, 73.29],
                'MSFI': [53.45, 43.17, 49.55, 34.62, 49.98, 53.46, 48.06]}},
    63: {'s1': {'cond': 'A', 'on': [0.625, 0.7, 0.55, 0.525, 0.5, 0.65, 0.675],
                'off_run': [0.592, 0.627, 0.633, 0.73, 0.628, 0.559, 0.664], 'off_sess': 0.578,
                'MBSR': [26.37, 74.16, 50.25, 70.10, 21.92, 43.93, 33.03],
                'MSFI': [23.99, 45.26, 35.16, 44.61, 63.25, 36.68, 59.50]},
         's2': {'cond': 'B', 'on': [0.5, 0.775, 0.725, 0.625, 0.7, 0.7],
                'off_run': [0.608, 0.762, 0.798, 0.914, 0.883, 0.814], 'off_sess': 0.764,
                'MBSR': [15.83, 62.16, '...', 71.48, 80.01, 84.56, 72.27],
                'MSFI': [41.39, 46.92, '...', 68.88, 71.73, 71.42, 66.49]}},
    64: {'s1': {'cond': 'A', 'on': [0.55, 0.55, 0.5, 0.45, 0.575, 0.615, 0.675],
                'off_run': [0.586, 0.684, 0.731, 0.777, 0.764, 0.728, 0.802], 'off_sess': 0.724,
                'MBSR': [12.68, 40.29, 46.21, 72.47, 49.93, 77.78, 66.01],
                'MSFI': [33.31, 34.85, 58.87, 67.28, 75.51, 52.77, 58.86]},
         's2': {'cond': 'B', 'on': [0.5, 0.675, 0.725, 0.575, 0.75, 0.45, 0.6],
                'off_run': [0.741, 0.731, 0.681, 0.664, 0.702, 0.661, 0.636], 'off_sess': 0.713,
                'MBSR': [65.14, 59.39, 63.47, 21.63, 61.59, 48.40, 31.36],
                'MSFI': [69.66, 49.52, 54.18, 32.23, 68.92, 60.12, 33.07]}},
    65: {'s1': {'cond': 'A', 'on': [0.5, 0.625, 0.575, 0.5, 0.5, 0.475, 0.475],
                'off_run': [0.65, 0.616, 0.613, 0.656, 0.58, 0.636, 0.611], 'off_sess': 0.582,
                'MBSR': [26.23, 24.70, 17.45, 24.83, 29.03, 53.08, 14.63],
                'MSFI': [46.31, 33.20, 40.63, 40.96, 40.72, 67.87, 38.54]},
         's2': {'cond': 'B', 'on': [0.575, 0.525, 0.475, 0.525, 0.525, 0.4, 0.475],
                'off_run': [0.689, 0.638, 0.567, 0.636, 0.678, 0.625, 0.6], 'off_sess': 0.629,
                'MBSR': [56.46, 57.32, 32.10, 55.64, 62.41, 49.64, 45.26],
                'MSFI': [43.20, 46.72, 39.16, 53.32, 53.61, 47.59, 38.12]}},
    68: {'s1': {'cond': 'A', 'on': [0.55, 0.475, 0.6, 0.675, 0.45, 0.525, 0.55],
                'off_run': [0.561, 0.583, 0.637, 0.661, 0.547, 0.653, 0.694], 'off_sess': 0.557,
                'MBSR': [20.52, 26.87, 52.76, 56.30, 27.31, 50.18, 24.12],
                'MSFI': [34.59, 27.69, 41.47, 46.63, 37.04, 46.23, 35.39]},
         's2': {'cond': 'B', 'on': [0.525, 0.575, 0.6, 0.575, 0.575, 0.45, 0.55],
                'off_run': [0.642, 0.652, 0.616, 0.605, 0.623, 0.589, 0.594], 'off_sess': 0.597,
                'MBSR': [36.30, 43.77, 13.99, 38.03, 57.80, 18.10, 37.96],
                'MSFI': [59.81, 42.08, 37.94, 50.59, 60.57, 36.48, 30.40]}},
    69: {'s1': {'cond': 'A', 'on': [0.575, 0.5, 0.775, 0.95, 0.95, 0.9, 1.0],
                'off_run': [0.739, 0.759, 0.837, 0.895, 0.945, 0.891, 0.956], 'off_sess': 0.748,
                'MBSR': [78.69, 64.52, 62.02, 72.79, 60.55, 48.32, 35.93],
                'MSFI': [66.37, 72.16, 66.47, 67.08, 87.38, 54.84, 61.58]},
         's2': {'cond': 'B', 'on': [0.675, 0.85, 0.95, 0.925, 0.95, 0.95],
                'off_run': [0.756, 0.905, 0.969, 0.947, 0.984, 0.952], 'off_sess': 0.91,
                'MBSR': [28.61, 40.47, '...', 50.52, 29.91, 17.91, 42.66],
                'MSFI': [57.06, 72.43, '...', 87.94, 56.80, 74.13, 70.81]}},
    70: {'s1': {'cond': 'A', 'on': [0.475, 0.55, 0.575, 0.725, 0.625, 0.5, 0.625],
                'off_run': [0.728, 0.622, 0.622, 0.689, 0.616, 0.58, 0.661], 'off_sess': 0.618,
                'MBSR': [51.23, 37.45, 22.64, 50.13, 37.37, 30.42, 57.00],
                'MSFI': [55.78, 40.74, 38.04, 58.76, 46.74, 31.58, 42.39]},
         's2': {'cond': 'B', 'on': [0.475, 0.55, 0.775, 0.75, 0.675, 0.675, 0.825],
                'off_run': [0.686, 0.802, 0.822, 0.836, 0.797, 0.822, 0.728], 'off_sess': 0.831,
                'MBSR': [48.63, 74.56, 81.05, 82.82, 83.75, 77.76, 79.00],
                'MSFI': [57.65, 83.13, 65.51, 65.60, 74.16, 74.55, 85.02]}}
}


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
        log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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


def get_stats(data_list):
    """計算平均值與標準誤 (SEM) 用於繪製誤差陰影區間"""
    arr = np.array(data_list)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)  # 有效樣本數
    sem = std / np.sqrt(n)
    return mean, sem


def plot_trend(ax, data1, label1, color1, data2, label2, color2, title):
    runs_labels = [f'Run {i}' for i in range(1, 8)]
    base_font_size = 15
    m1, se1 = get_stats(data1)
    m2, se2 = get_stats(data2)

    # 繪製主線與陰影誤差區間
    ax.plot(runs_labels, m1, marker='o', color=color1, linewidth=3.5, label=label1, markersize=8)
    ax.fill_between(runs_labels, m1 - se1, m1 + se1, color=color1, alpha=0.15)

    ax.plot(runs_labels, m2, marker='s', color=color2, linewidth=3.5, label=label2, markersize=8)
    ax.fill_between(runs_labels, m2 - se2, m2 + se2, color=color2, alpha=0.15)

    # 圖表美化
    ax.set_title(title, fontsize=base_font_size + 4, fontweight='bold', pad=15)
    ax.set_xlabel('Training Runs', fontsize=base_font_size)
    ax.set_ylabel('Accuracy (%)', fontsize=base_font_size)

    # 鎖定 Y 軸範圍讓圖有相同的比較基準
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=base_font_size - 1)


def plot_bar_box(data1, label1, color1, data2, label2, color2, title, ylabel, ylim=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    d1 = np.array(data1)[~np.isnan(data1)]
    d2 = np.array(data2)[~np.isnan(data2)]

    is_paired = (len(d1) == len(d2))

    # 統計檢定
    if is_paired:
        t_stat, p_val = stats.ttest_rel(d2, d1)
    else:
        t_stat, p_val = stats.ttest_ind(d2, d1)

    # Bar & Boxplot
    means, sems = [np.mean(d1), np.mean(d2)], [stats.sem(d1), stats.sem(d2)]
    ax.bar([1, 2], means, yerr=sems, color=[color1, color2], alpha=0.5, capsize=8, width=0.5)
    ax.boxplot([d1, d2], positions=[1, 2], widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='none', color='black', lw=1.5),
               medianprops=dict(color='black', lw=2), showfliers=False)

    # 連線 (Paired)
    if is_paired:
        for i in range(len(d1)):
            ax.plot([1, 2], [d1[i], d2[i]], color='gray', alpha=0.3, lw=1, marker='o', ms=5)

    # 標示顯著性
    sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"\n{title}")
    print(f"t = {t_stat:.4f}, p = {p_val:.6f}")

    y_max = max(np.max(d1), np.max(d2)) + (10 if np.max(d1) > 10 else 0.05)

    ax.plot([1, 1, 2, 2],
            [y_max - 1, y_max, y_max, y_max - 1],
            lw=1.5, color='black')

    ax.text(1.5,
            y_max + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"{sig_str}\n(p={p_val:.4f})",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label1, label2], fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=30)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_metric_visualizations(mbsr_data, msfi_data, ids):
    """將算好的 MBSR 與 MSFI 字典資料抽出，繪製趨勢圖與箱型圖，最後存成圖片"""
    s1_mbsr_runs, s2_mbsr_runs = [], []
    s1_msfi_runs, s2_msfi_runs = [], []
    s1_mbsr_avg, s2_mbsr_avg = [], []
    s1_msfi_avg, s2_msfi_avg = [], []

    # 新增：CondA 與 CondB 的儲存列
    condA_mbsr_runs, condB_mbsr_runs = [], []
    condA_msfi_runs, condB_msfi_runs = [], []
    condA_mbsr_avg, condB_mbsr_avg = [], []
    condA_msfi_avg, condB_msfi_avg = [], []

    def extract_and_shift(data_dict, session, sub):
        # 現在已經是精確對應 Run 1~7，不需要再平移
        shifted_array = [np.nan] * 7
        for r in range(1, 8):
            if r in data_dict[session][sub]:
                shifted_array[r - 1] = data_dict[session][sub][r]
        return shifted_array

    # 萃取資料
    for sub in ids:
        sub_str = str(sub)
        sub_int = int(sub)
        if sub_int not in raw_data_13:
            continue
        # Session 提取
        sub_s1_mbsr = extract_and_shift(mbsr_data, 's1', sub_str)
        sub_s2_mbsr = extract_and_shift(mbsr_data, 's2', sub_str)
        s1_mbsr_runs.append(sub_s1_mbsr)
        s2_mbsr_runs.append(sub_s2_mbsr)
        s1_mbsr_avg.append(np.nanmean(sub_s1_mbsr))
        s2_mbsr_avg.append(np.nanmean(sub_s2_mbsr))

        sub_s1_msfi = extract_and_shift(msfi_data, 's1', sub_str)
        sub_s2_msfi = extract_and_shift(msfi_data, 's2', sub_str)
        s1_msfi_runs.append(sub_s1_msfi)
        s2_msfi_runs.append(sub_s2_msfi)
        s1_msfi_avg.append(np.nanmean(sub_s1_msfi))
        s2_msfi_avg.append(np.nanmean(sub_s2_msfi))

        # Condition 提取 (根據 raw_data_13 的對應關係)
        cond_s1 = raw_data_13[sub_int]['s1']['cond']

        if cond_s1 == 'A':
            # S1 是 A, S2 就是 B
            condA_mbsr_runs.append(sub_s1_mbsr)
            condB_mbsr_runs.append(sub_s2_mbsr)
            condA_msfi_runs.append(sub_s1_msfi)
            condB_msfi_runs.append(sub_s2_msfi)
            condA_mbsr_avg.append(np.nanmean(sub_s1_mbsr))
            condB_mbsr_avg.append(np.nanmean(sub_s2_mbsr))
            condA_msfi_avg.append(np.nanmean(sub_s1_msfi))
            condB_msfi_avg.append(np.nanmean(sub_s2_msfi))
        else:
            # S1 是 B, S2 就是 A
            condB_mbsr_runs.append(sub_s1_mbsr)
            condA_mbsr_runs.append(sub_s2_mbsr)
            condB_msfi_runs.append(sub_s1_msfi)
            condA_msfi_runs.append(sub_s2_msfi)
            condB_mbsr_avg.append(np.nanmean(sub_s1_mbsr))
            condA_mbsr_avg.append(np.nanmean(sub_s2_mbsr))
            condB_msfi_avg.append(np.nanmean(sub_s1_msfi))
            condA_msfi_avg.append(np.nanmean(sub_s2_msfi))

    # ==========================================
    # 建立輸出資料夾
    # ==========================================
    out_dir = "./metric_plot"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📂 正在將指標圖片儲存至 {out_dir}/ ...")

    # 統計 與後續的 rel 結果一樣
    print("LE")
    s2_mbsr_runs_avg = [np.nanmean(sublist) for sublist in s2_mbsr_runs]
    s1_mbsr_runs_avg = [np.nanmean(sublist) for sublist in s1_mbsr_runs]
    t, p = stats.ttest_1samp(np.array(s2_mbsr_runs_avg) - np.array(s1_mbsr_runs_avg), 0)
    print(f"MBSR session vs 0: avg: {np.nanmean(np.array(s2_mbsr_runs_avg) - np.array(s1_mbsr_runs_avg)):.4f}, "
          f"t: {t:.4f}, p: {p:.4f}")

    s2_msfi_runs_avg = [np.nanmean(sublist) for sublist in s2_msfi_runs]
    s1_msfi_runs_avg = [np.nanmean(sublist) for sublist in s1_msfi_runs]
    t, p = stats.ttest_1samp(np.array(s2_msfi_runs_avg) - np.array(s1_msfi_runs_avg), 0)
    print(f"MSFI session vs 0: avg: {np.nanmean(np.array(s2_msfi_runs_avg) - np.array(s1_msfi_runs_avg)):.4f}, "
          f"t: {t:.4f}, p: {p:.4f}")
    print()
    print("AE")
    condB_mbsr_runs_avg = [np.nanmean(sublist) for sublist in condB_mbsr_runs]
    condA_mbsr_runs_avg = [np.nanmean(sublist) for sublist in condA_mbsr_runs]
    t, p = stats.ttest_1samp(np.array(condB_mbsr_runs_avg) - np.array(condA_mbsr_runs_avg), 0)
    print(f"MBSR AE vs 0: avg: {np.nanmean(np.array(condB_mbsr_runs_avg) - np.array(condA_mbsr_runs_avg)):.4f}, "
          f"t: {t:.4f}, p: {p:.4f}")

    condB_msfi_runs_avg = [np.nanmean(sublist) for sublist in condB_msfi_runs]
    condA_msfi_runs_avg = [np.nanmean(sublist) for sublist in condA_msfi_runs]
    t, p = stats.ttest_1samp(np.array(condB_msfi_runs_avg) - np.array(condA_msfi_runs_avg), 0)
    print(f"MSFI AE vs 0: avg: {np.nanmean(np.array(condB_msfi_runs_avg) - np.array(condA_msfi_runs_avg)):.4f}, "
          f"t: {t:.4f}, p: {p:.4f}")
    # ==========================================
    # 繪製 1：折線圖 (Trend)
    # ==========================================
    # --- Session 1 vs 2 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, s1_mbsr_runs, 'Session 1', '#9467bd', s2_mbsr_runs, 'Session 2', '#8c564b', 'MBSR Trend (S1 vs S2)')
    ax.set_ylabel('MBSR (%)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MBSR_Trend_S1_S2.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, s1_msfi_runs, 'Session 1', '#9467bd', s2_msfi_runs, 'Session 2', '#8c564b', 'MSFI Trend (S1 vs S2)')
    ax.set_ylabel('MSFI (%)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MSFI_Trend_S1_S2.png'), dpi=200)
    plt.close(fig)

    # --- Cond A vs B ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, condA_mbsr_runs, 'Condition A', '#2ca02c', condB_mbsr_runs, 'Condition B', '#d62728',
               'MBSR Trend (Cond A vs B)')
    ax.set_ylabel('MBSR (%)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MBSR_Trend_CondAB.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trend(ax, condA_msfi_runs, 'Condition A', '#2ca02c', condB_msfi_runs, 'Condition B', '#d62728',
               'MSFI Trend (Cond A vs B)')
    ax.set_ylabel('MSFI (%)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'MSFI_Trend_CondAB.png'), dpi=200)
    plt.close(fig)

    # ==========================================
    # 繪製 2：柱狀箱型圖 (Bar + Boxplot)
    # ==========================================
    # --- Session 1 vs 2 ---
    plot_bar_box(s1_mbsr_avg, 'Session 1', '#9467bd', s2_mbsr_avg, 'Session 2', '#8c564b',
                 'MBSR (S1 vs S2)', 'MBSR (%)', ylim=(0, 110),
                 )  # save_path=os.path.join(out_dir, 'MBSR_BarBox_S1_S2.png')
    plot_bar_box(s1_msfi_avg, 'Session 1', '#9467bd', s2_msfi_avg, 'Session 2', '#8c564b',
                 'MSFI (S1 vs S2)', 'MSFI (%)', ylim=(0, 110),
                 )  # save_path=os.path.join(out_dir, 'MSFI_BarBox_S1_S2.png')

    # --- Cond A vs B ---
    plot_bar_box(condA_mbsr_avg, 'Static', '#2ca02c', condB_mbsr_avg, 'Condition B', '#d62728',
                 'MBSR (Static vs Adaptive)', 'MBSR (%)', ylim=(0, 110),
                 )  # save_path=os.path.join(out_dir, 'MBSR_BarBox_CondAB.png')
    plot_bar_box(condA_msfi_avg, 'Static', '#2ca02c', condB_msfi_avg, 'Adaptive', '#d62728',
                 'MSFI (Static vs Adaptive)', 'MSFI (%)', ylim=(0, 110),
                 )  # save_path=os.path.join(out_dir, 'MSFI_BarBox_CondAB.png')

    print("✅ 所有對比圖片儲存完畢！")


def calculate_and_print_metrics():
    ids = [35, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 54, 55, 57, 58, 63, 64, 65, 68, 69, 70]
    sessions = ["s1", "s2"]

    print("\n從 raw_data_13 解析量化指標中...\n")

    mbsr_data = defaultdict(lambda: defaultdict(dict))
    msfi_data = defaultdict(lambda: defaultdict(dict))

    # 解析字典
    for sub_int, sessions_dict in raw_data_13.items():
        sub = str(sub_int)
        if sub_int not in raw_data_13:
            continue
        for session in sessions:
            if session in sessions_dict:
                mbsr_list = sessions_dict[session].get('MBSR', [])
                msfi_list = sessions_dict[session].get('MSFI', [])

                for i, val in enumerate(mbsr_list):
                    if val != '...':
                        mbsr_data[session][sub][i + 1] = float(val)

                for i, val in enumerate(msfi_list):
                    if val != '...':
                        msfi_data[session][sub][i + 1] = float(val)

    # 重新映射 ID (從小到大: S1 ~ S24)
    sorted_ids = sorted([str(i) for i in ids], key=lambda x: int(x))
    id_map = {sub: f"S{i + 1}" for i, sub in enumerate(sorted_ids)}

    # ------------------------------------------
    # 輸出 Run 級別的表格 (分 Session 與 指標)
    # ------------------------------------------
    metrics_to_print = [("MBSR", mbsr_data), ("MSFI", msfi_data)]

    for metric_name, data_dict in metrics_to_print:
        for session in sessions:
            print(f"\n### 表格: {metric_name} (%) - Session {session.upper()}")
            print("| **id** | **run 1** | **run 2** | **run 3** | **run 4** | **run 5** | **run 6** | **run 7** |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

            for sub in sorted_ids:
                row_str = f"| {id_map[sub]} |"
                for r in range(1, 8):
                    val = data_dict[session][sub].get(r, None)
                    if val is not None:
                        row_str += f" {val:.2f} |"
                    else:
                        row_str += " ... |"
                print(row_str)

    # ------------------------------------------
    # 輸出 終極統計表格：將整個 Session 的 Run 平均
    # ------------------------------------------
    print("\n### 終極統計表格：各 Session 整體平均指標 (%)")
    print("| **id** | **S1 MBSR** | **S2 MBSR** | **S1 MSFI** | **S2 MSFI** |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    avg_s1_mbsr, avg_s2_mbsr = [], []
    avg_s1_msfi, avg_s2_msfi = [], []

    # 新增：Cond 統計列表
    avg_condA_mbsr, avg_condB_mbsr = [], []
    avg_condA_msfi, avg_condB_msfi = [], []
    cond_table_rows = []

    # 新增：儲存 Sequence Delta (S2 - S1) 用於組間檢定
    delta_A_mbsr, delta_B_mbsr = [], []
    delta_A_msfi, delta_B_msfi = [], []

    for sub in sorted_ids:
        sub_int = int(sub)
        if sub_int not in raw_data_13:
            continue

        s1_mbsr_vals = list(mbsr_data['s1'][sub].values())
        s2_mbsr_vals = list(mbsr_data['s2'][sub].values())
        s1_msfi_vals = list(msfi_data['s1'][sub].values())
        s2_msfi_vals = list(msfi_data['s2'][sub].values())

        s1_mbsr = np.mean(s1_mbsr_vals) if s1_mbsr_vals else np.nan
        s2_mbsr = np.mean(s2_mbsr_vals) if s2_mbsr_vals else np.nan
        s1_msfi = np.mean(s1_msfi_vals) if s1_msfi_vals else np.nan
        s2_msfi = np.mean(s2_msfi_vals) if s2_msfi_vals else np.nan

        # 蒐集全體平均用
        if not np.isnan(s1_mbsr): avg_s1_mbsr.append(s1_mbsr)
        if not np.isnan(s2_mbsr): avg_s2_mbsr.append(s2_mbsr)
        if not np.isnan(s1_msfi): avg_s1_msfi.append(s1_msfi)
        if not np.isnan(s2_msfi): avg_s2_msfi.append(s2_msfi)

        # 判斷 Cond 分類並映射，同時計算 Delta (S2 - S1)
        cond_s1 = raw_data_13[sub_int]['s1']['cond']
        if cond_s1 == 'A':
            condA_mbsr, condB_mbsr = s1_mbsr, s2_mbsr
            condA_msfi, condB_msfi = s1_msfi, s2_msfi
            # Session 1 是 A，記錄 Sequence A 的差值
            delta_A_mbsr.append(s2_mbsr - s1_mbsr)
            delta_A_msfi.append(s2_msfi - s1_msfi)
        else:
            condB_mbsr, condA_mbsr = s1_mbsr, s2_mbsr
            condB_msfi, condA_msfi = s1_msfi, s2_msfi
            # Session 1 是 B，記錄 Sequence B 的差值
            delta_B_mbsr.append(s2_mbsr - s1_mbsr)
            delta_B_msfi.append(s2_msfi - s1_msfi)

        if not np.isnan(condA_mbsr): avg_condA_mbsr.append(condA_mbsr)
        if not np.isnan(condB_mbsr): avg_condB_mbsr.append(condB_mbsr)
        if not np.isnan(condA_msfi): avg_condA_msfi.append(condA_msfi)
        if not np.isnan(condB_msfi): avg_condB_msfi.append(condB_msfi)

        s1_mbsr_str = f"{s1_mbsr:.2f}" if not np.isnan(s1_mbsr) else "..."
        s2_mbsr_str = f"{s2_mbsr:.2f}" if not np.isnan(s2_mbsr) else "..."
        s1_msfi_str = f"{s1_msfi:.2f}" if not np.isnan(s1_msfi) else "..."
        s2_msfi_str = f"{s2_msfi:.2f}" if not np.isnan(s2_msfi) else "..."

        condA_mbsr_str = f"{condA_mbsr:.2f}" if not np.isnan(condA_mbsr) else "..."
        condB_mbsr_str = f"{condB_mbsr:.2f}" if not np.isnan(condB_mbsr) else "..."
        condA_msfi_str = f"{condA_msfi:.2f}" if not np.isnan(condA_msfi) else "..."
        condB_msfi_str = f"{condB_msfi:.2f}" if not np.isnan(condB_msfi) else "..."

        print(f"| {id_map[sub]} | {s1_mbsr_str} | {s2_mbsr_str} | {s1_msfi_str} | {s2_msfi_str} |")
        cond_table_rows.append(
            f"| {id_map[sub]} | {condA_mbsr_str} | {condB_mbsr_str} | {condA_msfi_str} | {condB_msfi_str} |")

    # 全局總平均 (S1 vs S2)
    print(
        f"| **Mean** | **{np.mean(avg_s1_mbsr):.2f}** | **{np.mean(avg_s2_mbsr):.2f}** | **{np.mean(avg_s1_msfi):.2f}** | **{np.mean(avg_s2_msfi):.2f}** |")

    # ------------------------------------------
    # 輸出 終極統計表格：將整個 Condition 的 Run 平均
    # ------------------------------------------
    print("\n### 終極統計表格：各 Condition (A vs B) 整體平均指標 (%)")
    print("| **id** | **CondA MBSR** | **CondB MBSR** | **CondA MSFI** | **CondB MSFI** |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for row in cond_table_rows:
        print(row)
    print(
        f"| **Mean** | **{np.mean(avg_condA_mbsr):.2f}** | **{np.mean(avg_condB_mbsr):.2f}** | **{np.mean(avg_condA_msfi):.2f}** | **{np.mean(avg_condB_msfi):.2f}** |")

    # # ------------------------------------------
    # # 新增：順序效應 (Sequence Effect) 檢定
    # # ------------------------------------------
    # print("\n### 順序效應檢定：Delta (S2 - S1) 的 Independent t-test")
    # # 檢驗 Seq A (A -> B) 與 Seq B (B -> A) 的進步幅度是否有顯著差異
    #
    # t_mbsr, p_mbsr = stats.ttest_ind(delta_A_mbsr, delta_B_mbsr, nan_policy='omit')
    # print(f"MBSR Delta (Seq A vs Seq B) -> t: {t_mbsr:.4f}, p: {p_mbsr:.4f}")
    #
    # t_msfi, p_msfi = stats.ttest_ind(delta_A_msfi, delta_B_msfi, nan_policy='omit')
    # print(f"MSFI Delta (Seq A vs Seq B) -> t: {t_msfi:.4f}, p: {p_msfi:.4f}")
    # ------------------------------------------
    # 新增：順序效應 (Sequence Effect) 檢定與 Table 輸出
    # ------------------------------------------
    print("\n### 順序效應檢定：Delta (S2 - S1) 的 Independent t-test")
    # 檢驗 Seq A (A -> B, Group 1) 與 Seq B (B -> A, Group 2) 的進步幅度是否有顯著差異

    t_mbsr, p_mbsr = stats.ttest_ind(delta_A_mbsr, delta_B_mbsr, nan_policy='omit')
    t_msfi, p_msfi = stats.ttest_ind(delta_A_msfi, delta_B_msfi, nan_policy='omit')

    # 計算 Group 1 與 Group 2 的 Delta 平均值 (Mean) 與標準差 (SD)
    # 使用 ddof=1 來計算樣本標準差 (Sample Standard Deviation)
    mean_delta_A_mbsr = np.nanmean(delta_A_mbsr)
    std_delta_A_mbsr = np.nanstd(delta_A_mbsr, ddof=1)

    mean_delta_B_mbsr = np.nanmean(delta_B_mbsr)
    std_delta_B_mbsr = np.nanstd(delta_B_mbsr, ddof=1)

    diff_mbsr = mean_delta_A_mbsr - mean_delta_B_mbsr

    mean_delta_A_msfi = np.nanmean(delta_A_msfi)
    std_delta_A_msfi = np.nanstd(delta_A_msfi, ddof=1)

    mean_delta_B_msfi = np.nanmean(delta_B_msfi)
    std_delta_B_msfi = np.nanstd(delta_B_msfi, ddof=1)

    diff_msfi = mean_delta_A_msfi - mean_delta_B_msfi

    # 輸出整理好的 Table
    print("\n" + "=" * 105)
    print("📊 Table: Between-group comparison (Group 1 vs Group 2)")
    print("=" * 105)
    print(
        f"| {'Metric':<10} | {'Group 1 Mean ± SD':<20} | {'Group 2 Mean ± SD':<20} | {'Mean Diff (G1−G2)':<20} | {'t':<8} | {'p':<8} |")
    print(f"|{'-' * 12}|{'-' * 22}|{'-' * 22}|{'-' * 22}|{'-' * 10}|{'-' * 10}|")

    # 預先格式化 Mean ± SD 字串
    g1_mbsr_str = f"{mean_delta_A_mbsr:.2f} ± {std_delta_A_mbsr:.2f}"
    g2_mbsr_str = f"{mean_delta_B_mbsr:.2f} ± {std_delta_B_mbsr:.2f}"

    g1_msfi_str = f"{mean_delta_A_msfi:.2f} ± {std_delta_A_msfi:.2f}"
    g2_msfi_str = f"{mean_delta_B_msfi:.2f} ± {std_delta_B_msfi:.2f}"

    print(
        f"| {'MBSR':<10} | {g1_mbsr_str:>20} | {g2_mbsr_str:>20} | {diff_mbsr:>20.2f} | {t_mbsr:>8.4f} | {p_mbsr:>8.4f} |")
    print(
        f"| {'MSFI':<10} | {g1_msfi_str:>20} | {g2_msfi_str:>20} | {diff_msfi:>20.2f} | {t_msfi:>8.4f} | {p_msfi:>8.4f} |")
    print("=" * 105)

    # 繪製視覺化圖表
    # plot_metric_visualizations(mbsr_data, msfi_data, ids)


def print_subject_ranks():
    print("\n" + "=" * 105)
    print("🏆 受試者排名 (Session 1 & Session 2) - 數值越高，名次越前 (1~24)")
    print("=" * 105)

    s1_data = []
    s2_data = []

    # 重新映射 ID (從小到大: S1 ~ S24)
    ids = sorted([sub for sub in raw_data_13.keys()])
    id_map = {str(sub): f"S{i + 1}" for i, sub in enumerate(ids)}

    for sub in ids:
        sub_str = str(sub)
        for sess, data_list in [('s1', s1_data), ('s2', s2_data)]:
            if sess in raw_data_13[sub]:
                sess_data = raw_data_13[sub][sess]
                off_sess = sess_data.get('off_sess', np.nan)

                # 計算平均值 (排除 '...')
                mbsr_vals = [float(v) for v in sess_data.get('MBSR', []) if v != '...']
                mbsr_avg = np.mean(mbsr_vals) if mbsr_vals else np.nan

                msfi_vals = [float(v) for v in sess_data.get('MSFI', []) if v != '...']
                msfi_avg = np.mean(msfi_vals) if msfi_vals else np.nan

                data_list.append({
                    'sub_name': id_map[sub_str],
                    'sub_id': int(id_map[sub_str][1:]),  # 用於最後排序 S1~S24
                    'off_sess': off_sess,
                    'mbsr_avg': mbsr_avg,
                    'msfi_avg': msfi_avg
                })

    # 計算排名並印出
    for sess, data_list in [('Session 1', s1_data), ('Session 2', s2_data)]:
        # 1. 根據 off_sess 排序並給予名次
        data_list.sort(key=lambda x: x['off_sess'], reverse=True)
        for i, d in enumerate(data_list):
            d['off_sess_rank'] = i + 1

        # 2. 根據 MBSR Avg 排序並給予名次
        data_list.sort(key=lambda x: x['mbsr_avg'], reverse=True)
        for i, d in enumerate(data_list):
            d['mbsr_rank'] = i + 1

        # 3. 根據 MSFI Avg 排序並給予名次
        data_list.sort(key=lambda x: x['msfi_avg'], reverse=True)
        for i, d in enumerate(data_list):
            d['msfi_rank'] = i + 1

        # 4. 恢復依受試者編號 (S1~S24) 排序，以利表格整齊呈現
        data_list.sort(key=lambda x: x['sub_id'])

        print(f"\n### {sess} Rankings")
        print(
            f"| {'Subject':<9} | {'Accuracy':<10} | {'Rank':<6} | {'MBSR Avg(%)':<12} | {'Rank':<6} | {'MSFI Avg(%)':<12} | {'Rank':<6} |")
        print(
            f"| {':---':<9} | {':---:':<10} | {':---:':<6} | {':---:':<12} | {':---:':<6} | {':---:':<12} | {':---:':<6} |")
        for d in data_list:
            off_sess_str = f"{d['off_sess']:.3f}" if not np.isnan(d['off_sess']) else "NaN"
            mbsr_str = f"{d['mbsr_avg']:.2f}" if not np.isnan(d['mbsr_avg']) else "NaN"
            msfi_str = f"{d['msfi_avg']:.2f}" if not np.isnan(d['msfi_avg']) else "NaN"

            # print(
            #     f"| {d['sub_name']:<9} | {off_sess_str:<10} | {d['off_sess_rank']:<6} | {mbsr_str:<12} | {d['mbsr_rank']:<6} | {msfi_str:<12} | {d['msfi_rank']:<6} |")
            print(
                f"| {d['sub_name']:<9} | {off_sess_str:<10} | {d['off_sess_rank']:<6} | {mbsr_str:<12} | {d['mbsr_rank']:<6} | {msfi_str:<12} | {d['msfi_rank']:<6} |")

@tee_log("compute_saliency_metric.txt")
def main():
    calculate_and_print_metrics()
    print_subject_ranks()


if __name__ == "__main__":
    main()
