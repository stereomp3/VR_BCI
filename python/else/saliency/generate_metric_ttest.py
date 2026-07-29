"""
t test (假定符合常態分佈)
Wilcoxon t test (無母數檢定，看數值排名，不會被離群值影響)
chatGPT 說可以使用 Two-way mixed ANOVA 和 Linear Mixed Model 來分析，所以加入到 table 3
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
    # 37: {'s1': {'cond': 'A', 'on': [0.45, 0.5, 0.6, 0.6, 0.625, 0.95, 0.897],
    #             'off_run': [0.613, 0.614, 0.83, 0.731, 0.7, 0.994, 0.998], 'off_sess': 0.735,
    #             'MBSR': [25.97, 15.76, 37.83, 55.35, 29.42, 78.49, 79.90],
    #             'MSFI': [29.90, 44.40, 29.50, 16.74, 37.61, 53.53, 23.67]},
    #      's2': {'cond': 'B', 'on': [0.55, 0.9, 0.975, 0.925, 0.95, 0.975],
    #             'off_run': [0.855, 0.986, 0.994, 0.995, 1.0, 0.998], 'off_sess': 0.965,
    #             'MBSR': [37.90, 68.53, '...', 75.41, 70.44, 76.43, 80.45],
    #             'MSFI': [51.86, 39.81, '...', 43.70, 35.95, 44.14, 29.08]}},
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
    # 45: {'s1': {'cond': 'B', 'on': [0.425, 0.425, 0.575, 0.525, 0.5, 0.625, 0.45],
    #             'off_run': [0.597, 0.675, 0.708, 0.658, 0.68, 0.616, 0.72], 'off_sess': 0.615,
    #             'MBSR': [43.24, 51.22, 50.71, 18.65, 69.62, 13.87, 62.72],
    #             'MSFI': [34.21, 56.77, 45.68, 27.42, 44.17, 44.92, 55.21]},
    #      's2': {'cond': 'A', 'on': [0.6, 0.8, 0.85, 0.75, 0.925, 0.9],
    #             'off_run': [0.637, 0.841, 0.947, 0.881, 0.98, 0.978], 'off_sess': 0.83,
    #             'MBSR': [31.16, 66.86, '...', 75.17, 74.48, 59.48, 40.48],
    #             'MSFI': [33.58, 49.55, '...', 51.59, 35.15, 41.26, 34.92]}},
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


# ==========================================
# 統計 Helpers：Wilcoxon & t-test
# ==========================================
def safe_wilcoxon(x, y=None):
    if y is not None:
        diff = np.array(x) - np.array(y)
    else:
        diff = np.array(x)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.all(diff == 0):
        return np.nan, np.nan
    res = stats.wilcoxon(diff)
    return res.statistic, res.pvalue


def safe_mannwhitneyu(x, y):
    x, y = np.array(x), np.array(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    res = stats.mannwhitneyu(x, y)
    return res.statistic, res.pvalue


def safe_ttest_1samp(x, popmean=0):
    x = np.array(x)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    res = stats.ttest_1samp(x, popmean)
    return res.statistic, res.pvalue


def safe_ttest_ind(x, y):
    x = np.array(x)
    y = np.array(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    res = stats.ttest_ind(x, y)
    return res.statistic, res.pvalue


# ==========================================
# LMM 與 Mixed ANOVA 分析函式 (新增)
# ==========================================
def run_lmm_analysis():
    """將字典轉換為 DataFrame 並執行 LMM 分析"""
    try:
        import pandas as pd
        # smf 是一個包裝入口，可以用公式語法快速建立模型，會透過 patsy 把像是 y ~ x1 + x2 轉化成 design matrix
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n[提示] 系統缺少 pandas 或 statsmodels 套件，無法執行 LMM 分析。請使用 pip install pandas statsmodels 安裝。")
        return

    # 1. 將字典整理成 Long-format DataFrame
    rows = []
    for sub_int, data in raw_data_13.items():
        sub = str(sub_int)
        # 標記 Sequence：若 Session 1 是 Cond A，代表 Sequence 是 A->B
        seq = 'Seq_AB' if data['s1']['cond'] == 'A' else 'Seq_BA'

        for sess in ['s1', 's2']:
            cond = data[sess]['cond']
            acc = data[sess]['off_sess'] * 100

            mbsr_raw = [float(x) for x in data[sess].get('MBSR', []) if x != '...']
            msfi_raw = [float(x) for x in data[sess].get('MSFI', []) if x != '...']
            mbsr = np.mean(mbsr_raw) if mbsr_raw else np.nan
            msfi = np.mean(msfi_raw) if msfi_raw else np.nan

            rows.append({
                'Subject': sub,
                'Session': sess.upper(),
                'Condition': cond,
                'Sequence': seq,
                'Accuracy': acc,
                'MBSR': mbsr,
                'MSFI': msfi
            })

    df = pd.DataFrame(rows)
    print(df)
    # 2. 輸出 LMM 表格
    print("\n" + "=" * 120)
    print("📊 Table 3: Linear Mixed Model (LMM) & Two-way Mixed ANOVA")
    print("   * 模型公式: Metric ~ Session * Sequence + (1|Subject)")
    print("   * [Session 主效應] = 學習效應 (Learning Effect, LE)")
    print("   * [Sequence 主效應] = 順序/組別效應 (Carryover Effect)")
    print("   * [Session × Sequence 交互作用] = 系統效應 (Adaptive Effect, AE) - 等價於 Condition 主效應")
    print("=" * 120)
    print(f"| {'Metric':<10} | {'Session (LE) p-val':<20} | {'Sequence p-val':<20} | {'Interaction (AE) p-val':<22} |")
    print(f"|{'-' * 12}|{'-' * 22}|{'-' * 22}|{'-' * 24}|")

    for m in ['Accuracy', 'MBSR', 'MSFI']:
        df_clean = df.dropna(subset=[m])
        if len(df_clean) == 0: continue

        try:
            # 建立 Two-way Mixed ANOVA 的 Linear Mixed Model (LMM) 模型
            # 使用 C() 確保類別變數被正確解析，會使用 one-hot，像是把 S1 轉成 0，S2 轉成 1
            md = smf.mixedlm(f"{m} ~ C(Session) * C(Sequence)", df_clean, groups=df_clean["Subject"])
            mdf = md.fit(disp=False)

            # 從模型中提取對應的 p-value
            # statsmodels 會將類別變數轉為例如 "C(Session)[T.S2]"
            p_sess = mdf.pvalues.get("C(Session)[T.S2]", np.nan)

            # 尋找 Sequence 變數的 key
            seq_key = [k for k in mdf.pvalues.keys() if "C(Sequence)" in k and ":" not in k]
            p_seq = mdf.pvalues.get(seq_key[0], np.nan) if seq_key else np.nan

            # 尋找 Interaction (交互作用) 的 key
            int_key = [k for k in mdf.pvalues.keys() if ":" in k]
            p_int = mdf.pvalues.get(int_key[0], np.nan) if int_key else np.nan

            # 格式化函數，小於 0.05 標上星號
            def fmt_p(p):
                if pd.isna(p): return "N/A"
                return f"{p:.4f}" + ("*" if p < 0.05 else "")

            print(f"| {m:<10} | {fmt_p(p_sess):<20} | {fmt_p(p_seq):<20} | {fmt_p(p_int):<22} |")

        except Exception as e:
            print(f"| {m:<10} | 模型擬合失敗: {str(e)[:40]:<60} |")

    print("=" * 120)


# ==========================================
# 主要繪表邏輯 (提取 Metrics 並印出兩種檢定)
# ==========================================
def calculate_and_print_tables():
    ids = sorted([sub_int for sub_int in raw_data_13.keys()])

    g1_deltas = {'Accuracy': [], 'MBSR': [], 'MSFI': []}
    g2_deltas = {'Accuracy': [], 'MBSR': [], 'MSFI': []}

    le_vals = {'Accuracy': [], 'MBSR': [], 'MSFI': []}
    ae_vals = {'Accuracy': [], 'MBSR': [], 'MSFI': []}

    for sub in ids:
        data = raw_data_13[sub]

        s1_acc = data['s1']['off_sess'] * 100
        s2_acc = data['s2']['off_sess'] * 100

        s1_mbsr_raw = [float(x) for x in data['s1'].get('MBSR', []) if x != '...']
        s2_mbsr_raw = [float(x) for x in data['s2'].get('MBSR', []) if x != '...']
        s1_mbsr = np.mean(s1_mbsr_raw) if s1_mbsr_raw else np.nan
        s2_mbsr = np.mean(s2_mbsr_raw) if s2_mbsr_raw else np.nan

        s1_msfi_raw = [float(x) for x in data['s1'].get('MSFI', []) if x != '...']
        s2_msfi_raw = [float(x) for x in data['s2'].get('MSFI', []) if x != '...']
        s1_msfi = np.mean(s1_msfi_raw) if s1_msfi_raw else np.nan
        s2_msfi = np.mean(s2_msfi_raw) if s2_msfi_raw else np.nan

        cond_s1 = data['s1']['cond']

        if cond_s1 == 'A':  # Group 1 (Seq A: A -> B)
            g1_deltas['Accuracy'].append(s2_acc - s1_acc)
            g1_deltas['MBSR'].append(s2_mbsr - s1_mbsr)
            g1_deltas['MSFI'].append(s2_msfi - s1_msfi)

            ae_vals['Accuracy'].append(s2_acc - s1_acc)
            ae_vals['MBSR'].append(s2_mbsr - s1_mbsr)
            ae_vals['MSFI'].append(s2_msfi - s1_msfi)

        else:  # Group 2 (Seq B: B -> A)
            g2_deltas['Accuracy'].append(s2_acc - s1_acc)
            g2_deltas['MBSR'].append(s2_mbsr - s1_mbsr)
            g2_deltas['MSFI'].append(s2_msfi - s1_msfi)

            ae_vals['Accuracy'].append(s1_acc - s2_acc)
            ae_vals['MBSR'].append(s1_mbsr - s2_mbsr)
            ae_vals['MSFI'].append(s1_msfi - s2_msfi)

        le_vals['Accuracy'].append(s2_acc - s1_acc)
        le_vals['MBSR'].append(s2_mbsr - s1_mbsr)
        le_vals['MSFI'].append(s2_msfi - s1_msfi)

    metrics = ['Accuracy', 'MBSR', 'MSFI']

    print("\n" + "=" * 120)
    print("📊 Table 1: Between-group comparison (Group 1 vs Group 2) for Sequence Effect")
    print("   * W-Test = Mann-Whitney U test (Wilcoxon rank-sum test)")
    print("   * t-Test = Independent Two-Sample t-test")
    print("=" * 120)
    print(
        f"| {'Metric':<10} | {'Group 1 Mean ± SD':<18} | {'Group 2 Mean ± SD':<18} | {'Mean Diff':<12} | {'U-Stat':<8} | {'W-p-val':<8} | {'t-Stat':<8} | {'t-p-val':<8} |")
    print(f"|{'-' * 12}|{'-' * 20}|{'-' * 20}|{'-' * 14}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|")

    for m in metrics:
        v1 = np.array(g1_deltas[m])
        v2 = np.array(g2_deltas[m])

        v1 = v1[~np.isnan(v1)]
        v2 = v2[~np.isnan(v2)]

        m1, std1 = np.mean(v1), np.std(v1, ddof=1)
        m2, std2 = np.mean(v2), np.std(v2, ddof=1)
        diff = m1 - m2

        w_stat, w_p = safe_mannwhitneyu(v1, v2)
        t_stat, t_p = safe_ttest_ind(v1, v2)

        g1_str = f"{m1:.2f} ± {std1:.2f}"
        g2_str = f"{m2:.2f} ± {std2:.2f}"

        print(
            f"| {m:<10} | {g1_str:>18} | {g2_str:>18} | {diff:>12.2f} | {w_stat:>8.2f} | {w_p:>8.4f} | {t_stat:>8.2f} | {t_p:>8.4f} |")
    print("=" * 120)

    print("\n" + "=" * 125)
    print("📊 Table 2: Overall Learning Effect (LE) and Adaptive Effect (AE)")
    print("   * W-Test = Wilcoxon Signed-Rank Test against 0")
    print("   * t-Test = One-Sample t-test against 0")
    print("   * LE Diff = Session 2 - Session 1  |  AE Diff = Adaptive(Cond B) - Static(Cond A)")
    print("=" * 125)
    print(
        f"| {'Metric':<10} | {'LE Diff':<8} | {'LE W-Stat':<9} | {'LE W-p':<8} | {'LE t-Stat':<9} | {'LE t-p':<8} | {'AE Diff':<8} | {'AE W-Stat':<9} | {'AE W-p':<8} | {'AE t-Stat':<9} | {'AE t-p':<8} |")
    print(
        f"|{'-' * 12}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 11}|{'-' * 10}|")

    for m in metrics:
        le_arr = np.array(le_vals[m])
        ae_arr = np.array(ae_vals[m])

        le_arr = le_arr[~np.isnan(le_arr)]
        ae_arr = ae_arr[~np.isnan(ae_arr)]

        le_diff = np.mean(le_arr)
        le_w_stat, le_w_p = safe_wilcoxon(le_arr)
        le_t_stat, le_t_p = safe_ttest_1samp(le_arr)

        ae_diff = np.mean(ae_arr)
        ae_w_stat, ae_w_p = safe_wilcoxon(ae_arr)
        ae_t_stat, ae_t_p = safe_ttest_1samp(ae_arr)

        print(
            f"| {m:<10} | {le_diff:>8.2f} | {le_w_stat:>9.2f} | {le_w_p:>8.4f} | {le_t_stat:>9.2f} | {le_t_p:>8.4f} | {ae_diff:>8.2f} | {ae_w_stat:>9.2f} | {ae_w_p:>8.4f} | {ae_t_stat:>9.2f} | {ae_t_p:>8.4f} |")
    print("=" * 125)

    # 執行新增的 LMM 模型分析
    run_lmm_analysis()


if __name__ == "__main__":
    calculate_and_print_tables()
