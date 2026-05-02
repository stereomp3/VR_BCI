"""
Pointer-based 檔案讀取工具
用於 Calibration 場景中，記錄上次讀取位置，只讀取新增的資料，
避免每次 calibration 都重新建立檔案。

Author: Antigravity
Date: 2026-02-26
"""
import numpy as np
import re
import datetime
import time
import main.Utils.config as config


def readCsvFromPointer(path, channelIndex, linePointer=0):
    """
    從指定行數 (linePointer) 開始讀取 EEG CSV 資料。
    跳過 11 行 header 和 linePointer 行已讀的資料行。

    :param path: CSV 檔案路徑
    :param channelIndex: 要讀取的 channel index 列表
    :param linePointer: 上次讀取到的資料行數（不含 header）
    :return: (timestamps, eeg, newLinePointer)
        - timestamps: np.array, 時間戳
        - eeg: np.array, shape=(n_samples, n_channels)
        - newLinePointer: int, 更新後的 pointer
    """
    timestampsList = []
    eegList = []
    headerLines = 11  # CSV header 固定 11 行

    print(f"{config.TAGS.INFO.value} [DEBUG] readCsvFromPointer: path={path}, linePointer={linePointer}")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # 跳過 header
        for _ in range(headerLines):
            next(f, None)
        # 跳過已讀的資料行
        for _ in range(linePointer):
            next(f, None)
        # 讀取新的資料行
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= max(channelIndex) + 1:
                try:
                    timestampsList.append(float(parts[0]))
                    eegList.append([float(parts[x]) for x in channelIndex])
                except (ValueError, IndexError):
                    continue  # 跳過格式錯誤或不完整的行（可能是正在寫入的行）

    newDataCount = len(timestampsList)
    newLinePointer = linePointer + newDataCount

    print(f"{config.TAGS.INFO.value} [DEBUG] readCsvFromPointer: 讀取 {newDataCount} 筆新資料, "
          f"newLinePointer={newLinePointer}")

    if newDataCount == 0:
        return np.array([]), np.array([]).reshape(0, len(channelIndex)), newLinePointer

    return np.array(timestampsList), np.array(eegList), newLinePointer


def readLogFromPointer(path, linePointer=0):
    """
    從指定行數 (linePointer) 開始讀取 Unity LOG 檔案，
    解析 Trial START/CUT/END 格式。

    :param path: LOG 檔案路徑
    :param linePointer: 上次讀取到的行數
    :return: (trials, newLinePointer, newLineCount)
        - trials: dict, 和 EEGDataLoader._parse_log_file 格式相同
        - newLinePointer: int, 更新後的 pointer
        - newLineCount: int, 本次新讀取的行數
    """
    trials = {}
    patTrial = re.compile(
        r'Trial\s+(\d+)\s+(START|CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?',
        re.IGNORECASE
    )
    patEyes = re.compile(
        r'(Close eyes|Open eyes):\s*([\d\.]+)',
        re.IGNORECASE
    )
    newLineCount = 0

    print(f"{config.TAGS.INFO.value} [DEBUG] readLogFromPointer: path={path}, linePointer={linePointer}")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # 跳過已讀的行
        for _ in range(linePointer):
            next(f, None)
        # 讀取新的行
        for line in f:
            newLineCount += 1
            stripped = line.strip()
            if not stripped:
                continue
            m = patTrial.match(stripped)
            if m:
                idx = int(m.group(1))
                typ = m.group(2).lower()
                ts = float(m.group(3))
                label = int(m.group(4)) if m.group(4) is not None else None

                trialEntry = trials.setdefault(idx, {})
                if typ == 'cut':
                    trialEntry.setdefault('cut', []).append(ts)
                else:
                    trialEntry[typ] = ts
                if label is not None:
                    trialEntry['label'] = label

    newLinePointer = linePointer + newLineCount

    print(f"{config.TAGS.INFO.value} [DEBUG] readLogFromPointer: 讀取 {newLineCount} 行新資料, "
          f"解析 {len(trials)} 個 trials, newLinePointer={newLinePointer}")

    return trials, newLinePointer, newLineCount


def extractRecordEpoch(path):
    """
    從 CSV header 提取 Record datetime 的 epoch 時間戳。
    與 EEGDataLoader._extract_record_epoch 邏輯相同。

    :param path: CSV 檔案路徑
    :return: float, epoch 時間戳
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(10):
            line = f.readline()
            if 'Record datetime' in line:
                m = re.search(r'Record datetime:\s*([0-9\- :\.]+)', line)
                localTimezoneOffset = time.localtime().tm_gmtoff
                tz = datetime.timezone(datetime.timedelta(seconds=localTimezoneOffset))
                dt = datetime.datetime.strptime(
                    m.group(1).strip(), '%Y-%m-%d %H:%M:%S.%f'
                ).replace(tzinfo=tz)
                return dt.timestamp()
    raise ValueError('找不到 Record datetime')
