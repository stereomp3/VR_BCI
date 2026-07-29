import os
import re
from collections import defaultdict

# 定義檔案路徑 **如果使用 13 需要改下面的 mi_22 變成 mi**
file_offline_runs = "training_log_20260416_22.txt"
file_offline_session = "training_log_20260416_22_all.txt"
base_dir = r"D:/CECNL_lab/lab_project/VR/VR-BCI_beat_saber_python/re_make_txt/log_data"
log_name = "log.txt"

# 建立儲存結構: data[id_num][session_num] = {'offline_run': {}, 'offline_session': '...', 'online_run': {}}
data = defaultdict(lambda: defaultdict(lambda: {'offline_run': {}, 'offline_session': '...', 'online_run': {}}))

# ==========================================
# 1. 處理 training_log_20260312.txt (Offline acc 6 run)
# ==========================================
if os.path.exists(file_offline_runs):
    with open(file_offline_runs, 'r', encoding='utf-8') as f:
        curr_id, curr_sess, curr_run = None, None, None
        for line in f:
            match_new_data = re.search(r'new data .*/(\d+)/(s\d+)/run(\d+)/mi_22\.pt', line)
            if match_new_data:
                curr_id = match_new_data.group(1)
                curr_sess = match_new_data.group(2)
                curr_run = int(match_new_data.group(3))
            if curr_id and curr_sess and curr_run:
                match_acc = re.search(r'avg_val acc:\s*([0-9.]+)', line)

                if match_acc:
                    acc = float(match_acc.group(1))
                    data[curr_id][curr_sess]['offline_run'][curr_run] = f"{acc:.3f}"
                    curr_run = None

                # ==========================================
# 2. 處理 training_log_20260312_all.txt (Offline acc 1 session)
# ==========================================
if os.path.exists(file_offline_session):
    with open(file_offline_session, 'r', encoding='utf-8') as f:
        curr_id, curr_sess = None, None
        for line in f:
            match_subject = re.search(r'subject (\d+), session (s\d+) start', line)
            if match_subject:
                curr_id = match_subject.group(1)
                curr_sess = match_subject.group(2)

            if curr_id and curr_sess:
                match_acc = re.search(r'avg_val acc:\s*([0-9.]+)', line)
                if match_acc:
                    acc = float(match_acc.group(1))
                    data[curr_id][curr_sess]['offline_session'] = f"{acc:.3f}"
                    curr_sess = None

                # ==========================================
# 3. 處理 base_dir 下各個 log.txt (Online acc 6 run)
# ==========================================
if os.path.exists(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if log_name in files:
            file_path = os.path.join(root, log_name)
            path_match = re.search(r'[\\/](\d+)[\\/](s\d+)[\\/]log\.txt$', file_path)

            if path_match:
                curr_id = path_match.group(1)
                curr_sess = path_match.group(2)

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    run_active_id = None
                    for line in f:
                        run_match = re.search(r'目前 Run 編號:\s*(\d+)', line)
                        if run_match:
                            run_active_id = int(run_match.group(1))

                        if run_active_id is not None:
                            acc_match = re.search(r"Received from \('127\.0\.0\.1', .*?\):.*?=\s*([0-9.]+)", line)
                            if acc_match:
                                acc = float(acc_match.group(1))
                                data[curr_id][curr_sess]['online_run'][run_active_id] = f"{acc:.3f}"
                                run_active_id = None
            else:
                pass  # 可開啟 debug 輸出 print(f"[Debug] 無法解析路徑格式: {file_path}")


# ==========================================
# 4. 格式化為嵌套字典 (raw_data = {...}) 包含 0.666 處理
# ==========================================
def get_list_format(run_data, max_run):
    """
    將資料轉為 List，若缺失或為 '...' 則替換為 0.666
    並回傳一個 Boolean 值 (has_tmp) 標記是否有替換發生
    """
    if max_run == 0:
        return [], False
    vals = []
    has_tmp = False
    for i in range(1, max_run + 1):
        if i in run_data and str(run_data[i]) != '...':
            vals.append(float(run_data[i]))
        else:
            vals.append(0.666)
            has_tmp = True
    return vals, has_tmp


# 建立受試者 Session 1 的條件對照表 (基於先前的表格資料)
s1_cond_map = {
    '35': 'B', '37': 'A', '38': 'B', '40': 'B', '41': 'B', '42': 'A',
    '43': 'A', '44': 'B', '45': 'B', '47': 'B', '48': 'B', '50': 'A',
    '51': 'B', '52': 'B', '54': 'B', '55': 'A', '57': 'B', '58': 'A',
    '63': 'A', '64': 'A', '65': 'A', '68': 'A', '69': 'A', '70': 'A'
}

print("raw_data = {")
sorted_subjects = sorted(data.keys(), key=lambda x: int(x))

for i, subject_id in enumerate(sorted_subjects):
    sorted_sessions = sorted(data[subject_id].keys())

    for j, session_id in enumerate(sorted_sessions):
        info = data[subject_id][session_id]

        # 找最大 run
        online_keys = list(info['online_run'].keys())
        offline_keys = list(info['offline_run'].keys())
        all_keys = online_keys + offline_keys
        max_run = max(all_keys) if all_keys else 0

        # 取出陣列並偵測是否有 0.666 tmp
        online_list, on_tmp = get_list_format(info['online_run'], max_run)
        offline_list, off_tmp = get_list_format(info['offline_run'], max_run)

        # 處理 session 單一數值
        offline_sess = info['offline_session']
        if str(offline_sess) == '...':
            offline_sess_val = 0.666
            sess_tmp = True
        else:
            offline_sess_val = float(offline_sess)
            sess_tmp = False

        # 判定 Condition
        base_cond = s1_cond_map.get(subject_id, 'A')
        if session_id == 's1':
            cond = base_cond
        else:
            cond = 'B' if base_cond == 'A' else 'A'

        # ------------------------------------------
        # 建立上半部字串 (online)
        # ------------------------------------------
        on_str = f"'cond': '{cond}', 'on': {online_list},"
        if on_tmp:
            on_str += "  # 0.666 tmp"

        # ------------------------------------------
        # 建立下半部字串 (offline) 與結尾括號處理
        # ------------------------------------------
        off_str = f"'off_run': {offline_list}, 'off_sess': {offline_sess_val}"

        is_last_sess = (j == len(sorted_sessions) - 1)
        is_last_subj = (i == len(sorted_subjects) - 1)

        # 決定結尾要幾個大括號跟逗號
        if is_last_sess:
            off_str += "}}"
            if not is_last_subj:
                off_str += ","
        else:
            off_str += "},"

        # 最後才加上註解，確保不會把括號註解掉
        if off_tmp or sess_tmp:
            off_str += "  # 0.666 tmp"

        # ------------------------------------------
        # 執行列印
        # ------------------------------------------
        if j == 0:
            print(f"    {subject_id}: {{'{session_id}': {{{on_str}")
        else:
            print(f"         '{session_id}': {{{on_str}")

        print(f"                {off_str}")

print("}")