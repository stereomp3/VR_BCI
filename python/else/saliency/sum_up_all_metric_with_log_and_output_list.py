import os
import re
from collections import defaultdict


# ============================================================
# 0. 基本設定
# ============================================================

# Offline：每個 run 的 accuracy
# 如果使用 13 channels，記得依你的檔案內容把 mi_22.pt 改成 mi.pt
file_offline_runs = "training_log_20260416_22.txt"

# Offline：整個 session 的 accuracy
file_offline_session = "training_log_20260416_22_all.txt"

# MBSR / MSFI 統計結果 txt
file_metric = "compute_saliency_metric.txt"

# Online log 資料夾
base_dir = (
    r"D:/CECNL_lab/lab_project/VR/"
    r"VR-BCI_beat_saber_python/re_make_txt/log_data"
)

log_name = "log.txt"

# Run 數量
NUM_RUNS = 7

# 缺失資料的暫時值
TMP_VALUE = 0.666


# ============================================================
# 1. S1 ~ S24 對應真正受試者 ID
# ============================================================
#
# metric_result.txt：
#
# S1  -> 35
# S2  -> 37
# S3  -> 38
# ...
# S24 -> 70
#
# 如果你的排序有改，只需要改這裡。
# ============================================================

subject_order = [
    "35",
    "37",
    "38",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "47",
    "48",
    "50",
    "51",
    "52",
    "54",
    "55",
    "57",
    "58",
    "63",
    "64",
    "65",
    "68",
    "69",
    "70",
]


# ============================================================
# 2. Session 1 Condition 對照表
# ============================================================
#
# Session 2 自動反轉：
#
# A -> B
# B -> A
#
# ============================================================

s1_cond_map = {
    "35": "B",
    "37": "A",
    "38": "B",
    "40": "B",
    "41": "B",
    "42": "A",
    "43": "A",
    "44": "B",
    "45": "B",
    "47": "B",
    "48": "B",
    "50": "A",
    "51": "B",
    "52": "B",
    "54": "B",
    "55": "A",
    "57": "B",
    "58": "A",
    "63": "A",
    "64": "A",
    "65": "A",
    "68": "A",
    "69": "A",
    "70": "A",
}


# ============================================================
# 3. 建立資料結構
# ============================================================

def create_default_session():
    return {
        "offline_run": {},
        "offline_session": "...",
        "online_run": {},

        # 新增的量化指標
        "MBSR": [],
        "MSFI": [],

        # 是否有缺失值
        "MBSR_tmp": False,
        "MSFI_tmp": False,
    }


data = defaultdict(
    lambda: defaultdict(create_default_session)
)


# ============================================================
# 4. 工具函式
# ============================================================

def parse_metric_value(value):
    """
    將 MBSR / MSFI 表格中的值轉成 float。

    例如：
        "28.64" -> 28.64
        "..."   -> 0.666
        ""      -> 0.666

    回傳：
        value, is_tmp
    """

    value = str(value).strip()

    if (
        value == ""
        or value == "..."
        or value.lower() == "nan"
        or value.lower() == "none"
    ):
        return TMP_VALUE, True

    try:
        return float(value), False

    except (ValueError, TypeError):
        return TMP_VALUE, True


def get_list_format(run_data, max_run=NUM_RUNS):
    """
    將：

        {
            1: 0.525,
            2: 0.550,
            4: 0.600
        }

    轉成：

        [
            0.525,
            0.550,
            0.666,
            0.600,
            ...
        ]

    缺少的 run 使用 TMP_VALUE。
    """

    vals = []
    has_tmp = False

    for run_id in range(1, max_run + 1):

        if (
            run_id in run_data
            and str(run_data[run_id]).strip() != "..."
        ):

            try:
                vals.append(float(run_data[run_id]))

            except (ValueError, TypeError):
                vals.append(TMP_VALUE)
                has_tmp = True

        else:
            vals.append(TMP_VALUE)
            has_tmp = True

    return vals, has_tmp


def format_normal_number(value, max_digits=3):
    """
    將一般 accuracy 數值輸出得比較乾淨。

    例如：
        0.600 -> 0.6
        0.525 -> 0.525
        0.550 -> 0.55
    """

    if abs(value - TMP_VALUE) < 1e-10:
        return "0.666"

    text = f"{value:.{max_digits}f}"

    text = text.rstrip("0").rstrip(".")

    if "." not in text:
        text += ".0"

    return text


def format_accuracy_list(values):
    """
    Online / Offline accuracy list
    """

    return "[" + ", ".join(
        format_normal_number(v)
        for v in values
    ) + "]"


def format_metric_list(values):
    """
    MBSR / MSFI 保留原本百分比的兩位小數。

    例如：
        28.64
        40.70
        33.80

    TMP_VALUE 維持：
        0.666
        改 ...
    """

    result = []

    for value in values:

        if abs(value - TMP_VALUE) < 1e-10:
            result.append('"..."')

        else:
            result.append(f"{value:.2f}")

    return "[" + ", ".join(result) + "]"


def reverse_condition(cond):
    """
    A <-> B
    """

    if cond == "A":
        return "B"

    return "A"


# ============================================================
# 5. 讀取 Offline 每個 Run
# ============================================================

print("=" * 80)
print("1. 讀取 Offline Run")
print("=" * 80)


if os.path.exists(file_offline_runs):

    with open(
        file_offline_runs,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        curr_id = None
        curr_sess = None
        curr_run = None

        for line in f:

            # =================================================
            # 例如：
            #
            # new data .../35/s1/run1/mi_22.pt
            #
            # 同時支援 / 與 \
            # =================================================

            match_new_data = re.search(
                r"new data .*?[\\/]"
                r"(\d+)[\\/]"
                r"(s\d+)[\\/]"
                r"run(\d+)[\\/]"
                r"mi_22\.pt",
                line,
                re.IGNORECASE,
            )

            if match_new_data:

                curr_id = match_new_data.group(1)

                curr_sess = (
                    match_new_data
                    .group(2)
                    .lower()
                )

                curr_run = int(
                    match_new_data.group(3)
                )

                continue


            # =================================================
            # 找 avg_val acc
            # =================================================

            if (
                curr_id is not None
                and curr_sess is not None
                and curr_run is not None
            ):

                match_acc = re.search(
                    r"avg_val acc:\s*([0-9.]+)",
                    line,
                    re.IGNORECASE,
                )

                if match_acc:

                    acc = float(
                        match_acc.group(1)
                    )

                    data[
                        curr_id
                    ][
                        curr_sess
                    ][
                        "offline_run"
                    ][
                        curr_run
                    ] = acc

                    curr_run = None

    print(
        f"讀取完成：{file_offline_runs}"
    )

else:

    print(
        f"[Warning] 找不到檔案：{file_offline_runs}"
    )


# ============================================================
# 6. 讀取 Offline 整個 Session
# ============================================================

print()
print("=" * 80)
print("2. 讀取 Offline Session")
print("=" * 80)


if os.path.exists(file_offline_session):

    with open(
        file_offline_session,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        curr_id = None
        curr_sess = None

        for line in f:

            # =================================================
            # subject 35, session s1 start
            # =================================================

            match_subject = re.search(
                r"subject\s+(\d+),"
                r"\s*session\s+(s\d+)"
                r"\s+start",
                line,
                re.IGNORECASE,
            )

            if match_subject:

                curr_id = (
                    match_subject
                    .group(1)
                )

                curr_sess = (
                    match_subject
                    .group(2)
                    .lower()
                )

                continue


            # =================================================
            # avg_val acc
            # =================================================

            if (
                curr_id is not None
                and curr_sess is not None
            ):

                match_acc = re.search(
                    r"avg_val acc:\s*([0-9.]+)",
                    line,
                    re.IGNORECASE,
                )

                if match_acc:

                    acc = float(
                        match_acc.group(1)
                    )

                    data[
                        curr_id
                    ][
                        curr_sess
                    ][
                        "offline_session"
                    ] = acc

                    curr_id = None
                    curr_sess = None

    print(
        f"讀取完成：{file_offline_session}"
    )

else:

    print(
        f"[Warning] 找不到檔案："
        f"{file_offline_session}"
    )


# ============================================================
# 7. 讀取 Online log.txt
# ============================================================

print()
print("=" * 80)
print("3. 讀取 Online Run")
print("=" * 80)


if os.path.exists(base_dir):

    online_file_count = 0

    for root, dirs, files in os.walk(base_dir):

        if log_name not in files:
            continue

        file_path = os.path.join(
            root,
            log_name,
        )

        # =====================================================
        # 找：
        #
        # .../35/s1/log.txt
        #
        # =====================================================

        path_match = re.search(
            r"[\\/](\d+)"
            r"[\\/](s\d+)"
            r"[\\/]log\.txt$",
            file_path,
            re.IGNORECASE,
        )

        if not path_match:
            continue

        curr_id = path_match.group(1)

        curr_sess = (
            path_match
            .group(2)
            .lower()
        )

        online_file_count += 1


        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            run_active_id = None

            for line in f:

                # =============================================
                # 目前 Run 編號: 1
                # =============================================

                run_match = re.search(
                    r"目前\s*Run\s*編號:\s*(\d+)",
                    line,
                    re.IGNORECASE,
                )

                if run_match:

                    run_active_id = int(
                        run_match.group(1)
                    )

                    continue


                # =============================================
                # Received from ('127.0.0.1', ...)
                #
                # ... = 0.525
                #
                # =============================================

                if run_active_id is not None:

                    acc_match = re.search(
                        r"Received from "
                        r"\('127\.0\.0\.1', .*?\)"
                        r":.*?=\s*([0-9.]+)",
                        line,
                        re.IGNORECASE,
                    )

                    if acc_match:

                        acc = float(
                            acc_match.group(1)
                        )

                        data[
                            curr_id
                        ][
                            curr_sess
                        ][
                            "online_run"
                        ][
                            run_active_id
                        ] = acc

                        run_active_id = None


    print(
        f"找到並讀取 {online_file_count} 個 log.txt"
    )

else:

    print(
        f"[Warning] 找不到 Online log 資料夾："
        f"{base_dir}"
    )


# ============================================================
# 8. 解析 MBSR / MSFI Markdown 表格
# ============================================================

def read_metric_txt(file_path):

    """
    讀取：

    ### 表格: MBSR (%) - Session S1
    ### 表格: MBSR (%) - Session S2
    ### 表格: MSFI (%) - Session S1
    ### 表格: MSFI (%) - Session S2

    最後得到：

    metric_data = {

        "MBSR": {
            "s1": {
                "S1": {
                    "values": [...]
                }
            },

            "s2": {...}
        },

        "MSFI": {
            "s1": {...},
            "s2": {...}
        }
    }
    """

    metric_data = {

        "MBSR": {
            "s1": {},
            "s2": {},
        },

        "MSFI": {
            "s1": {},
            "s2": {},
        },
    }


    if not os.path.exists(file_path):

        print(
            f"[Warning] 找不到 Metric txt："
            f"{file_path}"
        )

        return metric_data


    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        lines = f.readlines()


    current_metric = None
    current_session = None


    for line in lines:

        stripped = line.strip()


        # =====================================================
        # 判斷進入哪一張表
        # =====================================================

        title_match = re.search(
            r"表格\s*:\s*"
            r"(MBSR|MSFI)"
            r"\s*\(%\)"
            r"\s*-\s*"
            r"Session\s+(S1|S2)",
            stripped,
            re.IGNORECASE,
        )


        if title_match:

            current_metric = (
                title_match
                .group(1)
                .upper()
            )

            current_session = (
                title_match
                .group(2)
                .lower()
            )

            continue


        # =====================================================
        # 遇到新的 ###，
        # 如果不是 MBSR/MSFI table，
        # 就離開當前 table。
        # =====================================================

        if stripped.startswith("###"):

            if not re.search(
                r"表格\s*:\s*(MBSR|MSFI)",
                stripped,
                re.IGNORECASE,
            ):

                current_metric = None
                current_session = None

            continue


        # =====================================================
        # 沒有進入目標表格
        # =====================================================

        if (
            current_metric is None
            or current_session is None
        ):
            continue


        # =====================================================
        # 只處理 Markdown Table Row
        # =====================================================

        if not stripped.startswith("|"):
            continue


        # header 不要
        if "**id**" in stripped:
            continue


        # separator 不要
        if re.match(
            r"^\|\s*:?-+",
            stripped,
        ):
            continue


        # =====================================================
        # 拆欄位
        # =====================================================

        cols = [
            item.strip()
            for item in stripped.strip("|").split("|")
        ]


        if len(cols) < 2:
            continue


        row_id = cols[0].strip()


        # =====================================================
        # 只接受 S1 ~ S24
        #
        # 不接受 Mean
        # =====================================================

        if not re.fullmatch(
            r"S\d+",
            row_id,
            re.IGNORECASE,
        ):
            continue


        row_id = row_id.upper()


        values = []
        has_tmp = False


        # =====================================================
        # 只取前 7 個 Run
        #
        # 即使像 MSFI Session S2：
        #
        # | run1 | ... | run7 | 空白 |
        #
        # 多出欄位，也不影響。
        # =====================================================

        for run_index in range(NUM_RUNS):

            col_index = run_index + 1


            if col_index < len(cols):

                raw_value = cols[
                    col_index
                ]

            else:

                raw_value = "..."


            value, is_tmp = parse_metric_value(
                raw_value
            )


            values.append(value)


            if is_tmp:
                has_tmp = True


        metric_data[
            current_metric
        ][
            current_session
        ][
            row_id
        ] = {

            "values": values,

            "has_tmp": has_tmp,
        }


    return metric_data


# ============================================================
# 9. 讀取 Metric TXT
# ============================================================

print()
print("=" * 80)
print("4. 讀取 MBSR / MSFI")
print("=" * 80)


metric_data = read_metric_txt(
    file_metric
)


for metric_name in [
    "MBSR",
    "MSFI",
]:

    for session_id in [
        "s1",
        "s2",
    ]:

        count = len(
            metric_data[
                metric_name
            ][
                session_id
            ]
        )

        print(
            f"{metric_name} "
            f"{session_id}: "
            f"{count} 位受試者"
        )


# ============================================================
# 10. 將 MBSR / MSFI 對應到真正 subject ID
# ============================================================

print()
print("=" * 80)
print("5. 對應 MBSR / MSFI 到 Subject ID")
print("=" * 80)


for subject_index, subject_id in enumerate(
    subject_order
):

    # ========================================================
    # subject_order[0] = 35
    #
    # -> metric 裡面的 S1
    #
    # subject_order[1] = 37
    #
    # -> metric 裡面的 S2
    # ========================================================

    metric_subject_id = (
        f"S{subject_index + 1}"
    )


    for session_id in [
        "s1",
        "s2",
    ]:


        # =====================================================
        # MBSR
        # =====================================================

        mbsr_info = (
            metric_data
            .get("MBSR", {})
            .get(session_id, {})
            .get(metric_subject_id)
        )


        if mbsr_info is not None:

            data[
                subject_id
            ][
                session_id
            ][
                "MBSR"
            ] = mbsr_info["values"]


            data[
                subject_id
            ][
                session_id
            ][
                "MBSR_tmp"
            ] = mbsr_info["has_tmp"]


        else:

            data[
                subject_id
            ][
                session_id
            ][
                "MBSR"
            ] = (
                [TMP_VALUE] * NUM_RUNS
            )


            data[
                subject_id
            ][
                session_id
            ][
                "MBSR_tmp"
            ] = True


        # =====================================================
        # MSFI
        # =====================================================

        msfi_info = (
            metric_data
            .get("MSFI", {})
            .get(session_id, {})
            .get(metric_subject_id)
        )


        if msfi_info is not None:

            data[
                subject_id
            ][
                session_id
            ][
                "MSFI"
            ] = msfi_info["values"]


            data[
                subject_id
            ][
                session_id
            ][
                "MSFI_tmp"
            ] = msfi_info["has_tmp"]


        else:

            data[
                subject_id
            ][
                session_id
            ][
                "MSFI"
            ] = (
                [TMP_VALUE] * NUM_RUNS
            )


            data[
                subject_id
            ][
                session_id
            ][
                "MSFI_tmp"
            ] = True


print("對應完成")


# ============================================================
# 11. 建立真正的 raw_data dictionary
# ============================================================

raw_data = {}


for subject_id in subject_order:

    raw_data[
        int(subject_id)
    ] = {}


    for session_id in [
        "s1",
        "s2",
    ]:

        info = data[
            subject_id
        ][
            session_id
        ]


        # =====================================================
        # Online Runs
        # =====================================================

        online_list, online_tmp = get_list_format(
            info["online_run"],
            NUM_RUNS,
        )


        # =====================================================
        # Offline Runs
        # =====================================================

        offline_list, offline_tmp = get_list_format(
            info["offline_run"],
            NUM_RUNS,
        )


        # =====================================================
        # Offline Session
        # =====================================================

        offline_session = info[
            "offline_session"
        ]


        if (
            offline_session is None
            or str(offline_session).strip() == "..."
        ):

            offline_session_value = TMP_VALUE

        else:

            try:

                offline_session_value = float(
                    offline_session
                )

            except (
                ValueError,
                TypeError,
            ):

                offline_session_value = TMP_VALUE


        # =====================================================
        # Condition
        # =====================================================

        s1_cond = s1_cond_map.get(
            subject_id,
            "A",
        )


        if session_id == "s1":

            cond = s1_cond

        else:

            cond = reverse_condition(
                s1_cond
            )


        # =====================================================
        # MBSR
        # =====================================================

        mbsr = info.get(
            "MBSR",
            [TMP_VALUE] * NUM_RUNS,
        )


        # =====================================================
        # MSFI
        # =====================================================

        msfi = info.get(
            "MSFI",
            [TMP_VALUE] * NUM_RUNS,
        )


        # =====================================================
        # 建立 Session
        # =====================================================

        raw_data[
            int(subject_id)
        ][
            session_id
        ] = {

            "cond": cond,

            "on": online_list,

            "off_run": offline_list,

            "off_sess": offline_session_value,

            "MBSR": mbsr,

            "MSFI": msfi,
        }


# ============================================================
# 12. 自訂輸出 raw_data
# ============================================================
#
# 不直接 pprint，
# 因為我們希望：
#
# MBSR / MSFI 顯示兩位小數
#
# 例如：
#
# 40.70
# 33.80
#
# 而不是：
#
# 40.7
# 33.8
#
# ============================================================

def print_raw_data(raw_data):

    print()
    print("=" * 80)
    print("6. 最終 raw_data")
    print("=" * 80)
    print()

    print("raw_data = {")


    subject_ids = list(
        raw_data.keys()
    )


    for subject_index, subject_id in enumerate(
        subject_ids
    ):

        subject_data = raw_data[
            subject_id
        ]


        session_ids = [
            "s1",
            "s2",
        ]


        for session_index, session_id in enumerate(
            session_ids
        ):

            info = subject_data[
                session_id
            ]


            # =================================================
            # 格式化
            # =================================================

            on_str = format_accuracy_list(
                info["on"]
            )


            off_run_str = format_accuracy_list(
                info["off_run"]
            )


            off_sess_str = (
                format_normal_number(
                    info["off_sess"]
                )
            )


            mbsr_str = format_metric_list(
                info["MBSR"]
            )


            msfi_str = format_metric_list(
                info["MSFI"]
            )


            # =================================================
            # 第一個 Session
            # =================================================

            if session_index == 0:

                print(
                    f"    {subject_id}: "
                    f"{{'{session_id}': "
                    f"{{'cond': '{info['cond']}', "
                    f"'on': {on_str},"
                )

            else:

                print(
                    f"         "
                    f"'{session_id}': "
                    f"{{'cond': '{info['cond']}', "
                    f"'on': {on_str},"
                )


            # =================================================
            # Offline
            # =================================================

            print(
                f"                "
                f"'off_run': {off_run_str}, "
                f"'off_sess': {off_sess_str},"
            )


            # =================================================
            # MBSR
            # =================================================

            mbsr_line = (
                f"                "
                f"'MBSR': {mbsr_str},"
            )


            if any(
                abs(v - TMP_VALUE) < 1e-10
                for v in info["MBSR"]
            ):
                mbsr_line += (
                    "" # # 0.666 tmp
                )


            print(
                mbsr_line
            )


            # =================================================
            # MSFI
            # =================================================

            msfi_line = (
                f"                "
                f"'MSFI': {msfi_str}"
            )


            if any(
                abs(v - TMP_VALUE) < 1e-10
                for v in info["MSFI"]
            ):
                msfi_line += (
                    "" # 0.666 tmp
                )


            # =================================================
            # 括號
            # =================================================

            is_last_session = (
                session_index
                == len(session_ids) - 1
            )

            is_last_subject = (
                subject_index
                == len(subject_ids) - 1
            )


            if is_last_session:

                # 關閉：
                #
                # session }
                # subject }
                #
                msfi_line += "}}"


                if not is_last_subject:
                    msfi_line += ","


            else:

                # 關閉 session
                msfi_line += "},"


            print(
                msfi_line
            )


    print("}")


# ============================================================
# 13. 執行輸出
# ============================================================

print_raw_data(
    raw_data
)