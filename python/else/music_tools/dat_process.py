import json


def convert_dat_file(
        input_path,
        output_path,
        min_interval_sec=2.0,
        bpm=120.00,
        original_x=True,
        original_y=True,
        original_type=True,
        uniform_angle=0
):
    # 每秒幾拍
    beats_per_second = bpm / 60.0
    min_interval_beats = min_interval_sec * beats_per_second

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(data.keys())
    original_notes = data['_notes']
    original_notes.sort(key=lambda n: n['_time'])  # 依照時間排序

    converted_notes = []
    last_beat = -min_interval_beats  # 初始為負的確保第一個 note 可以加入

    for note in original_notes:
        current_beat = note['_time']
        if current_beat - last_beat < min_interval_beats:
            continue  # 跳過太近的 note

        new_note = {
            '_time': current_beat,
            '_lineIndex': note['_lineIndex'],
            '_lineLayer': note['_lineLayer'],
            '_type': note['_type'],
            '_cutDirection': note['_cutDirection']
        }

        # 調整 X 軸位置
        x = note['_lineIndex']
        if not original_x:
            x = 1 if x in [0, 1] else 2
        new_note['_lineIndex'] = x

        y = note['_lineLayer']
        if not original_y:
            y = 0
        new_note['_lineLayer'] = y

        _type = note['_type']
        if not original_type:
            _type = 0 if x == 1 or 0 else 1

        new_note['_type'] = _type

        # 統一角度 0
        new_note['_angleOffset'] = uniform_angle  # 或可用 "_angleOffset"、"_angle" 視版本而定

        # 砍擊方向：0 = 上，1 = 下，2 = 左，3 = 右，4~7 為斜砍，8 = 任意
        new_note['_cutDirection'] = 0  # 全部往下砍
        # if x in [0, 1]:  # 左邊
        #     new_note['_cutDirection'] = 2  # 往左砍
        # elif x in [2, 3]:
        #     new_note['_cutDirection'] = 3  # 往右砍

        converted_notes.append(new_note)
        last_beat = current_beat

    if len(converted_notes) % 2 == 1:  # 基數的話刪除第一個變為偶數，讓 label 可以平均
        del converted_notes[0]
    print(f"trials: {len(converted_notes)}")
    # 保留其他欄位
    data['_notes'] = converted_notes

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted file saved to: {output_path}")


# input_path = "data/SwordLand/Hard.dat"

# output_path = "data/SwordLand/MI_Hard.dat"
# output_path = "data/SwordLand/MI_Hard_5s.dat"

# input_path = "data/Golden/ExpertStandard_v2.dat"
# output_path = "data/Golden/MI_ExpertStandard_v2_5s.dat"

file_name = "ALL_OUT"
game_level = "ExpertPlusStandard"
# game_level = "ExpertStandard_v2"
input_path = f"data/{file_name}/{game_level}.dat"
# output_path = f"data/{file_name}/MI_{game_level}_5s.dat"
output_path = f"data/{file_name}/MI_{game_level}_07s.dat"
info_path = f"data/{file_name}/info.dat"
# 讀取 JSON 檔案
with open(info_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 取得 BPM
bpm = data["_beatsPerMinute"]
print("bpm: ", bpm)
convert_dat_file(
    input_path=input_path,
    output_path=output_path,
    min_interval_sec=0.7,  # 2.0s MI_Hard 4s MI_Hard_4s # 20251230 新版本 0.8
    bpm=bpm,
    original_x=False,  # 是否保留原始 x, False 為修改 x 為 0 1 為 1, 2 3 為 2 (左邊就左邊一個，右邊就右邊一個)
    original_y=False,  # 是否保留原始 y, False 為修改 y 為固定 1
    original_type=False,  # 是否保留原始 type, False 為修改 type 為左邊就紅色右邊就藍色
    uniform_angle=0  # 所有 angle 設為 0
)
