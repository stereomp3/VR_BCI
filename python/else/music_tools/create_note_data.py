import json
from pathlib import Path
from PIL import Image
import random

# ============================================================
# 基本設定
# ============================================================

FILE_NAME = "Generated_MI"
IMG_NAME = "image.png"
OUTPUT_PATH = Path(f"data/{FILE_NAME}/MI.dat")
INFO_OUTPUT_PATH = Path(f"data/{FILE_NAME}/Info.dat")
IMG_OUTPUT_PATH = Path(f"data/{FILE_NAME}/{IMG_NAME}")

# BPM 音樂設定 統一為 60
BPM = 60.0

# Trial 數量
TRIAL_COUNT = 48

# 每個 Trial 有幾個 Note # 因為預設會跳過一個 Note 所以這邊要多 +1
NOTES_PER_TRIAL = 6

# ------------------------------------------------------------
# Trial 內 Note 的間隔
#
# 單位：Beat
#
# BPM = 120
# 1 beat = 0.5 秒
#
# 例如：
# 1.0 beat = 0.5 秒
# 1.5 beat = 0.75 秒
# 2.0 beat = 1.0 秒
# ------------------------------------------------------------
NOTE_INTERVAL_BEATS = 1

# ------------------------------------------------------------
# Trial 與下一個 Trial 的額外間隔
#
# 例如：
#
# Trial 1 最後一個 note
# ↓
# TRIAL_GAP_BEATS
# ↓
# Trial 2 第一個 note
# ------------------------------------------------------------
TRIAL_GAP_BEATS = 1.0  # 因為 Unity 那邊會跳過一個，所以 interval 要再加上 NOTE_INTERVAL_BEATS

# 第一個 Note 開始時間
START_BEAT = 4.0

# ------------------------------------------------------------
# Note 位置
#
# 左邊 = 1
# 右邊 = 2
#
# 你的 Unity：
#
# position = lineIndex - 1.5
#
# 所以：
# x=1 -> -0.5
# x=2 -> +0.5
# ------------------------------------------------------------
LEFT_X = 1
RIGHT_X = 2

# Y 位置
NOTE_Y = 0

# Cut Direction
#
# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right
# 4 = Up-Left
# 5 = Up-Right
# 6 = Down-Left
# 7 = Down-Right
# 8 = Any
CUT_DIRECTION = 0

# Angle
ANGLE_OFFSET = 0


# ============================================================
# 產生單一 Note
# ============================================================

def create_note(
        time_beat: float,
        line_index: int,
        line_layer: int = NOTE_Y,
        note_type: int = 0,
        cut_direction: int = CUT_DIRECTION,
):
    return {
        "_time": round(time_beat, 4),
        "_lineIndex": line_index,
        "_lineLayer": line_layer,
        "_type": note_type,
        "_cutDirection": cut_direction,
        "_angleOffset": ANGLE_OFFSET,
    }


# ============================================================
# 產生 Beatmap
# ============================================================

def generate_beatmap():
    notes = []

    current_time = START_BEAT

    for trial_index in range(TRIAL_COUNT):

        # ----------------------------------------------------
        # Trial 左右方向
        #
        # 偶數 Trial → 左紅
        # 奇數 Trial → 右藍
        #
        # 如果想隨機，可以在這裡修改
        # ----------------------------------------------------

        if trial_index % 2 == 0:
            line_index = LEFT_X
            note_type = 0  # Red
            side = "LEFT"
        else:
            line_index = RIGHT_X
            note_type = 1  # Blue
            side = "RIGHT"

        trial_notes = []

        # ----------------------------------------------------
        # 產生這個 Trial 的 5 個 Note
        # ----------------------------------------------------

        for note_index in range(NOTES_PER_TRIAL):
            note_time = (
                    current_time
                    + note_index * NOTE_INTERVAL_BEATS
            )

            note = create_note(
                time_beat=note_time,
                line_index=line_index,
                line_layer=NOTE_Y,
                note_type=note_type,
                cut_direction=CUT_DIRECTION,
            )

            notes.append(note)
            trial_notes.append(note)

        # ----------------------------------------------------
        # 印出 Trial 資訊
        # ----------------------------------------------------

        first_time = trial_notes[0]["_time"]
        last_time = trial_notes[-1]["_time"]

        duration_beats = last_time - first_time

        duration_seconds = duration_beats * 60.0 / BPM

        print(
            f"Trial {trial_index + 1:02d} | "
            f"{side:5s} | "
            f"Notes={len(trial_notes)} | "
            f"Beat={first_time:.2f} -> {last_time:.2f} | "
            f"Duration={duration_seconds:.2f}s"
        )

        # ----------------------------------------------------
        # 下一個 Trial
        #
        # 最後一個 Note
        # +
        # Trial Gap
        # ----------------------------------------------------

        current_time = (
                trial_notes[-1]["_time"]
                + TRIAL_GAP_BEATS
        )

    # ========================================================
    # 建立 V2 Beatmap
    # ========================================================

    beatmap = {
        "_version": "2.2.0",

        "_beatsPerMinute": BPM,

        "_customData": {},

        "_notes": notes,

        "_obstacles": [],

        "_events": [],

        "_BPMChanges": [],
    }

    return beatmap


# ============================================================
# 儲存
# ============================================================

def save_beatmap(data, output_path: Path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)

    print(f"Output : {output_path}")
    print(f"BPM    : {BPM}")
    print(f"Trials : {TRIAL_COUNT}")
    print(f"Notes / Trial : {NOTES_PER_TRIAL}")
    print(f"Total Notes   : {len(data['_notes'])}")

    total_notes = TRIAL_COUNT * NOTES_PER_TRIAL

    assert len(data["_notes"]) == total_notes

    print()
    print(
        f"Expected Notes : {TRIAL_COUNT} × "
        f"{NOTES_PER_TRIAL} = {total_notes}"
    )

    # print()
    # print("First 5 Notes:")
    #
    # print(
    #     json.dumps(
    #         data["_notes"][:5],
    #         ensure_ascii=False,
    #         indent=2
    #     )
    # )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    beatmap = generate_beatmap()

    save_beatmap(
        beatmap,
        OUTPUT_PATH
    )

    info_data = {
        "_version": "2.0.0",
        "_songName": "AI",
        "_songAuthorName": "AI",
        "_levelAuthorName": "AI",
        "_beatsPerMinute": BPM,
        "_songFilename": "song.mp3",  # 使用 https://suno.com 生成
        "_coverImageFilename": IMG_NAME,
    }

    with open(INFO_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)



    # 隨機產生 RGB 顏色
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # 建立 1×1 像素的圖片
    image = Image.new("RGB", (1, 1), color)

    # 儲存成 PNG
    image.save(IMG_OUTPUT_PATH)

    print(f"產生的顏色：RGB{color}")

