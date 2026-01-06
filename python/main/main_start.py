import sys
from datetime import datetime
from functools import wraps
import main.Utils.config as config
from main.game_state import Game


# 自動儲存 log
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
    """裝飾器：將 print 輸出同時存檔與顯示"""
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


# ----------------------------
# Main flow
# ----------------------------
@tee_log(f"{config.BASE_FILE}log.txt")
def main_flow():
    # marker_outlet = LSL.setup_lsl_outlet(config.TO_UNITY_LSL_STREAM)  # to unity
    game = Game()
    game.start()


if __name__ == "__main__":
    try:
        main_flow()
    except KeyboardInterrupt:
        print(f"{config.TAGS.WARNING.value} Interrupted by user. Exiting.")
