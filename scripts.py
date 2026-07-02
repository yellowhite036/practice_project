# run_pipeline.py
import itertools
import subprocess
import sys
import time
from pathlib import Path

# 專案根目錄（這支腳本所在的位置）
ROOT = Path(__file__).resolve().parent

# Stage1 五種模式
MODES = {
    "1": {
        "name": "GSOC",
        "workdir": ROOT / "Stage1" / "GSOC",
        "script": "GSOC.py",
        "output": ROOT / "Stage1" / "GSOC" / "GSOC.mp4",
    },
    "2": {
        "name": "KNN",
        "workdir": ROOT / "Stage1" / "KNN",
        "script": "KNN.py",
        "output": ROOT / "Stage1" / "KNN" / "KNN.mp4",
    },
    "3": {
        "name": "MOG2",
        "workdir": ROOT / "Stage1" / "MOG2",
        "script": "MOG2.py",
        "output": ROOT / "Stage1" / "MOG2" / "MOG2.mp4",
    },
    "4": {
        "name": "OPTICAL_FLOW",
        "workdir": ROOT / "Stage1" / "OPTICAL_FLOW",
        "script": "OPTICAL_FLOW.py",
        "output": ROOT / "Stage1" / "OPTICAL_FLOW" / "OPTICAL_FLOW.mp4",
    },
    "5": {
        "name": "YOLO",
        "workdir": ROOT / "Stage1" / "YOLO",
        "script": "YOLO.py",
        "output": ROOT / "Stage1" / "YOLO" / "YOLO.mp4",
    },
}

MODES1 = {
    "1": {
        "name": "DENSE_OPTICAL_FLOW",
        "workdir": ROOT / "Stage5" / "code",
        "script": "dense_optical_flow.py",
        "output": ROOT / "Stage5" / "code" / "DENSE_OPTICAL_FLOW.mp4",
    },
    "2": {
        "name": "SPARSE_OPTICAL_FLOW",
        "workdir": ROOT / "Stage5" / "code",
        "script": "sparse_optical_flow.py",
        "output": ROOT / "Stage5" / "code" / "SPARSE_OPTICAL_FLOW.mp4",
    },
}

MERGE_SCRIPT = ROOT / "merge_video.py"
MERGE_OUTPUT = ROOT / "output.mp4"

def parse_selection(raw: str, valid_keys: dict) -> list:
    """
    解析像 "1+2"、"1,3"、"1 2" 這樣的輸入，
    回傳去重後、依輸入順序排列的合法 key 清單。
    """
    # 把常見分隔符號都轉成空白，再依空白切割
    normalized = raw.replace("+", " ").replace(",", " ")
    tokens = normalized.split()

    selected = []
    for token in tokens:
        if token not in valid_keys:
            return None  # 有不合法的選項，讓外層重新輸入
        if token not in selected:
            selected.append(token)

    return selected if selected else None


def select_modes():
    print("=" * 60)
    print("請選擇 Stage1 前處理模式（可複選，例如 1+2 或 1,3,5）")
    print("1. GSOC")
    print("2. KNN")
    print("3. MOG2")
    print("4. OPTICAL_FLOW")
    print("5. YOLO")
    print("=" * 60)

    while True:
        raw = input("請輸入 (1~5，可用 + 或 , 分隔多個)：").strip()
        selected = parse_selection(raw, MODES)
        if selected:
            return [MODES[k] for k in selected]

        print("輸入格式錯誤，請確認選項介於 1~5 之間，例如：1+2")


def select_modes1():
    print("=" * 60)
    print("請選擇 Stage5 前處理模式（可複選，例如 1+2）")
    print("1. DENSE_OPTICAL_FLOW")
    print("2. SPARSE_OPTICAL_FLOW")
    print("=" * 60)

    while True:
        raw = input("請輸入 (1~2，可用 + 或 , 分隔多個)：").strip()
        selected = parse_selection(raw, MODES1)
        if selected:
            return [MODES1[k] for k in selected]

        print("輸入格式錯誤，請確認選項介於 1~2 之間，例如：1+2")


def format_duration(seconds: float) -> str:
    """用於畫面顯示的時分秒格式"""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 時 {minutes} 分 {secs} 秒"
    if minutes:
        return f"{minutes} 分 {secs} 秒"
    return f"{secs} 秒"


def format_duration_for_filename(seconds: float) -> str:
    """用於檔名的 分.秒 格式，例如 218 秒 -> '3.38'"""
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}.{secs:02d}"


def run_one_combo(mode: dict, mode5: dict) -> tuple:
    """
    執行單一 (Stage1 模式, Stage5 模式) 組合的完整 pipeline。
    回傳 (是否成功, 最終輸出路徑或錯誤訊息, 耗時秒數)。
    """
    combo_start = time.time()
    stage4_output = ROOT / "Stage4" / "code" / "YOLO_result.mp4"

    steps = [
        (
            f"Stage1 - {mode['name']}",
            mode["workdir"],
            ["python", mode["script"]],
            mode["output"],
        ),
        (
            "Stage4 - YOLO 車輛追蹤",
            ROOT / "Stage4" / "code",
            [
                "python",
                "vehicle_yolo_tracker.py",
                "--source",
                str(mode["output"]),
                "--output",
                "YOLO_result.mp4",
            ],
            stage4_output,
        ),
        (
            f"Stage5 - {mode5['name']}",
            mode5["workdir"],
            [
                "python",
                mode5["script"],
                "--video",
                str(stage4_output),
                "--output",
                str(mode5["output"]),
            ],
            mode5["output"],
        ),
    ]

    for name, workdir, command, expected_output in steps:
        print(f"\n{'=' * 60}")
        print(f" 組合：{mode['name']} + {mode5['name']}")
        print(f" 執行：{name}")
        print(f" 目錄：{workdir}")
        print(f" 指令：{' '.join(command)}")
        print(f"{'=' * 60}")

        if not workdir.exists():
            return False, f"找不到目錄：{workdir}", time.time() - combo_start

        result = subprocess.run(command, cwd=str(workdir))

        if result.returncode != 0:
            return False, f"「{name}」執行失敗（結束代碼 {result.returncode}）", time.time() - combo_start

        if not expected_output.exists():
            return False, f"「{name}」找不到預期輸出檔案：{expected_output}", time.time() - combo_start

        if expected_output.stat().st_size == 0:
            return False, f"「{name}」輸出檔案大小為 0：{expected_output}", time.time() - combo_start

        print(f"「{name}」完成，輸出檔案：{expected_output}（{expected_output.stat().st_size / 1024:.1f} KB）")

    combo_elapsed = time.time() - combo_start

    # ---- 重新命名為「stage1模式-stage5模式-耗時.mp4」 ----
    stage1_short = mode["name"].lower()
    stage5_short = mode5["name"].lower().replace("_optical_flow", "")
    duration_tag = format_duration_for_filename(combo_elapsed)

    final_name = f"{stage1_short}-{stage5_short}-{duration_tag}.mp4"
    final_path = mode5["output"].parent / final_name

    mode5["output"].rename(final_path)

    return True, final_path, combo_elapsed

def run_merge_step() -> tuple:
    """
    在所有組合執行完畢後，呼叫 merge_video.py 把 Stage5/code
    資料夾內的影片合併成一支 3x3 預覽影片。
    回傳 (是否成功, 輸出路徑或錯誤訊息, 耗時秒數)。
    """
    print(f"\n{'=' * 60}")
    print(" 執行：merge_video.py（合併所有結果影片）")
    print(f" 目錄：{ROOT}")
    print(f"{'=' * 60}")

    start = time.time()

    if not MERGE_SCRIPT.exists():
        return False, f"找不到合併腳本：{MERGE_SCRIPT}", time.time() - start

    result = subprocess.run(["python", MERGE_SCRIPT.name], cwd=str(ROOT))

    elapsed = time.time() - start

    if result.returncode != 0:
        return False, f"merge_video.py 執行失敗（結束代碼 {result.returncode}）", elapsed

    if not MERGE_OUTPUT.exists():
        return False, f"找不到預期輸出檔案：{MERGE_OUTPUT}", elapsed

    if MERGE_OUTPUT.stat().st_size == 0:
        return False, f"輸出檔案大小為 0：{MERGE_OUTPUT}", elapsed

    print(f"合併完成，輸出檔案：{MERGE_OUTPUT}（{MERGE_OUTPUT.stat().st_size / 1024:.1f} KB）")
    return True, MERGE_OUTPUT, elapsed

def main():
    pipeline_start = time.time()
    modes = select_modes()
    modes5 = select_modes1()

    combos = list(itertools.product(modes, modes5))

    print(f"\n共選擇 {len(modes)} 個 Stage1 模式 x {len(modes5)} 個 Stage5 模式 "
          f"= {len(combos)} 組合，將依序執行。\n")
    for i, (m1, m5) in enumerate(combos, start=1):
        print(f"  {i}. {m1['name']} + {m5['name']}")

    results = []  # (mode1_name, mode5_name, success, info, elapsed)

    for i, (mode, mode5) in enumerate(combos, start=1):
        print(f"\n{'#' * 60}")
        print(f"# 開始第 {i}/{len(combos)} 組合：{mode['name']} + {mode5['name']}")
        print(f"{'#' * 60}")

        success, info, elapsed = run_one_combo(mode, mode5)
        results.append((mode["name"], mode5["name"], success, info, elapsed))

        if success:
            print(f"\n組合完成：{mode['name']} + {mode5['name']}，"
                  f"耗時 {format_duration(elapsed)}，輸出：{info}")
        else:
            print(f"\n組合失敗：{mode['name']} + {mode5['name']}，原因：{info}")
            print("將繼續執行下一個組合...")

    pipeline_elapsed = time.time() - pipeline_start

    # ---- 總結報告 ----
    print(f"\n{'=' * 60}")
    print("所有組合執行完畢，總結如下：")
    print(f"{'=' * 60}")

    success_count = 0
    for m1_name, m5_name, success, info, elapsed in results:
        status = "成功" if success else "失敗"
        if success:
            success_count += 1
        print(f"[{status}] {m1_name} + {m5_name}（耗時 {format_duration(elapsed)}） -> {info}")

    print(f"\n總計：{success_count}/{len(results)} 組合成功")
    print(f"整體總花費時間：{format_duration(pipeline_elapsed)}")
    print(f"{'=' * 60}")

    # ---- 新增：合併所有結果影片 ----
    merge_success = True
    if success_count > 0:
        merge_success, merge_info, merge_elapsed = run_merge_step()
        if merge_success:
            print(f"\n影片合併完成，輸出：{merge_info}")
        else:
            print(f"\n影片合併失敗，原因：{merge_info}")
    else:
        print("\n沒有任何組合成功，略過影片合併步驟。")

    if success_count < len(results) or not merge_success:
        sys.exit(1)


if __name__ == "__main__":
    main()