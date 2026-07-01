# run_pipeline.py
import subprocess
import sys
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


def select_mode():
    print("=" * 60)
    print("請選擇 Stage1 前處理模式")
    print("1. GSOC")
    print("2. KNN")
    print("3. MOG2")
    print("4. OPTICAL_FLOW")
    print("5. YOLO")
    print("=" * 60)

    while True:
        choice = input("請輸入 (1~5)：").strip()
        if choice in MODES:
            return MODES[choice]

        print(" 請輸入 1~5")


def main():
    mode = select_mode()

    steps = [
        (
            f"Stage1 - {mode['name']}",
            mode["workdir"],
            ["python", mode["script"]],
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
        ),
    ]

    for name, workdir, command in steps:
        print(f"\n{'=' * 60}")
        print(f" 執行：{name}")
        print(f" 目錄：{workdir}")
        print(f" 指令：{' '.join(command)}")
        print(f"{'=' * 60}")

        if not workdir.exists():
            print(f"找不到目錄：{workdir}")
            sys.exit(1)

        result = subprocess.run(command, cwd=str(workdir))

        if result.returncode != 0:
            print(f"「{name}」執行失敗（結束代碼 {result.returncode}）")
            sys.exit(result.returncode)

        print(f"「{name}」完成")

    print(f"\n{'=' * 60}")
    print("Pipeline 全部執行完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()