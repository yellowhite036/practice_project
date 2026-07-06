import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import time

DEFAULT_SOURCE = "../../123.mp4"
DEFAULT_OUTPUT = "GSOC.mp4"

parser = argparse.ArgumentParser(description="Frame Differencing 去背")
parser.add_argument("--source", default=DEFAULT_SOURCE, help="輸入影片路徑")
parser.add_argument("--output", default=DEFAULT_OUTPUT, help="去背結果輸出路徑")
args = parser.parse_args()

INPUT_VIDEO = args.source
OUTPUT_FOREGROUND = args.output

# 合併影片存到 combine 子資料夾
_output_path = Path(OUTPUT_FOREGROUND)
COMBINE_DIR = _output_path.parent / "combine"
COMBINE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_COMBINED = str(COMBINE_DIR / f"{_output_path.stem}_combined{_output_path.suffix}")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print(f"無法開啟影片：{INPUT_VIDEO}")
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_combined = cv2.VideoWriter(OUTPUT_COMBINED, fourcc, fps, (W * 2, H))
out_foreground = cv2.VideoWriter(OUTPUT_FOREGROUND, fourcc, fps, (W, H))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ret, prev_frame = cap.read()
if not ret:
    print("無法讀取第一幀")
    sys.exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 影格差分
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # 自適應二值化
    _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 形態學處理
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # 應用遮罩
    fg_frame = cv2.bilateralFilter(frame, 9, 75, 75)
    fg_result = cv2.bitwise_and(fg_frame, fg_frame, mask=fg_mask)

    # 左右對照
    left = frame.copy()
    right = fg_result.copy()
    cv2.putText(left, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(right, "FrameDiff FG", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    out_combined.write(np.hstack([left, right]))

    out_foreground.write(fg_result)

    prev_gray = curr_gray.copy()

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        print(f"進度：{frame_count} / {total_frames} 幀 | 已處理 {elapsed:.1f} 秒")

cap.release()
out_combined.release()
out_foreground.release()

total_elapsed = time.time() - start_time
print(f"Frame Differencing 完成！")
print(f"   • 去背版：{OUTPUT_FOREGROUND}")
print(f"   • 合併對照版：{OUTPUT_COMBINED}")
print(f"   總耗時 {total_elapsed:.1f} 秒")