import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import time
from ultralytics import YOLO

DEFAULT_SOURCE = "../../123.mp4"
DEFAULT_OUTPUT = "YOLO.mp4"

parser = argparse.ArgumentParser(description="YOLOv8-seg 車輛去背")
parser.add_argument("--source", default=DEFAULT_SOURCE, help="輸入影片路徑")
parser.add_argument("--output", default=DEFAULT_OUTPUT, help="去背結果輸出路徑")
args = parser.parse_args()

INPUT_VIDEO = args.source
OUTPUT_FOREGROUND = args.output

# ==================== 修改：合併影片輸出到 combine 子資料夾 ====================
_output_path = Path(OUTPUT_FOREGROUND)
COMBINE_DIR = _output_path.parent / "combine"
COMBINE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMBINED = str(COMBINE_DIR / f"{_output_path.stem}_combined{_output_path.suffix}")
# ============================================================================

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print(f"  無法開啟影片：{INPUT_VIDEO}")
    sys.exit(1)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_combined = cv2.VideoWriter(OUTPUT_COMBINED, fourcc, fps, (W * 2, H))
out_foreground = cv2.VideoWriter(OUTPUT_FOREGROUND, fourcc, fps, (W, H))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]

    fg_mask = np.zeros((H, W), dtype=np.uint8)

    if result.masks is not None:
        for i, cls in enumerate(result.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:

                mask = result.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                fg_mask = np.maximum(fg_mask, (mask * 255).astype(np.uint8))

                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(result.boxes.conf[i])
                label = f"{VEHICLE_CLASSES[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (box[0], box[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    fg_frame = cv2.bilateralFilter(frame, 9, 75, 75)
    fg_result = cv2.bitwise_and(fg_frame, fg_frame, mask=fg_mask)

    # 輸出一：左右拼接對照（存到 combine 資料夾）
    left = frame.copy()
    right = fg_result.copy()
    cv2.putText(left, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(right, "YOLOv8-seg FG", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    out_combined.write(np.hstack([left, right]))

    # 輸出二：只有去背結果（維持原位置）
    out_foreground.write(fg_result)

    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        print(f"進度：{frame_count} / {total_frames} 幀 | 已處理 {elapsed:.1f} 秒")

cap.release()
out_combined.release()
out_foreground.release()

total_elapsed = time.time() - start_time
print(f"完成！")
print(f"   • 去背版：{OUTPUT_FOREGROUND}")
print(f"   • 合併對照版：{OUTPUT_COMBINED}")
print(f"   總耗時 {total_elapsed:.1f} 秒")