import cv2
import numpy as np
import time

INPUT_VIDEO = "202.TS"
OUTPUT_VIDEO = "303.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("❌ 無法開啟影片")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W * 2, H))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.time()  # 開始計時

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = mag_norm.astype(np.uint8)

    _, fg_mask = cv2.threshold(mag_norm, 2, 255, cv2.THRESH_BINARY)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    fg_frame  = cv2.bilateralFilter(frame, 9, 75, 75)
    fg_result = cv2.bitwise_and(fg_frame, fg_frame, mask=fg_mask)

    left  = frame.copy()
    right = fg_result.copy()
    cv2.putText(left,  "Original",        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(right, "Optical Flow FG", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    out.write(np.hstack([left, right]))

    prev_gray = curr_gray

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        print(f"進度：{frame_count} / {total_frames} 幀 | 已處理 {elapsed:.1f} 秒")

cap.release()
out.release()

total_elapsed = time.time() - start_time
print(f"✅ 完成，輸出：{OUTPUT_VIDEO}，總耗時 {total_elapsed:.1f} 秒")