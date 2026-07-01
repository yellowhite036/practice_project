import cv2
import numpy as np
import time

INPUT_VIDEO = "../../123.mp4"
OUTPUT_VIDEO = "gsoc.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("❌ 無法開啟影片")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W * 2, H))

bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC(
    mc=0,
    nSamples=20,
    replaceRate=0.003,
    propagationRate=0.01,
    hitsThreshold=32,
    alpha=0.01,
    beta=0.0022,
    blinkingSupressionDecay=0.1,
    blinkingSupressionMultiplier=0.1,
    noiseRemovalThresholdFacBG=0.0004,
    noiseRemovalThresholdFacFG=0.0008
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("建立背景模型中...")
for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    bg_subtractor.apply(frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.time()  # 開始計時

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_subtractor.apply(frame)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    fg_frame  = cv2.bilateralFilter(frame, 9, 75, 75)
    fg_result = cv2.bitwise_and(fg_frame, fg_frame, mask=fg_mask)

    left  = frame.copy()
    right = fg_result.copy()
    cv2.putText(left,  "Original",   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(right, "Foreground", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    out.write(np.hstack([left, right]))

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        print(f"進度：{frame_count} / {total_frames} 幀 | 已處理 {elapsed:.1f} 秒")

cap.release()
out.release()

total_elapsed = time.time() - start_time
print(f"✅ 完成，輸出：{OUTPUT_VIDEO}，總耗時 {total_elapsed:.1f} 秒")