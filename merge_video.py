import cv2
import numpy as np
import glob
import os

folder_path = 'results'

video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
video_paths = []
for ext in video_extensions:
    video_paths.extend(glob.glob(os.path.join(folder_path, ext)))

if not video_paths:
    print("沒有找到影片！")
    exit()

print(f"找到 {len(video_paths)} 個影片")
video_paths = video_paths[:9]
caps = [cv2.VideoCapture(path) for path in video_paths]

fps = 30
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (1440, 1080))

def resize_keep_ratio(frame, target_w=480, target_h=360):
    if frame is None or frame.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

frame_count = 0
while True:
    frames = []
    all_ended = True
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            all_ended = False
        processed = resize_keep_ratio(frame)
        
        filename = os.path.basename(video_paths[i])
        cv2.putText(processed, filename[:28], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed, filename[:28], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        frames.append(processed)
    
    if all_ended and frame_count > 0:
        break
    
    while len(frames) < 9:
        frames.append(np.zeros((360, 480, 3), dtype=np.uint8))
    
    row1 = np.hstack(frames[0:3])
    row2 = np.hstack(frames[3:6])
    row3 = np.hstack(frames[6:9])
    combined = np.vstack((row1, row2, row3))
    
    out.write(combined)
    cv2.imshow('3x3 Preview (Press q to stop)', combined)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"已處理第 {frame_count} 幀", end='\r')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n匯出完成！output.mp4 （共 {frame_count} 幀）")