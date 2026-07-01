#!/usr/bin/env python3
"""
ORB 特徵點抓取 - 影片版本
用法:
  python orb_video.py --input input.mp4 --output output.mp4
  python orb_video.py --input input.mp4 --preview          # 即時預覽（不存檔）
  python orb_video.py --input input.mp4 --output out.mp4 --preview
"""

import cv2
import sys
import time

def process_video(input_path: str,
                  output_path: str | None = None,
                  preview: bool = False,
                  nfeatures: int = 500,
                  scale_factor: float = 1.2,
                  nlevels: int = 8,
                  edge_threshold: int = 31,
                  score_type: int = cv2.ORB_HARRIS_SCORE,
                  draw_rich: bool = True):
    """
    對影片每幀執行 ORB 特徵點偵測並畫出關鍵點。

    參數:
        input_path    : 輸入影片路徑
        output_path   : 輸出影片路徑（None 則不存檔）
        preview       : 是否開啟即時預覽視窗
        nfeatures     : 最大特徵點數量
        scale_factor  : 影像金字塔縮放比例
        nlevels       : 金字塔層數
        edge_threshold : 邊緣閾值（與 patch_size 相同時最佳）
        score_type    : 評分方式 (ORB_HARRIS_SCORE / ORB_FAST_SCORE)
        draw_rich     : True = 畫出方向與大小；False = 只畫圓點
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[資訊] {width}x{height} @ {fps:.1f}fps，共 {total} 幀")

    # 建立 ORB 偵測器
    orb = cv2.ORB_create(
        nfeatures     = nfeatures,
        scaleFactor   = scale_factor,
        nlevels       = nlevels,
        edgeThreshold = edge_threshold,
        scoreType     = score_type,
    )

    # 建立輸出 VideoWriter
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[錯誤] 無法建立輸出檔案: {output_path}")
            sys.exit(1)
        print(f"[資訊] 輸出路徑: {output_path}")

    draw_flag = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                 if draw_rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = orb.detect(gray, None)
        keypoints, _ = orb.compute(gray, keypoints)  # 同時計算描述子

        # 畫關鍵點（綠色小點）
        vis = frame.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

        # 左上角 HUD
        n_kp = len(keypoints)
        elapsed = time.time() - t0
        cur_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"Keypoints: {n_kp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Frame: {frame_idx+1}/{total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"FPS: {cur_fps:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, f"Elapsed: {elapsed:.1f}s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if writer:
            writer.write(vis)

        if preview:
            cv2.imshow("ORB Feature Detection", vis)
            # 按 q 或 Esc 提前結束
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[資訊] 使用者中斷")
                break

        frame_idx += 1

        # 每 100 幀印進度
        if frame_idx % 100 == 0:
            print(f"  處理中... {frame_idx}/{total} ({100*frame_idx/total:.1f}%)")

    # 清理
    cap.release()
    if writer:
        writer.release()
        print(f"[完成] 已輸出 {frame_idx} 幀至: {output_path}")
    if preview:
        cv2.destroyAllWindows()

    total_time = time.time() - t0
    print(f"[完成] 共處理 {frame_idx} 幀，耗時 {total_time:.1f}s ({frame_idx/total_time:.1f} fps)")


def main():
    INPUT_PATH  = "../../123.mp4"
    OUTPUT_PATH = "orb.mp4"

    process_video(
        input_path    = INPUT_PATH,
        output_path   = OUTPUT_PATH,
        preview       = False,
        nfeatures     = 5000,
        scale_factor  = 1.2,
        nlevels       = 8,
        edge_threshold= 31,
        score_type    = cv2.ORB_HARRIS_SCORE,
        draw_rich     = True,
    )


if __name__ == "__main__":
    main()