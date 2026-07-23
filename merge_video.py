import cv2
import numpy as np
import os
import argparse


def resize_keep_ratio(frame, target_w, target_h):
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


def main():
    parser = argparse.ArgumentParser(description="將原始影片與結果影片左右並排合併，方便比較")
    parser.add_argument("--original", required=True, help="原始影片路徑（放在左邊）")
    parser.add_argument("--processed", required=True, help="結果影片路徑（放在右邊）")
    parser.add_argument("--output", default="output.mp4", help="輸出檔案路徑")
    parser.add_argument("--width", type=int, default=960, help="單邊影片寬度（輸出總寬度會是 2 倍）")
    parser.add_argument("--height", type=int, default=720, help="單邊影片高度")
    parser.add_argument("--fps", type=int, default=30, help="輸出影片的 FPS")
    parser.add_argument("--no-preview", action="store_true", help="不開預覽視窗（自動化 / 無畫面環境請加上）")
    args = parser.parse_args()

    if not os.path.exists(args.original):
        print(f"找不到原始影片：{args.original}")
        return
    if not os.path.exists(args.processed):
        print(f"找不到結果影片：{args.processed}")
        return

    cap_original = cv2.VideoCapture(args.original)
    cap_processed = cv2.VideoCapture(args.processed)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_w, out_h = args.width, args.height
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (out_w * 2, out_h))

    label_original = os.path.basename(args.original)[:28]
    label_processed = os.path.basename(args.processed)[:28]

    frame_count = 0
    while True:
        ret_o, frame_o = cap_original.read()
        ret_p, frame_p = cap_processed.read()

        if not ret_o and not ret_p:
            break

        left = resize_keep_ratio(frame_o if ret_o else None, out_w, out_h)
        right = resize_keep_ratio(frame_p if ret_p else None, out_w, out_h)

        for frame, label in ((left, label_original), (right, label_processed)):
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

        combined = np.hstack((left, right))
        out.write(combined)

        if not args.no_preview:
            cv2.imshow('Compare (Press q to stop)', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已處理第 {frame_count} 幀", end='\r')

    cap_original.release()
    cap_processed.release()
    out.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    print(f"\n匯出完成！{args.output}（共 {frame_count} 幀）")


if __name__ == "__main__":
    main()