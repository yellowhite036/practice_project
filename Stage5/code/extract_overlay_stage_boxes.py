import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="從已疊加的藍/橘偵測框影片中抽出 Stage3/Stage4 候選框。")
    parser.add_argument("--video", required=True, help="輸入影片路徑")
    parser.add_argument("--stage3-output", default="outputs/stage3_overlay_boxes.json", help="藍色框輸出 JSON")
    parser.add_argument("--stage4-output", default="outputs/stage4_overlay_boxes.json", help="橘色框輸出 JSON")
    parser.add_argument("--sample-step", type=int, default=1, help="每幾幀抽一次；1 代表每幀")
    parser.add_argument("--min-width", type=int, default=30, help="候選框最小寬度")
    parser.add_argument("--min-height", type=int, default=25, help="候選框最小高度")
    parser.add_argument("--max-width", type=int, default=380, help="候選框最大寬度")
    parser.add_argument("--max-height", type=int, default=180, help="候選框最大高度")
    parser.add_argument("--max-height-ratio", type=float, default=0.55, help="候選框最大高度比例")
    parser.add_argument("--roi-x1", type=int, default=420, help="道路 ROI 左界")
    parser.add_argument("--roi-y1", type=int, default=360, help="道路 ROI 上界")
    parser.add_argument("--roi-x2", type=int, default=1450, help="道路 ROI 右界")
    parser.add_argument("--roi-y2", type=int, default=860, help="道路 ROI 下界")
    parser.add_argument("--progress-step", type=int, default=30, help="每處理幾幀更新一次進度顯示")
    return parser.parse_args()


def color_masks(frame: np.ndarray) -> Dict[str, np.ndarray]:
    b, g, r = cv2.split(frame)

    # The overlaid annotation boxes are near pure BGR blue/orange. BGR gating is
    # intentionally stricter than HSV so road lane paint is not mistaken as boxes.
    blue = (
        (b > 170)
        & (g > 70)
        & (g < 190)
        & (r < 90)
        & ((b.astype(np.int16) - r.astype(np.int16)) > 100)
    ).astype(np.uint8) * 255
    orange = (
        (r > 210)
        & (g > 70)
        & (g < 190)
        & (b < 80)
        & ((r.astype(np.int16) - b.astype(np.int16)) > 150)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return {
        "stage3": cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel),
        "stage4": cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel),
    }


def find_boxes(mask: np.ndarray, frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> List[BBox]:
    height, width = frame_shape[:2]
    roi_x1 = max(0, min(width - 1, args.roi_x1))
    roi_x2 = max(0, min(width, args.roi_x2))
    roi_y1 = max(0, min(height - 1, args.roi_y1))
    roi_y2 = max(0, min(height, args.roi_y2))
    road_mask = np.zeros_like(mask)
    road_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
    mask = cv2.bitwise_and(mask, road_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < args.min_width or h < args.min_height:
            continue
        if w > args.max_width or h > args.max_height:
            continue
        if h > height * args.max_height_ratio:
            continue
        if w > width * 0.45:
            continue

        # Annotation labels often touch the top border; trim them away when the box is unusually tall.
        if h > w * 0.85:
            y = y + int(h * 0.22)
            h = int(h * 0.78)

        boxes.append((x, y, x + w, y + h))

    return suppress_nested_boxes(boxes)


def suppress_nested_boxes(boxes: List[BBox]) -> List[BBox]:
    kept: List[BBox] = []
    for box in sorted(boxes, key=area, reverse=True):
        if all(iou(box, other) < 0.65 for other in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: (b[0], b[1]))


def area(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = area((ix1, iy1, ix2, iy2))
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def append_frame(records: Dict[int, List[dict]], frame_index: int, boxes: List[BBox]) -> None:
    if not boxes:
        return
    records[frame_index].extend(
        {
            "bbox": list(box),
            "bbox_format": "xyxy",
            "label": "vehicle",
        }
        for box in boxes
    )


def to_frame_list(records: Dict[int, List[dict]]) -> List[dict]:
    return [{"frame_index": idx, "boxes": boxes} for idx, boxes in sorted(records.items())]


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(frame_index: int, total_frames: int, start_time: float) -> None:
    elapsed = time.time() - start_time

    if total_frames > 0:
        percent = min(100.0, frame_index / total_frames * 100)
        message = (
            f"\r已處理 {frame_index}/{total_frames} 幀 "
            f"({percent:5.1f}%)  已耗時 {format_eta(elapsed)}"
        )
    else:
        # 無法取得總幀數時(例如某些串流),只顯示計數與耗時
        message = f"\r已處理 {frame_index} 幀  已耗時 {format_eta(elapsed)}"

    sys.stdout.write(message)
    sys.stdout.flush()


def run() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stage3_records: Dict[int, List[dict]] = defaultdict(list)
    stage4_records: Dict[int, List[dict]] = defaultdict(list)

    frame_index = 0
    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.sample_step == 0:
            masks = color_masks(frame)
            append_frame(stage3_records, frame_index, find_boxes(masks["stage3"], frame.shape, args))
            append_frame(stage4_records, frame_index, find_boxes(masks["stage4"], frame.shape, args))
        frame_index += 1

        if frame_index % args.progress_step == 0:
            print_progress(frame_index, total_frames, start_time)

    print_progress(frame_index, total_frames, start_time)
    sys.stdout.write("\n")
    sys.stdout.flush()

    cap.release()

    stage3_path = Path(args.stage3_output)
    stage4_path = Path(args.stage4_output)
    stage3_path.parent.mkdir(parents=True, exist_ok=True)
    stage4_path.parent.mkdir(parents=True, exist_ok=True)
    stage3_path.write_text(json.dumps(to_frame_list(stage3_records), indent=2), encoding="utf-8")
    stage4_path.write_text(json.dumps(to_frame_list(stage4_records), indent=2), encoding="utf-8")

    print(f"frames={frame_index}")
    print(f"stage3_frames={len(stage3_records)} boxes={sum(len(v) for v in stage3_records.values())}")
    print(f"stage4_frames={len(stage4_records)} boxes={sum(len(v) for v in stage4_records.values())}")
    print(f"saved={stage3_path}")
    print(f"saved={stage4_path}")


if __name__ == "__main__":
    run()