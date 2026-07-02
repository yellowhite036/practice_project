import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


@dataclass
class Detection:
    frame_index: int
    bbox: BBox
    track_id: Optional[str] = None
    source_stage: str = "stage4"
    label: Optional[str] = "vehicle"

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="偵測影片中的橘色標註框(stage4)並用稀疏光流(LK)繪製追蹤箭頭與車輛歷史軌跡。"
    )
    parser.add_argument("--video", default="../../123.mp4", help="輸入影片路徑")
    parser.add_argument("--output", default="stage4_flow_tracks.mp4", help="輸出影片路徑")
    parser.add_argument("--boxes-output", help="若指定路徑，額外把 stage4 偵測框存成 JSON")

    # 框偵測相關參數
    parser.add_argument("--sample-step", type=int, default=1, help="每幾幀偵測一次橘色框；1 代表每幀")
    parser.add_argument("--min-width", type=int, default=30, help="候選框最小寬度")
    parser.add_argument("--min-height", type=int, default=25, help="候選框最小高度")
    parser.add_argument("--max-width", type=int, default=380, help="候選框最大寬度")
    parser.add_argument("--max-height", type=int, default=180, help="候選框最大高度")
    parser.add_argument("--max-height-ratio", type=float, default=0.55, help="候選框最大高度比例")
    parser.add_argument("--roi-x1", type=int, default=420, help="道路 ROI 左界")
    parser.add_argument("--roi-y1", type=int, default=360, help="道路 ROI 上界")
    parser.add_argument("--roi-x2", type=int, default=1450, help="道路 ROI 右界")
    parser.add_argument("--roi-y2", type=int, default=860, help="道路 ROI 下界")

    # 稀疏光流(Lucas-Kanade)相關參數
    parser.add_argument("--max-corners", type=int, default=40, help="每個框內最多取幾個特徵角點")
    parser.add_argument("--quality-level", type=float, default=0.05, help="goodFeaturesToTrack 品質門檻(越低取點越多)")
    parser.add_argument("--min-distance", type=int, default=5, help="角點之間最小間距(像素)")
    parser.add_argument("--lk-win-size", type=int, default=21, help="LK 光流搜尋視窗大小")
    parser.add_argument("--lk-max-level", type=int, default=3, help="LK 金字塔層數")
    parser.add_argument("--show-points", action="store_true", help="額外畫出追蹤的特徵角點")

    # 追蹤/繪製共用參數
    parser.add_argument("--history", type=int, default=60, help="每個物件保留幾幀中心點軌跡")
    parser.add_argument("--match-distance", type=float, default=80.0, help="無 track_id 時的中心點最近鄰匹配距離")
    parser.add_argument("--arrow-scale", type=float, default=5.0, help="光流箭頭放大倍率")
    parser.add_argument("--min-flow", type=float, default=0.35, help="低於此長度的平均位移視為靜止/不可靠")
    parser.add_argument(
        "--lane-split-x",
        type=int,
        help="左右車道分界的影像 x 座標；未指定時使用畫面寬度的一半",
    )
    parser.add_argument(
        "--left-approach-axis",
        choices=("x", "y"),
        default="y",
        help="左側車道迎面而來用哪個位移軸判斷正負號",
    )
    parser.add_argument(
        "--left-approach-sign",
        choices=("positive", "negative"),
        default="positive",
        help="左側車道迎面而來時該軸的正負號，影像座標 y 正向為向下",
    )
    parser.add_argument(
        "--right-depart-axis",
        choices=("x", "y"),
        default="y",
        help="右側車道背對駛離用哪個位移軸判斷正負號",
    )
    parser.add_argument(
        "--right-depart-sign",
        choices=("positive", "negative"),
        default="negative",
        help="右側車道背對駛離時該軸的正負號，影像座標 y 負向為向上",
    )
    parser.add_argument("--preview", action="store_true", help="處理時顯示視窗，按 q 離開")
    parser.add_argument("--progress-step", type=int, default=30, help="每處理幾幀更新一次進度顯示")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage4 橘色框偵測
# ---------------------------------------------------------------------------

def orange_mask(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame)
    # 標註框接近純 BGR 橘色；用 BGR 門檻而非 HSV，避免道路標線被誤判。
    orange = (
        (r > 210) & (g > 70) & (g < 190) & (b < 80)
        & ((r.astype(np.int16) - b.astype(np.int16)) > 150)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel)


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
        # 標註文字常貼在框頂端；框特別高時裁掉上緣。
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


# ---------------------------------------------------------------------------
# 稀疏光流(Lucas-Kanade)追蹤
# ---------------------------------------------------------------------------

def clip_bbox(bbox: BBox, width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sparse_flow_for_bbox(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    bbox: BBox,
    args: argparse.Namespace,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """在 bbox 範圍內取角點，用 LK 光流追蹤到目前幀，回傳平均位移 (dx, dy, strength, 追蹤到的點)。"""
    x1, y1, x2, y2 = bbox
    roi = prev_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, None

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=args.max_corners,
        qualityLevel=args.quality_level,
        minDistance=args.min_distance,
    )
    if corners is None or len(corners) == 0:
        return 0.0, 0.0, 0.0, None

    # 角點座標平移回全畫面座標系
    corners[:, 0, 0] += x1
    corners[:, 0, 1] += y1

    win_size = (args.lk_win_size, args.lk_win_size)
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, corners, None,
        winSize=win_size, maxLevel=args.lk_max_level, criteria=LK_CRITERIA,
    )
    if next_pts is None or status is None:
        return 0.0, 0.0, 0.0, None

    status = status.reshape(-1)
    good_old = corners[status == 1].reshape(-1, 2)
    good_new = next_pts[status == 1].reshape(-1, 2)
    if len(good_old) == 0:
        return 0.0, 0.0, 0.0, None

    displacement = good_new - good_old
    mag = np.linalg.norm(displacement, axis=1)

    if len(mag) > 1:
        threshold = max(0.05, float(np.percentile(mag, 60)))
        mask = mag >= threshold
        if not np.any(mask):
            mask = np.ones_like(mag, dtype=bool)
    else:
        mask = np.ones_like(mag, dtype=bool)

    dx = float(np.mean(displacement[mask, 0]))
    dy = float(np.mean(displacement[mask, 1]))
    strength = float(np.mean(mag[mask]))
    return dx, dy, strength, good_new


def classify_direction(
    center: Point,
    frame_width: int,
    dx: float,
    dy: float,
    min_flow: float,
    args: argparse.Namespace,
) -> Tuple[str, Tuple[int, int, int]]:
    magnitude = float(np.hypot(dx, dy))
    if magnitude < min_flow:
        return "static/weak", (160, 160, 160)

    split_x = args.lane_split_x if args.lane_split_x is not None else frame_width / 2
    is_left_lane = center[0] < split_x
    if is_left_lane:
        value = dx if args.left_approach_axis == "x" else dy
        if sign_matches(value, args.left_approach_sign):
            return "left:oncoming", (30, 30, 240)
        return "left:opposite", (0, 190, 255)

    value = dx if args.right_depart_axis == "x" else dy
    if sign_matches(value, args.right_depart_sign):
        return "right:departing", (240, 120, 20)
    return "right:opposite", (0, 220, 255)


def sign_matches(value: float, expected: str) -> bool:
    return value > 0 if expected == "positive" else value < 0


class TrackStore:
    def __init__(self, max_history: int, max_match_distance: float) -> None:
        self.max_history = max_history
        self.max_match_distance = max_match_distance
        self.histories: Dict[str, deque[Point]] = defaultdict(lambda: deque(maxlen=max_history))
        self.last_centers: Dict[str, Point] = {}
        self.used_ids_in_frame: set[str] = set()
        self.next_id = 1

    def begin_frame(self) -> None:
        self.used_ids_in_frame.clear()

    def resolve_id(self, detection: Detection) -> str:
        if detection.track_id is not None:
            return f"{detection.source_stage}:{detection.track_id}"

        center = detection.center
        best_id = None
        best_distance = self.max_match_distance
        for track_id, last_center in self.last_centers.items():
            if track_id in self.used_ids_in_frame:
                continue
            distance = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]))
            if distance < best_distance:
                best_id = track_id
                best_distance = distance

        if best_id is not None:
            return best_id

        track_id = f"{detection.source_stage}:auto-{self.next_id}"
        self.next_id += 1
        return track_id

    def update(self, track_id: str, center: Point) -> None:
        self.histories[track_id].append(center)
        self.last_centers[track_id] = center
        self.used_ids_in_frame.add(track_id)


def draw_track(frame: np.ndarray, points: "deque[Point]", color: Tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        thickness = 1 + int(3 * idx / max(1, len(points) - 1))
        cv2.line(frame, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    track_id: str,
    dx: float,
    dy: float,
    label: str,
    color: Tuple[int, int, int],
    arrow_scale: float,
) -> None:
    x1, y1, x2, y2 = detection.bbox
    center = detection.center
    end = (int(center[0] + dx * arrow_scale), int(center[1] + dy * arrow_scale))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(frame, center, end, color, 3, cv2.LINE_AA, tipLength=0.35)

    short_id = track_id.split(":", 1)[-1]
    text = f"{detection.source_stage} {short_id} {label} dx={dx:+.2f} dy={dy:+.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_points(frame: np.ndarray, points: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
    if points is None:
        return
    for px, py in points:
        cv2.circle(frame, (int(px), int(py)), 2, color, -1, cv2.LINE_AA)


def draw_legend(frame: np.ndarray) -> None:
    items = [
        ("left:oncoming", (30, 30, 240)),
        ("right:departing", (240, 120, 20)),
        ("opposite/unknown", (0, 220, 255)),
        ("static/weak", (160, 160, 160)),
    ]
    x, y = 12, 24
    for text, color in items:
        cv2.rectangle(frame, (x, y - 12), (x + 14, y + 2), color, -1)
        cv2.putText(frame, text, (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 22


# ---------------------------------------------------------------------------
# 進度顯示
# ---------------------------------------------------------------------------

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
        message = f"\r已處理 {frame_index} 幀  已耗時 {format_eta(elapsed)}"
    sys.stdout.write(message)
    sys.stdout.flush()


def to_frame_list(records: Dict[int, List[dict]]) -> List[dict]:
    return [{"frame_index": idx, "boxes": boxes} for idx, boxes in sorted(records.items())]


# ---------------------------------------------------------------------------
# 主流程：單一迴圈同時完成 stage4 偵測 + 稀疏光流(LK)追蹤繪製
# ---------------------------------------------------------------------------

def run() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ok, prev_frame = cap.read()
    if not ok:
        raise RuntimeError("Input video has no frames.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    tracks = TrackStore(args.history, args.match_distance)
    stage4_records: Dict[int, List[dict]] = defaultdict(list) if args.boxes_output else None
    box_count_total = 0

    frame_index = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Stage4 橘色框偵測 ---
        detections: List[Detection] = []
        if frame_index % args.sample_step == 0:
            mask = orange_mask(frame)
            for box in find_boxes(mask, frame.shape, args):
                detections.append(Detection(frame_index=frame_index, bbox=box))
                box_count_total += 1
            if stage4_records is not None and detections:
                stage4_records[frame_index].extend(
                    {"bbox": list(d.bbox), "bbox_format": "xyxy", "label": "vehicle"} for d in detections
                )

        # --- 稀疏光流(LK)追蹤與繪製 ---
        canvas = frame.copy()
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (80, 80, 80), 1, cv2.LINE_AA)
        tracks.begin_frame()

        for detection in detections:
            clipped = clip_bbox(detection.bbox, width, height)
            if clipped is None:
                continue
            detection.bbox = clipped

            dx, dy, _strength, tracked_points = sparse_flow_for_bbox(prev_gray, gray, clipped, args)
            track_id = tracks.resolve_id(detection)
            tracks.update(track_id, detection.center)
            label, color = classify_direction(detection.center, width, dx, dy, args.min_flow, args)

            draw_track(canvas, tracks.histories[track_id], color)
            draw_detection(canvas, detection, track_id, dx, dy, label, color, args.arrow_scale)
            if args.show_points:
                draw_points(canvas, tracked_points, color)

        draw_legend(canvas)
        writer.write(canvas)

        if args.preview:
            cv2.imshow("stage4 sparse flow tracks", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = gray

        if frame_index % args.progress_step == 0:
            print_progress(frame_index, total_frames, start_time)

    print_progress(frame_index, total_frames, start_time)
    sys.stdout.write("\n")
    sys.stdout.flush()

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    if stage4_records is not None:
        boxes_path = Path(args.boxes_output)
        boxes_path.parent.mkdir(parents=True, exist_ok=True)
        boxes_path.write_text(json.dumps(to_frame_list(stage4_records), indent=2), encoding="utf-8")
        print(f"saved_boxes={boxes_path}")

    print(f"frames={frame_index}")
    print(f"stage4_boxes_total={box_count_total}")
    print(f"saved_video={output_path}")


if __name__ == "__main__":
    run()