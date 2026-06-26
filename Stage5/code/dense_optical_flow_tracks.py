import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    frame_index: int
    bbox: BBox
    track_id: Optional[str]
    source_stage: str
    score: Optional[float] = None
    label: Optional[str] = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="局部密集光流遮罩、Stage3/Stage4 物件框光流箭頭與車輛歷史軌跡繪製。"
    )
    parser.add_argument("--video", required=True, help="輸入影片路徑")
    parser.add_argument("--stage3", help="Stage3 偵測框 JSON/CSV")
    parser.add_argument("--stage4", help="Stage4 偵測框 JSON/CSV")
    parser.add_argument("--output", default="outputs/flow_tracks.mp4", help="輸出影片路徑")
    parser.add_argument(
        "--use-stages",
        default="stage3,stage4",
        help="要作為遮罩的來源，例如 stage3,stage4 或只填 stage4",
    )
    parser.add_argument("--history", type=int, default=60, help="每個物件保留幾幀中心點軌跡")
    parser.add_argument("--match-distance", type=float, default=80.0, help="無 track_id 時的中心點最近鄰匹配距離")
    parser.add_argument("--arrow-scale", type=float, default=5.0, help="光流箭頭放大倍率")
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=1.0,
        help="密集光流計算縮放比例；0.5 可大幅加速 1080p 影片，輸出仍為原尺寸",
    )
    parser.add_argument("--min-flow", type=float, default=0.35, help="低於此長度的平均光流視為靜止/不可靠")
    parser.add_argument(
        "--lane-split-x",
        type=int,
        help="左右車道分界的影像 x 座標；未指定時使用畫面寬度的一半",
    )
    parser.add_argument(
        "--left-approach-axis",
        choices=("x", "y"),
        default="y",
        help="左側車道迎面而來用哪個光流軸判斷正負號",
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
        help="右側車道背對駛離用哪個光流軸判斷正負號",
    )
    parser.add_argument(
        "--right-depart-sign",
        choices=("positive", "negative"),
        default="negative",
        help="右側車道背對駛離時該軸的正負號，影像座標 y 負向為向上",
    )
    parser.add_argument("--preview", action="store_true", help="處理時顯示視窗，按 q 離開")
    return parser.parse_args()


def load_detections(path: Optional[str], stage_name: str) -> Dict[int, List[Detection]]:
    if not path:
        return defaultdict(list)

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{stage_name} detections not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        raw_items = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("frames", raw_items.get("detections", []))
        return _detections_from_json(raw_items, stage_name)

    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            return _detections_from_csv(csv.DictReader(f), stage_name)

    raise ValueError(f"Unsupported detection file format: {file_path.suffix}")


def _detections_from_json(raw_items: Iterable[dict], stage_name: str) -> Dict[int, List[Detection]]:
    detections: Dict[int, List[Detection]] = defaultdict(list)
    for item in raw_items:
        frame_index = int(item.get("frame_index", item.get("frame", item.get("frame_id", 0))))
        boxes = item.get("boxes", item.get("objects", item.get("detections")))
        if boxes is None:
            boxes = [item]

        for box_item in boxes:
            bbox = _extract_bbox(box_item)
            if bbox is None:
                continue
            detections[frame_index].append(
                Detection(
                    frame_index=frame_index,
                    bbox=bbox,
                    track_id=_optional_str(box_item.get("track_id", box_item.get("id"))),
                    source_stage=stage_name,
                    score=_optional_float(box_item.get("score", box_item.get("confidence"))),
                    label=_optional_str(box_item.get("label", box_item.get("class"))),
                )
            )
    return detections


def _detections_from_csv(rows: Iterable[dict], stage_name: str) -> Dict[int, List[Detection]]:
    detections: Dict[int, List[Detection]] = defaultdict(list)
    for row in rows:
        frame_index = int(row.get("frame_index") or row.get("frame") or row.get("frame_id") or 0)
        bbox = _extract_bbox(row)
        if bbox is None:
            continue
        detections[frame_index].append(
            Detection(
                frame_index=frame_index,
                bbox=bbox,
                track_id=_optional_str(row.get("track_id") or row.get("id")),
                source_stage=stage_name,
                score=_optional_float(row.get("score") or row.get("confidence")),
                label=_optional_str(row.get("label") or row.get("class")),
            )
        )
    return detections


def _extract_bbox(item: dict) -> Optional[BBox]:
    if "bbox" in item and isinstance(item["bbox"], (list, tuple)) and len(item["bbox"]) >= 4:
        x1, y1, a, b = [float(v) for v in item["bbox"][:4]]
        fmt = str(item.get("bbox_format", "xyxy")).lower()
        if fmt in ("xywh", "ltwh"):
            return _normalize_bbox(x1, y1, x1 + a, y1 + b)
        return _normalize_bbox(x1, y1, a, b)

    keys_xyxy = ("x1", "y1", "x2", "y2")
    if all(k in item and item[k] not in (None, "") for k in keys_xyxy):
        return _normalize_bbox(*(float(item[k]) for k in keys_xyxy))

    keys_xywh = ("x", "y", "w", "h")
    if all(k in item and item[k] not in (None, "") for k in keys_xywh):
        x, y, w, h = (float(item[k]) for k in keys_xywh)
        return _normalize_bbox(x, y, x + w, y + h)

    return None


def _normalize_bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    left, right = sorted((int(round(x1)), int(round(x2))))
    top, bottom = sorted((int(round(y1)), int(round(y2))))
    return left, top, right, bottom


def _optional_str(value: object) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def merge_detection_maps(*maps: Dict[int, List[Detection]]) -> Dict[int, List[Detection]]:
    merged: Dict[int, List[Detection]] = defaultdict(list)
    for det_map in maps:
        for frame_index, detections in det_map.items():
            merged[frame_index].extend(detections)
    return merged


def clip_bbox(bbox: BBox, width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mean_flow_for_bbox(flow: np.ndarray, bbox: BBox, flow_scale: float) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    if flow_scale != 1.0:
        x1 = int(round(x1 * flow_scale))
        y1 = int(round(y1 * flow_scale))
        x2 = int(round(x2 * flow_scale))
        y2 = int(round(y2 * flow_scale))
        x1 = max(0, min(flow.shape[1] - 1, x1))
        x2 = max(0, min(flow.shape[1], x2))
        y1 = max(0, min(flow.shape[0] - 1, y1))
        y2 = max(0, min(flow.shape[0], y2))
    local_flow = flow[y1:y2, x1:x2]
    if local_flow.size == 0:
        return 0.0, 0.0, 0.0

    mag = np.linalg.norm(local_flow, axis=2)
    threshold = max(0.05, float(np.percentile(mag, 60)))
    mask = mag >= threshold
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    dx = float(np.mean(local_flow[..., 0][mask])) / flow_scale
    dy = float(np.mean(local_flow[..., 1][mask])) / flow_scale
    strength = float(np.mean(mag[mask])) / flow_scale
    return dx, dy, strength


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


def draw_track(frame: np.ndarray, points: deque[Point], color: Tuple[int, int, int]) -> None:
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


def run() -> None:
    args = parse_args()
    if not 0.1 <= args.flow_scale <= 1.0:
        raise ValueError("--flow-scale must be between 0.1 and 1.0")
    use_stages = {stage.strip().lower() for stage in args.use_stages.split(",") if stage.strip()}

    stage3 = load_detections(args.stage3, "stage3") if "stage3" in use_stages else defaultdict(list)
    stage4 = load_detections(args.stage4, "stage4") if "stage4" in use_stages else defaultdict(list)
    detections_by_frame = merge_detection_maps(stage3, stage4)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    prev_gray_full = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if args.flow_scale != 1.0:
        prev_gray = cv2.resize(prev_gray_full, None, fx=args.flow_scale, fy=args.flow_scale, interpolation=cv2.INTER_AREA)
    else:
        prev_gray = prev_gray_full
    tracks = TrackStore(args.history, args.match_distance)
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.flow_scale != 1.0:
            gray = cv2.resize(gray_full, None, fx=args.flow_scale, fy=args.flow_scale, interpolation=cv2.INTER_AREA)
        else:
            gray = gray_full

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        canvas = frame.copy()
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (80, 80, 80), 1, cv2.LINE_AA)
        tracks.begin_frame()

        for detection in detections_by_frame.get(frame_index, []):
            clipped = clip_bbox(detection.bbox, width, height)
            if clipped is None:
                continue
            detection.bbox = clipped

            dx, dy, _strength = mean_flow_for_bbox(flow, clipped, args.flow_scale)
            track_id = tracks.resolve_id(detection)
            tracks.update(track_id, detection.center)
            label, color = classify_direction(detection.center, width, dx, dy, args.min_flow, args)

            draw_track(canvas, tracks.histories[track_id], color)
            draw_detection(canvas, detection, track_id, dx, dy, label, color, args.arrow_scale)

        draw_legend(canvas)
        writer.write(canvas)

        if args.preview:
            cv2.imshow("dense optical flow tracks", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = gray

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run()
