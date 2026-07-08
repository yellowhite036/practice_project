import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    frame_index: int
    bbox: BBox
    track_id: str
    label: str = "vehicle"
    roi: Optional[str] = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="讀取 YOLO輸出的 tracks.json（含 track_id/bbox），"
                    "用質心軌跡差分（取代光流）畫出每台車的移動方向箭頭與歷史軌跡。"
    )
    parser.add_argument("--video", default="../../123.mp4", help="輸入影片路徑（僅用來當畫布，不再用來偵測）")
    parser.add_argument("--output", default="direction_tracks.mp4", help="輸出影片路徑")
    parser.add_argument(
        "--tracks-input",
        default="tracks.json",
        help="YOLO_result.py 用 --tracks-output 產生的 JSON 檔路徑，"
             "格式為 [{frame_index, boxes:[{track_id, bbox, label, roi}, ...]}, ...]",
    )

    # 方向計算相關參數（取代原本的稀疏光流）
    parser.add_argument(
        "--direction-window",
        type=int,
        default=6,
        help="用「目前質心 - N 幀前的質心」估計移動方向，N 即此參數。"
             "數字越大，方向越平滑但反應越慢。",
    )
    parser.add_argument("--history", type=int, default=60, help="每個物件保留幾幀中心點軌跡（畫圖用）")
    parser.add_argument("--arrow-scale", type=float, default=5.0, help="方向箭頭放大倍率")
    parser.add_argument("--min-flow", type=float, default=0.35, help="低於此長度（每幀平均位移）視為靜止/不可靠")

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
# 讀取 tracks.json
# ---------------------------------------------------------------------------

def load_tracks_json(path: str) -> Dict[int, List[Detection]]:
    """讀取 tracks.json，轉成 {frame_index: [Detection, ...]}。"""
    tracks_path = Path(path)
    if not tracks_path.exists():
        raise RuntimeError(
            f"找不到追蹤結果檔案：{path}。"
        )

    with open(tracks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: Dict[int, List[Detection]] = defaultdict(list)
    for entry in data:
        frame_index = int(entry["frame_index"])
        for box in entry.get("boxes", []):
            x1, y1, x2, y2 = box["bbox"]
            frames[frame_index].append(
                Detection(
                    frame_index=frame_index,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    track_id=str(box.get("track_id")),
                    label=box.get("label", "vehicle"),
                    roi=box.get("roi"),
                )
            )
    return frames


# ---------------------------------------------------------------------------
# 方向計算（質心軌跡差分，取代原本的稀疏光流）
# ---------------------------------------------------------------------------

def compute_direction(history: "deque[Point]", window: int) -> Tuple[float, float]:
    """用 history[-1] 與 history[-1-window]（或更早的第一筆）的質心差，
    除以實際跨越的幀數，得到「每幀平均位移」(dx, dy)，取代光流的平均位移。"""
    if len(history) < 2:
        return 0.0, 0.0

    ref_idx = max(0, len(history) - 1 - window)
    x0, y0 = history[ref_idx]
    x1, y1 = history[-1]
    span = max(1, (len(history) - 1) - ref_idx)
    return (x1 - x0) / span, (y1 - y0) / span


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
    """維護每個 track_id 的質心歷史，供畫軌跡與算方向使用。
    track_id 直接沿用 YOLO給的結果，不再自己做最近鄰配對。"""

    def __init__(self, max_history: int) -> None:
        self.max_history = max_history
        self.histories: Dict[str, deque[Point]] = defaultdict(lambda: deque(maxlen=max_history))

    def update(self, track_id: str, center: Point) -> None:
        self.histories[track_id].append(center)


def draw_track(frame: np.ndarray, points: "deque[Point]", color: Tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        thickness = 1 + int(3 * idx / max(1, len(points) - 1))
        cv2.line(frame, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    dx: float,
    dy: float,
    label: str,
    color: Tuple[int, int, int],
    arrow_scale: float,
    min_arrow_length: float = 40.0,
) -> None:
    x1, y1, x2, y2 = detection.bbox
    center = detection.center

    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        end = center
    else:
        # 方向靠 (dx, dy) 正規化決定，長度則保底 min_arrow_length，
        # 避免車輛移動慢時箭頭太短看不清楚。
        draw_length = max(norm * arrow_scale, min_arrow_length)
        ux, uy = dx / norm, dy / norm
        end = (int(center[0] + ux * draw_length), int(center[1] + uy * draw_length))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)

    # 先畫一條較粗的黑色外框線，再疊加彩色箭頭，讓箭頭在複雜背景上更明顯
    cv2.arrowedLine(frame, center, end, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(frame, center, end, color, 4, cv2.LINE_AA, tipLength=0.4)

    roi_tag = f" {detection.roi}" if detection.roi else ""
    text = f"ID {detection.track_id} {detection.label}{roi_tag} {label} dx={dx:+.2f} dy={dy:+.2f}"
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


# ---------------------------------------------------------------------------
# 主流程：讀取 tracks.json + 質心軌跡差分算方向 + 繪製
# ---------------------------------------------------------------------------

def run() -> None:
    args = parse_args()

    frames_data = load_tracks_json(args.tracks_input)

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

    # history 至少要能容納 direction_window + 1 個點，方向計算才有足夠的參考點
    track_store = TrackStore(max_history=max(args.history, args.direction_window + 1))

    frame_index = 0
    start_time = time.time()
    detection_count_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = frames_data.get(frame_index, [])
        canvas = frame.copy()
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (80, 80, 80), 1, cv2.LINE_AA)

        for detection in detections:
            detection_count_total += 1
            track_store.update(detection.track_id, detection.center)
            history = track_store.histories[detection.track_id]

            dx, dy = compute_direction(history, args.direction_window)
            label, color = classify_direction(detection.center, width, dx, dy, args.min_flow, args)

            draw_track(canvas, history, color)
            draw_detection(canvas, detection, dx, dy, label, color, args.arrow_scale)

        draw_legend(canvas)
        writer.write(canvas)

        if args.preview:
            cv2.imshow("centroid direction tracks", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1
        if frame_index % args.progress_step == 0:
            print_progress(frame_index, total_frames, start_time)

    print_progress(frame_index, total_frames, start_time)
    sys.stdout.write("\n")
    sys.stdout.flush()

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"frames={frame_index}")
    print(f"detections_total={detection_count_total}")
    print(f"saved_video={output_path}")


if __name__ == "__main__":
    run()