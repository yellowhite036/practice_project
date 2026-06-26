from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    7: "truck",
}

DISPLAY_LABELS = {
    "car": "sedan/car",
    "motorcycle": "motorcycle",
    "truck": "truck",
}

COLORS = {
    "car": (38, 132, 255),
    "motorcycle": (20, 184, 114),
    "truck": (235, 132, 34),
}


@dataclass
class Detection:
    bbox: np.ndarray
    cls: str
    conf: float

    @property
    def centroid(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    cls: str
    conf: float
    missed: int = 0
    age: int = 1
    hits: int = 1
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    def update(self, detection: Detection) -> None:
        self.bbox = detection.bbox
        self.cls = detection.cls
        self.conf = detection.conf
        self.missed = 0
        self.age += 1
        self.hits += 1
        cx, cy = detection.centroid.astype(int)
        self.history.append((int(cx), int(cy)))
        self.history = self.history[-24:]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class IoUCentroidTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_centroid_distance: float = 90.0,
        max_missed: int = 18,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_centroid_distance = max_centroid_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        candidates: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            for det_idx, detection in enumerate(detections):
                if track.cls != detection.cls:
                    continue
                iou = box_iou(track.bbox, detection.bbox)
                dist = centroid_distance(track.centroid, detection.centroid)
                if iou >= self.iou_threshold or dist <= self.max_centroid_distance:
                    # Lower score is better. IoU dominates, distance breaks ties.
                    score = (1.0 - iou) + (dist / max(self.max_centroid_distance, 1.0)) * 0.2
                    candidates.append((score, track_id, det_idx))

        for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            matches.append((track_id, det_idx))
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_idx)

        for track_id, det_idx in matches:
            self.tracks[track_id].update(detections[det_idx])

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed += 1
            track.age += 1
            if track.missed > self.max_missed:
                del self.tracks[track_id]

        for det_idx in sorted(unmatched_detections):
            detection = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            cx, cy = detection.centroid.astype(int)
            self.tracks[track_id] = Track(
                track_id=track_id,
                bbox=detection.bbox,
                cls=detection.cls,
                conf=detection.conf,
                history=[(int(cx), int(cy))],
            )

        return sorted(self.tracks.values(), key=lambda track: track.track_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO vehicle classification plus cross-frame IoU/centroid tracking."
    )
    parser.add_argument("--source", required=True, help="Input video path, camera index, or stream URL.")
    parser.add_argument("--output", default="outputs/tracked_vehicles.mp4", help="Annotated output video path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--track-iou", type=float, default=0.25, help="Minimum IoU for same-ID matching.")
    parser.add_argument("--track-distance", type=float, default=90.0, help="Maximum centroid distance for fallback matching.")
    parser.add_argument("--max-missed", type=int, default=18, help="Frames to keep an unmatched track alive.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window.")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    capture_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def detections_from_yolo(result, conf_threshold: float) -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue
        bbox = box.xyxy.cpu().numpy()[0].astype(np.float32)
        detections.append(Detection(bbox=bbox, cls=VEHICLE_CLASSES[cls_id], conf=conf))
    return detections


def draw_tracks(frame: np.ndarray, tracks: Iterable[Track]) -> np.ndarray:
    for track in tracks:
        if track.missed > 0:
            continue
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = COLORS.get(track.cls, (255, 255, 255))
        label = f"ID {track.track_id} | {DISPLAY_LABELS[track.cls]} | {track.conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        text_w, text_h = text_size
        label_y = max(y1, text_h + baseline + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - baseline - 6),
            (x1 + text_w + 8, label_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (15, 20, 25),
            2,
            cv2.LINE_AA,
        )

        if len(track.history) > 1:
            cv2.polylines(frame, [np.array(track.history, dtype=np.int32)], False, color, 2)

    return frame


def main() -> None:
    args = parse_args()
    cap = open_capture(args.source)
    model = YOLO(args.model)
    tracker = IoUCentroidTracker(
        iou_threshold=args.track_iou,
        max_centroid_distance=args.track_distance,
        max_missed=args.max_missed,
    )

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

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        detections = detections_from_yolo(result, args.conf)
        tracks = tracker.update(detections)
        annotated = draw_tracks(frame, tracks)

        cv2.putText(
            annotated,
            f"Frame {frame_index} | active tracks: {sum(track.missed == 0 for track in tracks)}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(annotated)

        if args.show:
            cv2.imshow("YOLO vehicle tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
