from __future__ import annotations

import subprocess
import sys
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

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

ROI_FILL_ALPHA = 0.25  # ROI 半透明填色透明度（畫在輸出影片上）


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
    current_roi: str | None = None  # 目前所在的 ROI 名稱（沒在任何 ROI 內為 None）

    @property
    def centroid(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def bottom_center(self) -> np.ndarray:
        """bbox 底部中心點，通常比幾何中心更貼近車輛實際壓在路面上的位置。"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2], dtype=np.float32)

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


# ---------- ROI 相關 ----------

@dataclass
class RegionOfInterest:
    id: int
    name: str
    color: tuple[int, int, int]
    polygon: np.ndarray  # shape (N, 2), dtype=int32


def load_rois(path: str) -> list[RegionOfInterest]:
    """讀取 roi_editor.py 產生的 roi.json，回傳 ROI 物件清單。"""
    rois: list[RegionOfInterest] = []
    roi_path = Path(path)
    if not roi_path.exists():
        print(f"[警告] 找不到 ROI 檔案：{path}，將不進行車道判斷。")
        return rois

    with open(roi_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("rois", []):
        polygon = np.array(item.get("polygon", []), dtype=np.int32)
        if len(polygon) < 3:
            continue
        color = tuple(int(c) for c in item.get("color", (0, 255, 0)))
        rois.append(
            RegionOfInterest(
                id=item.get("id", 0),
                name=item.get("name", f"ROI{item.get('id', 0)}"),
                color=color,
                polygon=polygon,
            )
        )
    print(f"[資訊] 已載入 {len(rois)} 個 ROI：{[r.name for r in rois]}")
    return rois


def find_roi(point: np.ndarray, rois: list[RegionOfInterest]) -> str | None:
    """回傳 point 所在的第一個 ROI 名稱，都沒有落在任何 ROI 內則回傳 None。"""
    px, py = float(point[0]), float(point[1])
    for roi in rois:
        if cv2.pointPolygonTest(roi.polygon, (px, py), False) >= 0:
            return roi.name
    return None


def build_roi_mask(frame_shape: tuple[int, int], rois: list[RegionOfInterest]) -> np.ndarray | None:
    """依照所有 ROI 的多邊形聯集，建立一張黑白遮罩（ROI 內為白、其餘為黑）。
    若沒有任何 ROI，回傳 None，代表不做遮罩，YOLO 直接偵測全畫面。"""
    if not rois:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for roi in rois:
        cv2.fillPoly(mask, [roi.polygon], 255)
    return mask


def draw_rois(frame: np.ndarray, rois: list[RegionOfInterest], roi_counts: dict[str, int]) -> np.ndarray:
    """在畫面上畫出半透明 ROI 區域與累計進入數量，方便確認車道判斷是否正確。"""
    if not rois:
        return frame

    overlay = frame.copy()
    for roi in rois:
        cv2.fillPoly(overlay, [roi.polygon], roi.color)
        cv2.polylines(frame, [roi.polygon], True, roi.color, 2)
        label = f"{roi.name}: {roi_counts.get(roi.name, 0)}"
        label_pos = tuple(roi.polygon[0] + np.array([5, -8]))
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    roi.color, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0)


def cutout_tracked_windows(frame: np.ndarray, tracks: Iterable[Track]) -> np.ndarray:
    """建立一張全黑畫布，只把每個 track 目前的 bbox 範圍從原始 frame 複製過來，
    其餘背景維持全黑（矩形窗口去背版）。"""
    canvas = np.zeros_like(frame)
    h, w = frame.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        canvas[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return canvas


# ---------- 主程式 ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO vehicle classification plus cross-frame IoU/centroid tracking."
    )
    parser.add_argument("--source", default="../../123.mp4", help="Input video path, camera index, or stream URL.")
    parser.add_argument("--output", default="YOLO_result.mp4", help="Annotated output video path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--track-iou", type=float, default=0.25, help="Minimum IoU for same-ID matching.")
    parser.add_argument("--track-distance", type=float, default=90.0, help="Maximum centroid distance for fallback matching.")
    parser.add_argument("--max-missed", type=int, default=18, help="Frames to keep an unmatched track alive.")
    parser.add_argument("--roi", default="roi.json", help="roi_editor.py 產生的 ROI 檔案路徑。")
    parser.add_argument("--show", action="store_true", help="Show a live preview window.")
    parser.add_argument(
        "--roi-mode",
        choices=["mask", "filter", "off"],
        default="filter",
        help="mask: 偵測前先遮罩ROI外區域（版本一）；"
             "filter: 全畫面偵測，只顯示/計數ROI內的車輛（版本二）；"
             "off: 不做ROI限制，全部顯示",
    )
    parser.add_argument(
        "--cutout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="矩形窗口去背：只保留被追蹤車輛的 bbox 範圍，其餘背景全黑。"
            "若搭配 --roi-mode filter，ROI 外的車輛也會被塗黑。"
            "預設為開啟，若要關閉請加上 --no-cutout。",
    )
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
        roi_tag = f" | {track.current_roi}" if track.current_roi else ""
        label = f"ID {track.track_id} | {DISPLAY_LABELS[track.cls]} | {track.conf:.2f}{roi_tag}"

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

    roi_path = Path(args.roi)
    if not roi_path.exists():
        print(f"[提示] 找不到 ROI 檔案：{roi_path}，將啟動 roi.py 讓您先標註 ROI。")
        editor_script = Path(__file__).parent / "roi.py"
        result = subprocess.run(
            [sys.executable, str(editor_script), "--video", args.source, "--roi", str(roi_path)]
        )
        if result.returncode != 0 or not roi_path.exists():
            raise RuntimeError(
                f"未成功建立 ROI 檔案（{roi_path}），已中止程式。請重新執行並在編輯器中按 S 儲存。"
            )

    cap = open_capture(args.source)
    model = YOLO(args.model)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = IoUCentroidTracker(
        iou_threshold=args.track_iou,
        max_centroid_distance=args.track_distance,
        max_missed=args.max_missed,
    )

    rois = load_rois(args.roi)
    roi_counts: dict[str, int] = {roi.name: 0 for roi in rois}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_mask = build_roi_mask((height, width), rois)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    progress = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="Processing")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.roi_mode == "mask" and roi_mask is not None:
            detect_input = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            detect_input = frame  # filter 或 off 模式都用全畫面偵測

        result = model.predict(detect_input, conf=args.conf, iou=args.iou, verbose=False)[0]
        detections = detections_from_yolo(result, args.conf)
        tracks = tracker.update(detections)

        # 判斷每台車目前所在的 ROI，並統計「進入」次數（同一台車連續在同一 ROI 內只算一次）
        for track in tracks:
            if track.missed > 0:
                continue
            matched_roi = find_roi(track.bottom_center, rois)
            if matched_roi != track.current_roi and matched_roi is not None:
                roi_counts[matched_roi] = roi_counts.get(matched_roi, 0) + 1
            track.current_roi = matched_roi

        display_tracks = tracks
        if args.roi_mode == "filter" and rois:
            display_tracks = [t for t in tracks if t.current_roi is not None]

        # 決定要疊加畫框/文字的背景畫面：
        # 若開啟 --cutout，就用矩形窗口去背後的黑畫布（只保留 display_tracks 的 bbox 範圍）；
        # 否則沿用原始 frame，維持完整背景。
        if args.cutout:
            base_frame = cutout_tracked_windows(frame, display_tracks)
        else:
            base_frame = frame

        annotated = draw_rois(base_frame, rois, roi_counts)
        annotated = draw_tracks(annotated, display_tracks)

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
        progress.update(1)
    progress.close()
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved annotated video to: {output_path}")
    if roi_counts:
        print("各 ROI 進入次數統計：")
        for name, count in roi_counts.items():
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()