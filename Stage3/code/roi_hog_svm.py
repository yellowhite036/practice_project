from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROI = (950,520,300,250)
DEFAULT_HOG_SIZE = (128, 64)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, w, h


def clamp_roi(roi: tuple[int, int, int, int], frame: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def build_hog(win_size: tuple[int, int]) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def roi_from_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = clamp_roi(roi, frame)
    return frame[y : y + h, x : x + w]


def extract_hog(image: np.ndarray, hog: cv2.HOGDescriptor, win_size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, win_size, interpolation=cv2.INTER_AREA)
    resized = cv2.equalizeHist(resized)
    feature = hog.compute(resized)
    return feature.reshape(1, -1).astype(np.float32)


def iter_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def load_training_set(
    positive_dir: Path,
    negative_dir: Path,
    hog: cv2.HOGDescriptor,
    win_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[int] = []

    for label, folder in ((1, positive_dir), (-1, negative_dir)):
        paths = iter_images(folder)
        if not paths:
            raise FileNotFoundError(f"No training images found in {folder}")
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"skip unreadable image: {path}")
                continue
            samples.append(extract_hog(image, hog, win_size))
            labels.append(label)

    if not samples:
        raise RuntimeError("No readable training images were found")

    return np.vstack(samples), np.array(labels, dtype=np.int32)


def train(args: argparse.Namespace) -> None:
    hog = build_hog(args.hog_size)
    samples, labels = load_training_set(args.positive_dir, args.negative_dir, hog, args.hog_size)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(args.c)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(args.model))
    print(f"saved model: {args.model}")
    print(f"training samples: {len(labels)} ({np.sum(labels == 1)} positive, {np.sum(labels == -1)} negative)")


def collect(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % args.step == 0:
            roi_img = roi_from_frame(frame, args.roi)
            out_path = args.output_dir / f"roi_{count:06d}.jpg"
            cv2.imwrite(str(out_path), roi_img)
            saved += 1
        count += 1

    print(f"saved {saved} ROI samples to {args.output_dir}")


def preview(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {args.frame}")

    x, y, w, h = clamp_roi(args.roi, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(frame, f"ROI x={x} y={y} w={w} h={h}", (x, max(35, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(str(args.output), frame)
    print(f"saved ROI preview: {args.output}")


def detect(args: argparse.Namespace) -> None:
    svm = cv2.ml.SVM_load(str(args.model))
    if svm.empty():
        raise FileNotFoundError(f"Cannot load SVM model: {args.model}")

    hog = build_hog(args.hog_size)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi_img = roi_from_frame(frame, args.roi)
        feature = extract_hog(roi_img, hog, args.hog_size)
        _, pred = svm.predict(feature)
        _, raw = svm.predict(feature, flags=cv2.ml.StatModel_RAW_OUTPUT)
        is_positive = int(pred[0, 0]) == 1
        score = float(abs(raw[0, 0]))

        x, y, w, h = clamp_roi(args.roi, frame)
        color = (0, 220, 0) if is_positive else (0, 0, 255)
        label = "vehicle/in-ROI" if is_positive else "empty/negative"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            frame,
            f"{label} score={score:.2f}",
            (x, max(35, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        writer.write(frame)
        frame_index += 1

    writer.release()
    print(f"processed frames: {frame_index}")
    print(f"saved annotated video: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed ROI + HOG feature extraction + linear SVM classification for traffic video.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="Draw the fixed ROI on one frame.")
    add_video_roi_args(preview_parser)
    preview_parser.add_argument("--frame", type=int, default=120)
    preview_parser.add_argument("--output", type=Path, default=Path("roi_preview.jpg"))
    preview_parser.set_defaults(func=preview)

    collect_parser = subparsers.add_parser("collect", help="Export ROI crops for manual labeling.")
    add_video_roi_args(collect_parser)
    collect_parser.add_argument("--output-dir", type=Path, default=Path("samples/unlabeled"))
    collect_parser.add_argument("--step", type=int, default=15, help="Save one ROI crop every N frames.")
    collect_parser.set_defaults(func=collect)

    train_parser = subparsers.add_parser("train", help="Train a linear SVM from positive/negative ROI images.")
    train_parser.add_argument("--positive-dir", type=Path, default=Path("samples/positive"))
    train_parser.add_argument("--negative-dir", type=Path, default=Path("samples/negative"))
    train_parser.add_argument("--model", type=Path, default=Path("models/roi_hog_svm.yml"))
    train_parser.add_argument("--c", type=float, default=0.1)
    add_hog_arg(train_parser)
    train_parser.set_defaults(func=train)

    detect_parser = subparsers.add_parser("detect", help="Classify the fixed ROI in every video frame.")
    add_video_roi_args(detect_parser)
    detect_parser.add_argument("--model", type=Path, default=Path("models/roi_hog_svm.yml"))
    detect_parser.add_argument("--output", type=Path, default=Path("output_roi_hog_svm.mp4"))
    add_hog_arg(detect_parser)
    detect_parser.set_defaults(func=detect)

    args = parser.parse_args()
    args.func(args)


def parse_roi_size(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("HOG size must be width,height")
    width, height = parts
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("HOG width and height must be positive")
    return width, height


def add_video_roi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, default=Path("../../123.mp4"))
    parser.add_argument("--roi", type=parse_roi, default=DEFAULT_ROI, help="Fixed ROI as x,y,w,h. Default: 250,520,1350,310")


def add_hog_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hog-size", type=parse_roi_size, default=DEFAULT_HOG_SIZE, help="HOG resize size as width,height. Default: 128,64")


if __name__ == "__main__":
    main()
