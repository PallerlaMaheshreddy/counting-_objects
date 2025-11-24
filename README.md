# counting-_objects
#python code
#!/usr/bin/env python3
"""
Real-time people counter using YOLOv8 + OpenCV

Features:
- Works with webcam (default) or video file
- Counts number of detected "person" in each frame
- Optional ROI ("line area") so you only count people in that region
- Overlays bounding boxes + labels + people count

Usage examples:
    python3 people_counter.py                # use default webcam
    python3 people_counter.py --source 0    # explicitly webcam 0
    python3 people_counter.py --source video.mp4
    python3 people_counter.py --roi 0.2 0.6 0.8 0.95
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time people counter (YOLOv8 + OpenCV)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for webcam, or path to video file",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help=(
            "Optional ROI (Region Of Interest) as relative coords in [0,1]. "
            "Only people whose center lies in this box are counted.\n"
            "Example: --roi 0.2 0.6 0.8 0.95"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for detections (default: 0.4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path (default: yolov8n.pt = small & fast)",
    )
    return parser.parse_args()


def open_video_source(src_str: str):
    """
    Open webcam or video file based on source string.
    """
    try:
        # If it's an int-like string, treat as webcam index
        src = int(src_str)
    except ValueError:
        src = src_str

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {src_str}")
        sys.exit(1)

    print(f"[INFO] Opened video source: {src_str}")
    return cap


def main():
    args = parse_args()

    print("[INFO] Loading YOLO model:", args.model)
    model = YOLO(args.model)  # will download yolov8n.pt if not present

    cap = open_video_source(args.source)

    has_roi = args.roi is not None
    roi_rel = args.roi if has_roi else None

    print("\n[INFO] Press 'q' in the video window to quit.\n")

    last_fps_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames or cannot read frame. Exiting.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Calculate ROI in absolute pixel coordinates if given
        roi_abs = None
        if has_roi:
            x_min_rel, y_min_rel, x_max_rel, y_max_rel = roi_rel
            x_min = int(x_min_rel * w)
            y_min = int(y_min_rel * h)
            x_max = int(x_max_rel * w)
            y_max = int(y_max_rel * h)
            roi_abs = (x_min, y_min, x_max, y_max)
            # Draw ROI rectangle
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "LINE AREA",
                (x_min + 5, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Run YOLO inference
        results = model(frame, conf=args.conf, verbose=False)

        # We expect one result per frame
        result = results[0]

        people_count = 0

        # result.boxes contains bounding boxes
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # We only care about "person" class
                class_name = model.names.get(cls_id, str(cls_id))
                if class_name != "person":
                    continue

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Compute center for ROI test
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if inside ROI (if defined)
                inside_roi = True
                if roi_abs is not None:
                    rx1, ry1, rx2, ry2 = roi_abs
                    inside_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2

                if inside_roi:
                    people_count += 1
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = f"person {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 0),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    # Draw grey boxes for people outside the "line"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

        # Compute FPS every ~10 frames for a smoother display
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / max(now - last_fps_time, 1e-6)
            last_fps_time = now

        # Overlay info text
        count_text = (
            f"People in LINE: {people_count}"
            if has_roi
            else f"People in frame: {people_count}"
        )
        cv2.putText(
            frame,
            count_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("People Counter - Press 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
