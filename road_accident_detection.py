"""
REAL-TIME ROAD ACCIDENT DETECTION SYSTEM
USING DEEP LEARNING AND COMPUTER VISION

Prototype implementation:
- YOLOv8 detects vehicles in a live webcam feed or video file.
- OpenCV displays the annotated video.
- A simple bounding-box overlap heuristic flags possible collision events.
- Possible events are saved as timestamped snapshots.

IMPORTANT:
This is an educational prototype, not a safety-critical accident detector.
Bounding-box overlap can produce false alarms and cannot prove that a crash
occurred. For a reliable system, train/evaluate a temporal accident classifier
on labelled accident and non-accident video clips.

Install:
    pip install ultralytics opencv-python

Run with webcam:
    python road_accident_detection.py

Run with a video:
    python road_accident_detection.py --source road_video.mp4

Optional:
    python road_accident_detection.py --source 0 --model yolov8n.pt
"""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


# COCO class IDs used by YOLO for common road vehicles.
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def parse_source(source_text):
    """Convert a numeric source such as '0' to an integer webcam index."""
    return int(source_text) if source_text.isdigit() else source_text


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-based road vehicle detection and possible-collision alert prototype."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or path to a video file."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model weights. Default: yolov8n.pt"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum detection confidence (default: 0.40)."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.15,
        help="Vehicle-box overlap threshold for a possible collision alert."
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between saved alerts."
    )
    args = parser.parse_args()

    os.makedirs("accident_alerts", exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(args.model)

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {args.source}. "
            "Check the webcam or video file path."
        )

    last_alert_time = 0.0
    print("Detection started. Press Q to stop.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video ended or frame could not be read.")
                break

            # Track objects between frames. YOLO downloads the selected
            # pretrained weights on first use if they are not already present.
            results = model.track(
                frame,
                persist=True,
                conf=args.confidence,
                verbose=False
            )

            vehicle_boxes = []
            annotated_frame = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                for detection in boxes:
                    class_id = int(detection.cls[0].item())
                    confidence = float(detection.conf[0].item())

                    if class_id not in VEHICLE_CLASS_IDS:
                        continue

                    x1, y1, x2, y2 = map(
                        int, detection.xyxy[0].tolist()
                    )
                    vehicle_boxes.append((x1, y1, x2, y2))

                    class_name = model.names.get(class_id, "vehicle")
                    track_id = None
                    if detection.id is not None:
                        track_id = int(detection.id[0].item())

                    label = f"{class_name} {confidence:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"

                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        annotated_frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2
                    )

            # Heuristic: overlapping boxes from two different detected
            # vehicles may indicate a possible collision. This is only a
            # visual proximity cue and is not proof of an accident.
            possible_collision = False
            for first_index in range(len(vehicle_boxes)):
                for second_index in range(first_index + 1, len(vehicle_boxes)):
                    overlap = calculate_iou(
                        vehicle_boxes[first_index],
                        vehicle_boxes[second_index]
                    )
                    if overlap >= args.iou_threshold:
                        possible_collision = True
                        break
                if possible_collision:
                    break

            current_time = time.time()

            if possible_collision:
                cv2.putText(
                    annotated_frame,
                    "WARNING: POSSIBLE VEHICLE COLLISION",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if current_time - last_alert_time >= args.alert_cooldown:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(
                        "accident_alerts",
                        f"possible_collision_{timestamp}.jpg"
                    )
                    cv2.imwrite(image_path, annotated_frame)
                    print(f"Possible collision snapshot saved: {image_path}")
                    last_alert_time = current_time

            cv2.imshow(
                "Real-Time Road Accident Detection - Prototype",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


if __name__ == "__main__":
    main()
