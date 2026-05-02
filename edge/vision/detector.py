import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")

from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_LOCAL = _ROOT / "models" / "yolov8n.pt"
MODEL_WEIGHTS = str(_MODEL_LOCAL) if _MODEL_LOCAL.is_file() else "yolov8n.pt"

model = YOLO(MODEL_WEIGHTS)
CONF_THRESHOLD = 0.4
CLASSES = {"car", "bus", "truck", "motorcycle", "person"}


def detect_and_track(frame):
    results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name not in CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else None
        detections.append(
            {
                "class": cls_name,
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
                "id": track_id,
            }
        )

    return detections