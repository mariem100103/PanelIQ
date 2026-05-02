import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from edge.roi_utils import clamp_roi
from edge.vision.detector import detect_and_track

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _PROJECT_ROOT / "configs" / "panel_rois.yaml"
_roi_cache: dict[str, tuple[str, dict]] = {}

SEND_INTERVAL = 300
counter = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "person": 0}
seen_ids = set()
last_send = time.time()
window_start = datetime.now(timezone.utc).isoformat()


def _panel_roi(panel_id: str) -> tuple[str, dict]:
    if panel_id not in _roi_cache:
        cfg = yaml.safe_load(open(_CFG_PATH, "r", encoding="utf-8"))
        p = cfg["panels"][panel_id]
        _roi_cache[panel_id] = (p["panel_id"], p["roi"])
    return _roi_cache[panel_id]


def in_roi(bbox, roi_data):
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return roi_data["x1"] <= cx <= roi_data["x2"] and roi_data["y1"] <= cy <= roi_data["y2"]


def run_job_a(frame, panel_id: str):
    global counter, seen_ids, last_send, window_start

    pid, roi = _panel_roi(panel_id)
    roi = clamp_roi(roi, frame.shape[1], frame.shape[0])
    detections = detect_and_track(frame)

    for det in detections:
        if not in_roi(det["bbox"], roi):
            continue
        track_id = det.get("id")
        if track_id is not None and track_id in seen_ids:
            continue
        if track_id is not None:
            seen_ids.add(track_id)
        cls = det["class"]
        if cls in counter:
            counter[cls] += 1

    if time.time() - last_send >= SEND_INTERVAL:
        payload = {
            "panel_id": pid,
            "window_start": window_start,
            "window_end": datetime.now(timezone.utc).isoformat(),
            **counter,
        }
        try:
            requests.post("http://localhost:8000/ingest/vehicle-mix", json=payload, timeout=5)
        except Exception as e:
            print(f"[job_a][{panel_id}] Send failed: {e}")
        counter = {k: 0 for k in counter}
        seen_ids = set()
        last_send = time.time()
        window_start = datetime.now(timezone.utc).isoformat()
