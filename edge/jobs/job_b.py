import time
from pathlib import Path

import cv2
import requests
import yaml

from edge.roi_utils import clamp_roi
from edge.vision.ssim_check import compute_ssim
from panel_paths import resolve_panel_paths

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CFG_PATH = _PROJECT_ROOT / "configs" / "panel_rois.yaml"
cfg_thr = yaml.safe_load(open(_PROJECT_ROOT / "configs" / "thresholds.yaml", "r", encoding="utf-8"))
SSIM_THRESHOLD = cfg_thr.get("ssim_threshold", 0.5)

latest_frame = None


def set_latest_frame(frame):
    global latest_frame
    latest_frame = frame.copy()


def run_job_b_loop(interval_seconds: int, panel_id: str):
    cfg_roi = yaml.safe_load(open(_CFG_PATH, "r", encoding="utf-8"))
    roi = cfg_roi["panels"][panel_id]["roi"]
    paths = resolve_panel_paths(_PROJECT_ROOT, panel_id)
    reference_path = str(paths["reference"])

    while True:
        time.sleep(interval_seconds)
        if latest_frame is None:
            print(f"[job_b][{panel_id}] No frame available yet")
            continue
        r = clamp_roi(roi, latest_frame.shape[1], latest_frame.shape[0])
        panel_crop = latest_frame[r["y1"] : r["y2"], r["x1"] : r["x2"]]
        score = compute_ssim(panel_crop, ref_path=reference_path)
        print(f"[job_b][{panel_id}] SSIM score: {score:.3f}")
        if score < SSIM_THRESHOLD:
            payload = {
                "type": "ssim_anomaly",
                "ssim_score": round(score, 4),
                "confidence": round(1 - score, 4),
                "panel_id": panel_id,
            }
            try:
                requests.post(
                    "http://localhost:8000/ingest/integrity-alert",
                    json=payload,
                    timeout=5,
                )
            except Exception as e:
                print(f"[job_b][{panel_id}] Alert send failed: {e}")
