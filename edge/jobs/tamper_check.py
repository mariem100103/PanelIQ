from pathlib import Path

import cv2
import yaml

_THRESH_PATH = Path(__file__).resolve().parents[2] / "configs" / "thresholds.yaml"
cfg = yaml.safe_load(open(_THRESH_PATH, "r", encoding="utf-8"))
BLUR_THRESHOLD = cfg.get("tamper_blur_threshold", 20)
BRIGHTNESS_MIN = cfg.get("tamper_brightness_min", 10)
BRIGHTNESS_MAX = cfg.get("tamper_brightness_max", 245)


def check_tamper(frame, panel_id: str | None = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    is_blurred = lap_var < BLUR_THRESHOLD
    is_covered = brightness < BRIGHTNESS_MIN
    is_blinded = brightness > BRIGHTNESS_MAX
    if is_blurred or is_covered or is_blinded:
        prefix = f"[tamper][{panel_id}]" if panel_id else "[tamper]"
        print(f"{prefix} lap_var={lap_var:.1f}, brightness={brightness:.1f}")
        return True
    return False
