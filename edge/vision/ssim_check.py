from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim


def compute_ssim(
    frame_or_crop,
    ref_path: str | None = None,
    *,
    panel_id: str | None = None,
    project_root: Path | None = None,
) -> float:
    root = project_root or Path(__file__).resolve().parents[2]
    if ref_path is None:
        if not panel_id:
            raise ValueError("compute_ssim requires ref_path or panel_id")
        from panel_paths import resolve_panel_paths

        ref_path = str(resolve_panel_paths(root, panel_id)["reference"])

    ref = cv2.imread(ref_path)
    if ref is None:
        print(f"[ssim] Reference image not found at {ref_path}. Returning healthy score.")
        return 1.0

    h, w = frame_or_crop.shape[:2]
    if h == 0 or w == 0:
        return 1.0

    frame_gray = cv2.cvtColor(frame_or_crop, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.resize(frame_gray, (300, 300))
    ref_gray = cv2.resize(ref_gray, (300, 300))

    score, _ = ssim(frame_gray, ref_gray, full=True)
    return score
