"""Validate OpenCV + Ultralytics tracking against a sample video (offline-friendly)."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

ORDERED_CLASSES = ["car", "truck", "bus", "motorcycle", "person"]
SAMPLE_DETECTION_CAP = 30
SAMPLE_MAX_PER_FRAME = 4


def default_video_path(panel_id: str = "panel_001") -> Path:
    env = os.environ.get("VERIFY_VIDEO_PATH", "").strip()
    if env:
        return Path(env)
    from panel_paths import get_panel_paths

    return ROOT / get_panel_paths(panel_id, project_root=ROOT)["video"]


def run_video_model_check(
    video_path: Path | str | None = None,
    *,
    panel_id: str | None = None,
    max_frames: int = 48,
    stride: int = 8,
) -> dict[str, Any]:
    """
    Sample frames from the video and run the same track() pipeline as the edge.
    Returns counts, confidence summaries, and sample rows so the UI can show real model output.
    """
    if video_path is not None:
        path = Path(video_path)
    else:
        pid = panel_id or "panel_001"
        path = default_video_path(pid)
    out: dict[str, Any] = {
        "ok": False,
        "video_path": str(path.resolve()),
        "video": {},
        "run": {"max_frames": max_frames, "stride": stride},
        "frames_sampled": 0,
        "detections_total": 0,
        "detections_per_sampled_frame_avg": 0.0,
        "by_class": {},
        "by_class_avg_confidence": {},
        "unique_track_ids_seen": 0,
        "sample_detections": [],
        "error": None,
        "model_weights": None,
        "conf_threshold": None,
    }

    if not path.is_file():
        out["error"] = (
            f"No video file at this path. Add an MP4 under data/panels/<panel_id>/video.mp4 "
            f"or set env VERIFY_VIDEO_PATH. Tried: {path}"
        )
        return out

    try:
        import cv2
    except ImportError as e:
        out["error"] = f"opencv-python not installed: {e}"
        return out

    try:
        from edge.vision import detector as det_mod

        out["model_weights"] = getattr(det_mod, "MODEL_WEIGHTS", "yolov8n.pt")
        out["conf_threshold"] = getattr(det_mod, "CONF_THRESHOLD", None)
        detect_and_track = det_mod.detect_and_track
    except ImportError as e:
        out["error"] = (
            "Could not import edge detector. Start the backend from the project root "
            f"(same folder as `edge/` and `backend/`). Detail: {e}"
        )
        return out

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        out["error"] = "Could not open the video (missing codec, corrupt file, or wrong path)."
        return out

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out["video"] = {
        "width": w,
        "height": h,
        "fps_rounded": round(fps, 2) if fps else None,
        "reported_total_frames": total_frames_meta if total_frames_meta > 0 else None,
    }

    class_counts: Counter[str] = Counter()
    conf_sum: Counter[str] = Counter()
    conf_n: Counter[str] = Counter()
    track_ids_seen: set[int] = set()
    sample_rows: list[dict[str, Any]] = []

    sampled = 0
    frame_idx = 0
    total_dets = 0

    try:
        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            video_frame_at_sample = frame_idx
            dets = detect_and_track(frame)
            sampled += 1

            taken_this_frame = 0
            for d in dets:
                if len(sample_rows) >= SAMPLE_DETECTION_CAP:
                    break
                cls_name = d.get("class")
                cf = float(d.get("conf") or 0.0)
                if cls_name:
                    class_counts[cls_name] += 1
                    conf_sum[cls_name] += cf
                    conf_n[cls_name] += 1
                tid = d.get("id")
                if tid is not None:
                    track_ids_seen.add(int(tid))
                total_dets += 1

                if taken_this_frame < SAMPLE_MAX_PER_FRAME:
                    taken_this_frame += 1
                    bbox = d.get("bbox")
                    sample_rows.append(
                        {
                            "video_frame_index": video_frame_at_sample,
                            "sample_index": sampled,
                            "class": cls_name,
                            "confidence": round(cf, 4),
                            "track_id": int(tid) if tid is not None else None,
                            "bbox_xyxy": [int(x) for x in bbox] if bbox else None,
                        }
                    )

            frame_idx += 1
    finally:
        cap.release()

    out["frames_sampled"] = sampled
    out["detections_total"] = total_dets
    out["detections_per_sampled_frame_avg"] = round(
        total_dets / max(sampled, 1), 2
    )

    mix = {c: int(class_counts.get(c, 0)) for c in ORDERED_CLASSES}
    for k, v in class_counts.items():
        if k not in mix:
            mix[k] = v
    out["by_class"] = mix

    avg_conf: dict[str, float | None] = {}
    for c in ORDERED_CLASSES:
        if conf_n[c]:
            avg_conf[c] = round(conf_sum[c] / conf_n[c], 4)
        else:
            avg_conf[c] = None
    for k in conf_n:
        if k not in avg_conf:
            avg_conf[k] = round(conf_sum[k] / conf_n[k], 4)
    out["by_class_avg_confidence"] = avg_conf

    out["unique_track_ids_seen"] = len(track_ids_seen)
    out["sample_detections"] = sample_rows

    if sampled == 0:
        out["error"] = "No frames were read — video may be empty or unsupported."
        return out

    out["ok"] = True
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run_video_model_check(), indent=2))
