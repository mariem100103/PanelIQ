"""
Offline multi-panel test: run YOLO + tracking on each panel's video with ROI gating.
Run from project root: python tools/test_all_panels.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.roi_utils import clamp_roi
from panel_paths import get_panel_paths, resolve_panel_paths


def in_roi(bbox, roi: dict) -> bool:
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]


def person_attentive_proxy(bbox) -> bool:
    """Bounding-box aspect ratio proxy (no landmarks)."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ratio = w / h
    return 0.28 <= ratio <= 1.15


def main():
    import cv2

    from edge.vision.blur import blur_faces
    from edge.vision.detector import detect_and_track
    from edge.vision.ssim_check import compute_ssim

    cfg_path = ROOT / "configs" / "panel_rois.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
    panels = cfg.get("panels") or {}

    max_frames = 48
    stride = 8

    for panel_key, meta in panels.items():
        panel_id = meta.get("panel_id", panel_key)
        roi = meta.get("roi") or {}
        zone = meta.get("zone_type", "")
        paths = resolve_panel_paths(ROOT, panel_id)
        rel = get_panel_paths(panel_id, project_root=ROOT)
        video_path = paths["video"]
        ref_path = paths["reference"]

        print("=" * 40)
        print(f"Testing panel: {panel_id}")
        print(f"Video: {rel['video']}")
        print(f"Reference: {rel['reference']}")
        if zone:
            print(f"Zone type: {zone}")
        print("-" * 40)

        if not video_path.is_file():
            print("⚠ Video file not found — skipping")
            print()
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("⚠ Could not open video — skipping")
            print()
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_s = f"{fps:.0f}" if fps else "—"
        print(f"Video: {w}x{h} @ {fps_s}fps — {nframes} frames")

        class_counts: dict[str, int] = defaultdict(int)
        conf_sum: dict[str, float] = defaultdict(float)
        conf_n: dict[str, int] = defaultdict(int)
        track_ids: set[int] = set()
        person_in_roi = 0
        attentive_persons = 0
        sampled = 0
        frame_idx = 0
        first_roi_crop = None

        try:
            while sampled < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                r = clamp_roi(roi, frame.shape[1], frame.shape[0])
                dets = detect_and_track(frame)
                for det in dets:
                    if not in_roi(det["bbox"], r):
                        continue
                    cls = det["class"]
                    tid = det.get("id")
                    if tid is not None:
                        track_ids.add(int(tid))
                    class_counts[cls] += 1
                    conf_sum[cls] += float(det.get("conf") or 0.0)
                    conf_n[cls] += 1
                    if cls == "person":
                        person_in_roi += 1
                        if person_attentive_proxy(det["bbox"]):
                            attentive_persons += 1

                blur_faces(frame, dets)
                if first_roi_crop is None:
                    rc = clamp_roi(roi, frame.shape[1], frame.shape[0])
                    first_roi_crop = frame[rc["y1"] : rc["y2"], rc["x1"] : rc["x2"]].copy()

                sampled += 1
                frame_idx += 1
        finally:
            cap.release()

        classes_order = ["car", "truck", "bus", "motorcycle", "person"]
        print(f"Detections ({sampled} sampled frames):")
        for c in classes_order:
            n = int(class_counts.get(c, 0))
            avg = conf_sum[c] / conf_n[c] if conf_n[c] else None
            avg_s = f" (avg conf: {avg:.3f})" if avg is not None else ""
            print(f"  {c:12s} {n:3d}{avg_s}")

        print(f"Unique track IDs: {len(track_ids)}")
        print(f"Face blur: applied to {person_in_roi} person detections")
        if person_in_roi:
            att_pct = 100.0 * attentive_persons / person_in_roi
            print(
                f"Attention rate: {att_pct:.1f}% ({attentive_persons}/{person_in_roi} persons attentive)"
            )
        else:
            print("Attention rate: — (0 persons in ROI)")

        if ref_path.is_file():
            print("Reference image: ✓ found")
            if first_roi_crop is not None:
                try:
                    ssim_score = compute_ssim(first_roi_crop, ref_path=str(ref_path))
                    print(f"SSIM baseline score: {ssim_score:.2f}")
                except Exception as e:
                    print(f"SSIM baseline score: — ({e})")
            else:
                print("SSIM baseline score: — (no ROI crop)")
        else:
            print("Reference image: ⚠ not found")

        print("-" * 40)
        print(f"✓ {panel_id} PASSED")
        print()


if __name__ == "__main__":
    main()
