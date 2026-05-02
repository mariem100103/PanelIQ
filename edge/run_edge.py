import argparse
import os
import threading
from pathlib import Path

# Avoid Ultralytics blocking on DNS during import on slow/offline networks.
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2

from edge.jobs.job_a import run_job_a
from edge.jobs.job_b import run_job_b_loop, set_latest_frame
from edge.jobs.tamper_check import check_tamper
from panel_paths import resolve_panel_paths

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_B_INTERVAL_SECONDS = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="panel_001")
    args = parser.parse_args()
    panel_id = args.panel

    paths = resolve_panel_paths(_PROJECT_ROOT, panel_id)
    video_path = paths["video"]
    if not video_path.is_file():
        print(f"[run_edge] Video not found: {video_path}")
        return

    threading.Thread(
        target=run_job_b_loop,
        args=(JOB_B_INTERVAL_SECONDS, panel_id),
        daemon=True,
    ).start()

    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        set_latest_frame(frame)

        if check_tamper(frame, panel_id):
            print("[tamper] Camera tamper detected - skipping frame")
            continue

        run_job_a(frame, panel_id)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
