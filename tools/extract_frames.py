import argparse
from pathlib import Path

import cv2

from panel_paths import resolve_panel_paths

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="panel_001")
    args = parser.parse_args()
    video = resolve_panel_paths(ROOT, args.panel)["video"]
    out_dir = ROOT / "data" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"No video at {video}")
        return

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % 30 == 0:
            cv2.imwrite(str(out_dir / f"frame_{i}.jpg"), frame)

        i += 1

    cap.release()


if __name__ == "__main__":
    main()
