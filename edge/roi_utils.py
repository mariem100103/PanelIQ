"""Clamp panel ROI to frame bounds (avoids empty crops on small or odd-sized videos)."""


def clamp_roi(roi: dict, frame_width: int, frame_height: int) -> dict:
    w = max(1, int(frame_width))
    h = max(1, int(frame_height))
    x1 = max(0, min(int(roi["x1"]), w - 1))
    y1 = max(0, min(int(roi["y1"]), h - 1))
    x2 = max(x1 + 1, min(int(roi["x2"]), w))
    y2 = max(y1 + 1, min(int(roi["y2"]), h))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
