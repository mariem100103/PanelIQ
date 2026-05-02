"""Smoke tests for video + YOLO verification (skipped if no sample asset)."""

from pathlib import Path

import pytest

from panel_paths import get_panel_paths

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = ROOT / get_panel_paths("panel_001", project_root=ROOT)["video"]


@pytest.mark.skipif(not SAMPLE.is_file(), reason="data/panels/panel_001/video.mp4 missing")
def test_run_video_model_check_samples_frames():
    from backend.verify_pipeline import run_video_model_check

    result = run_video_model_check(str(SAMPLE), max_frames=4, stride=15)
    assert "ok" in result
    assert result.get("frames_sampled", 0) >= 1
