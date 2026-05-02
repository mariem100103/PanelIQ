"""Panel-scoped asset paths (video + SSIM reference). Shared by edge and backend."""

from pathlib import Path

import yaml


def _panel_media_from_config(project_root: Path, panel_id: str) -> tuple[str, str]:
    cfg_path = project_root / "configs" / "panel_rois.yaml"
    try:
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
    except OSError:
        return "video.mp4", "ref.jpg"
    panel = (cfg.get("panels") or {}).get(panel_id) or {}
    return (
        str(panel.get("video_file", "video.mp4")),
        str(panel.get("reference_file", "ref.jpg")),
    )


def get_panel_paths(panel_id: str, project_root: Path | None = None) -> dict[str, str]:
    """Relative POSIX-style paths under the repo (for API / UI strings)."""
    if project_root is not None:
        vf, rf = _panel_media_from_config(project_root, panel_id)
    else:
        vf, rf = "video.mp4", "ref.jpg"
    base = f"data/panels/{panel_id}"
    return {"video": f"{base}/{vf}", "reference": f"{base}/{rf}"}


def resolve_panel_paths(project_root: Path, panel_id: str) -> dict[str, Path]:
    vf, rf = _panel_media_from_config(project_root, panel_id)
    base = project_root / "data" / "panels" / panel_id
    return {"video": base / vf, "reference": base / rf}
