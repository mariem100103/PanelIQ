# DOOH Panel Intelligence Platform

This project contains:
- `edge/`: Raspberry Pi/edge pipeline for camera processing and periodic reporting.
- `backend/`: FastAPI service for ingesting readings, integrity alerts, and panel scoring.

## 1) Install requirements

From the project root:

```bash
python -m pip install -r requirements.txt
```

## 2) Download/cache YOLOv8n model

Run once (internet required) to download/cache the model:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 3) Copy model for offline use

After download, place/copy the model file to:

```text
models/yolov8n.pt
```

If running fully offline, ensure your detector points to this local path.

## 4) Record a reference frame from video

Use the frame extraction tool to create a clean panel reference image:

```bash
python tools/extract_frames.py
```

Save the chosen frame as:

```text
data/references/ref.jpg
```

## 5) Start backend

From the **project root** (so `backend` and `edge` imports resolve):

```bash
python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/** for the dashboard (readings, alerts, score, and a **Run video + model check** button).

Interactive OpenAPI: **http://127.0.0.1:8000/docs**

SQLite database file: **`data/readings.db`** (created next to the repo `data/` folder).

### Verify video + model without the dashboard

```bash
python -m backend.verify_pipeline
```

Optional: point checks at another file:

```powershell
set VERIFY_VIDEO_PATH=C:\path\to\clip.mp4
python -m backend.verify_pipeline
```

Query parameters for the HTTP check (used by the dashboard button): **GET** `/api/verify-video?max_frames=48&stride=8`

### Automated test (requires sample video + deps)

```bash
python -m pip install pytest
pytest tests/test_verify_pipeline.py -q
```

## 6) Start edge pipeline

```bash
python -m edge.run_edge
```

## Notes

- Default video input path is `data/videos/sample.mp4`.
- Panel ROI and panel id are configured in `configs/panel_rois.yaml`.
- Tamper and SSIM thresholds are configured in `configs/thresholds.yaml`.
