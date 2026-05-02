import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from backend.scoring import build_panel_score
from backend.verify_pipeline import default_video_path, run_video_model_check
from panel_paths import get_panel_paths, resolve_panel_paths

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="DOOH Panel Intelligence API")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

DB_PATH = PROJECT_ROOT / "data" / "readings.db"
os.makedirs(PROJECT_ROOT / "data", exist_ok=True)


def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vehicle_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        received_at TEXT,
        panel_id TEXT,
        window_start TEXT,
        window_end TEXT,
        car INTEGER, truck INTEGER, bus INTEGER,
        motorcycle INTEGER, person INTEGER
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        received_at TEXT,
        type TEXT,
        ssim_score REAL,
        confidence REAL
    )"""
    )
    alert_cols = {r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()}
    if "panel_id" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN panel_id TEXT DEFAULT ''")
    conn.commit()
    return conn


conn = get_conn()

_CFG_PATH = PROJECT_ROOT / "configs" / "panel_rois.yaml"


def load_panels_config() -> dict:
    try:
        cfg = yaml.safe_load(open(_CFG_PATH, "r", encoding="utf-8")) or {}
    except OSError:
        return {}
    return cfg.get("panels") or {}


def default_panel_id() -> str:
    panels = load_panels_config()
    if not panels:
        return "panel_001"
    return next(iter(panels))


def panel_zone_type(panel_id: str) -> str:
    try:
        return str(load_panels_config()[panel_id].get("zone_type", ""))
    except KeyError:
        return ""


def get_panel_meta(panel_id: str) -> dict | None:
    return load_panels_config().get(panel_id)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    pid = default_panel_id()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "default_panel_id": pid,
            "video_hint": str(default_video_path(pid)),
        },
    )


@app.get("/panels")
def list_panels():
    panels_cfg = load_panels_config()
    out = []
    for pid, meta in panels_cfg.items():
        paths = resolve_panel_paths(PROJECT_ROOT, pid)
        rel = get_panel_paths(pid, project_root=PROJECT_ROOT)
        out.append(
            {
                "panel_id": meta.get("panel_id", pid),
                "zone_type": meta.get("zone_type", ""),
                "has_video": paths["video"].is_file(),
                "has_reference": paths["reference"].is_file(),
                "video_path": rel["video"],
                "reference_path": rel["reference"],
            }
        )
    return out


@app.get("/api/verify-video")
def api_verify_video(panel_id: str = "panel_001", max_frames: int = 48, stride: int = 8):
    max_frames = max(1, min(max_frames, 200))
    stride = max(1, min(stride, 120))
    return run_video_model_check(panel_id=panel_id, max_frames=max_frames, stride=stride)


class VehicleMix(BaseModel):
    panel_id: str = "panel_001"
    window_start: str = ""
    window_end: str = ""
    car: int = 0
    truck: int = 0
    bus: int = 0
    motorcycle: int = 0
    person: int = 0


class IntegrityAlert(BaseModel):
    type: str
    ssim_score: float = 0.0
    confidence: float = 0.0
    panel_id: str = ""


@app.post("/ingest/vehicle-mix")
def ingest_vehicle(data: VehicleMix):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO vehicle_readings VALUES (NULL,?,?,?,?,?,?,?,?,?)",
        (
            now,
            data.panel_id,
            data.window_start,
            data.window_end,
            data.car,
            data.truck,
            data.bus,
            data.motorcycle,
            data.person,
        ),
    )
    conn.commit()
    return {"status": "ok", "received_at": now}


@app.post("/ingest/integrity-alert")
def ingest_alert(data: IntegrityAlert):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO alerts VALUES (NULL,?,?,?,?,?)",
        (now, data.type, data.ssim_score, data.confidence, data.panel_id),
    )
    conn.commit()
    print(f"[ALERT] {data.type} | panel={data.panel_id} | confidence={data.confidence:.2f}")
    return {"status": "alert received", "received_at": now}


@app.get("/data")
@app.get("/data")
def get_data(panel_id: str | None = Query(None)):
    rows = conn.execute("SELECT * FROM vehicle_readings ORDER BY id DESC LIMIT 100").fetchall()

    def row_to_dict(row):
        return {
            "id":           row[0],
            "received_at":  row[1],
            "panel_id":     row[2],
            "window_start": row[3],
            "window_end":   row[4],
            "car":          row[5],
            "truck":        row[6],
            "bus":          row[7],
            "motorcycle":   row[8],
            "person":       row[9],
        }

    def alert_to_dict(a):
        return {
            "id":          a[0],
            "received_at": a[1],
            "type":        a[2],
            "ssim_score":  a[3],
            "confidence":  a[4],
            "panel_id":    a[5] if len(a) > 5 else "",
        }

    if panel_id:
        alert_rows = conn.execute(
            "SELECT * FROM alerts WHERE panel_id=? ORDER BY id DESC LIMIT 50",
            (panel_id,),
        ).fetchall()
    else:
        alert_rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT 50"
        ).fetchall()

    readings = [row_to_dict(r) for r in rows]

    by_panel = defaultdict(list)
    for r in readings:
        by_panel[r["panel_id"]].append(r)

    cut = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    alerts_24h_count = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE received_at >= ?", (cut,)
    ).fetchone()[0]

    return {
        "readings_by_panel": dict(by_panel),
        "readings": readings,
        "alerts": [alert_to_dict(a) for a in alert_rows],
        "alerts_24h_count": alerts_24h_count,
    }


@app.get("/panels/{panel_id}/score")
def get_panel_score(panel_id: str):
    meta = get_panel_meta(panel_id)
    if not meta:
        return {"error": "unknown panel", "panel_id": panel_id}
    zt = str(meta.get("zone_type", "residential"))
    lat = float(meta.get("lat", 36.8065))
    lng = float(meta.get("lng", 10.1815))

    row = conn.execute(
        "SELECT * FROM vehicle_readings WHERE panel_id=? ORDER BY id DESC LIMIT 1",
        (panel_id,),
    ).fetchone()
    reading = None
    if row:
        reading = {
            "panel_id": row[2],
            "window_start": row[3],
            "window_end": row[4],
            "car": row[5],
            "truck": row[6],
            "bus": row[7],
            "motorcycle": row[8],
            "person": row[9],
        }

    return build_panel_score(panel_id, zt, lat, lng, reading)

