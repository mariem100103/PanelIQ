from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo

    TUNIS_TZ = ZoneInfo("Africa/Tunis")
except Exception:
    TUNIS_TZ = None

# Tunisia uses UTC+1 year-round (no DST). Used if ZoneInfo data is unavailable (e.g. missing tzdata on Windows).
TUNIS_FALLBACK_TZ = timezone(timedelta(hours=1))

import requests as req

# Load .env when this module is imported (backend main loads dotenv first; this is a fallback)
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[1]
    load_dotenv(_root / ".env")
except ImportError:
    pass

SES_VEHICLE_WEIGHTS = {
    "car": 1.0,
    "truck": 0.5,
    "bus": 0.3,
    "motorcycle": 0.7,
    "person": 0.2,
}

ZONE_PROFILES: dict[str, dict[str, Any]] = {
    "urban-commercial": {
        "ses_score": 7.5,
        "age_dominant": "25-44",
        "income": "middle-high",
    },
    "residential": {"ses_score": 5.0, "age_dominant": "30-50", "income": "middle"},
    "university": {"ses_score": 4.0, "age_dominant": "18-26", "income": "low-middle"},
    "industrial": {"ses_score": 3.5, "age_dominant": "25-45", "income": "low-middle"},
    "coastal-leisure": {"ses_score": 8.5, "age_dominant": "20-40", "income": "high"},
}

BASE_IMPRESSIONS_BY_ZONE = {
    "urban-commercial": 420,
    "residential": 180,
    "university": 260,
    "industrial": 140,
    "coastal-leisure": 310,
}

OWM_API_KEY = os.getenv("OWM_API_KEY", "")
BASE_CPM_TND = 8.0

# panel_id -> (weather_mult, weather_label, monotonic_expiry)
_weather_cache: dict[str, tuple[float, str, float]] = {}
_WEATHER_TTL_SEC = 30 * 60

CONTEXT_TIMEZONE_LABEL = (
    "Africa/Tunis" if TUNIS_TZ is not None else "UTC+01:00 (Tunisia fallback)"
)


def now_tunis_wall_clock() -> datetime:
    """Local Tunisian wall-clock time for rush-hour, Friday, and summer patterns."""
    if TUNIS_TZ is not None:
        return datetime.now(TUNIS_TZ)
    return datetime.now(TUNIS_FALLBACK_TZ)


def zone_profile(zone_type: str) -> dict[str, Any]:
    return ZONE_PROFILES.get(zone_type, ZONE_PROFILES["residential"])


def get_context_multiplier(dt: datetime) -> tuple[float, str]:
    """Expects *dt* in Tunisia local time (or any tz-aware clock used for road patterns)."""
    hour = dt.hour
    weekday = dt.weekday()
    month = dt.month
    if weekday == 4 and 11 <= hour <= 13:
        return 0.6, "Friday midday"
    if hour in (7, 8, 17, 18, 19):
        return 1.3, "Rush hour"
    if month in (7, 8) and 13 <= hour <= 16:
        return 0.75, "Summer afternoon"
    if hour >= 22 or hour < 6:
        return 0.4, "Night"
    return 1.0, "Normal"


def get_weather_multiplier(lat: float, lng: float, panel_id: str) -> tuple[float, str]:
    now = time.monotonic()
    cached = _weather_cache.get(panel_id)
    if cached is not None:
        mult, label, exp = cached
        if now < exp:
            return mult, label

    if not OWM_API_KEY.strip():
        _weather_cache[panel_id] = (1.0, "Unknown", now + _WEATHER_TTL_SEC)
        return 1.0, "Unknown"

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lng}&appid={OWM_API_KEY}"
        )
        data = req.get(url, timeout=3).json()
        wid = int(data["weather"][0]["id"])
        if wid < 600:
            mult, label = 0.8, "Rain"
        elif wid < 700:
            mult, label = 0.7, "Snow"
        else:
            mult, label = 1.0, "Clear"
    except Exception:
        mult, label = 1.0, "Unknown"

    _weather_cache[panel_id] = (mult, label, now + _WEATHER_TTL_SEC)
    return mult, label


def _weighted_vehicle_total(reading: dict) -> float:
    return (
        reading.get("car", 0)
        + reading.get("truck", 0) * 0.5
        + reading.get("bus", 0) * 0.3
        + reading.get("motorcycle", 0) * 0.7
    )


def _vehicle_class_mix_proxy(reading: dict) -> float:
    total = sum(reading.get(k, 0) for k in ["car", "truck", "bus", "motorcycle"])
    if total <= 0:
        return 0.5
    return sum(reading.get(k, 0) * SES_VEHICLE_WEIGHTS[k] for k in SES_VEHICLE_WEIGHTS if k != "person") / total


def build_panel_score(
    panel_id: str,
    zone_type: str,
    lat: float,
    lng: float,
    reading: dict | None,
    *,
    avg_occupancy: float = 1.4,
    visibility_factor: float = 0.8,
) -> dict[str, Any]:
    """Full score payload for GET /panels/{id}/score."""
    now = now_tunis_wall_clock()
    profile = zone_profile(zone_type)
    zone_ses = float(profile["ses_score"])
    age_dominant = profile["age_dominant"]
    income_bracket = profile["income"]

    time_mult, context_label = get_context_multiplier(now)
    weather_mult, weather_label = get_weather_multiplier(lat, lng, panel_id)
    zone_ses_weight = zone_ses / 10.0

    if reading is None:
        base = BASE_IMPRESSIONS_BY_ZONE.get(zone_type, 200)
        estimated = base * time_mult * weather_mult
        ses_score = round(zone_ses, 2)
        attentive_impressions = max(0, round(estimated * 0.12))
        attention_rate = round(0.12, 2)
        cpm = round(BASE_CPM_TND * ses_score / 5.0, 2)

        return {
            "panel_id": panel_id,
            "window_start": None,
            "window_end": None,
            "estimated_impressions": max(0, round(estimated)),
            "attentive_impressions": attentive_impressions,
            "attention_rate": attention_rate,
            "ses_score": ses_score,
            "age_dominant": age_dominant,
            "income_bracket": income_bracket,
            "zone_type": zone_type,
            "context_multiplier": round(time_mult, 2),
            "context_label": context_label,
            "weather_multiplier": round(weather_mult, 2),
            "weather_label": weather_label,
            "cpm_estimate_tnd": cpm,
            "data_available": False,
            "source": "zone-baseline",
            "vehicle_total": 0,
            "ses_proxy": round(_vehicle_class_mix_proxy({"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "person": 0}), 3),
            "context_timezone": CONTEXT_TIMEZONE_LABEL,
            "context_local_time": now.isoformat(),
        }

    total_vehicles = _weighted_vehicle_total(reading)
    pedestrians = reading.get("person", 0)
    ses_proxy = _vehicle_class_mix_proxy(reading)
    ses_score = round(zone_ses * 0.55 + ses_proxy * 10 * 0.45, 2)

    impressions = (
        total_vehicles
        * avg_occupancy
        * visibility_factor
        * zone_ses_weight
        * time_mult
        * weather_mult
    )
    impressions += pedestrians * 0.25 * visibility_factor * zone_ses_weight * time_mult * weather_mult

    estimated = max(0, round(impressions))
    attentive_impressions = max(
        0,
        round(
            (pedestrians * 0.65 + total_vehicles * 0.12)
            * visibility_factor
            * min(1.2, zone_ses_weight + 0.25)
        ),
    )
    attention_rate = round(attentive_impressions / max(estimated, 1), 3)
    cpm = round(BASE_CPM_TND * ses_score / 5.0, 2)

    return {
        "panel_id": reading.get("panel_id", panel_id),
        "window_start": reading.get("window_start"),
        "window_end": reading.get("window_end"),
        "estimated_impressions": estimated,
        "attentive_impressions": attentive_impressions,
        "attention_rate": attention_rate,
        "ses_score": ses_score,
        "age_dominant": age_dominant,
        "income_bracket": income_bracket,
        "zone_type": zone_type,
        "context_multiplier": round(time_mult, 2),
        "context_label": context_label,
        "weather_multiplier": round(weather_mult, 2),
        "weather_label": weather_label,
        "cpm_estimate_tnd": cpm,
        "data_available": True,
        "source": "edge-ingest",
        "vehicle_total": round(total_vehicles),
        "ses_proxy": round(ses_proxy, 3),
        "context_timezone": CONTEXT_TIMEZONE_LABEL,
        "context_local_time": now.isoformat(),
    }


def estimate_footfall(reading: dict, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper (legacy callers)."""
    pid = reading.get("panel_id") or ""
    return build_panel_score(
        pid,
        "urban-commercial",
        36.8065,
        10.1815,
        reading,
        **kwargs,
    )
