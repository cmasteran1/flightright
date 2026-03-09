# src/flightright/features/weather.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from flightright.core.time import AirportTime
from flightright.integrations.weather.open_meteo import OpenMeteoClient


KELVIN_OFFSET = 273.15


@dataclass(frozen=True)
class WeatherFeatures:
    # daily
    origin_temp_max_K: Optional[float]
    origin_temp_min_K: Optional[float]
    origin_daily_precip_sum_mm: Optional[float]
    origin_daily_windspeed_max_kmh: Optional[float]
    origin_daily_weathercode: Optional[int]
    # dep-hour
    origin_dep_temp_K: Optional[float]
    origin_dep_precip_mm: Optional[float]
    origin_dep_windspeed_kmh: Optional[float]
    origin_dep_hour_weathercode: Optional[int]


def _safe_get(arr, idx, default=None):
    try:
        return arr[idx]
    except Exception:
        return default


def build_weather_features_open_meteo(
    *,
    client: OpenMeteoClient,
    origin_lat: float,
    origin_lon: float,
    origin_tz: str,
    flight_dep_local_dt: datetime,
) -> WeatherFeatures:
    """
    Calls Open-Meteo for the flight's local date and picks:
      - the daily row matching the flight's local date
      - the hourly row closest to scheduled departure time
    Returns None fields if pieces are missing (so model selection can fall back).
    """
    at = AirportTime(origin_tz)
    day = flight_dep_local_dt.date()

    data = client.forecast_daily_hourly(
        latitude=origin_lat,
        longitude=origin_lon,
        tz=origin_tz,
        day=day,
    )

    # Daily block: match the requested local date explicitly
    daily = data.get("daily") or {}
    daily_times = daily.get("time") or []

    day_str = day.isoformat()
    daily_i = None
    for i, t in enumerate(daily_times):
        if str(t) == day_str:
            daily_i = i
            break

    d_temp_max = _safe_get(daily.get("temperature_2m_max"), daily_i)
    d_temp_min = _safe_get(daily.get("temperature_2m_min"), daily_i)
    d_precip = _safe_get(daily.get("precipitation_sum"), daily_i)
    d_wind = _safe_get(daily.get("windspeed_10m_max"), daily_i)
    d_code = _safe_get(daily.get("weathercode"), daily_i)

    # Hourly block
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []

    # Open-Meteo returns times in local tz when timezone param is set.
    # We'll find the closest hour.
    target = flight_dep_local_dt.replace(minute=0, second=0, microsecond=0)
    if target.tzinfo is None:
        target = target.replace(tzinfo=at.tz)

    best_i = None
    best_abs = None
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except Exception:
            continue

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=at.tz)

        diff = abs((dt - target).total_seconds())
        if best_abs is None or diff < best_abs:
            best_abs = diff
            best_i = i

    h_temp = _safe_get(hourly.get("temperature_2m"), best_i)
    h_precip = _safe_get(hourly.get("precipitation"), best_i)
    h_wind = _safe_get(hourly.get("windspeed_10m"), best_i)
    h_code = _safe_get(hourly.get("weathercode"), best_i)

    def toK(x):
        return None if x is None else float(x) + KELVIN_OFFSET

    return WeatherFeatures(
        origin_temp_max_K=toK(d_temp_max),
        origin_temp_min_K=toK(d_temp_min),
        origin_daily_precip_sum_mm=None if d_precip is None else float(d_precip),
        origin_daily_windspeed_max_kmh=None if d_wind is None else float(d_wind),
        origin_daily_weathercode=None if d_code is None else int(d_code),
        origin_dep_temp_K=toK(h_temp),
        origin_dep_precip_mm=None if h_precip is None else float(h_precip),
        origin_dep_windspeed_kmh=None if h_wind is None else float(h_wind),
        origin_dep_hour_weathercode=None if h_code is None else int(h_code),
    )