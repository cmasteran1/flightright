# src/runtime/providers/weather/fused_weather.py
from __future__ import annotations
import requests
from typing import Dict, Any, Optional

from .nws import NWSProvider  # reuse UA and hourly logic

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

def _safe_num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

class FusedWeatherProvider:
    """
    Fuse:
      - Daily numerics (Tmax/Tmin, precip_sum, wind/gust max) from Open-Meteo
      - Hourly + text codes from NWS (with correct °F→°C, mph→m/s conversions handled in NWS)
    All outputs in model-ready units: °C, m/s, mm.
    """

    def __init__(self, nws: Optional[NWSProvider] = None):
        self.nws = nws or NWSProvider()

    @staticmethod
    def _open_meteo_daily(lat: float, lon: float, local_date: str) -> Dict[str, Any]:
        """
        Pull one-day window centered on local_date (Open-Meteo allows timezone=auto).
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max",
                "windgusts_10m_max",
            ]),
            "forecast_days": 7,  # OM will return a multi-day window; we pick the date we need
        }
        r = requests.get(OPEN_METEO, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        out = {
            "temperature_2m_max": None,
            "temperature_2m_min": None,
            "precipitation_sum": 0.0,
            "windspeed_10m_max": None,
            "windgusts_10m_max": None,
        }
        d = (j.get("daily") or {})
        dates = d.get("time") or []
        if not dates:
            return out
        try:
            idx = dates.index(local_date)
        except ValueError:
            # If the exact date is missing, return defaults
            return out

        def pick(key, default=None):
            arr = d.get(key) or []
            if 0 <= idx < len(arr):
                return _safe_num(arr[idx]) if arr[idx] is not None else default
            return default

        out["temperature_2m_max"] = pick("temperature_2m_max")
        out["temperature_2m_min"] = pick("temperature_2m_min")
        out["precipitation_sum"]  = pick("precipitation_sum", 0.0)
        out["windspeed_10m_max"]  = pick("windspeed_10m_max")
        out["windgusts_10m_max"]  = pick("windgusts_10m_max")
        return out

    def get_features(
        self,
        lat: float,
        lon: float,
        local_date: str,
        local_hour: int
    ) -> Dict[str, Any]:
        """
        Merge daily (Open-Meteo) + hourly/codes (NWS).
        Keys returned (matching training):
          - temperature_2m_max/min, precipitation_sum, windspeed_10m_max, windgusts_10m_max (daily)
          - dep_temperature_2m, dep_windspeed_10m, dep_windgusts_10m, dep_precipitation (hourly)
          - wx_code_day, wx_code_hour
          - provider='fused(nws+om)', provider_tz, valid_from_utc, valid_to_utc
        """
        # Hourly/text first (gives us tz + availability window)
        nws = self.nws.get_hourly_daily_text(lat, lon, local_date, local_hour)

        # Daily numerics
        om = {}
        try:
            om = self._open_meteo_daily(lat, lon, local_date)
        except Exception:
            # Keep defaults if Open-Meteo unavailable
            om = {
                "temperature_2m_max": None,
                "temperature_2m_min": None,
                "precipitation_sum": 0.0,
                "windspeed_10m_max": None,
                "windgusts_10m_max": None,
            }

        out: Dict[str, Any] = {
            "provider": "fused(nws+om)",
            "provider_tz": nws.get("provider_tz"),
            "available": bool(nws.get("available") or any(v is not None for v in om.values())),
            "valid_from_utc": nws.get("valid_from_utc"),
            "valid_to_utc": nws.get("valid_to_utc"),

            # daily numerics
            "temperature_2m_max": om.get("temperature_2m_max"),
            "temperature_2m_min": om.get("temperature_2m_min"),
            "precipitation_sum":  om.get("precipitation_sum", 0.0),
            "windspeed_10m_max":  om.get("windspeed_10m_max"),
            "windgusts_10m_max":  om.get("windgusts_10m_max"),

            # hourly at target hour (already normalized in NWS wrapper)
            "dep_temperature_2m": nws.get("dep_temperature_2m"),
            "dep_windspeed_10m":  nws.get("dep_windspeed_10m"),
            "dep_windgusts_10m":  nws.get("dep_windgusts_10m"),
            "dep_precipitation":  nws.get("dep_precipitation", 0.0),

            # codes
            "wx_code_day":  nws.get("wx_code_day", "Unknown"),
            "wx_code_hour": nws.get("wx_code_hour", "Unknown"),
        }
        return out
