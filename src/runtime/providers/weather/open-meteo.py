# src/runtime/providers/weather/open_meteo.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import pandas as pd

@dataclass
class WeatherPoint:
    ts: pd.Timestamp
    t2m_max: float | None
    t2m_min: float | None
    precip_sum: float | None
    wind10_max: float | None
    gust10_max: float | None
    code: int | None

class OpenMeteoProvider:
    """
    Uses open-meteo.com forecast API (free, no key; good coverage).
    Parameters chosen to match features you train on.
    """
    BASE = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout=20):
        self.timeout = timeout

    def hourly_forecast(self, lat: float, lon: float, tz: str, start: str, end: str) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "precipitation",
                "windspeed_10m",
                "windgusts_10m",
                "weathercode"
            ]),
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum"
            ]),
            "timezone": tz,
            "start_date": start,  # "YYYY-MM-DD"
            "end_date": end
        }
        r = requests.get(self.BASE, params=params, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()

        # Hourly frame
        h = pd.DataFrame(j.get("hourly", {}))
        if "time" not in h:
            raise RuntimeError("Open-Meteo hourly returned no time array")
        h["ts"] = pd.to_datetime(h["time"])
        h = h.drop(columns=["time"])

        # Daily aggregates (optional use)
        d = pd.DataFrame(j.get("daily", {}))
        if "time" in d:
            d["date"] = pd.to_datetime(d["time"]).dt.date
            d = d.drop(columns=["time"])

        # Rename to your feature names
        rename = {
            "temperature_2m": "origin_dep_temperature_2m",
            "windspeed_10m": "origin_dep_windspeed_10m",
            "windgusts_10m": "origin_dep_windgusts_10m",
            "precipitation": "origin_dep_precipitation",
            "weathercode": "origin_dep_weathercode"
        }
        h = h.rename(columns=rename)

        # Return hourly frame; caller can window/summarize as needed
        return h
