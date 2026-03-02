# src/flightright/core/weather/open_meteo.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from flightright.core.http import HttpClient


@dataclass(frozen=True)
class OpenMeteoClient:
    """
    Uses Open-Meteo Forecast API (no key by default).

    Notes:
    - Open-Meteo variable names use snake_case (e.g., weather_code, wind_speed_10m).
    - If your downstream feature builder expects different names, it should map them there.
    - Units are explicitly requested to stay stable across API defaults.
    """
    http: HttpClient = HttpClient()
    base_url: str = "https://api.open-meteo.com/v1/forecast"

    def forecast_daily_hourly(
        self,
        *,
        latitude: float,
        longitude: float,
        tz: str,
        day: date,
        daily_vars: Optional[List[str]] = None,
        hourly_vars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Open-Meteo Forecast API variable names:
        # - Hourly: temperature_2m, precipitation, wind_speed_10m, weather_code
        # - Daily:  temperature_2m_max, temperature_2m_min, precipitation_sum, wind_speed_10m_max, weather_code
        #
        # See docs for canonical names.
        # https://open-meteo.com/en/docs
        daily_vars = daily_vars or [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "weather_code",
        ]
        hourly_vars = hourly_vars or [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "weather_code",
        ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": tz,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "daily": ",".join(daily_vars),
            "hourly": ",".join(hourly_vars),
            # lock units/time format so parsing stays deterministic
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timeformat": "iso8601",
        }
        return self.http.get_json(self.base_url, params=params)