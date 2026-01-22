""" example of CLI usage: python -m runtime.providers.weather.nws \
  --iata JFK \
  --airports_csv data/meta/airports.csv \
  --date 2026-02-10 \
  --dep-hour 18 \
  --pretty"""
# src/runtime/providers/weather/nws.py
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd

NWS_POINTS = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

# ---------- ISO helpers ----------

def _parse_iso_dt(s: str) -> Optional[datetime]:
    """Best-effort parser for ISO8601 datetime with timezone offset."""
    try:
        # Python 3.9: fromisoformat handles 'YYYY-MM-DDTHH:MM:SS±HH:MM'
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _parse_iso_duration_to_hours(dur: str) -> float:
    """
    Parse a subset of ISO-8601 durations (PnDTnHnM) into hours.
    Supports PTnH, PTnM, PnD, and combinations we actually see in NWS grid data.
    """
    if not dur or "P" not in dur:
        return 0.0
    # Examples: 'PT1H', 'PT3H', 'PT30M', 'P1D', 'P1DT6H'
    days = hours = minutes = 0.0
    try:
        # Split date/time parts
        dpart, tpart = dur, ""
        if "T" in dur:
            dpart, tpart = dur.split("T", 1)
        # Days
        if "D" in dpart:
            try:
                days = float(dpart.split("P")[1].split("D")[0])
            except Exception:
                pass
        # Hours/Minutes in time part
        if tpart:
            if "H" in tpart:
                try:
                    hours = float(tpart.split("H")[0].split("P")[-1].split("T")[-1].split("D")[-1])
                except Exception:
                    pass
            if "M" in tpart:
                try:
                    # get last number before 'M'
                    minutes = float(tpart.split("M")[0].split("H")[-1].split("P")[-1].split("T")[-1].split("D")[-1])
                except Exception:
                    pass
        return days * 24.0 + hours + minutes / 60.0
    except Exception:
        return 0.0

def _expand_valid_time(valid_time: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    validTime looks like '2026-01-22T06:00:00+00:00/PT1H' or '2026-01-22T00:00:00-05:00/PT3H'
    Return (start_dt, end_dt) with tzinfo.
    """
    try:
        start_s, dur = valid_time.split("/")
        start = _parse_iso_dt(start_s)
        if not start:
            return None, None
        hours = _parse_iso_duration_to_hours(dur)
        end = start + timedelta(hours=hours)
        return start, end
    except Exception:
        return None, None

# ---------- Units / conversions ----------

def _uom_to_si(value: Optional[float], uom: str) -> Optional[float]:
    """
    Map from NWS wmoUnit to SI-like values we want:
    - degC -> °C (no change), degF -> convert to °C
    - m_s-1 -> m/s; km_h-1 -> m/s
    - kg_m-2 -> mm water equivalent (1 kg/m^2 == 1 mm)
    - percent -> % as number 0..100 (leave as-is)
    """
    if value is None:
        return None
    try:
        if "degC" in uom:
            return float(value)
        if "degF" in uom:
            return (float(value) - 32.0) * 5.0 / 9.0
        if "m_s-1" in uom:
            return float(value)
        if "km_h-1" in uom or "km_h-1" in uom.replace("km_h−1", "km_h-1"):
            return float(value) * (1000.0 / 3600.0)
        if "kg_m-2" in uom:
            # Precip: 1 kg/m^2 ≈ 1 mm
            return float(value)
        if "percent" in uom:
            return float(value)
        # If unknown unit, return as-is
        return float(value)
    except Exception:
        return None

def _mph_to_ms(s: Any) -> Optional[float]:
    """Parse 'NN mph' or numeric mph to m/s."""
    if s is None:
        return None
    try:
        if isinstance(s, (int, float)):
            return float(s) * 0.44704
        text = str(s).strip()
        num = text.split()[0]
        return float(num) * 0.44704
    except Exception:
        return None

# ---------- Weather code mapping ----------

def _wx_code_from_text(text: str) -> str:
    """Map NWS shortForecast → coarse code string."""
    if not text:
        return "Unknown"
    t = text.lower()
    # Order matters (more severe first)
    if "thunder" in t:
        return "Thunderstorm"
    if "snow" in t or "blizzard" in t or "sleet" in t or "wintry" in t:
        return "Snow/Ice"
    if "rain" in t or "showers" in t or "drizzle" in t:
        return "Rain"
    if "fog" in t or "mist" in t or "haze" in t or "smoke" in t:
        return "Fog"
    if "windy" in t or "breezy" in t or "gust" in t:
        return "Wind"
    if "clear" in t or "sunny" in t:
        return "Clear"
    if "cloudy" in t or "overcast" in t or "clouds" in t or "partly" in t or "mostly" in t:
        return "Clouds"
    return "Other"

# ---------- Provider ----------

@dataclass
class NWSProvider:
    user_agent: str = ""

    def _headers(self) -> Dict[str, str]:
        ua = self.user_agent or os.getenv("NWS_USER_AGENT", "").strip()
        if not ua:
            ua = "flight-delay-model/0.1 (no-user-agent-set)"
        return {"User-Agent": ua, "Accept": "application/geo+json"}

    def _points(self, lat: float, lon: float) -> Dict[str, Any]:
        url = NWS_POINTS.format(lat=lat, lon=lon)
        r = requests.get(url, headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json()

    # ---- Grid extract helpers ----

    @staticmethod
    def _series_to_df(series_json: Dict[str, Any], provider_tz: str) -> pd.DataFrame:
        """
        Convert a grid series object (with "uom" and "values": [{"validTime","value"}...]) to a DataFrame
        with start, end (aware datetimes), local_date (YYYY-MM-DD), local_hour (0..23), and SI value.
        """
        if not series_json:
            return pd.DataFrame(columns=["start", "end", "value", "local_date", "local_hour"])
        uom = series_json.get("uom", "") or ""
        vals = series_json.get("values") or []
        rows: List[Dict[str, Any]] = []
        for it in vals:
            vt = it.get("validTime")
            val = it.get("value", None)
            if vt is None:
                continue
            start, end = _expand_valid_time(vt)
            if start is None or end is None:
                continue
            # SI conversion based on uom
            si_val = _uom_to_si(val, uom)
            # Convert start to local by provider tz offset via string; we can infer local date/hour from the ISO itself.
            # start already has tzinfo. We can format local pieces using its own offset; for grouping by calendar day,
            # we take the date string from start.isoformat().
            local_date = start.astimezone().date().isoformat()
            local_hour = start.astimezone().hour
            rows.append({
                "start": start, "end": end,
                "value": si_val,
                "local_date": local_date,
                "local_hour": local_hour
            })
        return pd.DataFrame(rows)

    def _get_grid_fields(self, grid_json: Dict[str, Any], provider_tz: str) -> Dict[str, pd.DataFrame]:
        props = grid_json.get("properties") or {}
        # Extract series we care about
        out: Dict[str, pd.DataFrame] = {}
        for key in (
            "temperature",               # hourly degC
            "windSpeed",                 # hourly m/s or km/h
            "windGust",                  # hourly m/s or km/h (may be None sometimes)
            "quantitativePrecipitation", # hourly kg/m^2 == mm
            "maxTemperature",            # daily degC
            "minTemperature",            # daily degC
        ):
            series = props.get(key)
            out[key] = self._series_to_df(series, provider_tz) if series else pd.DataFrame(
                columns=["start", "end", "value", "local_date", "local_hour"]
            )
        return out

    @staticmethod
    def _pick_hour(df: pd.DataFrame, local_date: str, dep_hour_local: int) -> Optional[float]:
        if df.empty:
            return None
        m = df[(df["local_date"] == local_date) & (df["local_hour"] == int(dep_hour_local))]
        if not m.empty:
            v = m["value"].iloc[0]
            return None if pd.isna(v) else float(v)
        return None

    @staticmethod
    def _daily_sum(df: pd.DataFrame, local_date: str) -> float:
        if df.empty:
            return 0.0
        m = df[df["local_date"] == local_date]
        if m.empty:
            return 0.0
        return float(pd.to_numeric(m["value"], errors="coerce").fillna(0.0).sum())

    @staticmethod
    def _daily_max(df: pd.DataFrame, local_date: str) -> Optional[float]:
        if df.empty:
            return None
        m = df[df["local_date"] == local_date]
        if m.empty:
            return None
        v = pd.to_numeric(m["value"], errors="coerce")
        if v.notna().any():
            return float(v.max())
        return None

    @staticmethod
    def _pick_daily(df: pd.DataFrame, local_date: str) -> Optional[float]:
        """For maxTemperature/minTemperature which are daily series in gridData."""
        if df.empty:
            return None
        m = df[df["local_date"] == local_date]
        if m.empty:
            return None
        v = pd.to_numeric(m["value"], errors="coerce")
        if v.notna().any():
            # If multiple segments exist, take the first non-null
            return float(v.dropna().iloc[0])
        return None

    # ---- Public API ----

    def get_features(
        self,
        lat: float,
        lon: float,
        local_date: str,
        dep_hour_local: int
    ) -> Dict[str, Any]:
        """
        Returns:
          Daily (gridData):
            - temperature_2m_max (°C), temperature_2m_min (°C)
            - precipitation_sum (mm)  [sum of hourly QPF for the day]
            - windspeed_10m_max (m/s), windgusts_10m_max (m/s)
            - wx_code_day (from daily text 'forecast')
          Hourly @ departure hour (gridData + hourly text):
            - dep_temperature_2m (°C), dep_windspeed_10m (m/s), dep_windgusts_10m (m/s), dep_precipitation (mm)
            - wx_code_hour (from hourly text 'forecastHourly')
          Also returns availability window (from hourly grid series) and provider_tz.
        """
        out: Dict[str, Any] = {
            "provider": "nws",
            "provider_tz": None,
            "available": False,
            "valid_from_utc": None,
            "valid_to_utc": None,

            "temperature_2m_max": None,
            "temperature_2m_min": None,
            "precipitation_sum": 0.0,
            "windspeed_10m_max": None,
            "windgusts_10m_max": None,

            "dep_temperature_2m": None,
            "dep_windspeed_10m": None,
            "dep_windgusts_10m": None,
            "dep_precipitation": 0.0,

            "wx_code_day": "Unknown",
            "wx_code_hour": "Unknown",
        }

        # Discover endpoints
        pts = self._points(lat, lon)
        pprops = pts.get("properties") or {}
        out["provider_tz"] = pprops.get("timeZone")
        url_hourly = pprops.get("forecastHourly")
        url_daily  = pprops.get("forecast")
        url_grid   = pprops.get("forecastGridData")

        # Fetch gridData (numeric)
        grid = None
        if url_grid:
            rg = requests.get(url_grid, headers=self._headers(), timeout=20)
            if rg.ok:
                grid = rg.json()

        # Availability window (use hourly temperature series span if present)
        if grid:
            g = self._get_grid_fields(grid, out["provider_tz"] or "")
            temp_df = g.get("temperature", pd.DataFrame())
            if not temp_df.empty:
                out["valid_from_utc"] = temp_df["start"].min().astimezone(timezone.utc).isoformat()
                out["valid_to_utc"]   = temp_df["end"].max().astimezone(timezone.utc).isoformat()

            # Daily numeric (prefer direct daily series where available)
            out["temperature_2m_max"] = self._pick_daily(g.get("maxTemperature", pd.DataFrame()), local_date)
            out["temperature_2m_min"] = self._pick_daily(g.get("minTemperature", pd.DataFrame()), local_date)

            # Daily precip sum from hourly QPF (grid publishes hourly QPF; summing over the calendar day)
            out["precipitation_sum"] = self._daily_sum(g.get("quantitativePrecipitation", pd.DataFrame()), local_date)

            # Daily max wind / gust from hourly (aggregation of provided hourly numeric forecast)
            out["windspeed_10m_max"] = self._daily_max(g.get("windSpeed", pd.DataFrame()), local_date)
            out["windgusts_10m_max"] = self._daily_max(g.get("windGust", pd.DataFrame()), local_date)

            # Hour of departure numeric
            out["dep_temperature_2m"] = self._pick_hour(g.get("temperature", pd.DataFrame()), local_date, dep_hour_local)
            out["dep_windspeed_10m"]  = self._pick_hour(g.get("windSpeed", pd.DataFrame()), local_date, dep_hour_local)
            out["dep_windgusts_10m"]  = self._pick_hour(g.get("windGust", pd.DataFrame()), local_date, dep_hour_local)
            out["dep_precipitation"]  = self._pick_hour(g.get("quantitativePrecipitation", pd.DataFrame()), local_date, dep_hour_local) or 0.0

        # Fetch text forecasts for codes (hourly + day/night)
        if url_hourly:
            rh = requests.get(url_hourly, headers=self._headers(), timeout=15)
            if rh.ok:
                hj = rh.json()
                periods = ((hj.get("properties") or {}).get("periods") or [])
                # match by 'YYYY-MM-DDTHH'
                target_prefix = f"{local_date}T{int(dep_hour_local):02d}"
                for p in periods:
                    st = p.get("startTime", "")
                    if st[:13] == target_prefix:
                        out["wx_code_hour"] = _wx_code_from_text(p.get("shortForecast", ""))
                        break

        if url_daily:
            rd = requests.get(url_daily, headers=self._headers(), timeout=15)
            if rd.ok:
                dj = rd.json()
                periods = ((dj.get("properties") or {}).get("periods") or [])
                # choose the first day/night period starting on local_date
                for p in periods:
                    st = p.get("startTime", "")
                    if st[:10] == local_date:
                        out["wx_code_day"] = _wx_code_from_text(p.get("shortForecast", ""))
                        break

        out["available"] = bool(grid or url_hourly or url_daily)
        return out

# Back-compat alias if older code imports this name
NWSWeatherProvider = NWSProvider

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iata", type=str, required=False, help="IATA code (for demo only if you map to lat/lon yourself)")
    ap.add_argument("--lat", type=float, help="Latitude (preferred for direct query)")
    ap.add_argument("--lon", type=float, help="Longitude (preferred for direct query)")
    ap.add_argument("--airports_csv", type=str, help="Optional airports.csv to resolve IATA→lat/lon")
    ap.add_argument("--date", type=str, required=True, help="Local date YYYY-MM-DD")
    ap.add_argument("--dep-hour", type=int, required=True, help="Local departure hour [0..23]")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    lat = args.lat
    lon = args.lon
    if (lat is None or lon is None) and args.iata and args.airports_csv:
        df = pd.read_csv(args.airports_csv)
        iata_col = next((c for c in df.columns if c.lower() in ("iata", "airport", "code")), None)
        lat_col  = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
        lon_col  = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long")), None)
        if not (iata_col and lat_col and lon_col):
            print("[ERROR] airports_csv missing IATA/Latitude/Longitude columns", file=sys.stderr)
            sys.exit(2)
        row = df[df[iata_col].astype(str).str.upper() == args.iata.upper()].head(1)
        if row.empty:
            print("[ERROR] IATA not found in airports_csv", file=sys.stderr)
            sys.exit(2)
        lat = float(row.iloc[0][lat_col])
        lon = float(row.iloc[0][lon_col])

    if lat is None or lon is None:
        print("[ERROR] Provide --lat/--lon or --iata with --airports_csv", file=sys.stderr)
        sys.exit(2)

    prov = NWSProvider(user_agent=os.getenv("NWS_USER_AGENT", "flight-delay-model/0.1 (no-user-agent-set)"))
    out = prov.get_features(lat, lon, args.date, args.dep_hour)
    if args.pretty:
        print(json.dumps(out, indent=2, sort_keys=False))
    else:
        print(json.dumps(out))

if __name__ == "__main__":
    main()
