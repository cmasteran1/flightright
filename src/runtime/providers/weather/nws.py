""" example of CLI usage: python -m runtime.providers.weather.nws \
  --iata JFK \
  --airports_csv data/meta/airports.csv \
  --date 2026-02-10 \
  --dep-hour 18 \
  --pretty"""
# src/runtime/providers/weather/nws.py
from __future__ import annotations
import os, sys, json, argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
import pandas as pd
# src/runtime/providers/weather/nws.py
from __future__ import annotations
import os
import sys
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

import requests
import pandas as pd

NWS_POINTS = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"

def _to_aware(dt_str: str) -> Optional[datetime]:
    try:
        # accepts "YYYY-MM-DDTHH:MM:SS-05:00" or with Z
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def _parse_duration_iso8601(s: str) -> timedelta:
    # minimal PT#H / PT#M / PT#H#M support
    # examples: "PT1H", "PT2H30M", "PT45M"
    if not s or not s.startswith("PT"):
        return timedelta(0)
    hours = 0
    mins = 0
    s2 = s[2:]
    # find H and M if present
    if "H" in s2:
        pre, rest = s2.split("H", 1)
        hours = int(pre) if pre else 0
        s2 = rest
    if "M" in s2:
        pre, _ = s2.split("M", 1)
        mins = int(pre) if pre else 0
    return timedelta(hours=hours, minutes=mins)

def _local_day_window(local_date: str, tz_offset_minutes: int) -> Tuple[datetime, datetime]:
    # build start/end datetimes in UTC corresponding to local midnight..midnight(+1)
    # We approximate by applying a fixed offset minutes for the provider TZ at query time.
    # Good enough for forecast purposes (DST changes between now and the day rarely bite within 7-day horizon).
    try:
        # interpret local_date at 00:00 local
        local_midnight = datetime.fromisoformat(local_date)  # naive date
    except Exception:
        # fallback: assume already date-only like "2026-01-28"
        y, m, d = map(int, local_date.split("-"))
        local_midnight = datetime(y, m, d)
    offset = timedelta(minutes=tz_offset_minutes)
    # Convert local → UTC by subtracting offset
    start_utc = (local_midnight - offset).replace(tzinfo=timezone.utc)
    end_utc   = (local_midnight + timedelta(days=1) - offset).replace(tzinfo=timezone.utc)
    return start_utc, end_utc

def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    # returns overlapping seconds
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    delta = (earliest_end - latest_start).total_seconds()
    return max(0.0, delta)

def _mph_to_ms(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v) * 0.44704
        # strings like "12 mph"
        s = str(v).strip().split()[0]
        return float(s) * 0.44704
    except Exception:
        return None

def _f_to_c(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return (float(v) - 32.0) * (5.0 / 9.0)
    except Exception:
        return None

def _safe_num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().split()[0]
        return float(s)
    except Exception:
        return None

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
        r = requests.get(url, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _wx_code_from_text(text: str) -> str:
        if not text:
            return "Unknown"
        t = text.lower()
        if "thunder" in t:
            return "Thunderstorm"
        if any(k in t for k in ("snow", "blizzard", "sleet", "wintry", "freez")):
            return "Snow/Ice"
        if any(k in t for k in ("rain", "shower", "drizzle")):
            return "Rain"
        if any(k in t for k in ("fog", "mist", "haze", "smoke")):
            return "Fog"
        if any(k in t for k in ("windy", "breezy", "gust")):
            return "Wind"
        if any(k in t for k in ("clear", "sunny")):
            return "Clear"
        if any(k in t for k in ("cloudy", "overcast", "clouds", "partly", "mostly")):
            return "Clouds"
        return "Other"

    @staticmethod
    def _pick_hourly_period(hourly_json: Dict[str, Any], local_date: str, local_hour: int) -> Optional[Dict[str, Any]]:
        periods = ((hourly_json.get("properties") or {}).get("periods") or [])
        # startTime is local with offset; simple string prefix match on YYYY-MM-DDTHH
        prefix = f"{local_date}T{int(local_hour):02d}"
        for p in periods:
            st = p.get("startTime", "")
            if st[:13] == prefix:
                return p
        return None

    def _grid_day_stats(
        self,
        grid_json: Dict[str, Any],
        tz_offset_minutes: int,
        local_date: str
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Returns (tmax_C, tmin_C, precip_mm) for the local_date using forecastGridData.
        """
        props = grid_json.get("properties") or {}
        start_utc, end_utc = _local_day_window(local_date, tz_offset_minutes)

        def accumulate(name: str, agg: str = "sum") -> Optional[float]:
            v = props.get(name)
            if not v or "values" not in v:
                return None
            vals = []
            for item in v["values"]:
                vt = item.get("validTime", "")
                if not vt or "/PT" not in vt:
                    continue
                t0_str, dur_str = vt.split("/")
                t0 = _to_aware(t0_str)
                if not t0:
                    continue
                dur = _parse_duration_iso8601(dur_str)
                t1 = t0 + dur
                ov = _overlaps(t0, t1, start_utc, end_utc)
                if ov > 0:
                    vals.append((ov, item.get("value")))
            if not vals:
                return None
            if agg == "sum":
                # for quantitativePrecipitation the units are kg/m^2 over the interval (≈ mm)
                s = 0.0
                for ov, val in vals:
                    x = _safe_num(val)
                    if x is not None:
                        s += x
                return s
            elif agg == "max":
                mx = None
                for ov, val in vals:
                    x = _safe_num(val)
                    if x is None:
                        continue
                    mx = x if mx is None else max(mx, x)
                return mx
            elif agg == "min":
                mn = None
                for ov, val in vals:
                    x = _safe_num(val)
                    if x is None:
                        continue
                    mn = x if mn is None else min(mn, x)
                return mn
            return None

        # maxTemperature/minTemperature are °F; convert to °C
        tmax_f = accumulate("maxTemperature", agg="max")
        tmin_f = accumulate("minTemperature", agg="min")
        precip_mm = accumulate("quantitativePrecipitation", agg="sum") or 0.0

        tmax_c = _f_to_c(tmax_f) if tmax_f is not None else None
        tmin_c = _f_to_c(tmin_f) if tmin_f is not None else None
        return tmax_c, tmin_c, float(precip_mm)

    def get_features(
        self,
        lat: float,
        lon: float,
        local_date: str,
        dep_hour_local: int
    ) -> Dict[str, Any]:
        """
        Returns daily + hourly features for NWS:
          Daily (local_date): temperature_2m_max/min (°C), precipitation_sum (mm),
                               windspeed_10m_max, windgusts_10m_max (m/s)
          Hourly @ dep_hour_local: dep_temperature_2m (°C), dep_windspeed_10m/dep_windgusts_10m (m/s),
                                   dep_precipitation (mm; 0 due to NWS hourly limitations),
          Weather codes: wx_code_day (coarse from daily text), wx_code_hour (from hourly text)
          Availability window from hourly feed.
        """
        out: Dict[str, Any] = {
            "provider": "nws",
            "provider_tz": None,
            "available": False,
            "valid_from_utc": None,
            "valid_to_utc": None,
        }

        # /points → URLs
        j = self._points(lat, lon)
        props = j.get("properties") or {}
        hourly_url = props.get("forecastHourly")
        daily_url  = props.get("forecast")
        grid_url   = props.get("forecastGridData")
        out["provider_tz"] = props.get("timeZone", "")

        # For the local-day window, approximate tz offset using the first hourly period offset
        tz_offset_minutes = 0

        hjson = djson = gjson = None

        if hourly_url:
            rh = requests.get(hourly_url, headers=self._headers(), timeout=20)
            if rh.ok:
                hjson = rh.json()
                periods = ((hjson.get("properties") or {}).get("periods") or [])
                if periods:
                    st0 = periods[0].get("startTime")
                    dt0 = _to_aware(st0)
                    if dt0:
                        tz_offset_minutes = int(dt0.utcoffset().total_seconds() / 60.0)

        if daily_url:
            rd = requests.get(daily_url, headers=self._headers(), timeout=20)
            if rd.ok:
                djson = rd.json()

        if grid_url:
            rg = requests.get(grid_url, headers=self._headers(), timeout=20)
            if rg.ok:
                gjson = rg.json()

        # availability window from hourly
        if hjson:
            periods = ((hjson.get("properties") or {}).get("periods") or [])
            if periods:
                out["valid_from_utc"] = periods[0].get("startTime")
                out["valid_to_utc"] = periods[-1].get("endTime")

        # daily coarse code from text forecast if available
        wx_day = "Unknown"
        if djson:
            d_per = ((djson.get("properties") or {}).get("periods") or [])
            for p in d_per:
                st = p.get("startTime", "")
                if st[:10] == local_date:
                    wx_day = self._wx_code_from_text(p.get("shortForecast", ""))
                    break

        # hourly @ dep hour
        dep_t = dep_w = dep_g = None
        dep_p = 0.0
        wx_hour = "Unknown"
        if hjson:
            p_hr = self._pick_hourly_period(hjson, local_date, dep_hour_local)
            if p_hr:
                dep_t = _safe_num(p_hr.get("temperature"))
                dep_w = _mph_to_ms(p_hr.get("windSpeed"))
                dep_g = _mph_to_ms(p_hr.get("windGust"))
                wx_hour = self._wx_code_from_text(p_hr.get("shortForecast", ""))
                dep_p = 0.0  # NWS hourly doesn’t provide mm reliably

        # wind/gust daily maxima from hourly series over local date
        wind_max_ms = None
        gust_max_ms = None
        if hjson:
            start_utc, end_utc = _local_day_window(local_date, tz_offset_minutes)
            vals_w, vals_g = [], []
            for p in ((hjson.get("properties") or {}).get("periods") or []):
                st = _to_aware(p.get("startTime", ""))
                if not st:
                    continue
                en = st + timedelta(hours=1)
                if _overlaps(st, en, start_utc, end_utc) > 0:
                    w = _mph_to_ms(p.get("windSpeed"))
                    g = _mph_to_ms(p.get("windGust"))
                    if w is not None:
                        vals_w.append(w)
                    if g is not None:
                        vals_g.append(g)
            wind_max_ms = max(vals_w) if vals_w else None
            gust_max_ms = max(vals_g) if vals_g else None

        # daily tmax/tmin + precip from forecastGridData
        tmax_c = tmin_c = None
        precip_mm = 0.0
        if gjson:
            tmax_c, tmin_c, precip_mm = self._grid_day_stats(gjson, tz_offset_minutes, local_date)

        out.update({
            # daily
            "temperature_2m_max": tmax_c,
            "temperature_2m_min": tmin_c,
            "precipitation_sum": precip_mm,
            "windspeed_10m_max": wind_max_ms,
            "windgusts_10m_max": gust_max_ms,
            "wx_code_day": wx_day,

            # hourly snapshot
            "dep_temperature_2m": dep_t if dep_t is None else (dep_t if abs(dep_t) > 60 else dep_t),  # dep_t is °F? No: NWS hourly "temperature" is in °F; convert to °C below.
            "dep_windspeed_10m": dep_w,
            "dep_windgusts_10m": dep_g,
            "dep_precipitation": dep_p,
            "wx_code_hour": wx_hour,
        })

        # Convert hourly temperature (which NWS hourly gives in °F) to °C
        if out["dep_temperature_2m"] is not None:
            out["dep_temperature_2m"] = _f_to_c(out["dep_temperature_2m"])

        # availability flag
        out["available"] = bool(hjson or djson or gjson)
        out["units_note"] = {
            "daily_temps": "°C (from grid: maxTemperature/minTemperature)",
            "daily_precip": "mm (grid quantitativePrecipitation; kg/m² ≈ mm)",
            "daily_wind/gust": "m/s (max over hourly mph)",
            "hourly_temp": "°C (converted from hourly °F)",
            "hourly_wind/gust": "m/s (from hourly mph)",
        }
        return out

# ---------------- CLI for ad-hoc testing ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iata", type=str, help="IATA for demo (requires --airports_csv).")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--airports_csv", type=str)
    ap.add_argument("--date", type=str, required=True, help="Local date YYYY-MM-DD")
    ap.add_argument("--dep-hour", type=int, required=True, help="Local hour [0..23]")
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
