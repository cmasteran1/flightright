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

NWS_POINTS = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

@dataclass
class NWSProvider:
    """
    Minimal NWS provider that returns:
      - daily aggregate placeholders + wx_code_day (from /forecast shortForecast)
      - hourly-at-departure fields + wx_code_hour (from /forecastHourly shortForecast)
      - availability window from hourly periods
    Units:
      - We standardize to °C for temps and m/s for winds if numeric present in JSON.
        (NWS hourly returns temperature (°F) and windSpeed like '12 mph' in many grids;
         we parse and convert when possible.)
    """
    user_agent: str = ""

    def _headers(self) -> Dict[str, str]:
        ua = self.user_agent or os.getenv("NWS_USER_AGENT", "").strip()
        if not ua:
            ua = "flight-delay-model/0.1 (no-user-agent-set)"
        return {"User-Agent": ua, "Accept": "application/geo+json, application/json"}

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
        if "thunder" in t: return "Thunderstorm"
        if any(k in t for k in ["snow", "blizzard", "sleet", "wintry"]): return "Snow/Ice"
        if any(k in t for k in ["rain", "showers", "drizzle"]): return "Rain"
        if any(k in t for k in ["fog", "mist", "haze", "smoke"]): return "Fog"
        if any(k in t for k in ["windy", "breezy", "gust"]): return "Wind"
        if any(k in t for k in ["clear", "sunny"]): return "Clear"
        if any(k in t for k in ["cloudy", "overcast", "clouds", "partly", "mostly"]): return "Clouds"
        return "Other"

    @staticmethod
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

    @staticmethod
    def _f_to_c(v_f) -> Optional[float]:
        n = NWSProvider._safe_num(v_f)
        return None if n is None else (n - 32.0) * (5.0/9.0)

    @staticmethod
    def _mph_to_ms(v) -> Optional[float]:
        n = NWSProvider._safe_num(v)
        return None if n is None else n * 0.44704

    @staticmethod
    def _pick_period_hourly(hourly_json: Dict[str, Any], local_iso_hour_prefix: str) -> Optional[Dict[str, Any]]:
        periods = ((hourly_json.get("properties") or {}).get("periods") or [])
        # hourly 'startTime' is local ISO with offset, e.g. '2026-01-27T09:00:00-07:00'
        pref = local_iso_hour_prefix[:13]  # 'YYYY-MM-DDTHH'
        for p in periods:
            st = p.get("startTime", "")
            if st[:13] == pref:
                return p
        return None

    @staticmethod
    def _pick_day_daily(daily_json: Dict[str, Any], local_date: str) -> Optional[Dict[str, Any]]:
        periods = ((daily_json.get("properties") or {}).get("periods") or [])
        # daily series alternates Day/Night with startTime also local ISO
        for p in periods:
            st = p.get("startTime", "")
            if st[:10] == local_date:
                return p
        return None

    def get_features(self, lat: float, lon: float, local_date: str, dep_hour_local: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider": "nws",
            "provider_tz": None,
            "available": False,
            "valid_from_utc": None,
            "valid_to_utc": None,
        }

        pjson = self._points(lat, lon)
        props = pjson.get("properties") or {}
        out["provider_tz"] = props.get("timeZone")
        url_hourly = props.get("forecastHourly")
        url_daily  = props.get("forecast")

        hjson = djson = None
        if url_hourly:
            rh = requests.get(url_hourly, headers=self._headers(), timeout=20)
            if rh.ok:
                hjson = rh.json()
        if url_daily:
            rd = requests.get(url_daily, headers=self._headers(), timeout=20)
            if rd.ok:
                djson = rd.json()

        # availability window from hourly
        if hjson:
            periods = ((hjson.get("properties") or {}).get("periods") or [])
            if periods:
                out["valid_from_utc"] = periods[0].get("startTime")
                out["valid_to_utc"]   = periods[-1].get("endTime")

        # daily (coarse): codes + placeholders for numeric summary
        out["temperature_2m_max"] = None
        out["temperature_2m_min"] = None
        out["precipitation_sum"]  = 0.0
        out["windspeed_10m_max"]  = None
        out["windgusts_10m_max"]  = None
        if djson:
            p_day = self._pick_day_daily(djson, local_date)
            if p_day:
                out["wx_code_day"] = self._wx_code_from_text(p_day.get("shortForecast", ""))
            else:
                out["wx_code_day"] = "Unknown"
        else:
            out["wx_code_day"] = "Unknown"

        # hourly at dep hour
        out["dep_temperature_2m"] = None
        out["dep_windspeed_10m"]  = None
        out["dep_windgusts_10m"]  = None
        out["dep_precipitation"]  = 0.0
        out["wx_code_hour"]       = "Unknown"
        if hjson:
            prefix = f"{local_date}T{int(dep_hour_local):02d}"
            p_hr = self._pick_period_hourly(hjson, prefix)
            if p_hr:
                # temperature often °F → convert to °C
                out["dep_temperature_2m"] = self._f_to_c(p_hr.get("temperature"))
                # windSpeed '12 mph' → m/s
                out["dep_windspeed_10m"]  = self._mph_to_ms(p_hr.get("windSpeed"))
                out["dep_windgusts_10m"]  = self._mph_to_ms(p_hr.get("windGust"))
                out["dep_precipitation"]  = 0.0
                out["wx_code_hour"] = self._wx_code_from_text(p_hr.get("shortForecast", ""))

        out["available"] = bool(hjson or djson)
        return out

# ---------------- CLI (optional quick test) ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iata", type=str)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--airports_csv", type=str)
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--dep-hour", type=int, required=True)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    lat = args.lat; lon = args.lon
    if (lat is None or lon is None) and args.iata and args.airports_csv:
        df = pd.read_csv(args.airports_csv)
        lower = {c.lower(): c for c in df.columns}
        iata_col = lower.get("iata"); lat_col = lower.get("latitude"); lon_col = lower.get("longitude")
        row = df[df[iata_col].astype(str).str.upper() == args.iata.upper()].head(1)
        if row.empty:
            print("[ERROR] IATA not found", file=sys.stderr); sys.exit(2)
        lat = float(row.iloc[0][lat_col]); lon = float(row.iloc[0][lon_col])

    if lat is None or lon is None:
        print("[ERROR] provide --lat/--lon or --iata + --airports_csv", file=sys.stderr); sys.exit(2)

    prov = NWSProvider(user_agent=os.getenv("NWS_USER_AGENT", "flight-delay-model/0.1 (no-user-agent-set)"))
    out = prov.get_features(lat, lon, args.date, args.dep_hour)
    print(json.dumps(out, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
