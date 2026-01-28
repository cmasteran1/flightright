# src/runtime/providers/history/aerodatabox_history.py
from __future__ import annotations
import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

import requests

from .base import HistoryProvider

def _dt(d: str) -> datetime:
    # YYYY-MM-DD (no tz) -> naive date at 00:00 origin-local (we only use date arithmetic)
    return datetime.strptime(d, "%Y-%m-%d")

def _http_headers():
    key = os.getenv("AERODATABOX_API_KEY", "").strip()
    return {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
        "Accept": "application/json",
    }

def _delay_minutes_from_leg(leg: Dict[str, Any]) -> Optional[float]:
    """
    Try to compute arrival delay minutes from a leg payload.
    Order of preference: actual vs scheduled → predicted vs scheduled.
    Returns None if insufficient info.
    """
    arr = (leg or {}).get("arrival") or {}
    sch = (arr.get("scheduledTime") or {}).get("utc") or (arr.get("scheduledTimeUtc"))
    act = (arr.get("actualTime") or {}).get("utc") or (arr.get("actualTimeUtc"))
    pred = (arr.get("predictedTime") or {}).get("utc") or (arr.get("predictedTimeUtc"))

    def _parse_utc(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        # formats like "2026-01-27 20:25Z" or ISO with 'Z'
        ss = s.replace("Z", "+00:00").replace(" ", "T")
        try:
            return datetime.fromisoformat(ss)
        except Exception:
            return None

    t_sch = _parse_utc(sch)
    t_act = _parse_utc(act)
    t_pred = _parse_utc(pred)

    if t_sch and t_act:
        return (t_act - t_sch).total_seconds() / 60.0
    if t_sch and t_pred:
        return (t_pred - t_sch).total_seconds() / 60.0
    return None

@dataclass
class AeroDataBoxHistory(HistoryProvider):
    """
    Uses AeroDataBox endpoints to gather small, targeted recent history.
    API calls (upper bound per prediction):
      - flight-number over last 14 days (<=14 calls)
      - OD daily schedules over last 7 days (<=7 calls)
      - carrier/day count at origin for prior day (1 call)
      - origin-only day summaries (<=7 calls)
      - dest-only day summaries (<=7 calls)
    Tip: You can aggressively cache these responses on disk if needed.
    """
    timeout: int = 15

    # --- core public ---
    def recent_metrics(self, *, airline_iata: str, flight_number: str, origin: str, dest: str,
                       asof_local_date: str) -> Dict[str, Any]:
        headers = _http_headers()
        if not headers["X-RapidAPI-Key"]:
            raise RuntimeError("AERODATABOX_API_KEY is not set")

        asof = _dt(asof_local_date)

        # 1) flightnum 14d delay mean (this exact flight number, any OD that matches origin/dest)
        fn_delays: List[float] = []
        for i in range(1, 15):
            day = (asof - timedelta(days=i)).strftime("%Y-%m-%d")
            leg = self._fetch_flightnum_day(airline_iata, flight_number, day, headers, origin, dest)
            if leg is None:
                continue
            dmin = _delay_minutes_from_leg(leg)
            if dmin is not None:
                fn_delays.append(dmin)

        # 2) OD 7d mean (any carrier)
        od_delays: List[float] = []
        for i in range(1, 8):
            day = (asof - timedelta(days=i)).strftime("%Y-%m-%d")
            legs = self._fetch_od_day(origin, dest, day, headers)
            for lg in legs:
                dmin = _delay_minutes_from_leg(lg)
                if dmin is not None:
                    od_delays.append(dmin)

        # 3) carrier 7d mean (this carrier, any OD)
        carrier_delays: List[float] = []
        for i in range(1, 8):
            day = (asof - timedelta(days=i)).strftime("%Y-%m-%d")
            legs = self._fetch_carrier_day(airline_iata, day, headers)
            for lg in legs:
                dmin = _delay_minutes_from_leg(lg)
                if dmin is not None:
                    carrier_delays.append(dmin)

        # 4) origin 7d mean (any carrier/dest departing origin)
        origin_delays: List[float] = []
        for i in range(1, 8):
            day = (asof - timedelta(days=i)).strftime("%Y-%m-%d")
            legs = self._fetch_airport_day(origin, day, headers, role="departures")
            for lg in legs:
                dmin = _delay_minutes_from_leg(lg)
                if dmin is not None:
                    origin_delays.append(dmin)

        # 5) dest 7d mean (arrivals into dest)
        dest_delays: List[float] = []
        for i in range(1, 8):
            day = (asof - timedelta(days=i)).strftime("%Y-%m-%d")
            legs = self._fetch_airport_day(dest, day, headers, role="arrivals")
            for lg in legs:
                dmin = _delay_minutes_from_leg(lg)
                if dmin is not None:
                    dest_delays.append(dmin)

        # 6) carrier flights prior day (count)
        prior_day = (asof - timedelta(days=1)).strftime("%Y-%m-%d")
        carrier_prior_legs = self._fetch_carrier_day(airline_iata, prior_day, headers)

        # Validate & build outputs
        def _mean_or_err(name: str, arr: List[float]) -> float:
            if not arr:
                raise RuntimeError(f"Missing recent history for {name} (no samples)")
            return float(sum(arr) / len(arr))

        out = {
            "carrier_flights_prior_day": len(carrier_prior_legs),
            "carrier_delay_7d_mean": _mean_or_err("carrier_delay_7d_mean", carrier_delays),
            "od_delay_7d_mean": _mean_or_err("od_delay_7d_mean", od_delays),
            "flightnum_delay_14d_mean": _mean_or_err("flightnum_delay_14d_mean", fn_delays),
            "origin_delay_7d_mean": _mean_or_err("origin_delay_7d_mean", origin_delays),
            "dest_delay_7d_mean": _mean_or_err("dest_delay_7d_mean", dest_delays),
        }
        return out

    # --- private fetch helpers (AeroDataBox) ---

    def _fetch_flightnum_day(self, airline: str, number: str, date: str, headers: Dict[str, str],
                             origin: str, dest: str) -> Optional[Dict[str, Any]]:
        """
        Grab the flight-number list for the day; select the leg matching OD if present.
        Endpoint: /flights/number/{AA}{100}/{YYYY-MM-DD}
        """
        url = f"https://aerodatabox.p.rapidapi.com/flights/number/{airline}{number}/{date}"
        r = requests.get(url, headers=headers, timeout=self.timeout)
        if not r.ok:
            return None
        data = r.json()
        if not isinstance(data, list):
            return None
        # Choose leg matching OD; if multiple, pick first.
        for leg in data:
            dep = ((leg.get("departure") or {}).get("airport") or {}).get("iata", "").upper()
            arr = ((leg.get("arrival") or {}).get("airport") or {}).get("iata", "").upper()
            if dep == origin.upper() and arr == dest.upper():
                return leg
        return None

    def _fetch_od_day(self, origin: str, dest: str, date: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        OD daily legs (any carrier).
        Endpoint: /flights/route/{ORIGIN}/{DEST}/{YYYY-MM-DD}   (if present)
        Fallback: filter airport departures for origin where arrival iata == dest.
        """
        # Try route endpoint (not all plans expose it; fall back to origin departures)
        route_url = f"https://aerodatabox.p.rapidapi.com/flights/route/{origin}/{dest}/{date}"
        r = requests.get(route_url, headers=headers, timeout=self.timeout)
        if r.ok:
            data = r.json()
            return data if isinstance(data, list) else []

        # Fallback via origin departures
        legs = self._fetch_airport_day(origin, date, headers, role="departures")
        return [lg for lg in legs if (((lg.get("arrival") or {}).get("airport") or {}).get("iata", "").upper() == dest.upper())]

    def _fetch_carrier_day(self, airline: str, date: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Carrier-wide legs for a date.
        Endpoint: /flights/airline/{IATA}/{YYYY-MM-DD}
        """
        url = f"https://aerodatabox.p.rapidapi.com/flights/airline/{airline}/{date}"
        r = requests.get(url, headers=headers, timeout=self.timeout)
        if not r.ok:
            return []
        data = r.json()
        return data if isinstance(data, list) else []

    def _fetch_airport_day(self, iata: str, date: str, headers: Dict[str, str], role: str) -> List[Dict[str, Any]]:
        """
        Airport arrivals/departures for a day.
        Endpoints:
          - Departures: /flights/airports/iata/{IATA}/{YYYY-MM-DD}/departures
          - Arrivals:   /flights/airports/iata/{IATA}/{YYYY-MM-DD}/arrivals
        """
        if role == "departures":
            url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{iata}/{date}/departures"
        else:
            url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{iata}/{date}/arrivals"
        r = requests.get(url, headers=headers, timeout=self.timeout)
        if not r.ok:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
