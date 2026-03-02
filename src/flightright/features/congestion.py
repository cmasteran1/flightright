# src/flightright/features/congestion.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from flightright.core.time import AirportTime


@dataclass(frozen=True)
class CongestionFeatures:
    origin_congestion_3h_total: Optional[int]
    origin_airline_congestion_3h_total: Optional[int]


def _parse_time24_to_local_dt(day_local: datetime, hhmm: str, tz_name: str) -> Optional[datetime]:
    """
    Future flights response has:
      sortTime: ISO8601 (UTC-ish)
      departureTime.time24: "06:00"
    We’ll build a local datetime on that date using time24.
    """
    try:
        hh, mm = hhmm.split(":")
        at = AirportTime(tz_name)
        dt = day_local.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0, tzinfo=at.tz)
        return dt
    except Exception:
        return None


def compute_congestion_3h(
    *,
    future_flights_json: Dict[str, Any],
    origin_tz: str,
    target_dep_local_dt: datetime,
    target_airline_iata: str,
) -> CongestionFeatures:
    """
    Counts scheduled departures within +/- 1.5 hours of target departure time.
    Also counts those for the same airline (by carrier.fs in FlightLabs response example).
    """
    data = future_flights_json.get("data") or []
    at = AirportTime(origin_tz)

    t0 = target_dep_local_dt
    window_start = t0 - timedelta(hours=1.5)
    window_end = t0 + timedelta(hours=1.5)

    total = 0
    airline_total = 0

    for row in data:
        dep_time24 = ((row.get("departureTime") or {}).get("time24")) or None
        if not dep_time24:
            continue
        dt = _parse_time24_to_local_dt(t0, dep_time24, origin_tz)
        if not dt:
            continue

        if window_start <= dt <= window_end:
            total += 1
            carrier_fs = ((row.get("carrier") or {}).get("fs")) or ""
            if carrier_fs.strip().upper() == target_airline_iata.strip().upper():
                airline_total += 1

    return CongestionFeatures(
        origin_congestion_3h_total=total,
        origin_airline_congestion_3h_total=airline_total,
    )