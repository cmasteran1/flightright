# src/flightright/features/flightnumber_history.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flightright.core.time import parse_iso_or_space_datetime
from flightright.integrations.flightlabs.client import FlightLabsClient


@dataclass(frozen=True)
class FlightNumberHistoryFeatures:
    flightnum_od_support_count_last28d: Optional[int]
    flightnum_od_low_support_last28d: Optional[int]
    flightnum_od_depdelay_mean_last7: Optional[float]
    flightnum_od_depdelay_mean_last14: Optional[float]
    flightnum_od_depdelay_mean_last21: Optional[float]
    flightnum_od_depdelay_mean_last28: Optional[float]


def _dep_delay_minutes_from_row(row: Dict[str, Any]) -> Optional[float]:
    """
    Compute departure delay minutes from scheduled vs revised/runway times.
    Prefer runwayTime if present, else revisedTime.
    """
    dep = row.get("departure") or {}
    sched = (dep.get("scheduledTime") or {}).get("local") or (dep.get("scheduledTime") or {}).get("utc")
    if not sched:
        return None

    actual = (dep.get("runwayTime") or {}).get("local") or (dep.get("runwayTime") or {}).get("utc")
    if not actual:
        actual = (dep.get("revisedTime") or {}).get("local") or (dep.get("revisedTime") or {}).get("utc")
    if not actual:
        return None

    try:
        dt_sched = parse_iso_or_space_datetime(sched)
        dt_actual = parse_iso_or_space_datetime(actual)
        return (dt_actual - dt_sched).total_seconds() / 60.0
    except Exception:
        return None


def _filter_by_od(row: Dict[str, Any], origin_iata: str, dest_iata: str) -> bool:
    dep_airport = ((row.get("departure") or {}).get("airport") or {}).get("iata") or ""
    arr_airport = ((row.get("arrival") or {}).get("airport") or {}).get("iata") or ""
    return dep_airport.strip().upper() == origin_iata.strip().upper() and arr_airport.strip().upper() == dest_iata.strip().upper()


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def build_flightnum_od_history_last28(
    *,
    client: FlightLabsClient,
    flight_number_request: str,   # e.g. "AA1821"
    origin_iata: str,
    dest_iata: str,
    anchor_day_local: date,       # "as-of" date in ORIGIN local time (typically flight date or call date)
    low_support_threshold: int = 5,
) -> FlightNumberHistoryFeatures:
    """
    Uses two FlightLabs calls (max 14d each) to assemble up to 28 days of recent history.
    NOTE: This is "most recent available" history; it won't include future days when customer books ahead.
    """
    # last 28 days not including anchor_day_local (you can choose to include; here we exclude same-day to avoid partial day)
    end1 = anchor_day_local - timedelta(days=1)
    start1 = anchor_day_local - timedelta(days=14)
    end2 = anchor_day_local - timedelta(days=15)
    start2 = anchor_day_local - timedelta(days=28)

    j1 = client.flight_data_by_date_number(flight_number_request, start1, end1)
    j2 = client.flight_data_by_date_number(flight_number_request, start2, end2)

    rows = []
    for j in (j1, j2):
        if not j.get("success", False):
            continue
        rows.extend(j.get("data") or [])

    # OD filter
    rows = [r for r in rows if _filter_by_od(r, origin_iata, dest_iata)]

    # compute delay per row
    dated_delays: List[Tuple[date, float]] = []
    for r in rows:
        d = _dep_delay_minutes_from_row(r)
        if d is None:
            continue
        # use scheduled local date for binning
        sched_local = ((r.get("departure") or {}).get("scheduledTime") or {}).get("local")
        if not sched_local:
            continue
        try:
            dt = parse_iso_or_space_datetime(sched_local)
            dated_delays.append((dt.date(), float(d)))
        except Exception:
            continue

    # bin into windows
    def collect(days: int) -> List[float]:
        start = anchor_day_local - timedelta(days=days)
        end = anchor_day_local - timedelta(days=1)
        return [val for (d, val) in dated_delays if start <= d <= end]

    last7 = collect(7)
    last14 = collect(14)
    last21 = collect(21)
    last28 = collect(28)

    support_28 = len(last28)

    return FlightNumberHistoryFeatures(
        flightnum_od_support_count_last28d=support_28,
        flightnum_od_low_support_last28d=(1 if (support_28 < low_support_threshold) else 0),
        flightnum_od_depdelay_mean_last7=_mean(last7),
        flightnum_od_depdelay_mean_last14=_mean(last14),
        flightnum_od_depdelay_mean_last21=_mean(last21),
        flightnum_od_depdelay_mean_last28=_mean(last28),
    )