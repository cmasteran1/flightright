# src/flightright/pipeline/build_features.py

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, Optional

from flightright.core.airports import load_airports_csv
from flightright.core.time import AirportTime
from flightright.features.congestion import compute_congestion_3h
from flightright.features.flightnumber_history import build_flightnum_od_history_last28
from flightright.features.weather import build_weather_features_open_meteo
from flightright.integrations.flightlabs.client import FlightLabsClient, normalize_flight_number
from flightright.integrations.weather.open_meteo import OpenMeteoClient


def verify_flight_exists_in_future_schedule(
    *,
    flightlabs: FlightLabsClient,
    origin_iata: str,
    flight_date_local: date,
    reporting_airline_iata: str,
    flight_number: str | int,
    dest_iata: str,
) -> Dict[str, Any]:
    """
    Verify the flight exists in the future schedule for that airport+day by matching:
      - carrier.fs == Reporting_Airline (IATA)
      - carrier.flightNumber == flight_number
      - airport.fs == Dest (IATA)
    Returns the matched schedule row (for departure/arrival time extraction).
    """
    j = flightlabs.future_flights(origin_iata, flight_date_local, type_="departure")
    if not j.get("success", False):
        raise RuntimeError(f"Future flights endpoint failed for {origin_iata} {flight_date_local}: {j}")

    target_al = str(reporting_airline_iata).strip().upper()
    target_fn = str(flight_number).strip().lstrip("0")  # be forgiving
    target_dest = str(dest_iata).strip().upper()

    for row in (j.get("data") or []):
        carrier = row.get("carrier") or {}
        airport = row.get("airport") or {}
        if (str(carrier.get("fs") or "").strip().upper() == target_al and
            str(carrier.get("flightNumber") or "").strip().lstrip("0") == target_fn and
            str(airport.get("fs") or "").strip().upper() == target_dest):
            return row

    raise ValueError(
        f"Flight not found in schedule: {target_al}{target_fn} {origin_iata}->{target_dest} on {flight_date_local}"
    )


def build_features_full_weather_plus_flightnum_history(
    *,
    airports_csv_path: str,
    flightlabs_access_key: str,
    origin_iata: str,
    dest_iata: str,
    reporting_airline_iata: str,
    flight_number: str | int,
    flight_date_local: date,
    sched_dep_time24: Optional[str] = None,  # if you already know "HH:MM" local, pass it; else inferred from schedule row
) -> Dict[str, Any]:
    """
    Builds a feature dict that is sufficient for:
      - full model if carrier/origin rolling history exists (later)
      - limited_history_model today (hourly weather + flightnum history)
      - daily_only_weather_model if hourly weather missing
    """
    airports = load_airports_csv(airports_csv_path)
    if origin_iata not in airports:
        raise KeyError(f"Origin {origin_iata} not in airports.csv at {airports_csv_path}")
    origin = airports[origin_iata]

    flightlabs = FlightLabsClient(access_key=flightlabs_access_key)
    om = OpenMeteoClient()

    # 1) verify exists; use schedule row to get dep time if needed
    schedule_row = verify_flight_exists_in_future_schedule(
        flightlabs=flightlabs,
        origin_iata=origin_iata,
        flight_date_local=flight_date_local,
        reporting_airline_iata=reporting_airline_iata,
        flight_number=flight_number,
        dest_iata=dest_iata,
    )

    # departure time
    time24 = sched_dep_time24 or ((schedule_row.get("departureTime") or {}).get("time24"))
    if not time24:
        raise RuntimeError(f"Could not determine departure time24 from schedule row: {schedule_row}")

    hh, mm = time24.split(":")
    at = AirportTime(origin.timezone)
    dep_local_dt = datetime(
        year=flight_date_local.year,
        month=flight_date_local.month,
        day=flight_date_local.day,
        hour=int(hh),
        minute=int(mm),
        second=0,
        microsecond=0,
        tzinfo=at.tz,
    )

    # 2) congestion (from the schedule payload)
    congestion = compute_congestion_3h(
        future_flights_json={"success": True, "data": [schedule_row] + ([])},  # placeholder; caller usually has whole day
        origin_tz=origin.timezone,
        target_dep_local_dt=dep_local_dt,
        target_airline_iata=reporting_airline_iata,
    )
    # NOTE: If you want true congestion, pass the full future_flights JSON instead of only schedule_row.
    # Easiest: change compute_congestion_3h call site to reuse the full `future_flights` response.

    # 3) weather (Open-Meteo)
    w = build_weather_features_open_meteo(
        client=om,
        origin_lat=origin.latitude,
        origin_lon=origin.longitude,
        origin_tz=origin.timezone,
        flight_dep_local_dt=dep_local_dt,
    )

    # 4) flight number history (2 calls, max 14d each :contentReference[oaicite:4]{index=4})
    req_fnum = normalize_flight_number(reporting_airline_iata, flight_number)
    hist = build_flightnum_od_history_last28(
        client=flightlabs,
        flight_number_request=req_fnum,
        origin_iata=origin_iata,
        dest_iata=dest_iata,
        anchor_day_local=flight_date_local,
    )

    # 5) assemble base categorical/numeric that you can test now.
    feats: Dict[str, Any] = {}

    # categoricals you can build now (stubs where upstream provides them)
    feats["Origin"] = origin_iata
    feats["Dest"] = dest_iata
    feats["od_pair"] = f"{origin_iata}_{dest_iata}"
    feats["Reporting_Airline"] = reporting_airline_iata
    feats["sched_dep_hour"] = dep_local_dt.hour

    # These are pipeline-dependent; keep as None until you wire them:
    feats["dep_dow"] = dep_local_dt.weekday()  # 0=Mon
    feats["is_holiday"] = None
    feats["aircraft_type"] = None
    feats["has_recent_arrival_turn_5h"] = None

    # weather categoricals
    feats["origin_daily_weathercode"] = w.origin_daily_weathercode
    feats["origin_dep_hour_weathercode"] = w.origin_dep_hour_weathercode

    # weather numerics
    feats["origin_temp_max_K"] = w.origin_temp_max_K
    feats["origin_temp_min_K"] = w.origin_temp_min_K
    feats["origin_daily_precip_sum_mm"] = w.origin_daily_precip_sum_mm
    feats["origin_daily_windspeed_max_kmh"] = w.origin_daily_windspeed_max_kmh
    feats["origin_dep_temp_K"] = w.origin_dep_temp_K
    feats["origin_dep_precip_mm"] = w.origin_dep_precip_mm
    feats["origin_dep_windspeed_kmh"] = w.origin_dep_windspeed_kmh

    # flightnum history numerics
    feats.update(asdict(hist))

    # congestion numerics (placeholder note above)
    feats["origin_congestion_3h_total"] = congestion.origin_congestion_3h_total
    feats["origin_airline_congestion_3h_total"] = congestion.origin_airline_congestion_3h_total

    # tail/turn propagation features you said you’ll build from graph later
    feats["turn_time_hours"] = None
    feats["tail_leg_num_day"] = None
    feats["flightnum_hours_since_first_departure_today"] = None

    # long-term cached features (not testable yet)
    feats["carrier_depdelay_mean_last7"] = None
    feats["carrier_depdelay_mean_last14"] = None
    feats["carrier_depdelay_mean_last21"] = None
    feats["carrier_depdelay_mean_last28"] = None
    feats["carrier_origin_depdelay_mean_last7"] = None
    feats["carrier_origin_depdelay_mean_last14"] = None
    feats["carrier_origin_depdelay_mean_last21"] = None
    feats["carrier_origin_depdelay_mean_last28"] = None
    feats["origin_depdelay_mean_last7"] = None
    feats["origin_depdelay_mean_last14"] = None
    feats["origin_depdelay_mean_last21"] = None
    feats["origin_depdelay_mean_last28"] = None

    return feats