#!/usr/bin/env python3
"""
src/flightright/cli/predict.py

CLI that:
- verifies a requested flight exists (FlightLabs Future Flights Prediction)
- computes congestion around scheduled departure time (same response)
- fetches last-28d flight-number history (FlightLabs Flight Data by Date in two 14-day calls)
- fetches daily + hourly weather (Open-Meteo, no caching)
- selects which model to use based on feature availability
- loads the chosen model artifact and outputs a prediction + feature payload

Key constraints:
- FlightLabs only builds features; training schema is BTS-derived.
- Do NOT cache weather.
- Do NOT cache flight-number history; pull on demand.
- flightrightdata lives OUTSIDE repo: ../flightrightdata (relative to repo root).
- Optional: load models from S3-compatible storage (iDrive E2), with local caching.

Remote model layout (in bucket):
  s3://<bucket>/<prefix>/<MODEL_DIR>/
    bins_meta.json
    dep_delay_bins_bundle.joblib        (preferred)
    bep_delay_bins_bundle.joblib        (legacy typo support)
    prediction_samples.parquet
    registry.json
    resolved_features.json

MODEL_DIR naming convention:
  {AIRLINE}_{N}_{YEARS}_{WEATHERFLAG}_{HISTFLAG}
    AIRLINE: e.g. AA
    N: inferred from airport rankings lists (e.g. 50)
    YEARS: default "23-25" (override via --model-years)
    WEATHERFLAG: "weather+" if hourly weather available else "weather"
    HISTFLAG: "standard" if airport/carrier cache available else "minhist"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


# ----------------------------
# Paths / configuration helpers
# ----------------------------

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]  # .../src/flightright/cli/predict.py -> repo root


def _default_data_root() -> Path:
    return _repo_root().parent / "flightrightdata"


def _default_airports_csv() -> Path:
    return _default_data_root() / "data" / "meta" / "airports.csv"


def _default_rankings_dir() -> Path:
    return _default_data_root() / "data" / "meta" / "airport_rankings"


def _default_models_dir() -> Path:
    # local models live here if not remote
    return _default_data_root() / "data" / "models"


def _default_remote_cache_dir() -> Path:
    return _default_data_root() / "data" / ".cache" / "remote_models"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    return val if val is not None and val != "" else default


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class AirportInfo:
    iata: str
    lat: float
    lon: float
    tz: str


@dataclass(frozen=True)
class RequestSpec:
    origin: str
    dest: str
    airline_iata: str
    flight_number: str
    flight_date: date
    sched_dep_time_24h: Optional[str] = None


@dataclass
class Availability:
    has_hourly_weather: bool
    has_daily_weather: bool
    has_airport_carrier_history_cache: bool


# ----------------------------
# CSV loading (airports.csv)
# ----------------------------

def load_airports_csv(path: Path) -> Dict[str, AirportInfo]:
    import csv

    if not path.exists():
        raise FileNotFoundError(f"airports.csv not found at: {path}")

    out: Dict[str, AirportInfo] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iata = (row.get("IATA") or "").strip().upper()
            if not iata:
                continue
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                tz = (row.get("Timezone") or "").strip()
            except Exception:
                continue
            out[iata] = AirportInfo(iata=iata, lat=lat, lon=lon, tz=tz)
    return out


# ----------------------------
# Airport rankings parsing (AA_top_50.csv etc)
# ----------------------------

_RANKING_RE = re.compile(r"^(?P<airline>[A-Z0-9]{2})_top_(?P<n>\d+)\.csv$", re.IGNORECASE)

def _parse_one_line_airport_list(text: str) -> List[str]:
    """
    Files are 1-line like:
      "DEN","LAS","MDW",...
    """
    s = text.strip()
    if not s:
        return []
    parts = [p.strip().strip('"').strip("'").upper() for p in s.split(",")]
    return [p for p in parts if p]

def infer_training_airport_N(
    *,
    rankings_dir: Path,
    airline: str,
    origin: str,
    dest: str,
) -> int:
    """
    Choose the *smallest* N such that both origin and dest are in the airline's top_N list.
    """
    airline = airline.strip().upper()
    origin = origin.strip().upper()
    dest = dest.strip().upper()

    if not rankings_dir.exists():
        raise FileNotFoundError(f"airport_rankings directory not found: {rankings_dir}")

    candidates: List[Tuple[int, Path]] = []
    for p in rankings_dir.iterdir():
        if not p.is_file():
            continue
        m = _RANKING_RE.match(p.name)
        if not m:
            continue
        if m.group("airline").upper() != airline:
            continue
        n = int(m.group("n"))
        candidates.append((n, p))

    if not candidates:
        raise FileNotFoundError(f"No ranking files found for airline {airline} under {rankings_dir}")

    candidates.sort(key=lambda x: x[0])  # smallest N first

    for n, path in candidates:
        airports = _parse_one_line_airport_list(path.read_text(encoding="utf-8"))
        aset = set(airports)
        if origin in aset and dest in aset:
            return n

    # If none contain both, fail loudly
    raise FileNotFoundError(
        f"Origin/dest not both present in any {airline}_top_N.csv under {rankings_dir}. "
        f"(origin={origin}, dest={dest})"
    )


# ----------------------------
# FlightLabs clients
# ----------------------------

class FlightLabsClient:
    def __init__(self, access_key: str, timeout_s: float = 30.0) -> None:
        self.access_key = access_key
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def future_flights_departures(self, airport_iata: str, day: date) -> Dict[str, Any]:
        url = "https://app.goflightlabs.com/advanced-future-flights"
        params = {
            "access_key": self.access_key,
            "type": "departure",
            "iataCode": airport_iata,
            "date": day.isoformat(),
        }
        r = self.session.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def flight_data_by_date_number(self, flight_number_full: str, date_from: date, date_to: date) -> Dict[str, Any]:
        url = "https://app.goflightlabs.com/v2/flight"
        params = {
            "access_key": self.access_key,
            "search_by": "number",
            "flight_number": flight_number_full,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }
        r = self.session.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


# ----------------------------
# Future flights matching + congestion
# ----------------------------

def _parse_hhmm(s: str) -> Tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM: {s!r}")
    return int(parts[0]), int(parts[1])


def _match_future_flight(future_payload: Dict[str, Any], req: RequestSpec) -> Tuple[Dict[str, Any], datetime]:
    if not future_payload.get("success", False):
        raise RuntimeError(f"Future flights call failed: {future_payload}")

    rows = future_payload.get("data") or []
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"No future flights returned for {req.origin} on {req.flight_date}")

    want_num = req.flight_number.strip()
    want_dest = req.dest.strip().upper()

    candidates: List[Tuple[Dict[str, Any], datetime]] = []
    for row in rows:
        try:
            carrier = row.get("carrier") or {}
            num = str(carrier.get("flightNumber") or "").strip()
            dest_code = str((row.get("airport") or {}).get("fs") or "").strip().upper()
            if num != want_num:
                continue
            if dest_code != want_dest:
                continue

            sort_time = str(row.get("sortTime") or "").strip()
            if not sort_time:
                continue
            sched_utc = datetime.fromisoformat(sort_time.replace("Z", "+00:00")).astimezone(timezone.utc)

            if req.sched_dep_time_24h:
                hh, mm = _parse_hhmm(req.sched_dep_time_24h)
                if not (sched_utc.hour == hh and sched_utc.minute == mm):
                    continue

            candidates.append((row, sched_utc))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"Could not verify flight {req.airline_iata}{req.flight_number} {req.origin}->{req.dest} on {req.flight_date}. "
            f"(No match in future flights schedule for that airport-day.)"
        )

    candidates.sort(key=lambda x: x[1])
    return candidates[0][0], candidates[0][1]


def compute_congestion_3h(
    future_payload: Dict[str, Any],
    target_sched_dep_utc: datetime,
    airline_iata: str,
    flight_number: str,
) -> Tuple[int, int]:
    rows = future_payload.get("data") or []
    if not isinstance(rows, list):
        return 0, 0

    lo = target_sched_dep_utc - timedelta(hours=3)
    hi = target_sched_dep_utc + timedelta(hours=3)

    total = 0
    airline_total = 0

    want_airline = airline_iata.strip().upper()
    want_num = flight_number.strip()

    for row in rows:
        try:
            sort_time = str(row.get("sortTime") or "").strip()
            if not sort_time:
                continue
            sched_utc = datetime.fromisoformat(sort_time.replace("Z", "+00:00")).astimezone(timezone.utc)
            if sched_utc < lo or sched_utc > hi:
                continue

            total += 1
            carrier = row.get("carrier") or {}
            fs = str(carrier.get("fs") or "").strip().upper()
            num = str(carrier.get("flightNumber") or "").strip()
            if fs == want_airline and not (num == want_num and sched_utc == target_sched_dep_utc):
                airline_total += 1
        except Exception:
            continue

    return total, airline_total


# ----------------------------
# Flight-number rolling history (28d in two 14d calls)
# ----------------------------

def _parse_flightlabs_utc(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z") and " " in s:
        s2 = s.replace(" ", "T").replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s2)
        except Exception:
            return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _dep_delay_minutes(flight_row: Dict[str, Any]) -> Optional[float]:
    dep = flight_row.get("departure") or {}
    sched = _parse_flightlabs_utc(((dep.get("scheduledTime") or {}).get("utc")))
    if sched is None:
        return None

    runway = _parse_flightlabs_utc(((dep.get("runwayTime") or {}).get("utc")))
    revised = _parse_flightlabs_utc(((dep.get("revisedTime") or {}).get("utc")))

    actual = runway or revised
    if actual is None:
        return None

    return (actual - sched).total_seconds() / 60.0


def compute_flightnum_od_rolling_stats_last28d(
    client: FlightLabsClient,
    airline_iata: str,
    flight_number: str,
    origin: str,
    dest: str,
    as_of_local_date: date,
) -> Dict[str, Any]:
    full_num = f"{airline_iata.strip().upper()}{flight_number.strip()}"

    end = as_of_local_date
    start_14 = end - timedelta(days=13)
    start_28 = end - timedelta(days=27)
    mid = start_14 - timedelta(days=1)

    p1 = client.flight_data_by_date_number(full_num, start_14, end)
    p2 = client.flight_data_by_date_number(full_num, start_28, mid)

    rows: List[Dict[str, Any]] = []
    for p in (p1, p2):
        if p.get("success", False) and isinstance(p.get("data"), list):
            rows.extend(p["data"])

    o = origin.strip().upper()
    d = dest.strip().upper()

    samples: List[Tuple[date, float]] = []
    for r in rows:
        try:
            dep_air = ((r.get("departure") or {}).get("airport") or {})
            arr_air = ((r.get("arrival") or {}).get("airport") or {})
            if str(dep_air.get("iata") or "").strip().upper() != o:
                continue
            if str(arr_air.get("iata") or "").strip().upper() != d:
                continue

            dep_delay = _dep_delay_minutes(r)
            if dep_delay is None:
                continue

            sched_utc = _parse_flightlabs_utc((((r.get("departure") or {}).get("scheduledTime") or {}).get("utc")))
            if sched_utc is None:
                continue

            samples.append((sched_utc.date(), float(dep_delay)))
        except Exception:
            continue

    def mean_last_n(n: int) -> Optional[float]:
        lo = end - timedelta(days=n - 1)
        vals = [v for (dt, v) in samples if lo <= dt <= end]
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    support_28 = len([1 for (dt, _) in samples if start_28 <= dt <= end])
    low_support = 1 if support_28 < 5 else 0

    return {
        "flightnum_od_support_count_last28d": support_28,
        "flightnum_od_low_support_last28d": low_support,
        "flightnum_od_depdelay_mean_last7": mean_last_n(7),
        "flightnum_od_depdelay_mean_last14": mean_last_n(14),
        "flightnum_od_depdelay_mean_last21": mean_last_n(21),
        "flightnum_od_depdelay_mean_last28": mean_last_n(28),
    }


# ----------------------------
# Weather (Open-Meteo, no caching)
# ----------------------------

def _pick0(x: Any) -> Any:
    return x[0] if isinstance(x, list) and x else None

def _picki(x: Any, i: int) -> Any:
    return x[i] if isinstance(x, list) and 0 <= i < len(x) else None

def _c_to_k(c: Any) -> Optional[float]:
    if c is None:
        return None
    try:
        return float(c) + 273.15
    except Exception:
        return None

def openmeteo_daily(lat: float, lon: float, day: date) -> Optional[Dict[str, Any]]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
        "timezone": "UTC",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily") or {}
        if not daily:
            return None
        return {
            "origin_temp_max_K": _c_to_k(_pick0(daily.get("temperature_2m_max"))),
            "origin_temp_min_K": _c_to_k(_pick0(daily.get("temperature_2m_min"))),
            "origin_daily_precip_sum_mm": _pick0(daily.get("precipitation_sum")),
            "origin_daily_windspeed_max_kmh": _pick0(daily.get("windspeed_10m_max")),
            "origin_daily_weathercode": _pick0(daily.get("weathercode")),
        }
    except Exception:
        return None

def openmeteo_hourly_near_departure(lat: float, lon: float, dep_utc: datetime) -> Optional[Dict[str, Any]]:
    """
    FIX: compare parsed datetimes instead of string munging.
    We request timezone=UTC, so treat returned times as UTC.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    day = dep_utc.date()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
        "timezone": "UTC",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        hourly = j.get("hourly") or {}
        times = hourly.get("time") or []
        if not isinstance(times, list) or not times:
            return None

        target_dt = dep_utc.replace(minute=0, second=0, microsecond=0)

        idx: Optional[int] = None
        for i, t in enumerate(times):
            ts = str(t).strip()
            try:
                tdt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            except Exception:
                try:
                    tdt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    continue
            if tdt == target_dt:
                idx = i
                break

        if idx is None:
            return None

        return {
            "origin_dep_temp_K": _c_to_k(_picki(hourly.get("temperature_2m"), idx)),
            "origin_dep_precip_mm": _picki(hourly.get("precipitation"), idx),
            "origin_dep_windspeed_kmh": _picki(hourly.get("windspeed_10m"), idx),
            "origin_dep_hour_weathercode": _picki(hourly.get("weathercode"), idx),
        }
    except Exception:
        return None


# ----------------------------
# Model selection
# ----------------------------

def assess_availability(
    daily_weather: Optional[Dict[str, Any]],
    hourly_weather: Optional[Dict[str, Any]],
    airport_carrier_history_cache: Optional[Dict[str, Any]],
) -> Availability:
    return Availability(
        has_daily_weather=bool(daily_weather),
        has_hourly_weather=bool(hourly_weather),
        has_airport_carrier_history_cache=bool(airport_carrier_history_cache),
    )

def choose_model(av: Availability) -> str:
    if av.has_hourly_weather and av.has_airport_carrier_history_cache:
        return "full"
    if av.has_hourly_weather and not av.has_airport_carrier_history_cache:
        return "limited_history"
    if (not av.has_hourly_weather) and av.has_daily_weather and av.has_airport_carrier_history_cache:
        return "daily_weather"
    raise RuntimeError(
        "Could not choose a model with available features. "
        f"(hourly_weather={av.has_hourly_weather}, daily_weather={av.has_daily_weather}, "
        f"airport_carrier_history_cache={av.has_airport_carrier_history_cache})"
    )

def load_model_artifact(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    suffix = model_path.suffix.lower()
    if suffix in (".joblib", ".jl"):
        import joblib  # type: ignore
        return joblib.load(model_path)
    if suffix in (".pkl", ".pickle"):
        import pickle
        with model_path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported model file type: {model_path.suffix}")


# ----------------------------
# Remote model store (S3 / iDrive E2)
# ----------------------------

@dataclass(frozen=True)
class RemoteModelSpec:
    airline: str
    n_airports: int
    years: str
    weatherflag: str  # "weather" or "weather+"
    histflag: str     # "standard" or "minhist"

    @property
    def dir_name(self) -> str:
        return f"{self.airline}_{self.n_airports}_{self.years}_{self.weatherflag}_{self.histflag}"

class S3ModelStore:
    def __init__(self, *, bucket: str, prefix: str, cache_dir: Path, endpoint_url: Optional[str]) -> None:
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else (prefix + "/")
        self.cache_dir = cache_dir
        self.endpoint_url = endpoint_url

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            import boto3  # type: ignore
        except Exception as e:
            raise RuntimeError("boto3 is required for --use-remote-models. Install with: pip install boto3") from e

        self._boto3 = boto3

    def _client(self):
        # Uses AWS_PROFILE and standard AWS config/credentials resolution.
        session = self._boto3.Session()
        return session.client("s3", endpoint_url=self.endpoint_url)

    def _list_common_prefixes(self, *, s3_prefix: str) -> List[str]:
        """
        List immediate 'directories' under s3_prefix using delimiter='/'
        Returns prefixes like: models/AA_50_23-25_weather+_minhist/
        """
        cli = self._client()
        out: List[str] = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": s3_prefix,
                "Delimiter": "/",
                "MaxKeys": 1000,
            }
            if token:
                kwargs["ContinuationToken"] = token
            resp = cli.list_objects_v2(**kwargs)
            for cp in resp.get("CommonPrefixes", []) or []:
                p = cp.get("Prefix")
                if p:
                    out.append(str(p))
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return out

    def find_remote_model_dir_prefix(self, spec: RemoteModelSpec) -> str:
        want = self.prefix + spec.dir_name + "/"
        prefixes = self._list_common_prefixes(s3_prefix=self.prefix)
        if want in prefixes:
            return want
        raise FileNotFoundError(
            f"No matching remote model directory under s3://{self.bucket}/{self.prefix} for "
            f"(airline={spec.airline}, N={spec.n_airports}, weatherflag={spec.weatherflag}, histflag={spec.histflag}). "
            f"Expected prefix like: {want}"
        )

    def _download_key_if_needed(self, *, key: str, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size > 0:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        cli = self._client()
        cli.download_file(self.bucket, key, str(dest))

    def download_deploy_joblib(self, remote_dir_prefix: str) -> Path:
        """
        Download the deployable joblib artifact from the remote directory into local cache.
        Supports two possible filenames:
          - dep_delay_bins_bundle.joblib (preferred)
          - bep_delay_bins_bundle.joblib (legacy typo)
        """
        candidates = [
            remote_dir_prefix + "dep_delay_bins_bundle.joblib",
            remote_dir_prefix + "bep_delay_bins_bundle.joblib",
        ]

        # Try HEAD-like check via list_objects_v2 for each exact key
        cli = self._client()

        def key_exists(k: str) -> bool:
            resp = cli.list_objects_v2(Bucket=self.bucket, Prefix=k, MaxKeys=1)
            for obj in resp.get("Contents", []) or []:
                if obj.get("Key") == k:
                    return True
            return False

        chosen: Optional[str] = None
        for k in candidates:
            if key_exists(k):
                chosen = k
                break
        if not chosen:
            raise FileNotFoundError(
                "Could not find deployable joblib in remote dir. Tried: "
                + ", ".join([f"s3://{self.bucket}/{k}" for k in candidates])
            )

        local_path = self.cache_dir / chosen
        self._download_key_if_needed(key=chosen, dest=local_path)
        return local_path


# ----------------------------
# Feature assembly (BTS schema)
# ----------------------------

def build_feature_payload(
    req: RequestSpec,
    sched_dep_utc: datetime,
    congestion_total_3h: int,
    congestion_airline_3h: int,
    daily_weather: Optional[Dict[str, Any]],
    hourly_weather: Optional[Dict[str, Any]],
    flightnum_history: Dict[str, Any],
    airport_carrier_history_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    od_pair = f"{req.origin.strip().upper()}_{req.dest.strip().upper()}"
    dep_dow = int(sched_dep_utc.weekday())
    sched_dep_hour = int(sched_dep_utc.hour)

    feats: Dict[str, Any] = {
        "Origin": req.origin.strip().upper(),
        "Dest": req.dest.strip().upper(),
        "od_pair": od_pair,
        "Reporting_Airline": req.airline_iata.strip().upper(),
        "dep_dow": dep_dow,
        "sched_dep_hour": sched_dep_hour,
        "is_holiday": 0,
        "aircraft_type": "UNKNOWN",
        "has_recent_arrival_turn_5h": 0,
        "origin_congestion_3h_total": congestion_total_3h,
        "origin_airline_congestion_3h_total": congestion_airline_3h,
        "tail_leg_num_day": 0,
        "flightnum_hours_since_first_departure_today": 0.0,
        "turn_time_hours": 0.0,
    }

    if daily_weather:
        feats.update(daily_weather)
    if hourly_weather:
        feats.update(hourly_weather)

    feats.update(flightnum_history)

    if airport_carrier_history_cache:
        feats.update(airport_carrier_history_cache)

    return feats


# ----------------------------
# Prediction wrapper
# ----------------------------
def model_predict(model: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prediction wrapper that supports:
      A) sklearn-like estimators/pipelines: predict_proba / predict
      B) deployable bins wrapper object: predict_bin_proba / predict_expected_delay_minutes / etc.
      C) dict bundle (joblib) that contains calibrators/thresholds and optional metadata.

    Critical: CatBoost enforces categorical-vs-numeric feature types at inference.
    We must coerce categorical features to string/object to match training.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas and numpy are required to run predictions.") from e

    # -------------------------
    # Helpers (self-contained)
    # -------------------------
    def _as_str_series(s: "pd.Series") -> "pd.Series":
        s = s.astype("object")
        s = s.where(~pd.isna(s), "Unknown")
        s = s.map(lambda v: "Unknown" if str(v).strip().lower() in ("", "nan", "none") else str(v))
        return s.astype("object")

    def _coerce_frame(
        X: "pd.DataFrame",
        *,
        cat_cols: List[str],
        num_cols: List[str],
    ) -> "pd.DataFrame":
        # Ensure all expected columns exist
        for c in cat_cols + num_cols:
            if c not in X.columns:
                X[c] = np.nan

        # Coerce categoricals to string/object (this is the key fix)
        for c in cat_cols:
            X[c] = _as_str_series(X[c])

        # Coerce numerics to float
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().any():
                # single-row: just fill missing with 0.0
                X[c] = X[c].fillna(0.0)

        # Order columns deterministically (CatBoost cares about feature order)
        return X[cat_cols + num_cols].copy()

    def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
        p = p_ge.copy()
        for j in range(1, p.shape[1]):
            p[:, j] = np.minimum(p[:, j], p[:, j - 1])
        return p

    def ge_to_bins(p_ge15, p_ge30, p_ge45, p_ge60) -> np.ndarray:
        p_ge = enforce_monotone_ge_probs(np.vstack([p_ge15, p_ge30, p_ge45, p_ge60]).T)
        p15, p30, p45, p60 = p_ge[:, 0], p_ge[:, 1], p_ge[:, 2], p_ge[:, 3]

        p_lt15 = 1.0 - p15
        p_15_30 = np.maximum(0.0, p15 - p30)
        p_30_45 = np.maximum(0.0, p30 - p45)
        p_45_60 = np.maximum(0.0, p45 - p60)
        p_ge60 = np.maximum(0.0, p60)

        P = np.vstack([p_lt15, p_15_30, p_30_45, p_45_60, p_ge60]).T
        Z = P.sum(axis=1, keepdims=True)
        Z[Z == 0] = 1.0
        return (P / Z).astype(float)

    def _as_float_list(a: Any) -> List[float]:
        arr = np.asarray(a, dtype=float).reshape(-1)
        return [float(x) for x in arr.tolist()]

    # Default schema fallback (matches your project lists)
    DEFAULT_CATS = [
        "Origin",
        "Dest",
        "od_pair",
        "Reporting_Airline",
        "dep_dow",
        "sched_dep_hour",
        "is_holiday",
        "aircraft_type",
        "origin_daily_weathercode",
        "origin_dep_hour_weathercode",
        "has_recent_arrival_turn_5h",
    ]
    DEFAULT_NUMS = [
        "origin_temp_max_K",
        "origin_temp_min_K",
        "origin_daily_precip_sum_mm",
        "origin_daily_windspeed_max_kmh",
        "origin_dep_temp_K",
        "origin_dep_precip_mm",
        "origin_dep_windspeed_kmh",
        "flightnum_od_support_count_last28d",
        "flightnum_od_low_support_last28d",
        "flightnum_od_depdelay_mean_last7",
        "flightnum_od_depdelay_mean_last14",
        "flightnum_od_depdelay_mean_last21",
        "flightnum_od_depdelay_mean_last28",
        "carrier_depdelay_mean_last7",
        "carrier_depdelay_mean_last14",
        "carrier_depdelay_mean_last21",
        "carrier_depdelay_mean_last28",
        "carrier_origin_depdelay_mean_last7",
        "carrier_origin_depdelay_mean_last14",
        "carrier_origin_depdelay_mean_last21",
        "carrier_origin_depdelay_mean_last28",
        "origin_depdelay_mean_last7",
        "origin_depdelay_mean_last14",
        "origin_depdelay_mean_last21",
        "origin_depdelay_mean_last28",
        "turn_time_hours",
        "origin_congestion_3h_total",
        "origin_airline_congestion_3h_total",
        "tail_leg_num_day",
        "flightnum_hours_since_first_departure_today",
    ]

    # Build initial frame
    X_raw = pd.DataFrame([features])

    # ------------------------------------------------------------
    # C) Dict bundle artifact (your current case)
    # ------------------------------------------------------------
    if isinstance(model, dict):
        bundle = model

        # Pull schema if the bundle includes it; otherwise fallback.
        cat_cols = None
        num_cols = None
        if isinstance(bundle.get("resolved_features"), dict):
            rf = bundle["resolved_features"]
            if isinstance(rf.get("categorical"), list) and isinstance(rf.get("numeric"), list):
                cat_cols = list(rf["categorical"])
                num_cols = list(rf["numeric"])
        if cat_cols is None and isinstance(bundle.get("categorical_features"), list):
            cat_cols = list(bundle["categorical_features"])
        if num_cols is None and isinstance(bundle.get("numeric_features"), list):
            num_cols = list(bundle["numeric_features"])

        cat_cols = cat_cols or DEFAULT_CATS
        num_cols = num_cols or DEFAULT_NUMS

        X = _coerce_frame(X_raw.copy(), cat_cols=cat_cols, num_cols=num_cols)

        # Case C1: dict contains an actual wrapper/estimator under common keys
        for k in ("model", "artifact", "wrapper", "estimator"):
            if k in bundle:
                inner = bundle[k]
                return model_predict(inner, features)

        # Case C2: dict contains calibrators
        calibs = None
        for k in ("calibrators", "calibs", "models", "per_threshold_models"):
            if k in bundle and isinstance(bundle[k], dict):
                calibs = bundle[k]
                break

        thresholds = bundle.get("thresholds", None)
        if thresholds is None and calibs is not None:
            try:
                thresholds = sorted(int(x) for x in calibs.keys())
            except Exception:
                thresholds = None

        if calibs is None or thresholds is None:
            raise RuntimeError(
                "Loaded model is a dict, but I couldn't find calibrators/thresholds inside it. "
                f"Top-level keys={list(bundle.keys())}"
            )

        thr_list = [int(t) for t in list(thresholds)]
        if len(thr_list) != 4:
            raise RuntimeError(
                f"Bundle has calibrators but unexpected thresholds={thr_list}. "
                "Expected exactly 4 (e.g. [15,30,45,60])."
            )

        p_ge: Dict[int, float] = {}
        for t in thr_list:
            cal = calibs.get(t) if t in calibs else calibs.get(str(t))
            if cal is None:
                raise RuntimeError(f"Bundle missing calibrator for threshold {t}. Keys={list(calibs.keys())[:10]}")
            if not hasattr(cal, "predict_proba"):
                raise RuntimeError(f"Calibrator for thr={t} has no predict_proba(). type={type(cal)!r}")
            p = cal.predict_proba(X)[:, 1].astype(float)
            p_ge[int(t)] = float(p[0])

        t0, t1, t2, t3 = thr_list
        P = ge_to_bins(
            np.array([p_ge[t0]]),
            np.array([p_ge[t1]]),
            np.array([p_ge[t2]]),
            np.array([p_ge[t3]]),
        )

        bin_labels = bundle.get(
            "bin_labels",
            ["< 15 min", "15–30 min", "30–45 min", "45–60 min", "≥ 60 min"],
        )
        if not isinstance(bin_labels, list) or len(bin_labels) != 5:
            bin_labels = ["< 15 min", "15–30 min", "30–45 min", "45–60 min", "≥ 60 min"]

        w_min = bundle.get("bin_weights_minutes", [7.5, 22.5, 37.5, 52.5, 75.0])
        w_sev = bundle.get("severity_weights", [0.0, 1.0, 2.0, 3.0, 4.0])
        try:
            w_min_arr = np.asarray(w_min, dtype=float)
            w_sev_arr = np.asarray(w_sev, dtype=float)
            exp_min = float((P * w_min_arr[None, :]).sum(axis=1)[0])
            sev = float((P * w_sev_arr[None, :]).sum(axis=1)[0])
        except Exception:
            exp_min = None
            sev = None

        pred_idx = int(np.argmax(P, axis=1)[0])
        out: Dict[str, Any] = {
            "api": "dep_delay_bins_bundle_dict",
            "thresholds": thr_list,
            "p_ge": {str(k): float(v) for k, v in p_ge.items()},
            "bin_labels": bin_labels,
            "bin_proba": _as_float_list(P[0]),
            "predicted_bin": str(bin_labels[pred_idx]),
        }
        if exp_min is not None:
            out["expected_delay_minutes"] = exp_min
        if sev is not None:
            out["severity_score"] = sev
        return out

    # ------------------------------------------------------------
    # B) Deployable bins wrapper object (Pattern B)
    # ------------------------------------------------------------
    if hasattr(model, "predict_bin_proba") and callable(getattr(model, "predict_bin_proba")):
        # Try to get schema off the wrapper; otherwise fallback.
        cat_cols = getattr(model, "categorical_features", None) or DEFAULT_CATS
        num_cols = getattr(model, "numeric_features", None) or DEFAULT_NUMS
        X = _coerce_frame(X_raw.copy(), cat_cols=list(cat_cols), num_cols=list(num_cols))

        out: Dict[str, Any] = {"api": "dep_delay_bins_wrapper"}

        P = model.predict_bin_proba(X)
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != 1:
            raise RuntimeError(f"predict_bin_proba returned unexpected shape: {P.shape}")

        bin_labels = getattr(model, "bin_labels", None)
        if isinstance(bin_labels, list) and len(bin_labels) == P.shape[1]:
            out["bin_labels"] = bin_labels
        out["bin_proba"] = _as_float_list(P[0])

        if hasattr(model, "predict_bin_label") and callable(getattr(model, "predict_bin_label")):
            try:
                lbls = model.predict_bin_label(X)
                if isinstance(lbls, list) and len(lbls) == 1:
                    out["predicted_bin"] = str(lbls[0])
            except Exception:
                pass

        if hasattr(model, "predict_expected_delay_minutes") and callable(getattr(model, "predict_expected_delay_minutes")):
            try:
                edm = np.asarray(model.predict_expected_delay_minutes(X), dtype=float)
                if edm.shape[0] == 1:
                    out["expected_delay_minutes"] = float(edm[0])
            except Exception:
                pass

        if hasattr(model, "predict_severity_score") and callable(getattr(model, "predict_severity_score")):
            try:
                sev = np.asarray(model.predict_severity_score(X), dtype=float)
                if sev.shape[0] == 1:
                    out["severity_score"] = float(sev[0])
            except Exception:
                pass

        if hasattr(model, "predict_ge_proba") and callable(getattr(model, "predict_ge_proba")):
            try:
                pge = model.predict_ge_proba(X)
                if isinstance(pge, dict):
                    out["p_ge"] = {str(k): float(np.asarray(v, dtype=float)[0]) for k, v in pge.items()}
            except Exception:
                pass

        return out

    # ------------------------------------------------------------
    # A) sklearn-like estimators/pipelines
    # ------------------------------------------------------------
    # NOTE: if this is also CatBoost under the hood, coercing categoricals helps too.
    X = _coerce_frame(X_raw.copy(), cat_cols=DEFAULT_CATS, num_cols=DEFAULT_NUMS)

    out2: Dict[str, Any] = {"api": "sklearn_like"}

    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        proba = model.predict_proba(X)
        try:
            out2["proba"] = float(proba[0][1])
        except Exception:
            try:
                out2["proba"] = float(proba[0])
            except Exception:
                out2["proba"] = None

    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        pred = model.predict(X)
        try:
            out2["prediction"] = float(pred[0])
        except Exception:
            out2["prediction"] = pred[0] if pred is not None else None

    if out2.get("proba") is None and "prediction" not in out2:
        attrs = [a for a in ("predict", "predict_proba", "predict_bin_proba", "__call__") if hasattr(model, a)]
        raise RuntimeError(
            "Loaded model does not implement a recognized prediction API. "
            f"type={type(model)!r} attrs={attrs}"
        )

    return out2
# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate features via FlightLabs/Open-Meteo and run the best-available model.")

    p.add_argument("--origin", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--airline", required=True)
    p.add_argument("--flightnum", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--sched-dep", default=None)

    p.add_argument(
        "--airports-n",
        type=int,
        default=50,
        help="Training airport list size N (e.g., 50). Use this if you only have top_50 lists and don't want inference.",
    )
    p.add_argument(
        "--require-airports-in-ranking",
        action="store_true",
        help="If set, enforce that origin and dest are in the AIRLINE_top_N.csv list.",
    )

    p.add_argument("--airports-csv", default=str(_default_airports_csv()))
    p.add_argument("--airport-rankings-dir", default=str(_default_rankings_dir()))
    p.add_argument("--model-years", default="23-25", help="Directory tag for training years, e.g. 23-25")

    # Local model options
    p.add_argument("--models-dir", default=str(_default_models_dir()))
    p.add_argument("--model", default=None, choices=["full", "limited_history", "daily_weather"])
    p.add_argument("--model-path", default=None)

    # Remote model options
    p.add_argument("--use-remote-models", action="store_true", help="Load deployable model joblib from S3-compatible storage.")
    p.add_argument("--s3-bucket", default=None, help="S3 bucket name (e.g., flightright-models)")
    p.add_argument("--s3-prefix", default="models/", help="Prefix under the bucket (e.g., models/)")
    p.add_argument("--remote-cache-dir", default=str(_default_remote_cache_dir()), help="Local cache dir for downloaded models")
    p.add_argument("--s3-endpoint", default=None, help="Override endpoint URL (else uses env E2_ENDPOINT)")

    p.add_argument("--print-features", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()

    access_key = _env("FLIGHTLABS_ACCESS_KEY") or _env("GOFLIGHTLABS_ACCESS_KEY")
    if not access_key:
        print("ERROR: Missing FlightLabs access key. Set FLIGHTLABS_ACCESS_KEY in your environment/.env.", file=sys.stderr)
        return 2

    flight_date = date.fromisoformat(args.date)
    req = RequestSpec(
        origin=args.origin,
        dest=args.dest,
        airline_iata=args.airline,
        flight_number=args.flightnum,
        flight_date=flight_date,
        sched_dep_time_24h=args.sched_dep,
    )

    airports = load_airports_csv(Path(args.airports_csv))
    if req.origin.upper() not in airports:
        print(f"ERROR: Origin airport {req.origin} not found in airports.csv", file=sys.stderr)
        return 2
    origin_info = airports[req.origin.upper()]

    fl = FlightLabsClient(access_key=access_key)

    # 1) Verify flight exists + scheduled departure
    future = fl.future_flights_departures(req.origin.upper(), req.flight_date)
    _flight_row, sched_dep_utc = _match_future_flight(future, req)

    # 2) Congestion
    cong_total, cong_airline = compute_congestion_3h(
        future_payload=future,
        target_sched_dep_utc=sched_dep_utc,
        airline_iata=req.airline_iata,
        flight_number=req.flight_number,
    )

    # 3) Weather
    daily_w = openmeteo_daily(origin_info.lat, origin_info.lon, req.flight_date)
    hourly_w = openmeteo_hourly_near_departure(origin_info.lat, origin_info.lon, sched_dep_utc)

    # 4) Flight-number history
    flight_hist = compute_flightnum_od_rolling_stats_last28d(
        client=fl,
        airline_iata=req.airline_iata,
        flight_number=req.flight_number,
        origin=req.origin,
        dest=req.dest,
        as_of_local_date=req.flight_date,
    )

    # 5) Airport/carrier cache not built yet
    airport_carrier_cache = None

    av = assess_availability(daily_w, hourly_w, airport_carrier_cache)

    # Determine flags that affect model directory naming
    weatherflag = "weather+" if av.has_hourly_weather else "weather"
    histflag = "standard" if av.has_airport_carrier_history_cache else "minhist"

    rankings_dir = Path(args.airport_rankings_dir)
    n_airports = int(args.airports_n)

    if args.require_airports_in_ranking:
        ranking_file = rankings_dir / f"{req.airline_iata.upper()}_top_{n_airports}.csv"
        if not ranking_file.exists():
            raise FileNotFoundError(f"Ranking file not found: {ranking_file}")

        airports_list = _parse_one_line_airport_list(ranking_file.read_text(encoding="utf-8"))
        aset = set(airports_list)
        o = req.origin.upper()
        d = req.dest.upper()
        if o not in aset or d not in aset:
            raise FileNotFoundError(
                f"Origin/dest not both present in ranking list {ranking_file.name}. "
                f"(origin={o} in_list={o in aset}, dest={d} in_list={d in aset})"
            )

    # Choose which model "family" (this is separate from your dir naming)
    model_family = args.model if args.model else choose_model(av)

    # Load model (remote or local)
    if args.use_remote_models:
        if not args.s3_bucket:
            raise SystemExit("ERROR: --use-remote-models requires --s3-bucket")

        endpoint = args.s3_endpoint or _env("E2_ENDPOINT")
        store = S3ModelStore(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            cache_dir=Path(args.remote_cache_dir),
            endpoint_url=endpoint,
        )
        spec = RemoteModelSpec(
            airline=req.airline_iata.upper(),
            n_airports=int(n_airports),
            years=str(args.model_years),
            weatherflag=weatherflag,
            histflag=histflag,
        )
        remote_dir_prefix = store.find_remote_model_dir_prefix(spec)
        model_path = store.download_deploy_joblib(remote_dir_prefix)
        model = load_model_artifact(model_path)
        model_path_str = f"s3://{args.s3_bucket}/{remote_dir_prefix} (cached:{model_path})"
    else:
        # Local: keep your old convention unless you specify --model-path.
        # If you want local to also follow the directory naming convention, we can switch it later.
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            # fallback: old simple paths
            models_dir = Path(args.models_dir)
            mapping = {
                "full": models_dir / "model_full.joblib",
                "limited_history": models_dir / "model_limited_history.joblib",
                "daily_weather": models_dir / "model_daily_weather.joblib",
            }
            model_path = mapping[model_family]
        model = load_model_artifact(model_path)
        model_path_str = str(model_path)

    # Assemble feature payload
    features = build_feature_payload(
        req=req,
        sched_dep_utc=sched_dep_utc,
        congestion_total_3h=cong_total,
        congestion_airline_3h=cong_airline,
        daily_weather=daily_w,
        hourly_weather=hourly_w,
        flightnum_history=flight_hist,
        airport_carrier_history_cache=airport_carrier_cache,
    )

    # Predict
    pred = model_predict(model, features)

    output: Dict[str, Any] = {
        "ok": True,
        "chosen_model_family": model_family,
        "model_locator": model_path_str,
        "resolved_model_dir_spec": {
            "airline": req.airline_iata.upper(),
            "n_airports": int(n_airports),
            "years": str(args.model_years),
            "weatherflag": weatherflag,
            "histflag": histflag,
        },
        "availability": {
            "has_daily_weather": av.has_daily_weather,
            "has_hourly_weather": av.has_hourly_weather,
            "has_airport_carrier_history_cache": av.has_airport_carrier_history_cache,
        },
        "request": {
            "Origin": req.origin.upper(),
            "Dest": req.dest.upper(),
            "Reporting_Airline": req.airline_iata.upper(),
            "flight_number": f"{req.airline_iata.upper()}{req.flight_number}",
            "flight_date": req.flight_date.isoformat(),
            "scheduled_departure_utc": sched_dep_utc.isoformat(),
        },
        "prediction": pred,
    }
    if args.print_features:
        output["features"] = features

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())