# src/flightright/service/predictor.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from flightright.cli import predict as cli
from flightright.service.rollups import get_airline_stats, get_airport_stats

@dataclass(frozen=True)
class RemoteModelConfig:
    use_remote_models: bool = True
    s3_bucket: str = ""
    s3_prefix: str = "models/"
    s3_endpoint: Optional[str] = None
    remote_cache_dir: Optional[Path] = None


@dataclass(frozen=True)
class DataPaths:
    airports_csv: Path
    airport_rankings_dir: Path


@dataclass(frozen=True)
class ModelSpecDefaults:
    model_years: str = "23-25"
    airports_n: int = 50
    require_airports_in_ranking: bool = False


@dataclass(frozen=True)
class ServiceConfig:
    flightlabs_access_key: str
    data_paths: DataPaths
    remote_models: RemoteModelConfig
    model_defaults: ModelSpecDefaults = ModelSpecDefaults()


class FlightNeedsScheduleTimesError(ValueError):
    def __init__(
        self,
        *,
        airline_iata: str,
        flight_number: str,
        origin: str,
        dest: str,
        dep_date: date,
        future_match_error: str,
    ) -> None:
        super().__init__(
            f"Future-flight lookup failed for {airline_iata}{flight_number} "
            f"{origin}->{dest} on {dep_date.isoformat()}: {future_match_error}"
        )
        self.airline_iata = airline_iata
        self.flight_number = flight_number
        self.origin = origin
        self.dest = dest
        self.dep_date = dep_date
        self.future_match_error = future_match_error


_MODEL_LOAD_LOCK = Lock()


def _default_severity_band(score_0_to_4: float) -> str:
    if score_0_to_4 < 0.9:
        return "Safe"
    if score_0_to_4 < 1.8:
        return "Mild"
    if score_0_to_4 < 2.7:
        return "Caution"
    if score_0_to_4 < 3.6:
        return "High"
    return "Dangerous"


def _infer_airline_iata(airline: str) -> str:
    a = airline.strip()
    if len(a) == 2 and a.isalnum():
        return a.upper()

    name = a.lower().strip()
    mapping = {
        "southwest": "WN",
        "southwest airlines": "WN",
        "american": "AA",
        "american airlines": "AA",
        "delta": "DL",
        "delta air lines": "DL",
        "united": "UA",
        "united airlines": "UA",
        "jetblue": "B6",
        "alaska": "AS",
        "spirit": "NK",
        "frontier": "F9",
    }
    if name in mapping:
        return mapping[name]

    raise ValueError(
        "Could not infer airline IATA code from airline name. "
        "Please provide the 2-letter IATA code (e.g., WN)."
    )


@lru_cache(maxsize=16)
def _load_airports_cached(airports_csv_path: str) -> Dict[str, cli.AirportInfo]:
    return cli.load_airports_csv(Path(airports_csv_path))


@lru_cache(maxsize=8)
def _get_s3_store_cached(
    *,
    bucket: str,
    prefix: str,
    cache_dir: str,
    endpoint_url: Optional[str],
) -> cli.S3ModelStore:
    return cli.S3ModelStore(
        bucket=bucket,
        prefix=prefix,
        cache_dir=Path(cache_dir),
        endpoint_url=endpoint_url,
    )


def _load_joblib_with_mmap(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix in (".joblib", ".jl"):
        import joblib  # type: ignore

        try:
            return joblib.load(model_path, mmap_mode="r")
        except TypeError:
            return joblib.load(model_path)
        except Exception:
            return joblib.load(model_path)

    if suffix in (".pkl", ".pickle"):
        import pickle

        with model_path.open("rb") as f:
            return pickle.load(f)

    raise ValueError(f"Unsupported model file type: {model_path.suffix}")


@lru_cache(maxsize=32)
def _get_model_cached(
    *,
    bucket: str,
    prefix: str,
    cache_dir: str,
    endpoint_url: Optional[str],
    airline: str,
    n_airports: int,
    years: str,
    weatherflag: str,
    histflag: str,
) -> Tuple[Any, str, str]:
    store = _get_s3_store_cached(
        bucket=bucket,
        prefix=prefix,
        cache_dir=cache_dir,
        endpoint_url=endpoint_url,
    )
    spec = cli.RemoteModelSpec(
        airline=airline,
        n_airports=int(n_airports),
        years=str(years),
        weatherflag=str(weatherflag),
        histflag=str(histflag),
    )
    remote_dir_prefix = store.find_remote_model_dir_prefix(spec)
    model_path = store.download_deploy_joblib(remote_dir_prefix)

    with _MODEL_LOAD_LOCK:
        model = _load_joblib_with_mmap(model_path)

    model_path_str = f"s3://{bucket}/{remote_dir_prefix} (cached:{model_path})"
    return model, remote_dir_prefix, model_path_str


def _safe_model_id(
    airline: str,
    n_airports: int,
    years: str,
    weatherflag: str,
    histflag: str,
) -> str:
    return f"{airline}_{int(n_airports)}_{years}_{weatherflag}_{histflag}"


def _parse_hhmm_24h(value: str, *, field_name: str) -> time:
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except Exception as e:
        raise ValueError(f"{field_name} must be in HH:MM 24-hour format.") from e


def _local_hhmm_to_utc(dep_date: date, hhmm: str, tz_name: str) -> datetime:
    t = _parse_hhmm_24h(hhmm, field_name="Scheduled time")
    local_dt = datetime.combine(dep_date, t, tzinfo=ZoneInfo(tz_name))
    return local_dt.astimezone(timezone.utc)


def _arrival_local_hhmm_to_utc(
    dep_date: date,
    dep_hhmm: str,
    arr_hhmm: str,
    dep_tz_name: str,
    arr_tz_name: str,
) -> datetime:
    dep_utc = _local_hhmm_to_utc(dep_date, dep_hhmm, dep_tz_name)

    arr_t = _parse_hhmm_24h(arr_hhmm, field_name="Scheduled arrival time")
    arr_local = datetime.combine(dep_date, arr_t, tzinfo=ZoneInfo(arr_tz_name))
    arr_utc = arr_local.astimezone(timezone.utc)

    if arr_utc <= dep_utc:
        arr_local = arr_local + timedelta(days=1)
        arr_utc = arr_local.astimezone(timezone.utc)

    return arr_utc


def predict_departure(
    *,
    airline: str,
    flightnum: str,
    dep_date: date,
    origin: str,
    dest: str,
    cfg: ServiceConfig,
    sched_dep_time_24h: Optional[str] = None,
    sched_arr_time_24h: Optional[str] = None,
    include_weather: bool = True,
    include_flight_history: bool = True,
    include_airport_stats: bool = True,
    include_airline_stats: bool = True,
    include_features: bool = False,
    public_mode: bool = True,
) -> Dict[str, Any]:
    airline_iata = _infer_airline_iata(airline)

    req = cli.RequestSpec(
        origin=origin,
        dest=dest,
        airline_iata=airline_iata,
        flight_number=str(flightnum),
        flight_date=dep_date,
        sched_dep_time_24h=None,
    )

    airports_csv_path = str(Path(cfg.data_paths.airports_csv))
    airports = _load_airports_cached(airports_csv_path)

    if req.origin.upper() not in airports:
        raise ValueError(f"Origin airport {req.origin} not found in airports.csv")
    if req.dest.upper() not in airports:
        raise ValueError(f"Destination airport {req.dest} not found in airports.csv")

    origin_info = airports[req.origin.upper()]
    _dest_info = airports[req.dest.upper()]

    airport_timezones: Dict[str, str] = {
        code.upper(): info.tz
        for code, info in airports.items()
        if getattr(info, "tz", None)
    }

    fl = cli.FlightLabsClient(access_key=cfg.flightlabs_access_key)

    future = fl.future_flights_departures(req.origin.upper(), req.flight_date)

    _flight_row = None
    sched_dep_utc: Optional[datetime] = None
    sched_arr_utc: Optional[datetime] = None
    verification_source = "future_flights"
    future_match_error: Optional[str] = None
    warnings: list[dict[str, str]] = []

    try:
        _flight_row, sched_dep_utc = cli._match_future_flight(future, req)
    except Exception as e:
        future_match_error = str(e)

        if not sched_dep_time_24h or not sched_arr_time_24h:
            raise FlightNeedsScheduleTimesError(
                airline_iata=req.airline_iata,
                flight_number=req.flight_number,
                origin=req.origin.upper(),
                dest=req.dest.upper(),
                dep_date=req.flight_date,
                future_match_error=future_match_error,
            )

        dep_tz = airport_timezones.get(req.origin.upper())
        arr_tz = airport_timezones.get(req.dest.upper())
        if not dep_tz:
            raise RuntimeError(f"Missing timezone for origin airport {req.origin.upper()}")
        if not arr_tz:
            raise RuntimeError(f"Missing timezone for destination airport {req.dest.upper()}")

        sched_dep_utc = _local_hhmm_to_utc(req.flight_date, sched_dep_time_24h, dep_tz)
        sched_arr_utc = _arrival_local_hhmm_to_utc(
            req.flight_date,
            sched_dep_time_24h,
            sched_arr_time_24h,
            dep_tz,
            arr_tz,
        )
        verification_source = "user_schedule_fallback"
        warnings.append(
            {
                "code": "future_flight_not_found",
                "message": (
                    "Flight was not found in the future-flights lookup. "
                    "It may still exist; scheduled times were supplied by the user."
                ),
            }
        )

    assert sched_dep_utc is not None

    cong_total, cong_airline = cli.compute_congestion_3h(
        future_payload=future,
        target_sched_dep_utc=sched_dep_utc,
        airline_iata=req.airline_iata,
        flight_number=req.flight_number,
    )

    daily_w = (
        cli.openmeteo_daily(origin_info.lat, origin_info.lon, req.flight_date)
        if include_weather
        else None
    )
    hourly_w = (
        cli.openmeteo_hourly_near_departure(origin_info.lat, origin_info.lon, sched_dep_utc)
        if include_weather
        else None
    )

    flight_hist = (
        cli.compute_flightnum_od_rolling_stats_last28d(
            client=fl,
            airline_iata=req.airline_iata,
            flight_number=req.flight_number,
            origin=req.origin,
            dest=req.dest,
            as_of_local_date=req.flight_date,
        )
        if include_flight_history
        else {}
    )

    airport_carrier_cache = None
    av = cli.assess_availability(daily_w, hourly_w, airport_carrier_cache)

    weatherflag = "weather+" if av.has_hourly_weather else "weather"
    histflag = "standard" if av.has_airport_carrier_history_cache else "minhist"

    rankings_dir = Path(cfg.data_paths.airport_rankings_dir)
    n_airports = int(cfg.model_defaults.airports_n)

    if cfg.model_defaults.require_airports_in_ranking:
        ranking_file = rankings_dir / f"{req.airline_iata.upper()}_top_{n_airports}.csv"
        if not ranking_file.exists():
            raise FileNotFoundError(f"Ranking file not found: {ranking_file}")

        airports_list = cli._parse_one_line_airport_list(ranking_file.read_text(encoding="utf-8"))
        aset = set(airports_list)
        o = req.origin.upper()
        d = req.dest.upper()
        if o not in aset or d not in aset:
            raise FileNotFoundError(
                f"Origin/dest not both present in ranking list {ranking_file.name}. "
                f"(origin={o} in_list={o in aset}, dest={d} in_list={d in aset})"
            )

    model_family = cli.choose_model(av)

    if not cfg.remote_models.use_remote_models:
        raise RuntimeError("Website service expects remote models; set use_remote_models=True.")
    if not cfg.remote_models.s3_bucket:
        raise RuntimeError("Missing s3_bucket in ServiceConfig.remote_models")

    cache_dir = str(cfg.remote_models.remote_cache_dir or cli._default_remote_cache_dir())
    model, remote_dir_prefix, model_path_str = _get_model_cached(
        bucket=cfg.remote_models.s3_bucket,
        prefix=cfg.remote_models.s3_prefix,
        cache_dir=cache_dir,
        endpoint_url=cfg.remote_models.s3_endpoint,
        airline=req.airline_iata.upper(),
        n_airports=n_airports,
        years=str(cfg.model_defaults.model_years),
        weatherflag=weatherflag,
        histflag=histflag,
    )

    features = cli.build_feature_payload(
        req=req,
        sched_dep_utc=sched_dep_utc,
        congestion_total_3h=cong_total,
        congestion_airline_3h=cong_airline,
        daily_weather=daily_w,
        hourly_weather=hourly_w,
        flightnum_history=flight_hist,
        airport_carrier_history_cache=airport_carrier_cache,
    )

    pred = cli.model_predict(model, features)
    sev_score = pred.get("severity_score")
    if isinstance(sev_score, (int, float)):
        pred["severity_band"] = _default_severity_band(float(sev_score))

    model_id = _safe_model_id(
        airline=req.airline_iata.upper(),
        n_airports=int(n_airports),
        years=str(cfg.model_defaults.model_years),
        weatherflag=weatherflag,
        histflag=histflag,
    )

    airport_stats = None
    airline_stats = None

    if include_airport_stats:
        airport_stats = get_airport_stats(
            origin=req.origin.upper(),
            dest=req.dest.upper(),
            flight_date=req.flight_date,
        )

    if include_airline_stats:
        airline_stats = get_airline_stats(
            airline_iata=req.airline_iata.upper(),
            flight_date=req.flight_date,
        )

    out: Dict[str, Any] = {
        "ok": True,
        "chosen_model_family": model_family,
        "model": {
            "id": model_id,
            "resolved_model_dir_spec": {
                "airline": req.airline_iata.upper(),
                "n_airports": int(n_airports),
                "years": str(cfg.model_defaults.model_years),
                "weatherflag": weatherflag,
                "histflag": histflag,
            },
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
            "scheduled_arrival_utc": sched_arr_utc.isoformat() if sched_arr_utc else None,
            "verification_source": verification_source,
        },
        "prediction": pred,
        "warnings": warnings,
        "tabs": {
            "weather": {"daily": daily_w, "hourly": hourly_w} if include_weather else None,
            "flight_history": flight_hist if include_flight_history else None,
            "airport_stats": airport_stats,
            "airline_stats": airline_stats,
        },
    }

    if future_match_error:
        out["verification"] = {
            "source": verification_source,
            "exists": verification_source == "future_flights",
            "notes": [future_match_error],
            "scheduled_dep_time_24h": sched_dep_time_24h,
            "scheduled_arr_time_24h": sched_arr_time_24h,
        }

    if not public_mode:
        out["model_locator"] = model_path_str
        out["remote_dir_prefix"] = remote_dir_prefix

    if include_features:
        out["features"] = features

    return out