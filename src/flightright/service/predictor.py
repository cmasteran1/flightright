# src/flightright/service/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

# Reuse the battle-tested CLI building blocks for v0
from flightright.cli import predict as cli


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


def _default_severity_band(score_0_to_4: float) -> str:
    # Simple, explainable bands (frontend decides exact colors)
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
    # v0: accept "WN" etc; also accept names and map a small set.
    # Expand later (or integrate a proper mapping table).
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


def predict_departure(
    *,
    airline: str,
    flightnum: str,
    dep_date: date,
    origin: str,
    dest: str,
    cfg: ServiceConfig,
    # “tabs” toggles
    include_weather: bool = True,
    include_flight_history: bool = True,
    include_airport_stats: bool = True,
    include_airline_stats: bool = True,
    # security controls
    include_features: bool = False,
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

    # Load airport metadata
    airports = cli.load_airports_csv(Path(cfg.data_paths.airports_csv))
    if req.origin.upper() not in airports:
        raise ValueError(f"Origin airport {req.origin} not found in airports.csv")
    origin_info = airports[req.origin.upper()]

    fl = cli.FlightLabsClient(access_key=cfg.flightlabs_access_key)

    # 1) Verify flight exists + scheduled departure (UTC)
    future = fl.future_flights_departures(req.origin.upper(), req.flight_date)
    _flight_row, sched_dep_utc = cli._match_future_flight(future, req)

    # 2) Congestion (always used for prediction)
    cong_total, cong_airline = cli.compute_congestion_3h(
        future_payload=future,
        target_sched_dep_utc=sched_dep_utc,
        airline_iata=req.airline_iata,
        flight_number=req.flight_number,
    )

    # 3) Weather (optional “tab”, but also used as features if available)
    daily_w = cli.openmeteo_daily(origin_info.lat, origin_info.lon, req.flight_date) if include_weather else None
    hourly_w = (
        cli.openmeteo_hourly_near_departure(origin_info.lat, origin_info.lon, sched_dep_utc)
        if include_weather
        else None
    )

    # 4) Flight-number history (optional “tab”, but also used as features)
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

    # 5) Airport/airline caches (not implemented in CLI yet; keep hook)
    airport_carrier_cache = None

    av = cli.assess_availability(daily_w, hourly_w, airport_carrier_cache)

    weatherflag = "weather+" if av.has_hourly_weather else "weather"
    histflag = "standard" if av.has_airport_carrier_history_cache else "minhist"

    rankings_dir = Path(cfg.data_paths.airport_rankings_dir)
    n_airports = int(cfg.model_defaults.airports_n)

    # Optional strict ranking membership check (same as CLI)
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

    # Choose model family (CLI chooses based on availability)
    model_family = cli.choose_model(av)

    # Load model (remote only for v0 website)
    if not cfg.remote_models.use_remote_models:
        raise RuntimeError("Website service expects remote models; set use_remote_models=True.")

    if not cfg.remote_models.s3_bucket:
        raise RuntimeError("Missing s3_bucket in ServiceConfig.remote_models")

    store = cli.S3ModelStore(
        bucket=cfg.remote_models.s3_bucket,
        prefix=cfg.remote_models.s3_prefix,
        cache_dir=cfg.remote_models.remote_cache_dir or cli._default_remote_cache_dir(),
        endpoint_url=cfg.remote_models.s3_endpoint,
    )

    spec = cli.RemoteModelSpec(
        airline=req.airline_iata.upper(),
        n_airports=n_airports,
        years=str(cfg.model_defaults.model_years),
        weatherflag=weatherflag,
        histflag=histflag,
    )
    remote_dir_prefix = store.find_remote_model_dir_prefix(spec)
    model_path = store.download_deploy_joblib(remote_dir_prefix)
    model = cli.load_model_artifact(model_path)
    model_path_str = f"s3://{cfg.remote_models.s3_bucket}/{remote_dir_prefix} (cached:{model_path})"

    # Assemble features & predict (exact same math as CLI)
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

    # Add severity band (0..4 score computed inside CLI’s model_predict) :contentReference[oaicite:1]{index=1}
    sev_score = pred.get("severity_score")
    if isinstance(sev_score, (int, float)):
        pred["severity_band"] = _default_severity_band(float(sev_score))

    # Public payload (features omitted by default)
    out: Dict[str, Any] = {
        "ok": True,
        "chosen_model_family": model_family,
        "model_locator": model_path_str,
        "resolved_model_dir_spec": {
            "airline": req.airline_iata.upper(),
            "n_airports": int(n_airports),
            "years": str(cfg.model_defaults.model_years),
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
        # “tabs” data (you can expand these over time without breaking the main UI)
        "tabs": {
            "weather": {"daily": daily_w, "hourly": hourly_w} if include_weather else None,
            "flight_history": flight_hist if include_flight_history else None,
            # placeholders for your future enrichments:
            "airport_stats": {"origin": req.origin.upper()} if include_airport_stats else None,
            "airline_stats": {"airline": req.airline_iata.upper()} if include_airline_stats else None,
        },
    }
    if include_features:
        out["features"] = features

    return out