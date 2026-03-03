# src/flightright/api/app.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from flightright.service.bootstrap_meta import ensure_meta_files
from flightright.service.predictor import DataPaths, ModelSpecDefaults, RemoteModelConfig, ServiceConfig, predict_departure
from flightright.cli import predict as cli  # for warm() reuse


app = FastAPI(title="flightright", version="0.1.0")

@app.get("/")
def root():
    return {
        "service": "flightright",
        "status": "ok",
        "version": "0.1.0"
    }
# -------------------------
# Config from env
# -------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def build_config() -> ServiceConfig:
    access_key = _env("FLIGHTLABS_ACCESS_KEY") or _env("GOFLIGHTLABS_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("Missing FLIGHTLABS_ACCESS_KEY")

    # IMPORTANT for Fly volume: mount persistent storage at /data
    airports_csv = Path(_env("FLIGHTRIGHT_AIRPORTS_CSV", "/data/meta/airports.csv"))
    rankings_dir = Path(_env("FLIGHTRIGHT_RANKINGS_DIR", "/data/meta/airport_rankings"))

    s3_bucket = _env("FLIGHTRIGHT_S3_BUCKET", "")
    s3_prefix = _env("FLIGHTRIGHT_S3_PREFIX", "models/")
    s3_endpoint = _env("E2_ENDPOINT") or _env("FLIGHTRIGHT_S3_ENDPOINT")

    remote_cache_dir = Path(_env("FLIGHTRIGHT_REMOTE_CACHE_DIR", "/data/remote_models_cache"))

    airports_n = int(_env("FLIGHTRIGHT_AIRPORTS_N", "50"))
    model_years = _env("FLIGHTRIGHT_MODEL_YEARS", "23-25") or "23-25"

    return ServiceConfig(
        flightlabs_access_key=access_key,
        data_paths=DataPaths(airports_csv=airports_csv, airport_rankings_dir=rankings_dir),
        remote_models=RemoteModelConfig(
            use_remote_models=True,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            remote_cache_dir=remote_cache_dir,
        ),
        model_defaults=ModelSpecDefaults(
            model_years=model_years,
            airports_n=airports_n,
            require_airports_in_ranking=False,
        ),
    )


CFG = build_config()


# -------------------------
# Very small in-memory rate limiter (v0)
# -------------------------
@dataclass
class Bucket:
    tokens: float
    last: float


RATE_STATE: Dict[str, Bucket] = {}
RATE_PER_MIN = float(_env("FLIGHTRIGHT_RPM", "6"))        # 6 / min
BURST = float(_env("FLIGHTRIGHT_BURST", "3"))            # allow short bursts


def _take_token(key: str) -> None:
    now = time.time()
    b = RATE_STATE.get(key)
    if b is None:
        b = Bucket(tokens=BURST, last=now)
        RATE_STATE[key] = b

    # refill
    dt = max(0.0, now - b.last)
    b.last = now
    b.tokens = min(BURST, b.tokens + (RATE_PER_MIN / 60.0) * dt)

    if b.tokens < 1.0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")
    b.tokens -= 1.0


def _client_key(req: Request) -> str:
    # v0: per-IP limiting. Add session cookies later if you want.
    host = req.client.host if req.client else "unknown"
    return f"ip:{host}"


# -------------------------
# Schemas
# -------------------------
class IncludeSpec(BaseModel):
    weather: bool = True
    flight_history: bool = True
    airport_stats: bool = True
    airline_stats: bool = True


class PredictIn(BaseModel):
    airline: str = Field(..., description="Airline IATA (e.g. WN) or airline name (e.g. Southwest)")
    flightnum: str = Field(..., description="Flight number without airline prefix (e.g. 868)")
    date: date
    origin: str
    dest: str
    include: IncludeSpec = IncludeSpec()


# -------------------------
# Endpoints
# -------------------------

@app.on_event("startup")
def _startup_bootstrap_meta() -> None:
    bucket = os.environ.get("FLIGHTRIGHT_META_BUCKET") or os.environ.get("FLIGHTRIGHT_S3_BUCKET")
    if not bucket:
        raise RuntimeError("Missing FLIGHTRIGHT_META_BUCKET (or FLIGHTRIGHT_S3_BUCKET)")

    downloads = [
        ("meta/aircraft_registry_clean.csv", Path("/data/meta/aircraft_registry_clean.csv")),
        ("meta/airport_rankings/50_group_4_total.txt", Path("/data/meta/airport_rankings/50_group_4_total.txt")),
    ]
    ensure_meta_files(bucket=bucket, downloads=downloads)
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/predict")
def predict(inp: PredictIn, request: Request) -> Dict[str, Any]:
    _take_token(_client_key(request))

    try:
        return predict_departure(
            airline=inp.airline,
            flightnum=inp.flightnum,
            dep_date=inp.date,
            origin=inp.origin,
            dest=inp.dest,
            cfg=CFG,
            include_weather=inp.include.weather,
            include_flight_history=inp.include.flight_history,
            include_airport_stats=inp.include.airport_stats,
            include_airline_stats=inp.include.airline_stats,
            include_features=False,  # public mode hides raw features
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _require_admin(x_admin_key: Optional[str]) -> None:
    want = _env("FLIGHTRIGHT_ADMIN_KEY")
    if not want:
        raise HTTPException(status_code=500, detail="Admin key not configured on server.")
    if not x_admin_key or x_admin_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized.")


@app.post("/admin/predict")
def admin_predict(inp: PredictIn, request: Request, x_admin_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_admin(x_admin_key)
    _take_token(_client_key(request))  # still rate-limit admin by default

    try:
        return predict_departure(
            airline=inp.airline,
            flightnum=inp.flightnum,
            dep_date=inp.date,
            origin=inp.origin,
            dest=inp.dest,
            cfg=CFG,
            include_weather=inp.include.weather,
            include_flight_history=inp.include.flight_history,
            include_airport_stats=inp.include.airport_stats,
            include_airline_stats=inp.include.airline_stats,
            include_features=True,  # admin/debug includes raw features
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class WarmIn(BaseModel):
    airline_iata: str = Field(..., description="e.g. WN")
    n_airports: int = 50
    years: str = "23-25"
    weatherflag: str = "weather+"
    histflag: str = "minhist"


@app.post("/admin/warm")
def admin_warm(inp: WarmIn, x_admin_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_admin(x_admin_key)

    store = cli.S3ModelStore(
        bucket=CFG.remote_models.s3_bucket,
        prefix=CFG.remote_models.s3_prefix,
        cache_dir=CFG.remote_models.remote_cache_dir or cli._default_remote_cache_dir(),
        endpoint_url=CFG.remote_models.s3_endpoint,
    )
    spec = cli.RemoteModelSpec(
        airline=inp.airline_iata.upper(),
        n_airports=int(inp.n_airports),
        years=str(inp.years),
        weatherflag=str(inp.weatherflag),
        histflag=str(inp.histflag),
    )
    remote_dir_prefix = store.find_remote_model_dir_prefix(spec)
    model_path = store.download_deploy_joblib(remote_dir_prefix)

    return {
        "ok": True,
        "remote_dir_prefix": remote_dir_prefix,
        "cached_model_path": str(model_path),
    }