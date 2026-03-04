# src/flightright/api/app.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from flightright.cli import predict as cli  # for warm() reuse
from flightright.service.bootstrap_meta import ensure_meta_files
from flightright.service.predictor import (
    DataPaths,
    ModelSpecDefaults,
    RemoteModelConfig,
    ServiceConfig,
    predict_departure,
)

# -------------------------
# Environment
# -------------------------
ENV = os.getenv("ENV", "development").lower()
IS_PROD = ENV in {"prod", "production"}

# -------------------------
# App (docs hidden in prod)
# -------------------------
app = FastAPI(
    title="flightright",
    version="0.1.0",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)

# -------------------------
# Helpers
# -------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


# -------------------------
# Authorization (X-API-Key)
# -------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: Optional[str] = Depends(_api_key_header)) -> None:
    """
    Require X-API-Key for protected endpoints.
    - If FLIGHTRIGHT_API_KEY is not set on the server: fail closed with 500.
    - If wrong/missing key: 401.
    """
    want = _env("FLIGHTRIGHT_API_KEY")
    if not want:
        raise HTTPException(status_code=500, detail="Server auth not configured (missing FLIGHTRIGHT_API_KEY).")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _require_admin(x_admin_key: Optional[str]) -> None:
    want = _env("FLIGHTRIGHT_ADMIN_KEY")
    if not want:
        raise HTTPException(status_code=500, detail="Admin key not configured on server.")
    if not x_admin_key or x_admin_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized.")


# -------------------------
# Concurrency limiting (prevents RAM spikes / OOM)
# -------------------------
# How many in-flight predictions to allow per machine.
# Start conservative (1). If stable, try 2.
_MAX_INFLIGHT = int(_env("FLIGHTRIGHT_MAX_INFLIGHT", "1") or "1")
_PRED_SEM = BoundedSemaphore(value=max(1, _MAX_INFLIGHT))


class _PredSlot:
    def __enter__(self) -> None:
        ok = _PRED_SEM.acquire(blocking=False)
        if not ok:
            raise HTTPException(status_code=503, detail="Server is busy. Please retry in a moment.")

    def __exit__(self, exc_type, exc, tb) -> None:
        _PRED_SEM.release()


# -------------------------
# Config from env
# -------------------------
def build_config() -> ServiceConfig:
    access_key = _env("FLIGHTLABS_ACCESS_KEY") or _env("GOFLIGHTLABS_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("Missing FLIGHTLABS_ACCESS_KEY")

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
RATE_PER_MIN = float(_env("FLIGHTRIGHT_RPM", "6"))  # 6 / min
BURST = float(_env("FLIGHTRIGHT_BURST", "3"))       # allow short bursts


def _take_token(key: str) -> None:
    now = time.time()
    b = RATE_STATE.get(key)
    if b is None:
        b = Bucket(tokens=BURST, last=now)
        RATE_STATE[key] = b

    dt = max(0.0, now - b.last)
    b.last = now
    b.tokens = min(BURST, b.tokens + (RATE_PER_MIN / 60.0) * dt)

    if b.tokens < 1.0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")
    b.tokens -= 1.0


def _client_key(req: Request) -> str:
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


class WarmIn(BaseModel):
    airline_iata: str = Field(..., description="e.g. WN")
    n_airports: int = 50
    years: str = "23-25"
    weatherflag: str = "weather+"
    histflag: str = "minhist"


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def _startup_bootstrap_meta() -> None:
    try:
        bucket = os.environ.get("FLIGHTRIGHT_META_BUCKET") or os.environ.get("FLIGHTRIGHT_S3_BUCKET")
        if not bucket:
            # Don't crash the app; just log.
            print("WARN: Missing FLIGHTRIGHT_META_BUCKET (or FLIGHTRIGHT_S3_BUCKET); skipping meta bootstrap.")
            return

        downloads = [
            ("meta/aircraft_registry_clean.csv", Path("/data/meta/aircraft_registry_clean.csv")),
            ("meta/airport_rankings/50_group_4_total.txt", Path("/data/meta/airport_rankings/50_group_4_total.txt")),
        ]
        ensure_meta_files(bucket=bucket, downloads=downloads)

    except Exception as e:
        # Never prevent startup due to meta download issues
        print(f"WARN: meta bootstrap failed: {e!r}")
# -------------------------
# Routers
# -------------------------
public = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


@public.get("/")
def root() -> Dict[str, Any]:
    return {"service": "flightright", "status": "ok", "version": "0.1.0"}


@public.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@protected.post("/predict")
def predict(inp: PredictIn, request: Request) -> Dict[str, Any]:
    _take_token(_client_key(request))

    with _PredSlot():
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
                include_features=False,   # never for public
                public_mode=True,         # prevents model_locator leaks
            )
        except HTTPException:
            raise
        except Exception:
            import logging

            log = logging.getLogger("flightright")
            log.exception("predict failed")
            raise HTTPException(status_code=400, detail="Bad request.")


@protected.post("/admin/predict")
def admin_predict(
    inp: PredictIn,
    request: Request,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    _require_admin(x_admin_key)
    _take_token(_client_key(request))

    with _PredSlot():
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
                include_features=True,    # admin only
                public_mode=False,        # include model_locator + remote_dir_prefix
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@protected.post("/admin/warm")
def admin_warm(
    inp: WarmIn,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
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

    return {"ok": True, "remote_dir_prefix": remote_dir_prefix, "cached_model_path": str(model_path)}


app.include_router(public)
app.include_router(protected)