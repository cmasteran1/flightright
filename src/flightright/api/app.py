from __future__ import annotations

import logging
import os
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
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
# Logging
# -------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("flightright.api")

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


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _debug_allowed(x_admin_key: Optional[str]) -> bool:
    want = _env("FLIGHTRIGHT_ADMIN_KEY")
    return bool(want and x_admin_key and x_admin_key == want)


def _client_host(req: Request) -> str:
    return req.client.host if req.client else "unknown"


def _log_predict_exception(
    *,
    request_id: str,
    route_name: str,
    request: Request,
    inp: "PredictIn",
    exc: Exception,
) -> None:
    logger.exception(
        "predict failure request_id=%s route=%s client=%s airline=%s flightnum=%s date=%s origin=%s dest=%s include_weather=%s include_flight_history=%s include_airport_stats=%s include_airline_stats=%s exc_type=%s exc=%s",
        request_id,
        route_name,
        _client_host(request),
        inp.airline,
        inp.flightnum,
        inp.date.isoformat(),
        inp.origin,
        inp.dest,
        inp.include.weather,
        inp.include.flight_history,
        inp.include.airport_stats,
        inp.include.airline_stats,
        type(exc).__name__,
        str(exc),
    )


def _public_error_payload(
    *,
    request_id: str,
    user_title: str,
    user_message: str,
    user_action: str,
    debug_detail: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "request_id": request_id,
        "user_title": user_title,
        "user_message": user_message,
        "user_action": user_action,
    }
    if debug_detail and not IS_PROD:
        payload["debug_detail"] = debug_detail
    return payload


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
        raise HTTPException(
            status_code=500,
            detail="Server auth not configured (missing FLIGHTRIGHT_API_KEY).",
        )
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _require_admin(x_admin_key: Optional[str]) -> None:
    want = _env("FLIGHTRIGHT_ADMIN_KEY")
    if not want:
        raise HTTPException(status_code=500, detail="Admin key not configured on server.")
    if not x_admin_key or x_admin_key != want:
        raise HTTPException(status_code=401, detail="Unauthorized.")


# -------------------------
# Config from env
# -------------------------
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
        data_paths=DataPaths(
            airports_csv=airports_csv,
            airport_rankings_dir=rankings_dir,
        ),
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
BURST = float(_env("FLIGHTRIGHT_BURST", "3"))  # allow short bursts


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
    bucket = os.environ.get("FLIGHTRIGHT_META_BUCKET") or os.environ.get("FLIGHTRIGHT_S3_BUCKET")
    if not bucket:
        raise RuntimeError("Missing FLIGHTRIGHT_META_BUCKET (or FLIGHTRIGHT_S3_BUCKET)")

    downloads = [
        ("meta/aircraft_registry_clean.csv", Path("/data/meta/aircraft_registry_clean.csv")),
        ("meta/airport_rankings/50_group_4_total.txt", Path("/data/meta/airport_rankings/50_group_4_total.txt")),
    ]
    ensure_meta_files(bucket=bucket, downloads=downloads)


# -------------------------
# Routers
# - public: no auth
# - protected: require X-API-Key
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
def predict(
    inp: PredictIn,
    request: Request,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    _take_token(_client_key(request))
    request_id = _new_request_id()

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
            include_features=False,  # never for public
            public_mode=True,        # prevents model_locator leaks
        )
    except HTTPException:
        raise
    except Exception as e:
        _log_predict_exception(
            request_id=request_id,
            route_name="/predict",
            request=request,
            inp=inp,
            exc=e,
        )

        debug = _debug_allowed(x_admin_key)
        if debug:
            raise HTTPException(
                status_code=400,
                detail={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        raise HTTPException(
            status_code=400,
            detail=_public_error_payload(
                request_id=request_id,
                user_title="We couldn’t run this prediction",
                user_message="We were not able to generate a delay estimate for this flight.",
                user_action="Please double-check the airline, flight number, airports, and date, then try again.",
                debug_detail=str(e),
            ),
        )


@protected.post("/admin/predict")
def admin_predict(
    inp: PredictIn,
    request: Request,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    _require_admin(x_admin_key)
    _take_token(_client_key(request))  # still rate-limit admin by default
    request_id = _new_request_id()

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
            include_features=True,   # admin/debug includes raw features
            public_mode=False,       # admin gets internal model_locator
        )
    except HTTPException:
        raise
    except Exception as e:
        _log_predict_exception(
            request_id=request_id,
            route_name="/admin/predict",
            request=request,
            inp=inp,
            exc=e,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


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

    return {
        "ok": True,
        "remote_dir_prefix": remote_dir_prefix,
        "cached_model_path": str(model_path),
    }


@protected.get("/admin/cache")
def admin_cache(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> Dict[str, Any]:
    """
    Report what's on disk in the remote model cache.

    Requires BOTH:
    - X-API-Key (router dependency)
    - X-Admin-Key (this handler)
    """
    _require_admin(x_admin_key)

    cache_root = CFG.remote_models.remote_cache_dir or Path("/data/remote_models_cache")
    cache_root = Path(cache_root)

    files = []
    total_bytes = 0

    if cache_root.exists():
        # look for common artifact types
        patterns = ("*.joblib", "*.jl", "*.pkl", "*.pickle")
        seen = set()
        for pat in patterns:
            for p in cache_root.rglob(pat):
                if not p.is_file():
                    continue

                # de-dupe in case patterns overlap
                rp = str(p.resolve())
                if rp in seen:
                    continue
                seen.add(rp)

                st = p.stat()
                total_bytes += st.st_size
                files.append(
                    {
                        "path": str(p.relative_to(cache_root)),
                        "bytes": int(st.st_size),
                        "modified_utc": datetime.fromtimestamp(
                            st.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )

    files.sort(key=lambda x: x["bytes"], reverse=True)

    # Disk usage (best effort)
    try:
        du = shutil.disk_usage("/data")
        disk = {
            "path": "/data",
            "total_bytes": int(du.total),
            "used_bytes": int(du.used),
            "free_bytes": int(du.free),
        }
    except Exception:
        disk = {
            "path": "/data",
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
        }

    return {
        "ok": True,
        "cache_root": str(cache_root),
        "file_count": len(files),
        "total_cached_bytes": int(total_bytes),
        "disk": disk,
        "files": files,
    }


app.include_router(public)
app.include_router(protected)
