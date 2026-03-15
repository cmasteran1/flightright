from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import pandas as pd


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _s3_client():
    endpoint = _env("FLIGHTRIGHT_S3_ENDPOINT") or _env("E2_ENDPOINT")
    return boto3.client("s3", endpoint_url=endpoint)


def _rollups_local_path() -> Path:
    return Path(_env("FLIGHTRIGHT_ROLLUPS_CACHE", "/data/meta/latest_rollups.csv"))


def _ensure_rollups_csv() -> Path:
    bucket = _env("FLIGHTRIGHT_ROLLUPS_BUCKET") or _env("FLIGHTRIGHT_META_BUCKET") or _env("FLIGHTRIGHT_S3_BUCKET")
    key = _env("FLIGHTRIGHT_ROLLUPS_KEY")
    local_path = _rollups_local_path()

    if not bucket:
        raise RuntimeError(
            "Missing FLIGHTRIGHT_ROLLUPS_BUCKET (or FLIGHTRIGHT_META_BUCKET / FLIGHTRIGHT_S3_BUCKET)."
        )
    if not key:
        raise RuntimeError("Missing FLIGHTRIGHT_ROLLUPS_KEY.")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")

    s3 = _s3_client()
    try:
        s3.download_file(bucket, key, str(tmp))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download rollups CSV from bucket={bucket!r} key={key!r}: {e}"
        ) from e
    tmp.replace(local_path)
    return local_path


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float) and math.isnan(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _row_to_stats(row: pd.Series) -> Dict[str, Any]:
    return {
        "n_1d": _clean_scalar(row.get("dep_delay_1d_n")),
        "dep_delay_1d_mean": _clean_scalar(row.get("dep_delay_1d_mean")),
        "dep_delay_1d_median": _clean_scalar(row.get("dep_delay_1d_median")),
        "n_7d": _clean_scalar(row.get("dep_delay_7d_n")),
        "dep_delay_7d_mean": _clean_scalar(row.get("dep_delay_7d_mean")),
        "dep_delay_7d_median": _clean_scalar(row.get("dep_delay_7d_median")),
        "n_14d": _clean_scalar(row.get("dep_delay_14d_n")),
        "dep_delay_14d_mean": _clean_scalar(row.get("dep_delay_14d_mean")),
        "dep_delay_14d_median": _clean_scalar(row.get("dep_delay_14d_median")),
        "arr_n_1d": _clean_scalar(row.get("arr_delay_1d_n")),
        "arr_delay_1d_mean": _clean_scalar(row.get("arr_delay_1d_mean")),
        "arr_delay_1d_median": _clean_scalar(row.get("arr_delay_1d_median")),
        "arr_n_7d": _clean_scalar(row.get("arr_delay_7d_n")),
        "arr_delay_7d_mean": _clean_scalar(row.get("arr_delay_7d_mean")),
        "arr_delay_7d_median": _clean_scalar(row.get("arr_delay_7d_median")),
        "arr_n_14d": _clean_scalar(row.get("arr_delay_14d_n")),
        "arr_delay_14d_mean": _clean_scalar(row.get("arr_delay_14d_mean")),
        "arr_delay_14d_median": _clean_scalar(row.get("arr_delay_14d_median")),
    }


@lru_cache(maxsize=4)
def _load_rollups_cached(local_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(local_csv_path)
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df["group_type"] = df["group_type"].astype(str).str.strip()
    df["group_value"] = df["group_value"].astype(str).str.strip().str.upper()
    return df


def refresh_rollups() -> pd.DataFrame:
    csv_path = _ensure_rollups_csv()
    _load_rollups_cached.cache_clear()
    return _load_rollups_cached(str(csv_path))


def _latest_as_of_date_on_or_before(df: pd.DataFrame, target_date):
    eligible = df.loc[df["as_of_date"] <= target_date, "as_of_date"]
    if eligible.empty:
        return None
    return eligible.max()


def get_airport_stats(*, origin: str, dest: str, flight_date) -> Dict[str, Any]:
    csv_path = _ensure_rollups_csv()
    df = _load_rollups_cached(str(csv_path))

    as_of_date = _latest_as_of_date_on_or_before(df, flight_date)
    if as_of_date is None:
        return {"as_of_date": None, "origin": None, "dest": None, "missing": True}

    subset = df[
        (df["as_of_date"] == as_of_date)
        & (df["group_type"] == "airport_total")
        & (df["group_value"].isin([origin.upper(), dest.upper()]))
    ]

    out: Dict[str, Any] = {
        "as_of_date": as_of_date.isoformat(),
        "origin": None,
        "dest": None,
        "missing": False,
    }

    for _, row in subset.iterrows():
        rec = {"code": row["group_value"], **_row_to_stats(row)}
        if row["group_value"] == origin.upper():
            out["origin"] = rec
        elif row["group_value"] == dest.upper():
            out["dest"] = rec

    return out


def get_airline_stats(*, airline_iata: str, flight_date) -> Dict[str, Any]:
    csv_path = _ensure_rollups_csv()
    df = _load_rollups_cached(str(csv_path))

    as_of_date = _latest_as_of_date_on_or_before(df, flight_date)
    if as_of_date is None:
        return {"as_of_date": None, "airline": airline_iata.upper(), "stats": None, "missing": True}

    subset = df[
        (df["as_of_date"] == as_of_date)
        & (df["group_type"] == "airline_total")
        & (df["group_value"] == airline_iata.upper())
    ]

    if subset.empty:
        return {
            "as_of_date": as_of_date.isoformat(),
            "airline": airline_iata.upper(),
            "stats": None,
            "missing": True,
        }

    row = subset.iloc[0]
    return {
        "as_of_date": as_of_date.isoformat(),
        "airline": airline_iata.upper(),
        "stats": {"code": row["group_value"], **_row_to_stats(row)},
        "missing": False,
    }