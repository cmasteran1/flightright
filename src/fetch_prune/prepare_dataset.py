#!/usr/bin/env python3
# src/fetch_prune/prepare_dataset.py
#
# Run from REPO_ROOT:
#   python src/fetch_prune/prepare_dataset.py data/dep_arr_config.json
#
import sys
import json
import glob
from pathlib import Path
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path.cwd()  # you always run from REPO_ROOT
DATA_DIR = REPO_ROOT / "data"


# ------------------------------ path helpers ------------------------------

def _abspath(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)

def _format_path_template(p: str, *, target: str, thr: Optional[int] = None) -> str:
    # Supports {target} and {thr} placeholders.
    if thr is None:
        return p.format(target=target)
    return p.format(target=target, thr=thr)

def _expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for pat in patterns:
        full = str(_abspath(pat))
        hits = sorted(glob.glob(full))
        for h in hits:
            p = Path(h)
            if p.is_file() and p.suffix == ".parquet" and p not in seen:
                seen.add(p)
                files.append(p)
    return files

def _parse_date_strict(s: str, field: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(s, errors="raise")
    except Exception as e:
        raise ValueError(f"Invalid date for cfg['{field}']={s!r}: {e}")
    if pd.isna(ts):
        raise ValueError(f"Invalid date for cfg['{field}']={s!r}")
    return ts.normalize()


# ------------------------------ BTS canonicalization ------------------------------

def canonicalize_bts_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    def first_present(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    if "Reporting_Airline" not in df.columns:
        src = first_present(
            "IATA_Code_Operating_Airline",
            "Operating_Airline",
            "IATA_Code_Marketing_Airline",
            "Marketing_Airline_Network",
        )
        if src is not None:
            df["Reporting_Airline"] = df[src].astype(str)
            print(f"[INFO] Reporting_Airline derived from '{src}'")

    if "Flight_Number_Reporting_Airline" not in df.columns:
        src = first_present("Flight_Number_Operating_Airline", "Flight_Number_Marketing_Airline")
        if src is not None:
            df["Flight_Number_Reporting_Airline"] = df[src]
            print(f"[INFO] Flight_Number_Reporting_Airline derived from '{src}'")

    for col in ["Reporting_Airline", "Origin", "Dest", "Tail_Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def read_input_parquet(input_flights_path: List[str]) -> pd.DataFrame:
    files = _expand_inputs(input_flights_path)
    if not files:
        raise FileNotFoundError(
            "No parquet files found for cfg['input_flights_path'] patterns.\n"
            f"Patterns={input_flights_path}"
        )
    dfs = []
    for fp in files:
        print(f"[INFO] Reading: {fp}")
        dfs.append(pd.read_parquet(fp))
    return pd.concat(dfs, ignore_index=True)


# ------------------------------ airport metadata ------------------------------

def load_airport_meta(airports_csv_path: str) -> pd.DataFrame:
    p = _abspath(airports_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"airports_csv not found: {p}")
    meta = pd.read_csv(p, dtype=str)
    required = {"IATA", "Latitude", "Longitude", "Timezone"}
    missing = required - set(meta.columns)
    if missing:
        raise RuntimeError(f"{p} missing columns: {sorted(missing)}")

    meta = meta.copy()
    meta["IATA"] = meta["IATA"].astype(str).str.upper().str.strip()
    meta["Latitude"] = pd.to_numeric(meta["Latitude"], errors="coerce")
    meta["Longitude"] = pd.to_numeric(meta["Longitude"], errors="coerce")
    meta["Timezone"] = meta["Timezone"].astype(str).str.strip()

    # normalize "nan"/"" -> NaN
    meta.loc[meta["Timezone"].str.lower().isin(["nan", "none", ""]), "Timezone"] = np.nan
    return meta


def _validate_tz_strings_for_needed(meta: pd.DataFrame, needed_iata: List[str]) -> Dict[str, Tuple[float, float, str]]:
    m = meta.set_index("IATA", drop=False)

    bad = []
    out: Dict[str, Tuple[float, float, str]] = {}
    for ap in needed_iata:
        if ap not in m.index:
            bad.append((ap, "MISSING_ROW"))
            continue
        row = m.loc[ap]
        lat = row["Latitude"]
        lon = row["Longitude"]
        tz = row["Timezone"]

        if pd.isna(lat) or pd.isna(lon):
            bad.append((ap, f"BAD_LATLON({lat},{lon})"))
            continue
        if pd.isna(tz):
            bad.append((ap, "MISSING_TZ"))
            continue
        try:
            ZoneInfo(str(tz))
        except ZoneInfoNotFoundError:
            bad.append((ap, f"UNRECOGNIZED_TZ({tz!r})"))
            continue

        out[ap] = (float(lat), float(lon), str(tz))

    if bad:
        examples = ", ".join([f"{a}:{why}" for a, why in bad[:40]])
        raise RuntimeError(
            "[AIRPORT META ERROR] Bad airport metadata for airports required by your filtered dataset.\n"
            f"airports_csv={_abspath('data/meta/airports.csv')}\n"
            f"Examples: {examples}{' ...' if len(bad)>40 else ''}\n"
            "Fix Latitude/Longitude/Timezone (IANA tz names like 'America/Chicago') for these IATA codes."
        )
    return out


# ------------------------------ local scheduled times ------------------------------

def _mk_local_dt(flight_date: pd.Timestamp, hhmm, tz_name: str):
    if pd.isna(flight_date) or pd.isna(hhmm) or tz_name is None:
        return None
    try:
        s = str(int(hhmm)).zfill(4)
        hh = int(s[:2])
        mm = int(s[2:])
        return datetime.combine(flight_date.date(), dtime(hh, mm)).replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def add_timezone_local_times(df: pd.DataFrame, airports_meta: Dict[str, Tuple[float, float, str]]) -> pd.DataFrame:
    df = df.copy()
    df["Origin"] = df["Origin"].astype(str).str.upper()
    df["Dest"] = df["Dest"].astype(str).str.upper()

    df["Origin_TZ"] = df["Origin"].map(lambda a: airports_meta[a][2])
    df["Dest_TZ"] = df["Dest"].map(lambda a: airports_meta[a][2])

    dep_locals, arr_locals = [], []
    for r in df.itertuples(index=False):
        dep_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSDepTime"), getattr(r, "Origin_TZ"))
        arr_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSArrTime"), getattr(r, "Dest_TZ"))

        # overnight arrival (same schedule date but arrives after midnight local)
        if dep_dt and arr_dt:
            try:
                dep_hhmm = getattr(r, "CRSDepTime")
                arr_hhmm = getattr(r, "CRSArrTime")
                if int(str(int(arr_hhmm)).zfill(4)) < int(str(int(dep_hhmm)).zfill(4)):
                    arr_dt = arr_dt + timedelta(days=1)
            except Exception:
                pass

        dep_locals.append(dep_dt)
        arr_locals.append(arr_dt)

    df["dep_dt_local"] = dep_locals
    df["arr_dt_local"] = arr_locals
    df["dep_local_date"] = df["dep_dt_local"].apply(lambda x: x.date() if isinstance(x, datetime) else pd.NaT)
    df["arr_local_date"] = df["arr_dt_local"].apply(lambda x: x.date() if isinstance(x, datetime) else pd.NaT)
    return df


# ------------------------------ HTTP helper ------------------------------

def _req_with_backoff(url, params, max_tries=6):
    import requests
    import time as _time

    backoff = 1.8
    delay = 1.0
    last_status = None
    last_text = None
    last_exc = None

    for i in range(1, max_tries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            last_status = r.status_code
            if r.status_code == 200:
                return r

            try:
                last_text = r.text[:600]
            except Exception:
                last_text = "<no text>"

            if r.status_code in (429, 500, 502, 503, 504):
                print(f"[WARN] HTTP {r.status_code} attempt {i}/{max_tries}. sleep {delay:.1f}s")
                _time.sleep(delay)
                delay = min(90, delay * backoff)
                continue

            raise RuntimeError(f"Request failed HTTP {r.status_code}. Response: {last_text}")

        except Exception as e:
            last_exc = e
            print(f"[WARN] Request exception attempt {i}/{max_tries}: {repr(e)}. sleep {delay:.1f}s")
            _time.sleep(delay)
            delay = min(90, delay * backoff)

    raise RuntimeError(
        "Request failed after retries. "
        f"Last status={last_status}, last_response={last_text}, last_exception={repr(last_exc)}"
    )


# ------------------------------ weather (daily, origin only) ------------------------------

DAILY_WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "weathercode",
]

def _fetch_daily_weather(lat, lon, start, end, tz):
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _req_with_backoff(
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY_WEATHER_VARS),
            "timezone": tz,
        },
    )
    payload = r.json()
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        cols = ["time"] + DAILY_WEATHER_VARS
        return pd.DataFrame(columns=cols).assign(FlightDate=pd.to_datetime([])).drop(columns=["time"])

    w = pd.DataFrame(daily)
    w["FlightDate"] = pd.to_datetime(w["time"])
    return w.drop(columns=["time"])

def add_origin_weather_daily(df, airports_meta: Dict[str, Tuple[float, float, str]], cache_dir: str):
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    cache_dir = _abspath(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = df["FlightDate"].min().strftime("%Y-%m-%d")
    end = df["FlightDate"].max().strftime("%Y-%m-%d")

    needed = sorted(pd.unique(df["Origin"].astype(str).str.upper()))
    frames = []

    for ap in needed:
        lat, lon, tz = airports_meta[ap]
        safe_tz = tz.replace("/", "__")
        cache_path = cache_dir / f"{ap}_{start}_{end}_{safe_tz}_daily.parquet"

        if cache_path.exists():
            wx = pd.read_parquet(cache_path)
        else:
            wx = _fetch_daily_weather(lat, lon, start, end, tz)
            wx.to_parquet(cache_path, index=False)

        ow = wx.copy()
        ow["Origin"] = ap
        ow = ow.rename(columns={c: f"origin_{c}" for c in DAILY_WEATHER_VARS})
        frames.append(ow[["FlightDate", "Origin"] + [f"origin_{c}" for c in DAILY_WEATHER_VARS]])

    if frames:
        o = pd.concat(frames, ignore_index=True)
        df = df.merge(o, on=["FlightDate", "Origin"], how="left")

    return df


# ------------------------------ aircraft age (optional) ------------------------------

def add_aircraft_age(df: pd.DataFrame, reg_csv: str) -> pd.DataFrame:
    reg_path = _abspath(reg_csv)
    if not reg_path.exists():
        print(f"[WARN] aircraft registry not found at {reg_path}; skipping aircraft age.")
        df["Aircraft_Age_Years"] = np.nan
        df["Aircraft_Age_Bucket"] = "Unknown"
        return df

    reg = pd.read_csv(reg_path, dtype={"Tail_Number": str, "Year_Mfr": float})
    df = df.merge(reg, on="Tail_Number", how="left")
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    mask = df["Year_Mfr"].notna() & df["FlightDate"].notna()
    df.loc[mask, "Aircraft_Age_Years"] = df.loc[mask, "FlightDate"].dt.year - df.loc[mask, "Year_Mfr"]

    def bucket(y):
        if pd.isna(y):
            return "Unknown"
        y = float(y)
        if y <= 5:
            return "0-5"
        if y <= 10:
            return "6-10"
        if y <= 15:
            return "11-15"
        return "16+"

    df["Aircraft_Age_Bucket"] = df["Aircraft_Age_Years"].apply(bucket)
    return df


# ------------------------------ history pool (for later feature engineering) ------------------------------

def build_and_write_history_pool(cfg: dict) -> Optional[Path]:
    """
    History pool does NOT require timezones (we only use FlightDate-based lag features).
    It is written unbalanced; features_dep.py can use it for lag features.
    """
    hist_paths = cfg.get("history_flights_path") or cfg.get("input_flights_path")
    if not hist_paths:
        return None

    target = (cfg.get("target") or "dep").lower()
    out_path_cfg = cfg.get("history_output_path", f"data/intermediate/history_pool_{target}.parquet")
    out_path = _abspath(_format_path_template(out_path_cfg, target=target))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lookback_days = int(cfg.get("history_lookback_days", 60))

    hist = read_input_parquet(hist_paths)
    hist = canonicalize_bts_columns(hist)

    hist["FlightDate"] = pd.to_datetime(hist["FlightDate"], errors="coerce")

    # apply same filters as main pool (airport/airline/date), but without tz dependency
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        hist["Origin"] = hist["Origin"].astype(str).str.upper()
        hist["Dest"] = hist["Dest"].astype(str).str.upper()
        # NOTE: history is still both-in-set to match target population
        hist = hist[hist["Origin"].isin(airports) & hist["Dest"].isin(airports)]

    if cfg.get("single_airline"):
        hist = hist[hist["Reporting_Airline"] == cfg["single_airline"]]
    elif cfg.get("airlines"):
        hist = hist[hist["Reporting_Airline"].isin(cfg["airlines"])]

    if cfg.get("start_date") and cfg.get("end_date"):
        s = _parse_date_strict(cfg["start_date"], "start_date") - pd.Timedelta(days=lookback_days)
        e = _parse_date_strict(cfg["end_date"], "end_date")
        hist = hist[(hist["FlightDate"] >= s) & (hist["FlightDate"] <= e)]

    for c in ["DepDelayMinutes", "DepDel15"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    keep = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "DepDelayMinutes",
        "DepDel15",
    ]
    keep = [c for c in keep if c in hist.columns]
    hist = hist[keep].dropna(subset=["FlightDate", "Origin", "Dest", "Reporting_Airline"]).copy()

    if "DepDel15" not in hist.columns and "DepDelayMinutes" in hist.columns:
        hist["DepDel15"] = (hist["DepDelayMinutes"] >= 15).astype("Int8")

    hist.to_parquet(out_path, index=False)
    print(f"[OK] wrote history pool rows={len(hist)} -> {out_path}")
    return out_path


# ------------------------------ main ------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_dataset.py data/dep_arr_config.json")
        sys.exit(1)

    cfg_path = _abspath(sys.argv[1])
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    target = (cfg.get("target") or "dep").lower()
    if target not in ("dep", "arr"):
        raise ValueError("cfg.target must be 'dep' or 'arr'")

    airports_csv = cfg.get("airports_csv", "data/meta/airports.csv")
    meta_df = load_airport_meta(airports_csv)
    print(f"[INFO] airports_csv -> {_abspath(airports_csv)}  (exists={_abspath(airports_csv).exists()})")

    # load raw
    df = read_input_parquet(cfg["input_flights_path"])
    df = canonicalize_bts_columns(df)

    # normalize types
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    for c in ["CRSDepTime", "CRSArrTime", "DepDelayMinutes", "ArrDelayMinutes", "DepDel15", "ArrDel15"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # filters
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        df["Origin"] = df["Origin"].astype(str).str.upper()
        df["Dest"] = df["Dest"].astype(str).str.upper()
        # IMPORTANT: both Origin and Dest must be in set
        df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports)]

    if cfg.get("single_airline"):
        df = df[df["Reporting_Airline"] == cfg["single_airline"]]
    elif cfg.get("airlines"):
        df = df[df["Reporting_Airline"].isin(cfg["airlines"])]

    if cfg.get("start_date") and cfg.get("end_date"):
        s = _parse_date_strict(cfg["start_date"], "start_date")
        e = _parse_date_strict(cfg["end_date"], "end_date")
        df = df[(df["FlightDate"] >= s) & (df["FlightDate"] <= e)]

    if df.empty:
        raise RuntimeError("No rows after filtering (airports/airlines/date range).")

    # validate tz ONLY for airports present after filtering (needed airports)
    needed_iata = sorted(pd.unique(pd.concat([df["Origin"], df["Dest"]]).astype(str).str.upper()))
    airports_meta = _validate_tz_strings_for_needed(meta_df, needed_iata)

    # local scheduled times
    df = add_timezone_local_times(df, airports_meta=airports_meta)

    # optional aircraft age
    if bool(cfg.get("add_aircraft_age", True)):
        df = add_aircraft_age(df, reg_csv=cfg.get("aircraft_registry_csv", "data/aircraft_registry_clean.csv"))

    # weather (daily origin only)
    weather_cfg = cfg.get("weather") or {}
    if bool(weather_cfg.get("daily", True)):
        cache_dir = weather_cfg.get("cache_dir", "data/weather_cache")
        df = add_origin_weather_daily(df, airports_meta=airports_meta, cache_dir=cache_dir)

    # write enriched unbalanced
    out_enriched = cfg.get("output_enriched_unbalanced_path", "data/intermediate/enriched_{target}_unbalanced.parquet")
    out_enriched = _format_path_template(out_enriched, target=target)
    out_path = _abspath(out_enriched)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote enriched unbalanced rows={len(df)} -> {out_path}")

    # history pool (optional, but recommended for lag features)
    hp = build_and_write_history_pool(cfg)
    if hp is not None:
        print(f"[INFO] history pool -> {hp}")


if __name__ == "__main__":
    main()
