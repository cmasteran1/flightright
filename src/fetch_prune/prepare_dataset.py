# prepare_dataset.py
import sys
import json
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import glob
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path.cwd()  # you run from repo root


def _resolve_input_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / p).resolve()


def _resolve_output_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / p).resolve()


def _format_path_template(p: str, target: str) -> str:
    return p.format(target=target) if "{target}" in p else p


# ---------- input expansion ----------
_MONTH_FILE_RE = re.compile(r"Year=\d{4}/Month=\d{2}/flights\.parquet$")


def _smart_month_glob(entry: str) -> Optional[str]:
    if _MONTH_FILE_RE.search(entry):
        return re.sub(r"Month=\d{2}/flights\.parquet$", "Month=*/flights.parquet", entry)
    return None


def _expand_one_input(entry: str) -> List[Path]:
    out: List[Path] = []
    seen = set()

    def _add(paths):
        for p in paths:
            if p.suffix == ".parquet" and p.is_file() and p not in seen:
                out.append(p)
                seen.add(p)

    rp = _resolve_input_path(entry)

    if rp.is_file() and rp.suffix == ".parquet":
        _add([rp])
        return out

    if rp.is_dir():
        _add(list(rp.rglob("*.parquet")))
        return out

    if any(c in entry for c in "*?[]{}"):
        _add([Path(x) for x in glob.glob(str(_resolve_input_path(entry)))])
        return out

    mg = _smart_month_glob(entry)
    if mg:
        _add([Path(x) for x in glob.glob(str(_resolve_input_path(mg)))])
        return out

    return out


def _expand_inputs(input_list: List[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for ent in input_list:
        found = _expand_one_input(ent)
        if not found:
            print(f"[WARN] No parquet for '{ent}' (relative to REPO_ROOT).")
        for f in found:
            if f not in seen:
                files.append(f)
                seen.add(f)
    return sorted(files)


def read_input_parquet(input_flights_path: List[str]) -> pd.DataFrame:
    files = _expand_inputs(input_flights_path)
    if not files:
        raise FileNotFoundError("No valid parquet files found from input_flights_path list.")
    dfs = []
    for fp in files:
        print(f"[INFO] Reading: {fp}")
        dfs.append(pd.read_parquet(fp))
    return pd.concat(dfs, ignore_index=True)


# ---------- column canonicalization ----------
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


# ---------- airport meta (scoped + optional fallback tz) ----------
def load_airport_meta_scoped(
    airports_csv_path: str,
    required_airports: List[str],
    timezone_fallback: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[float, float, str]], List[str]]:
    """
    Load airport meta from airports.csv, but ONLY validate/require airports in `required_airports`.

    Returns:
      meta_map: {IATA: (lat, lon, tz)}
      used_fallback: [IATA,...] airports for which timezone_fallback was applied

    Rules for required airports:
      - Must exist with valid lat/lon
      - Timezone must be valid IANA tz OR (if missing/invalid) use timezone_fallback if provided
      - If fallback not provided -> raise with the exact bad/missing tz strings we saw
    """
    p = Path(airports_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"airports csv not found: {airports_csv_path}")

    meta = pd.read_csv(p, dtype=str)

    need = {"IATA", "Latitude", "Longitude", "Timezone"}
    miss = need - set(meta.columns)
    if miss:
        raise RuntimeError(f"{airports_csv_path} missing required columns: {sorted(miss)}")

    meta["IATA"] = meta["IATA"].astype(str).str.upper().str.strip()
    meta["Latitude_num"] = pd.to_numeric(meta["Latitude"], errors="coerce")
    meta["Longitude_num"] = pd.to_numeric(meta["Longitude"], errors="coerce")

    # Keep raw tz as string so we can show repr() on failure
    def _tz_clean(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x)
        s2 = s.strip()
        if s2.lower() in ("nan", "none", ""):
            return None
        return s2

    meta["Timezone_clean"] = meta["Timezone"].map(_tz_clean)

    req = sorted({str(a).upper().strip() for a in required_airports})

    # Index by IATA for fast lookup
    meta_idx = meta.set_index("IATA", drop=False)

    bad = []  # list of (IATA, tz_repr_or_msg)
    used_fallback = []
    out: Dict[str, Tuple[float, float, str]] = {}

    # If they provided a fallback, validate it once up front (so we don't silently accept junk)
    if timezone_fallback is not None:
        try:
            ZoneInfo(timezone_fallback)
        except Exception:
            raise RuntimeError(
                f"[AIRPORT META ERROR] cfg.timezone_fallback is not a valid IANA timezone: {repr(timezone_fallback)}"
            )

    for ap in req:
        if ap not in meta_idx.index:
            bad.append((ap, "missing row in airports.csv"))
            continue

        row = meta_idx.loc[ap]
        lat = row["Latitude_num"]
        lon = row["Longitude_num"]
        if pd.isna(lat) or pd.isna(lon):
            bad.append((ap, "missing/invalid Latitude or Longitude"))
            continue

        tz = row["Timezone_clean"]

        if tz is None:
            if timezone_fallback is not None:
                tz = timezone_fallback
                used_fallback.append(ap)
            else:
                bad.append((ap, repr(row["Timezone"])))
                continue

        # Validate tz
        try:
            ZoneInfo(str(tz))
        except Exception:
            if timezone_fallback is not None:
                tz = timezone_fallback
                used_fallback.append(ap)
            else:
                bad.append((ap, repr(tz)))
                continue

        out[ap] = (float(lat), float(lon), str(tz))

    if bad:
        sample = ", ".join([f"{a}:{msg}" for a, msg in bad[:50]])
        raise RuntimeError(
            "[AIRPORT META ERROR] airports.csv has missing/invalid fields for REQUIRED airports.\n"
            f"File: {airports_csv_path}\n"
            f"Examples (IATA:problem): {sample}\n"
            "Fix those airports in data/meta/airports.csv, or set cfg.timezone_fallback to proceed."
        )

    return out, used_fallback


def _mk_local_dt(flight_date: pd.Timestamp, hhmm, tz_name: str):
    if pd.isna(flight_date) or pd.isna(hhmm) or tz_name is None:
        return None
    try:
        s = str(int(hhmm)).zfill(4)
        hh = int(s[:2])
        mm = int(s[2:])
        return datetime.combine(flight_date.date(), time(hh, mm)).replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None


def add_timezone_local_times(df: pd.DataFrame, airports_csv: str, timezone_fallback: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df["Origin"] = df["Origin"].astype(str).str.upper()
    df["Dest"] = df["Dest"].astype(str).str.upper()

    needed = sorted(pd.unique(pd.concat([df["Origin"], df["Dest"]]).astype(str).str.upper()))
    meta, used_fallback = load_airport_meta_scoped(
        airports_csv_path=airports_csv,
        required_airports=needed,
        timezone_fallback=timezone_fallback,
    )
    if used_fallback:
        print(f"[WARN] Applied timezone_fallback to {len(used_fallback)} required airports (e.g. {used_fallback[:10]})")

    df["Origin_TZ"] = df["Origin"].map(lambda a: meta[a][2])
    df["Dest_TZ"] = df["Dest"].map(lambda a: meta[a][2])

    dep_locals, arr_locals = [], []
    for r in df.itertuples(index=False):
        dep_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSDepTime"), getattr(r, "Origin_TZ"))
        arr_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSArrTime"), getattr(r, "Dest_TZ"))

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


# ---------- aircraft age ----------
def add_aircraft_age(df, reg_csv: str):
    reg = pd.read_csv(reg_csv, dtype={"Tail_Number": str, "Year_Mfr": float})
    df = df.merge(reg, on="Tail_Number", how="left")
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    mask = df["Year_Mfr"].notna() & df["FlightDate"].notna()
    df.loc[mask, "Aircraft_Age_Years"] = df.loc[mask, "FlightDate"].dt.year - df.loc[mask, "Year_Mfr"]

    def bucket(y):
        if pd.isna(y):
            return "Unknown"
        if y <= 5:
            return "0-5"
        if y <= 10:
            return "6-10"
        if y <= 15:
            return "11-15"
        return "16+"

    df["Aircraft_Age_Bucket"] = df["Aircraft_Age_Years"].apply(bucket)
    return df


# ---------- HTTP helper with backoff ----------
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
            try:
                last_text = r.text[:500]
            except Exception:
                last_text = "<no text>"

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                wait = max(65.0, delay)
                print(f"[WARN] HTTP 429 attempt {i}/{max_tries}. sleep {wait:.1f}s")
                _time.sleep(wait)
                delay = min(120, delay * backoff)
                continue

            if r.status_code in (500, 502, 503, 504):
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
            continue

    raise RuntimeError(
        f"Request failed after retries. Last status={last_status}, last_response={last_text}, last_exception={repr(last_exc)}"
    )


# ---------- weather (daily, origin only) ----------
DAILY_WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "weathercode",
]


def _fetch_daily_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _req_with_backoff(
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY_WEATHER_VARS),
            "timezone": "auto",
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


def add_origin_weather_daily(df, airports_csv, cache_dir="data/weather_cache"):
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    needed = sorted(pd.unique(df["Origin"].astype(str).str.upper()))
    # For weather we only need lat/lon; tz can be garbage and we don't care
    meta, _ = load_airport_meta_scoped(
        airports_csv_path=airports_csv,
        required_airports=needed,
        timezone_fallback="America/New_York",  # harmless here; not used
    )

    cache_dir = _resolve_output_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = df["FlightDate"].min().strftime("%Y-%m-%d")
    end = df["FlightDate"].max().strftime("%Y-%m-%d")

    frames = []
    for ap in needed:
        lat, lon, _tz = meta[ap]
        cache_path = cache_dir / f"{ap}_{start}_{end}_daily.parquet"

        if cache_path.exists():
            wx = pd.read_parquet(cache_path)
        else:
            wx = _fetch_daily_weather(lat, lon, start, end)
            wx.to_parquet(cache_path, index=False)

        ow = wx.copy()
        ow["Origin"] = ap
        ow = ow.rename(columns={c: f"origin_{c}" for c in DAILY_WEATHER_VARS})
        frames.append(ow[["FlightDate", "Origin"] + [f"origin_{c}" for c in DAILY_WEATHER_VARS]])

    if frames:
        o = pd.concat(frames, ignore_index=True)
        df = df.merge(o, on=["FlightDate", "Origin"], how="left")

    return df


# ---------- weather (hourly, origin departure hour only) ----------
HOURLY_VARS = ["temperature_2m", "windspeed_10m", "windgusts_10m", "precipitation", "weathercode"]


def _month_chunks(start_dt, end_dt):
    s = pd.Timestamp(start_dt).normalize()
    e = pd.Timestamp(end_dt).normalize()
    cur = pd.Timestamp(s.replace(day=1))
    out = []
    while cur <= e:
        month_start = cur
        month_end = (cur + pd.offsets.MonthEnd(0)).normalize()
        out.append((max(s, month_start).strftime("%Y-%m-%d"), min(e, month_end).strftime("%Y-%m-%d")))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out


def _fetch_hourly_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _req_with_backoff(
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "auto",
        },
    )
    payload = r.json().get("hourly", {})
    if not payload or "time" not in payload:
        cols = ["time"] + HOURLY_VARS
        return pd.DataFrame(columns=cols).assign(time_local=pd.to_datetime([])).drop(columns=["time"])

    w = pd.DataFrame(payload)
    w["time_local"] = pd.to_datetime(w["time"])
    return w.drop(columns=["time"])


def _fetch_hourly_weather_all(lat, lon, start_dt, end_dt, cache_root="data/weather_cache/hourly"):
    frames = []
    cache_root = _resolve_output_path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    for s, e in _month_chunks(start_dt, end_dt):
        cache_path = cache_root / f"{lat:.4f}_{lon:.4f}_{s}_{e}_hourly.parquet"
        if cache_path.exists():
            w = pd.read_parquet(cache_path)
        else:
            w = _fetch_hourly_weather(lat, lon, s, e)
            w.to_parquet(cache_path, index=False)
        if not w.empty:
            frames.append(w)

    if not frames:
        return pd.DataFrame(columns=["time_local"] + HOURLY_VARS)

    wx = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time_local"])
    return wx.reset_index(drop=True)


def add_origin_weather_hourly_at_dep_hour(df, airports_csv, cache_dir="data/weather_cache"):
    df = df.copy()
    df["Origin"] = df["Origin"].astype(str).str.upper()

    needed = sorted(pd.unique(df["Origin"].astype(str).str.upper()))
    # For weather we only need lat/lon; tz can be garbage and we don't care
    meta, _ = load_airport_meta_scoped(
        airports_csv_path=airports_csv,
        required_airports=needed,
        timezone_fallback="America/New_York",  # harmless here; not used
    )

    def _floor_hour(dt):
        if not isinstance(dt, datetime):
            return pd.NaT
        return dt.replace(minute=0, second=0, microsecond=0)

    df["dep_hour_local"] = df["dep_dt_local"].apply(_floor_hour)
    dep_vals = df["dep_hour_local"].dropna()
    if dep_vals.empty:
        return df

    start_dt = pd.Timestamp(min(dep_vals).date()) - pd.Timedelta(days=1)
    end_dt = pd.Timestamp(max(dep_vals).date()) + pd.Timedelta(days=1)

    frames = []
    for ap in needed:
        lat, lon, _tz = meta[ap]
        wx = _fetch_hourly_weather_all(
            lat, lon, start_dt, end_dt, cache_root=str(_resolve_output_path(cache_dir) / "hourly")
        )

        ow = wx.copy()
        for c in HOURLY_VARS:
            ow.rename(columns={c: f"origin_dep_{c}"}, inplace=True)
        ow["Origin"] = ap
        ow["dep_hour_local_naive"] = pd.to_datetime(ow["time_local"])
        ow = ow.drop(columns=["time_local"])
        frames.append(ow[["Origin", "dep_hour_local_naive"] + [f"origin_dep_{c}" for c in HOURLY_VARS]])

    origin_hourly = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Origin", "dep_hour_local_naive"])

    df["dep_hour_local_naive"] = df["dep_hour_local"].apply(
        lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) else pd.NaT
    )

    df = df.merge(origin_hourly, on=["Origin", "dep_hour_local_naive"], how="left")
    return df


# ---------- label + balancing ----------
def add_dep_label(df: pd.DataFrame, thr_min: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["DepDelayMinutes"] = pd.to_numeric(df.get("DepDelayMinutes"), errors="coerce")
    df["y_dep15"] = (df["DepDelayMinutes"] >= thr_min).astype("Int8")
    return df


def balance_50_50(df: pd.DataFrame, label_col="y_dep15", seed=42, max_rows=None) -> pd.DataFrame:
    df = df.copy()
    y = df[label_col]
    df = df[y.notna()].copy()
    y = df[label_col].astype(int)

    a = df[y == 0]
    b = df[y == 1]
    if len(a) == 0 or len(b) == 0:
        print("[WARN] Cannot balance: one class empty.")
        return df

    n = min(len(a), len(b))
    a = a.sample(n=n, random_state=seed)
    b = b.sample(n=n, random_state=seed)
    out = pd.concat([a, b], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    print(f"[INFO] Balanced 50/50 -> {len(out)} rows (n0=n1={n})")
    return out


def build_and_write_history_pool(cfg: dict) -> Path:
    hist_paths = cfg.get("history_flights_path") or cfg.get("input_flights_path")
    if not hist_paths:
        raise RuntimeError("Need cfg.history_flights_path or cfg.input_flights_path")

    lookback_days = int(cfg.get("history_lookback_days", 60))
    out_path_cfg = cfg.get("history_output_path", "data/intermediate/history_pool_dep.parquet")
    out_path = _resolve_output_path(out_path_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hist = read_input_parquet(hist_paths)
    hist = canonicalize_bts_columns(hist)

    hist["FlightDate"] = pd.to_datetime(hist["FlightDate"], errors="coerce")
    for c in ["DepDelayMinutes", "DepDel15"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    if cfg.get("start_date") and cfg.get("end_date"):
        start = pd.to_datetime(cfg["start_date"]) - pd.Timedelta(days=lookback_days)
        end = pd.to_datetime(cfg["end_date"])
        hist = hist[(hist["FlightDate"] >= start) & (hist["FlightDate"] <= end)]

    # Airports filter: Origin AND Dest must be in cfg["airports"]
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        hist["Origin"] = hist["Origin"].astype(str).str.upper()
        hist["Dest"] = hist["Dest"].astype(str).str.upper()
        hist = hist[hist["Origin"].isin(airports) & hist["Dest"].isin(airports)]

    if cfg.get("single_airline"):
        hist = hist[hist["Reporting_Airline"] == cfg["single_airline"]]
    elif cfg.get("airlines"):
        hist = hist[hist["Reporting_Airline"].isin(cfg["airlines"])]

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


# ---------- main ----------
def main():
    if len(sys.argv) != 2:
        print("Usage: python src/fetch_prune/prepare_dataset.py data/dep_arr_config.json")
        sys.exit(1)

    cfg_path = _resolve_input_path(sys.argv[1])
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    target = (cfg.get("target") or "dep").lower()
    if target not in ("dep", "arr"):
        raise ValueError("cfg.target must be 'dep' or 'arr'")

    airports_csv = str(_resolve_input_path(cfg.get("airports_csv", "data/meta/airports.csv")))
    print(f"[INFO] airports_csv -> {airports_csv}  (exists={Path(airports_csv).exists()})")

    df = read_input_parquet(cfg["input_flights_path"])
    df = canonicalize_bts_columns(df)

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    for c in ["CRSDepTime", "CRSArrTime", "DepDelayMinutes", "DepDel15"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Airports filter: Origin AND Dest must be in cfg["airports"]
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        df["Origin"] = df["Origin"].astype(str).str.upper()
        df["Dest"] = df["Dest"].astype(str).str.upper()
        df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports)]

    if cfg.get("single_airline"):
        df = df[df["Reporting_Airline"] == cfg["single_airline"]]
    elif cfg.get("airlines"):
        df = df[df["Reporting_Airline"].isin(cfg["airlines"])]

    if cfg.get("start_date") and cfg.get("end_date"):
        mask = (df["FlightDate"] >= pd.to_datetime(cfg["start_date"])) & (df["FlightDate"] <= pd.to_datetime(cfg["end_date"]))
        df = df[mask]

    if df.empty:
        raise RuntimeError("No rows after filtering.")

    # Timezone behavior:
    # - Only validates tz for airports actually present in df (after filters)
    # - If cfg.timezone_fallback is set, applies it ONLY where required airports are missing/bad
    tz_fallback = cfg.get("timezone_fallback")  # e.g. "America/New_York" or None
    df = add_timezone_local_times(df, airports_csv=airports_csv, timezone_fallback=tz_fallback)

    # Aircraft age (optional)
    if cfg.get("add_aircraft_age", True):
        reg_csv = str(_resolve_input_path(cfg.get("aircraft_registry_csv", "data/aircraft_registry_clean.csv")))
        if Path(reg_csv).exists():
            df = add_aircraft_age(df, reg_csv=reg_csv)
        else:
            print(f"[WARN] aircraft registry not found at {reg_csv}; skipping aircraft age.")

    # Weather
    weather_cfg = cfg.get("weather") or {}
    cache_dir = str(_resolve_output_path(weather_cfg.get("cache_dir", "data/weather_cache")))

    if weather_cfg.get("daily", True):
        df = add_origin_weather_daily(df, airports_csv=airports_csv, cache_dir=cache_dir)

    if weather_cfg.get("hourly", True):
        df = add_origin_weather_hourly_at_dep_hour(df, airports_csv=airports_csv, cache_dir=cache_dir)

    # Build history pool (full, unbalanced)
    history_path = build_and_write_history_pool(cfg)
    print(f"[INFO] history pool -> {history_path}")

    # Label + balance (dep)
    if target == "dep":
        df = add_dep_label(df, thr_min=int(cfg.get("dep_delay_threshold_min", 15)))
        bal = cfg.get("balance") or {}
        if bal.get("enabled", True):
            df = balance_50_50(
                df,
                label_col="y_dep15",
                seed=int(bal.get("seed", 42)),
                max_rows=bal.get("max_rows"),
            )

    out_cfg = cfg.get("output_intermediate_path", "data/intermediate/flights_enriched_{target}.parquet")
    out_cfg = _format_path_template(out_cfg, target)
    out_path = _resolve_output_path(out_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
