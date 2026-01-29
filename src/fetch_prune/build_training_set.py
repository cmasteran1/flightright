# build_training_set.py
import sys
import json
from pathlib import Path
from datetime import datetime, time, timedelta, date, timezone
from zoneinfo import ZoneInfo
import time as _time

import numpy as np
import pandas as pd

from pathlib import Path
import glob
import re
from typing import Optional, List  # <-- Py 3.9 compatibility for type hints

# ---------- multi-root path bases ----------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR  = SCRIPT_PATH.parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
ALT_ROOT    = REPO_ROOT.parent / "pythonProject2"

BASES = [
    Path.cwd(),
    SCRIPT_DIR,
    REPO_ROOT, REPO_ROOT / "src",
    ALT_ROOT,  ALT_ROOT  / "src",
    SCRIPT_DIR.parent / "data",
    REPO_ROOT / "src" / "data",
]


def _resolve_candidates(rel: str):
    return [(b / rel).resolve() for b in BASES]

def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    tried = []
    for cand in _resolve_candidates(p):
        tried.append(str(cand))
        if cand.exists():
            return cand
    fallback = (Path.cwd() / p).resolve()
    print(f"[PATH DEBUG] Could not find '{p}'. Tried:")
    for t in tried:
        print(f"  - {t}")
    print(f"  - (fallback) {fallback}")
    return fallback



def _resolve_output_path(p: str) -> Path:
    """Prefer writing relative outputs under REPO_ROOT (don’t require existence)."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (REPO_ROOT / p).resolve()

# ---------- input expansion (files/dirs/globs & Month=*/ fallback) ----------
_MONTH_FILE_RE = re.compile(r"Year=\d{4}/Month=\d{2}/flights\.parquet$")

def _smart_month_glob(entry: str) -> Optional[str]:  # <-- Py 3.9: Optional[str]
    if _MONTH_FILE_RE.search(entry):
        return re.sub(r"Month=\d{2}/flights\.parquet$", "Month=*/flights.parquet", entry)
    return None

def _expand_one_input(entry: str) -> List[Path]:     # <-- Py 3.9: List[Path]
    out, seen = [], set()

    def _add(paths):
        for p in paths:
            if p.suffix == ".parquet" and p.is_file() and p not in seen:
                out.append(p); seen.add(p)

    rp = _resolve_path(entry)
    if rp.is_file() and rp.suffix == ".parquet":
        _add([rp]); return out
    if rp.is_dir():
        _add(list(rp.rglob("*.parquet")))
        if out: return out

    if any(c in entry for c in "*?[]{}"):
        for base in BASES:
            _add([Path(x) for x in glob.glob(str((base / entry).resolve()))])
        if out: return out

    mg = _smart_month_glob(entry)
    if mg:
        for base in BASES:
            _add([Path(x) for x in glob.glob(str((base / mg).resolve()))])
        if out: return out

    for cand in _resolve_candidates(entry):
        if cand.is_file() and cand.suffix == ".parquet":
            _add([cand])
    return out

def _expand_inputs(input_list: List[str]) -> List[Path]:  # <-- Py 3.9: List[str] -> List[Path]
    files, seen = [], set()
    for ent in input_list:
        found = _expand_one_input(ent)
        if not found:
            # concise warning; avoid 24 lines of month spam
            examples = ", ".join(str(c) for c in _resolve_candidates(ent)[:2])
            print(f"[WARN] No parquet for '{ent}' (searched e.g. {examples} …)")
        for f in found:
            if f not in seen:
                files.append(f); seen.add(f)
    return sorted(files)


# Optional: timezone inference if tz is missing in airport JSON
try:
    from timezonefinder import TimezoneFinder
    _TF = TimezoneFinder()
except Exception:
    _TF = None

# --------------------------------------------------------------------
# Required canonical columns (we'll derive these from 2023–24 headers)
# --------------------------------------------------------------------
REQUIRED_COLS = [
    "FlightDate", "Origin", "Dest",
    "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    "CRSDepTime", "CRSArrTime",
    "ArrDelay", "ArrDel15",
]

def assert_required(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}. "
            "Ensure your raw BTS cache kept these fields."
        )

# ----------------- IO helpers -----------------
def read_input_parquet(input_flights_path: List[str]) -> pd.DataFrame:
    files = _expand_inputs(input_flights_path)
    if not files:
        print("[ERROR] No valid parquet files found from input_flights_path.")
        print("Tips: use globs like 'data/raw_bts/Year=2023/Month=*/flights.parquet' "
              "or list the parent Year dir(s). This script searches CWD, script dir, "
              "repo root, old repo root, and their /src variants.")
        raise FileNotFoundError("No valid parquet files found from input_flights_path list.")

    dfs = []
    for fp in files:
        try:
            # Optional pretty path in logs:
            rel = None
            for base in BASES:
                try:
                    rel = fp.relative_to(base); break
                except Exception:
                    pass
            print(f"[INFO] Reading: {rel if rel else fp}")
            dfs.append(pd.read_parquet(fp))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")

    if not dfs:
        raise FileNotFoundError("All candidate parquet files failed to read.")
    return pd.concat(dfs, ignore_index=True)


# ----------------- Column canonicalization for 2023–2024 BTS -----------------
def canonicalize_bts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize 2023–2024 BTS headers to canonical fields expected by the pipeline:
      - Reporting_Airline  (IATA code string)
      - Flight_Number_Reporting_Airline
    Also strips column-name whitespace.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    def first_present(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    # Reporting_Airline (IATA code)
    if "Reporting_Airline" not in df.columns:
        src = first_present(
            "IATA_Code_Operating_Airline",
            "Operating_Airline",
            "IATA_Code_Marketing_Airline",
            "Marketing_Airline_Network"
        )
        if src is not None:
            df["Reporting_Airline"] = df[src].astype(str)
            print(f"[INFO] Reporting_Airline derived from '{src}'")

    # Flight_Number_Reporting_Airline
    if "Flight_Number_Reporting_Airline" not in df.columns:
        src = first_present("Flight_Number_Operating_Airline", "Flight_Number_Marketing_Airline")
        if src is not None:
            df["Flight_Number_Reporting_Airline"] = df[src]
            print(f"[INFO] Flight_Number_Reporting_Airline derived from '{src}'")

    # Minimal harmonization of core columns
    for col in ["Reporting_Airline", "Origin", "Dest", "Tail_Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

# ----------------- small utils -----------------
def hhmm_to_min(x):
    try:
        s = str(int(x)).zfill(4)
        return int(s[:2]) * 60 + int(s[2:])
    except Exception:
        return np.nan

def hhmm_to_hour(x):
    try:
        s = str(int(x)).zfill(4)
        return int(s[:2])
    except Exception:
        return np.nan

# ----------------- optional final downsampling -----------------
def _temp_binary_label_for_stratify(df: pd.DataFrame) -> pd.Series:
    y15 = pd.to_numeric(df.get("ArrDel15"), errors="coerce")
    if y15.notna().any():
        y = y15.fillna(0).astype(int)
    else:
        y = (pd.to_numeric(df.get("ArrDelay"), errors="coerce") >= 15).astype(int)
    return y

def downsample_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Limit final dataset size after all feature joins/filters.
    Config keys:
      - sample_max_rows (int): cap on total rows
      - sample_strategy (str): "uniform" (default) or "stratified_by_label"
      - sample_seed (int): RNG seed (default 42)
    """
    max_rows = cfg.get("sample_max_rows") or cfg.get("limit_rows")
    if not max_rows or max_rows <= 0:
        return df

    n = len(df)
    if n <= max_rows:
        print(f"[INFO] Sampling skipped (rows={n} <= sample_max_rows={max_rows}).")
        return df

    strategy = (cfg.get("sample_strategy") or "uniform").lower()
    seed = int(cfg.get("sample_seed", 42))

    if strategy == "stratified_by_label":
        # Build a temporary binary label for stratification
        y = _temp_binary_label_for_stratify(df)
        # guard: if one class is empty, fall back to uniform
        vc = y.value_counts(dropna=True)
        if len(vc) < 2 or (vc == 0).any():
            print("[WARN] Stratified sampling fallback to uniform (one class empty).")
            sampled = df.sample(n=max_rows, random_state=seed)
        else:
            # proportional allocation by class
            frac = max_rows / float(n)
            parts = []
            for cls, g in df.groupby(y, sort=False):
                take = max(1, int(round(len(g) * frac)))
                take = min(take, len(g))  # safety
                parts.append(g.sample(n=take, random_state=seed))
            sampled = pd.concat(parts, ignore_index=True)
            # exact cap if rounding went a bit over
            if len(sampled) > max_rows:
                sampled = sampled.sample(n=max_rows, random_state=seed)
    else:
        # uniform
        sampled = df.sample(n=max_rows, random_state=seed)

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Downsampled rows: {n} -> {len(sampled)}  (strategy={strategy})")
    return sampled


# ----------------- airport coords + tz -----------------
def _coerce_airport_coords_with_tz(airport_json_path: str):
    """
    Returns dict: { IATA (upper): (lat, lon, tz_name) }.

    Policy (strict):
      - Every airport must have a valid (lat, lon) and an IANA tz name.
      - If tz is missing but lat/lon present, we compute it with timezonefinder.
      - If still missing (or lat/lon missing), we RAISE a RuntimeError listing offenders.
    """
    with open(airport_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "airport_coords" in data:
        data = data["airport_coords"]

    out = {}
    resolved_tz = 0

    for k, v in data.items():
        ap = str(k).upper()
        lat = lon = tz = None
        if isinstance(v, dict):
            lat = v.get("lat") or v.get("latitude")
            lon = v.get("lon") or v.get("lng") or v.get("longitude")
            tz  = v.get("tz")
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            lat, lon = v[0], v[1]
            if len(v) >= 3:
                tz = v[2]
        else:
            # malformed entry; leave as None to be caught below
            pass

        # last-resort manual tz map (kept; but we still crash later if not set)
        manual = {
            "JFK":"America/New_York", "LGA":"America/New_York", "EWR":"America/New_York",
            "BOS":"America/New_York", "PHL":"America/New_York", "BWI":"America/New_York",
            "ATL":"America/New_York", "MCO":"America/New_York", "MIA":"America/New_York",
            "ORD":"America/Chicago",  "DFW":"America/Chicago",  "IAH":"America/Chicago",
            "DEN":"America/Denver",   "PHX":"America/Phoenix",
            "LAS":"America/Los_Angeles", "LAX":"America/Los_Angeles", "SFO":"America/Los_Angeles",
            "SEA":"America/Los_Angeles", "PDX":"America/Los_Angeles",
        }
        if tz is None and ap in manual:
            tz = manual[ap]

        # If tz still missing but coords present, try to compute
        if tz is None and _TF is not None and lat is not None and lon is not None:
            try:
                tz_guess = _TF.timezone_at(lng=float(lon), lat=float(lat))
                if tz_guess:
                    tz = tz_guess
                    resolved_tz += 1
            except Exception:
                pass

        # Record; validation below
        out[ap] = (
            float(lat) if lat is not None else None,
            float(lon) if lon is not None else None,
            tz
        )

    if resolved_tz:
        print(f"[INFO] tz resolved by timezonefinder: {resolved_tz}")

    # Validate: every airport must have lat,lon,tz
    missing_latlon = [ap for ap,(lat,lon,tz) in out.items() if lat is None or lon is None]
    missing_tz     = [ap for ap,(lat,lon,tz) in out.items() if tz is None]

    msgs = []
    if missing_latlon:
        msgs.append(f"missing lat/lon for: {missing_latlon[:20]}{' ...' if len(missing_latlon)>20 else ''}")
    if missing_tz:
        msgs.append(f"missing timezone for: {missing_tz[:20]}{' ...' if len(missing_tz)>20 else ''}")

    if msgs:
        raise RuntimeError(
            "[AIRPORT TZ ERROR] Airport coordinate/timezone resolution failed:\n  - " +
            "\n  - ".join(msgs) +
            "\nFix airport_json entries to include lat/lon and (preferably) 'tz' or ensure timezonefinder can infer it."
        )

    return out

def _mk_local_dt(flight_date: pd.Timestamp, hhmm, tz_name: str):
    """Combine FlightDate (local calendar) + HHMM into tz-aware datetime."""
    if pd.isna(flight_date) or pd.isna(hhmm) or tz_name is None:
        return None
    try:
        s = str(int(hhmm)).zfill(4)
        hh = int(s[:2]); mm = int(s[2:])
        return datetime.combine(flight_date.date(), time(hh, mm)).replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        return None

def add_timezone_local_times(df, airport_json: str):
    """
    Adds timezone-aware scheduled local times:
      - dep_dt_local (Origin tz)
      - arr_dt_local (Dest tz; adds +1 day if arr HHMM < dep HHMM)
      - dep_local_date / arr_local_date (local calendar dates)
      - dep_bin30_local / arr_bin30_local (local 30-min bins)

    STRICT: Will raise if any airport lacks tz/coords (enforced by _coerce_airport_coords_with_tz).
    """
    coords_tz = _coerce_airport_coords_with_tz(airport_json)

    df = df.copy()
    df["Origin"] = df["Origin"].astype(str).str.upper()
    df["Dest"]   = df["Dest"].astype(str).str.upper()

    df["Origin_TZ"] = df["Origin"].map(lambda a: coords_tz[a][2])
    df["Dest_TZ"]   = df["Dest"].map(lambda a: coords_tz[a][2])

    dep_locals, arr_locals = [], []
    for r in df.itertuples(index=False):
        dep_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSDepTime"), getattr(r, "Origin_TZ"))
        arr_dt = _mk_local_dt(getattr(r, "FlightDate"), getattr(r, "CRSArrTime"), getattr(r, "Dest_TZ"))

        if dep_dt and arr_dt:
            dep_hhmm = getattr(r, "CRSDepTime")
            arr_hhmm = getattr(r, "CRSArrTime")
            try:
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

    def _to_bin30(dt):
        if not isinstance(dt, datetime):
            return pd.NA
        return (dt.hour * 60 + dt.minute) // 30

    df["dep_bin30_local"] = df["dep_dt_local"].apply(_to_bin30).astype("Int64")
    df["arr_bin30_local"] = df["arr_dt_local"].apply(_to_bin30).astype("Int64")

    # Logging
    dep_cov = df["dep_dt_local"].notna().mean()
    arr_cov = df["arr_dt_local"].notna().mean()
    print(f"[INFO] dep_dt_local coverage: {dep_cov:.3f}  |  arr_dt_local coverage: {arr_cov:.3f}")
    if dep_cov < 0.9:
        ex = df.loc[df["dep_dt_local"].isna(), "Origin"].astype(str).str.upper().value_counts().head(5).index.tolist()
        print(f"[WARN] dep_dt_local NaT examples (Origin): {ex}")
    if arr_cov < 0.9:
        ex = df.loc[df["arr_dt_local"].isna(), "Dest"].astype(str).str.upper().value_counts().head(5).index.tolist()
        print(f"[WARN] arr_dt_local NaT examples (Dest):   {ex}")

    return df

# ----------------- congestion counts (LOCAL) -----------------
def add_simple_congestion(df):
    """
    Local-time congestion signals:
      - dep_count_origin_bin: # flights departing from Origin in same local 30-min dep_bin
      - arr_count_dest_trail_2h: # flights scheduled to ARRIVE into Dest in previous 120 local minutes
      - carrier_flights_prior_day: # flights by same carrier nationwide earlier that local day (by dep_dt_local)

    New:
      - origin_dep_count_trail_60m / 120m
      - origin_dep_count_carrier_trail_120m
    """
    df = df.copy()

    # dep_count_origin_bin (LOCAL)
    dep_src = df.dropna(subset=["dep_bin30_local"])
    dep_cnt = (
        dep_src.groupby(["Origin","dep_local_date","dep_bin30_local"])
               .size().rename("dep_count_origin_bin").reset_index()
    )
    df = df.merge(dep_cnt, on=["Origin","dep_local_date","dep_bin30_local"], how="left")

    # arr_count_dest_trail_2h (LOCAL)
    df["arr_count_dest_trail_2h"] = np.nan
    mask = df["arr_dt_local"].notna()
    for (dest, d), idx in df[mask].sort_values(["Dest","arr_local_date","arr_dt_local"]).groupby(["Dest","arr_local_date"]).groups.items():
        arr_minutes = df.loc[idx, "arr_dt_local"].apply(lambda x: x.hour*60 + x.minute).to_numpy()
        cum = np.arange(1, len(arr_minutes) + 1)
        lower = np.searchsorted(arr_minutes, arr_minutes - 120, side="left")
        win = cum - np.where(lower > 0, cum[lower - 1], 0)
        df.loc[idx, "arr_count_dest_trail_2h"] = win

    # New: origin trailing departures
    for col in ["origin_dep_count_trail_60m","origin_dep_count_trail_120m","origin_dep_count_carrier_trail_120m"]:
        df[col] = np.nan

    mask = df["dep_dt_local"].notna()
    gcols = ["Origin","dep_local_date"]
    for (origin, d), idx in df[mask].sort_values(gcols + ["dep_dt_local"]).groupby(gcols).groups.items():
        dep_minutes = df.loc[idx, "dep_dt_local"].apply(lambda x: x.hour*60 + x.minute).to_numpy()

        for win_m, out_col in [(60,"origin_dep_count_trail_60m"), (120,"origin_dep_count_trail_120m")]:
            cum = np.arange(1, len(dep_minutes) + 1)
            lower = np.searchsorted(dep_minutes, dep_minutes - win_m, side="left")
            win = cum - np.where(lower > 0, cum[lower - 1], 0)
            df.loc[idx, out_col] = win

        carriers = df.loc[idx, "Reporting_Airline"].astype(str).to_numpy()
        for carr in pd.unique(carriers):
            carr_mask = carriers == carr
            carr_idx = df.loc[idx].index[carr_mask]
            dep_m_c = dep_minutes[carr_mask]
            cum = np.arange(1, len(dep_m_c) + 1)
            lower = np.searchsorted(dep_m_c, dep_m_c - 120, side="left")
            win = cum - np.where(lower > 0, cum[lower - 1], 0)
            df.loc[carr_idx, "origin_dep_count_carrier_trail_120m"] = win

    # carrier_flights_prior_day (LOCAL by dep_dt_local)
    df = df.sort_values(["Reporting_Airline","dep_local_date","dep_dt_local"])
    df["carrier_flights_prior_day"] = (
        df.groupby(["Reporting_Airline","dep_local_date"])
          .cumcount()
          .astype("Int64")
    )

    return df

# ----------------- taxi rolling baselines (LOCAL) -----------------
def add_taxi_congestion_rolling(df, window_days=7, bin_minutes=30):
    """
    Local-time rolling baselines:
      - origin_taxiout_avg_{window_days}d_bin (by Origin, dep_bin30_local)
      - dest_taxiin_avg_{window_days}d_bin (by Dest, arr_bin30_local)
    """
    df = df.copy()
    # Origin TaxiOut
    if "TaxiOut" in df.columns:
        o = df[["Origin","dep_local_date","dep_bin30_local","TaxiOut"]].dropna().copy()
        if not o.empty:
            o_day = (o.groupby(["Origin","dep_local_date","dep_bin30_local"], as_index=False)["TaxiOut"].mean()
                       .sort_values(["Origin","dep_bin30_local","dep_local_date"]))
            o_day[f"origin_taxiout_avg_{window_days}d_bin"] = (
                o_day.groupby(["Origin","dep_bin30_local"])["TaxiOut"]
                     .apply(lambda s: s.shift(1).rolling(window=window_days, min_periods=1).mean())
                     .reset_index(level=[0,1], drop=True)
            )
            df = df.merge(
                o_day[["Origin","dep_local_date","dep_bin30_local",f"origin_taxiout_avg_{window_days}d_bin"]],
                on=["Origin","dep_local_date","dep_bin30_local"], how="left"
            )

    # Dest TaxiIn
    if "TaxiIn" in df.columns:
        d = df[["Dest","arr_local_date","arr_bin30_local","TaxiIn"]].dropna().copy()
        if not d.empty:
            d_day = (d.groupby(["Dest","arr_local_date","arr_bin30_local"], as_index=False)["TaxiIn"].mean()
                       .sort_values(["Dest","arr_bin30_local","arr_local_date"]))
            d_day[f"dest_taxiin_avg_{window_days}d_bin"] = (
                d_day.groupby(["Dest","arr_bin30_local"])["TaxiIn"]
                     .apply(lambda s: s.shift(1).rolling(window=window_days, min_periods=1).mean())
                     .reset_index(level=[0,1], drop=True)
            )
            df = df.merge(
                d_day[["Dest","arr_local_date","arr_bin30_local",f"dest_taxiin_avg_{window_days}d_bin"]],
                on=["Dest","arr_local_date","arr_bin30_local"], how="left"
            )
    return df

# ----------------- enriched historical baselines (row-count rolling) -----------------
def add_history_baselines(df,
                          carrier_days=7,
                          od_days=7,
                          flightnum_days=14,
                          origin_days=7,
                          dest_days=7):
    """
    Adds lagged rolling mean delay features (no leakage: shift(1) before rolling):
      - carrier_delay_{carrier_days}d_mean         (by Reporting_Airline)
      - od_delay_{od_days}d_mean                   (by Origin, Dest)
      - flightnum_delay_{flightnum_days}d_mean     (by Reporting_Airline, Flight_Number_Reporting_Airline)
      - origin_delay_{origin_days}d_mean           (by Origin)
      - dest_delay_{dest_days}d_mean               (by Dest)
    Uses row-count rolling windows after daily aggregation.
    """
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["ArrDelay"] = pd.to_numeric(df.get("ArrDelay"), errors="coerce")

    out = df

    # Carrier
    carr = df[["Reporting_Airline","FlightDate","ArrDelay"]].dropna().copy()
    if not carr.empty:
        carr_daily = (carr.groupby(["Reporting_Airline","FlightDate"], as_index=False)["ArrDelay"].mean()
                          .sort_values(["Reporting_Airline","FlightDate"]))
        carr_daily[f"carrier_delay_{carrier_days}d_mean"] = (
            carr_daily.groupby("Reporting_Airline")["ArrDelay"]
                      .apply(lambda s: s.shift(1).rolling(window=carrier_days, min_periods=1).mean())
                      .reset_index(level=0, drop=True)
        )
        out = out.merge(
            carr_daily[["Reporting_Airline","FlightDate",f"carrier_delay_{carrier_days}d_mean"]],
            on=["Reporting_Airline","FlightDate"], how="left"
        )

    # OD pair
    od = df[["Origin","Dest","FlightDate","ArrDelay"]].dropna().copy()
    if not od.empty:
        od_daily = (od.groupby(["Origin","Dest","FlightDate"], as_index=False)["ArrDelay"].mean()
                        .sort_values(["Origin","Dest","FlightDate"]))
        od_daily[f"od_delay_{od_days}d_mean"] = (
            od_daily.groupby(["Origin","Dest"])["ArrDelay"]
                    .apply(lambda s: s.shift(1).rolling(window=od_days, min_periods=1).mean())
                    .reset_index(level=[0,1], drop=True)
        )
        out = out.merge(
            od_daily[["Origin","Dest","FlightDate",f"od_delay_{od_days}d_mean"]],
            on=["Origin","Dest","FlightDate"], how="left"
        )

    # Flight number (carrier + flight number)
    if "Flight_Number_Reporting_Airline" in df.columns:
        fn = df[["Reporting_Airline","Flight_Number_Reporting_Airline","FlightDate","ArrDelay"]].dropna().copy()
        if not fn.empty:
            fn_daily = (fn.groupby(["Reporting_Airline","Flight_Number_Reporting_Airline","FlightDate"], as_index=False)["ArrDelay"].mean()
                           .sort_values(["Reporting_Airline","Flight_Number_Reporting_Airline","FlightDate"]))
            fn_daily[f"flightnum_delay_{flightnum_days}d_mean"] = (
                fn_daily.groupby(["Reporting_Airline","Flight_Number_Reporting_Airline"])["ArrDelay"]
                        .apply(lambda s: s.shift(1).rolling(window=flightnum_days, min_periods=3).mean())
                        .reset_index(level=[0,1], drop=True)
            )
            out = out.merge(
                fn_daily[["Reporting_Airline","Flight_Number_Reporting_Airline","FlightDate",
                          f"flightnum_delay_{flightnum_days}d_mean"]],
                on=["Reporting_Airline","Flight_Number_Reporting_Airline","FlightDate"], how="left"
            )
    else:
        out[f"flightnum_delay_{flightnum_days}d_mean"] = np.nan

    # Origin airport
    org = df[["Origin","FlightDate","ArrDelay"]].dropna().copy()
    if not org.empty:
        org_daily = (org.groupby(["Origin","FlightDate"], as_index=False)["ArrDelay"].mean()
                        .sort_values(["Origin","FlightDate"]))
        org_daily[f"origin_delay_{origin_days}d_mean"] = (
            org_daily.groupby("Origin")["ArrDelay"]
                     .apply(lambda s: s.shift(1).rolling(window=origin_days, min_periods=1).mean())
                     .reset_index(level=0, drop=True)
        )
        out = out.merge(
            org_daily[["Origin","FlightDate",f"origin_delay_{origin_days}d_mean"]],
            on=["Origin","FlightDate"], how="left"
        )

    # Destination airport
    dst = df[["Dest","FlightDate","ArrDelay"]].dropna().copy()
    if not dst.empty:
        dst_daily = (dst.groupby(["Dest","FlightDate"], as_index=False)["ArrDelay"].mean()
                        .sort_values(["Dest","FlightDate"]))
        dst_daily[f"dest_delay_{dest_days}d_mean"] = (
            dst_daily.groupby("Dest")["ArrDelay"]
                     .apply(lambda s: s.shift(1).rolling(window=dest_days, min_periods=1).mean())
                     .reset_index(level=0, drop=True)
        )
        out = out.merge(
            dst_daily[["Dest","FlightDate",f"dest_delay_{dest_days}d_mean"]],
            on=["Dest","FlightDate"], how="left"
        )

    return out

# ----------------- weather (daily) with per-airport TZ -----------------
DAILY_WEATHER_VARS = [
    "temperature_2m_max","temperature_2m_min","precipitation_sum",
    "windspeed_10m_max","windgusts_10m_max","weathercode"
]

def _fetch_daily_weather(lat, lon, start, end, tz):
    import requests
    if tz is None:
        raise RuntimeError("Daily weather fetch requires per-airport timezone (tz is None).")
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_WEATHER_VARS),
        "timezone": tz
    }, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    w = pd.DataFrame(d)
    w["FlightDate"] = pd.to_datetime(w["time"])
    return w.drop(columns=["time"])

def _get_weather_for_airports_daily(df, coords_map, cache_dir="../../data/weather_cache"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    start = df["FlightDate"].min().strftime("%Y-%m-%d")
    end   = df["FlightDate"].max().strftime("%Y-%m-%d")

    needed = pd.unique(pd.concat([df["Origin"], df["Dest"]]).astype(str).str.upper())
    origin_frames, dest_frames = [], []

    for ap in needed:
        lat, lon, tz = coords_map[ap]  # strict: keys guaranteed, tz validated
        safe_tz = tz.replace("/", "__")
        cache_path = Path(cache_dir) / f"{ap}_{start}_{end}_{safe_tz}_daily.parquet"

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            wx = pd.read_parquet(cache_path)
        else:
            wx = _fetch_daily_weather(lat, lon, start, end, tz=tz)
            wx.to_parquet(cache_path, index=False)

        ow = wx.copy(); ow["Origin"] = ap
        ow = ow.rename(columns={c: f"origin_{c}" for c in DAILY_WEATHER_VARS})
        origin_frames.append(ow[["FlightDate","Origin"] + [f"origin_{c}" for c in DAILY_WEATHER_VARS]])

        dw = wx.copy(); dw["Dest"] = ap
        dw = dw.rename(columns={c: f"dest_{c}" for c in DAILY_WEATHER_VARS})
        dest_frames.append(dw[["FlightDate","Dest"] + [f"dest_{c}" for c in DAILY_WEATHER_VARS]])

    origin_df = pd.concat(origin_frames, ignore_index=True) if origin_frames else pd.DataFrame()
    dest_df   = pd.concat(dest_frames,   ignore_index=True) if dest_frames   else pd.DataFrame()
    return origin_df, dest_df

def add_origin_dest_weather_daily(df, airport_json, cache_dir="data/weather_cache"):
    coords_map = _coerce_airport_coords_with_tz(airport_json)
    ow, dw = _get_weather_for_airports_daily(df, coords_map, cache_dir=cache_dir)
    if not ow.empty:
        df = df.merge(ow, on=["FlightDate","Origin"], how="left")
    if not dw.empty:
        df = df.merge(dw, on=["FlightDate","Dest"],   how="left")
    return df

# ----------------- weather (hourly) aligned to local dep/arr hours -----------------
HOURLY_VARS = [
    "temperature_2m",      # °C
    "windspeed_10m",       # km/h
    "windgusts_10m",       # km/h
    "precipitation",       # mm
    "weathercode"          # WMO code (int)
]

def _month_chunks(start_dt, end_dt):
    s = pd.Timestamp(start_dt).normalize()
    e = pd.Timestamp(end_dt).normalize()
    cur = pd.Timestamp(s.replace(day=1))
    out = []
    while cur <= e:
        month_start = cur
        month_end   = (cur + pd.offsets.MonthEnd(0)).normalize()
        out.append((
            max(s, month_start).strftime("%Y-%m-%d"),
            min(e, month_end).strftime("%Y-%m-%d"),
        ))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out

def _req_with_backoff(url, params, max_tries=5):
    import requests
    backoff = 1.5
    delay = 1.0
    for i in range(max_tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            _time.sleep(delay)
            delay = min(60, delay * backoff)
            continue
        r.raise_for_status()
    r.raise_for_status()

def _fetch_hourly_weather(lat, lon, start, end, tz):
    """Fetch one chunk of hourly series from Open-Meteo archive."""
    if tz is None:
        raise RuntimeError("Hourly weather fetch requires per-airport timezone (tz is None).")
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _req_with_backoff(url, {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": tz
    })
    d = r.json().get("hourly", {})
    if not d:
        cols = ["time"] + HOURLY_VARS
        return pd.DataFrame(columns=cols).assign(time_local=pd.to_datetime([])).drop(columns=["time"])
    w = pd.DataFrame(d)
    # Open-Meteo returns "time" in local clock per tz param.
    w["time_local"] = pd.to_datetime(w["time"])
    return w.drop(columns=["time"])

def _fetch_hourly_weather_all(lat, lon, start_dt, end_dt, tz):
    """Fetch across [start_dt, end_dt] using monthly chunks, cache per chunk, concat."""
    frames = []
    start_s, end_s = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    safe_tz = tz.replace("/", "__")
    cache_root = Path("../../data/weather_cache/hourly")
    cache_root.mkdir(parents=True, exist_ok=True)

    for s, e in _month_chunks(start_dt, end_dt):
        cache_path = cache_root / f"{lat:.4f}_{lon:.4f}_{s}_{e}_{safe_tz}.parquet"
        if cache_path.exists():
            w = pd.read_parquet(cache_path)
        else:
            w = _fetch_hourly_weather(lat, lon, s, e, tz=tz)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            w.to_parquet(cache_path, index=False)
        if not w.empty:
            frames.append(w)

    if not frames:
        return pd.DataFrame(columns=["time_local"] + HOURLY_VARS)
    wx = pd.concat(frames, ignore_index=True)
    wx = wx.drop_duplicates(subset=["time_local"])
    mask = (wx["time_local"] >= pd.to_datetime(start_s)) & (wx["time_local"] <= pd.to_datetime(end_s) + pd.Timedelta(days=1))
    return wx.loc[mask].reset_index(drop=True)

def _get_dep_arr_time_bounds(df: pd.DataFrame):
    # Use dep/arr local directly; they are tz-aware datetimes
    dep_vals = df["dep_dt_local"].dropna()
    arr_vals = df["arr_dt_local"].dropna()
    if dep_vals.empty and arr_vals.empty:
        # fallback to FlightDate
        s = pd.to_datetime(df["FlightDate"].min())
        e = pd.to_datetime(df["FlightDate"].max())
        return (s.normalize() - pd.Timedelta(days=1)), (e.normalize() + pd.Timedelta(days=1))
    # Min/max in local tz; then just take naive dates for bounds
    dep_min = min(dep_vals)
    arr_max = max(arr_vals)
    start = pd.Timestamp(dep_min.date())
    end   = pd.Timestamp(arr_max.date())
    return (start - pd.Timedelta(days=1)), (end + pd.Timedelta(days=1))

def _build_dest_lagged_features(dw: pd.DataFrame, cfg: dict):
    """
    Build lagged-window features at destination:
      - any/max weather code >3 in prev {1h,2h,4h}
      - precip sum prev {1h,3h,4h}
      - wind/gust max prev {1h,3h,4h}
      - minutes_since last wx>3 within 6h  (FIXED: numpy timedelta division to minutes)
      - composite 'stressed' flag + binned minutes-since
    """
    w = dw.copy().sort_values(["Dest","time_local"]).reset_index(drop=True)

    def roll_back_any_gt3(ts, codes, win):
        out = np.zeros(len(ts), dtype="int8")
        j = 0
        for i in range(len(ts)):
            t_left = ts[i] - win
            while j < i and ts[j] < t_left:
                j += 1
            out[i] = 1 if (codes[j:i] > 3).any() else 0
        return out

    def roll_back_max(ts, arr, win):
        out = np.full(len(ts), np.nan, dtype=float)
        j = 0
        for i in range(len(ts)):
            t_left = ts[i] - win
            while j < i and ts[j] < t_left:
                j += 1
            if i > j:
                out[i] = np.nanmax(arr[j:i])
        return out

    def roll_back_sum(ts, arr, win):
        out = np.full(len(ts), np.nan, dtype=float)
        j = 0
        for i in range(len(ts)):
            t_left = ts[i] - win
            while j < i and ts[j] < t_left:
                j += 1
            if i > j:
                out[i] = np.nansum(arr[j:i])
        return out

    frames = []
    for dest, g in w.groupby("Dest", sort=False):
        ts   = pd.to_datetime(g["time_local"]).to_numpy()  # numpy datetime64[ns]
        code = pd.to_numeric(g["dest_arr_weathercode"], errors="coerce").to_numpy()
        prcp = pd.to_numeric(g["dest_arr_precipitation"], errors="coerce").to_numpy()
        wspd = pd.to_numeric(g["dest_arr_windspeed_10m"], errors="coerce").to_numpy()
        gust = pd.to_numeric(g["dest_arr_windgusts_10m"], errors="coerce").to_numpy()

        out = g.copy()

        # windows
        one_h = np.timedelta64(1, 'h')
        two_h = np.timedelta64(2, 'h')
        three_h = np.timedelta64(3, 'h')
        four_h = np.timedelta64(4, 'h')
        six_h = np.timedelta64(6, 'h')

        # any wx>3 prev windows
        out["dest_arr_wx_any_gt3_prev_1h"] = roll_back_any_gt3(ts, code, one_h)
        out["dest_arr_wx_any_gt3_prev_2h"] = roll_back_any_gt3(ts, code, two_h)
        out["dest_arr_wx_any_gt3_prev_4h"] = roll_back_any_gt3(ts, code, four_h)

        # max code prev windows
        out["dest_arr_wx_max_code_prev_1h"] = roll_back_max(ts, code, one_h)
        out["dest_arr_wx_max_code_prev_2h"] = roll_back_max(ts, code, two_h)
        out["dest_arr_wx_max_code_prev_4h"] = roll_back_max(ts, code, four_h)

        # precip sum prev windows
        out["dest_arr_precip_sum_prev_1h"] = roll_back_sum(ts, prcp, one_h)
        out["dest_arr_precip_sum_prev_3h"] = roll_back_sum(ts, prcp, three_h)
        out["dest_arr_precip_sum_prev_4h"] = roll_back_sum(ts, prcp, four_h)

        # wind/gust max prev windows
        out["dest_arr_wind_max_prev_1h"] = roll_back_max(ts, wspd, one_h)
        out["dest_arr_wind_max_prev_3h"] = roll_back_max(ts, wspd, three_h)
        out["dest_arr_wind_max_prev_4h"] = roll_back_max(ts, wspd, four_h)

        out["dest_arr_gust_max_prev_1h"] = roll_back_max(ts, gust, one_h)
        out["dest_arr_gust_max_prev_3h"] = roll_back_max(ts, gust, three_h)
        out["dest_arr_gust_max_prev_4h"] = roll_back_max(ts, gust, four_h)

        # minutes since last wx>3 within 6h  (numpy timedelta64 -> minutes)
        last_hit_idx = -np.ones(len(ts), dtype=int)
        last = -1
        for i in range(len(ts)):
            if code[i] > 3:
                last = i
            last_hit_idx[i] = last

        minutes_since = np.full(len(ts), np.nan, dtype=float)
        for i in range(len(ts)):
            j = last_hit_idx[i]
            if j >= 0 and (ts[i] - ts[j]) <= six_h:
                minutes_since[i] = (ts[i] - ts[j]) / np.timedelta64(1, 'm')  # <-- Py 3.9-safe
        out["dest_arr_minutes_since_wx_gt3_6h"] = minutes_since

        frames.append(out)

    z = pd.concat(frames, ignore_index=True)

    # Composite “stressed” flag
    z["dest_arr_wx_stressed"] = (
        (z["dest_arr_wx_any_gt3_prev_2h"].fillna(0).astype(int) > 0)
        | (z["dest_arr_precip_sum_prev_3h"].fillna(0) > 0.5)
        | (z["dest_arr_gust_max_prev_3h"].fillna(0) > 35)
    ).astype("int8")

    # Bins for minutes-since
    z["dest_arr_minutes_since_wx_gt3_6h_bin"] = pd.cut(
        z["dest_arr_minutes_since_wx_gt3_6h"],
        bins=[-1, 30, 90, 180, 1e12],
        labels=["<=30","30-90","90-180",">180"]
    ).astype("object")

    return z

def _get_weather_for_airports_hourly(df, coords_map, cache_dir="data/weather_cache"):
    start_dt, end_dt = _get_dep_arr_time_bounds(df)

    needed = pd.unique(pd.concat([df["Origin"], df["Dest"]]).astype(str).str.upper())
    per_origin, per_dest = {}, {}

    for ap in needed:
        lat, lon, tz = coords_map[ap]  # strict: tz must exist
        wx = _fetch_hourly_weather_all(lat, lon, start_dt, end_dt, tz=tz)

        ow = wx.copy()
        for c in HOURLY_VARS:
            ow.rename(columns={c: f"origin_dep_{c}"}, inplace=True)
        ow["Origin"] = ap
        per_origin[ap] = ow[["time_local","Origin"] + [f"origin_dep_{c}" for c in HOURLY_VARS]]

        dw = wx.copy()
        for c in HOURLY_VARS:
            dw.rename(columns={c: f"dest_arr_{c}"}, inplace=True)
        dw["Dest"] = ap

        # --- build lagged dest features here ---
        dw_aug = _build_dest_lagged_features(dw, cfg={})
        per_dest[ap] = dw_aug[["time_local","Dest"] + [c for c in dw_aug.columns if c.startswith("dest_arr_")]]

    origin_df = pd.concat(per_origin.values(), ignore_index=True) if per_origin else pd.DataFrame()
    dest_df   = pd.concat(per_dest.values(),   ignore_index=True) if per_dest   else pd.DataFrame()
    return origin_df, dest_df

def add_origin_dest_weather_hourly(df, airport_json, cache_dir="data/weather_cache"):
    """
    Align hourly weather to:
      - Origin departure hour (dep_dt_local floored to hour)
      - Dest arrival hour (arr_dt_local floored to hour)
    Uses each airport's own time zone for the hourly series (strict).
    """
    coords_map = _coerce_airport_coords_with_tz(airport_json)
    ow, dw = _get_weather_for_airports_hourly(df, coords_map, cache_dir=cache_dir)
    if ow.empty and dw.empty:
        print("[WARN] Hourly weather frames are empty; skipping hourly merge.")
        return df

    df = df.copy()

    def _floor_hour(dt):
        if not isinstance(dt, datetime):
            return pd.NaT
        return dt.replace(minute=0, second=0, microsecond=0)

    df["dep_hour_local"] = df["dep_dt_local"].apply(_floor_hour)
    df["arr_hour_local"] = df["arr_dt_local"].apply(_floor_hour)

    def _naive(dt):
        if not isinstance(dt, datetime):
            return pd.NaT
        return dt.replace(tzinfo=None)

    df["dep_hour_local_naive"] = df["dep_hour_local"].apply(_naive)
    df["arr_hour_local_naive"] = df["arr_hour_local"].apply(_naive)

    if not ow.empty:
        df = df.merge(
            ow.rename(columns={"time_local": "dep_hour_local_naive"}),
            on=["Origin","dep_hour_local_naive"],
            how="left"
        )
    if not dw.empty:
        df = df.merge(
            dw.rename(columns={"time_local": "arr_hour_local_naive"}),
            on=["Dest","arr_hour_local_naive"],
            how="left"
        )

    # Coverage logs so you can verify success
    for c in ["origin_dep_weathercode","dest_arr_weathercode"]:
        if c in df.columns:
            rate = df[c].notna().mean()
            print(f"[INFO] {c} non-null rate: {rate:.3f}")

    return df

# ----------------- label balancing -----------------
def balance_on_time_delayed(df, seed=42, max_rows=None):
    """
    Build a clean binary label from ArrDel15 and/or ArrDelay>=15.
    Then downsample to 1:1.
    """
    df = df.copy()
    df["ArrDel15"] = pd.to_numeric(df.get("ArrDel15"), errors="coerce")
    df["ArrDelay"] = pd.to_numeric(df.get("ArrDelay"), errors="coerce")

    if df["ArrDel15"].notna().any():
        label = df["ArrDel15"].astype("Float64")
        fallback = (df["ArrDelay"] >= 15).astype("Float64")
        label = label.where(label.notna(), fallback)
    else:
        if "ArrDelay" not in df.columns:
            raise RuntimeError("Neither ArrDel15 nor ArrDelay available to derive labels.")
        label = (df["ArrDelay"] >= 15).astype("Float64")

    before = len(df)
    mask_labeled = label.notna()
    if mask_labeled.sum() == 0:
        raise RuntimeError("No labeled rows after combining ArrDel15 and ArrDelay.")
    if mask_labeled.sum() < before:
        print(f"[WARN] Dropping {before - mask_labeled.sum()} rows with no label (missing ArrDel15 and ArrDelay).")

    df = df.loc[mask_labeled].copy()
    y = label.loc[mask_labeled].astype(int)

    on_time = df[y == 0]
    delayed = df[y == 1]
    if len(on_time) == 0 or len(delayed) == 0:
        print("[WARN] Cannot balance: one class empty after labeling.")
        return df

    n = min(len(on_time), len(delayed))
    on_time_s = on_time.sample(n=n, random_state=seed)
    delayed_s = delayed.sample(n=n, random_state=seed)

    balanced = pd.concat([on_time_s, delayed_s], ignore_index=True)
    if max_rows and len(balanced) > max_rows:
        balanced = balanced.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Balanced to 1:1 -> on_time={n}, delayed={n}, total={len(balanced)}")
    return balanced

def add_aircraft_age(df, reg_csv="data/aircraft_registry_clean.csv"):
    reg = pd.read_csv(reg_csv, dtype={"Tail_Number": str, "Year_Mfr": float})
    df = df.merge(reg, on="Tail_Number", how="left")
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    mask = df["Year_Mfr"].notna() & df["FlightDate"].notna()
    df.loc[mask, "Aircraft_Age_Years"] = df.loc[mask, "FlightDate"].dt.year - df.loc[mask, "Year_Mfr"]

    def bucket(y):
        if pd.isna(y): return "Unknown"
        if y <= 5:     return "0-5"
        if y <= 10:    return "6-10"
        if y <= 15:    return "11-15"
        return "16+"

    df["Aircraft_Age_Bucket"] = df["Aircraft_Age_Years"].apply(bucket)
    return df

# ----------------- delay reason filters & helpers -----------------
def apply_delay_reason_filters(df, cfg):
    """
    Exclude flights where the delay was driven by specific reasons.
      config:
        "exclude_delay_reasons": ["late_aircraft","security","carrier"]  # options
        "delay_threshold_minutes": 15
    Logic: if ArrDelayMinutes >= threshold and that reason column > 0, drop the row.
    """
    reasons = set((cfg.get("exclude_delay_reasons") or []))
    if not reasons:
        return df

    df = df.copy()
    for col in ["ArrDelayMinutes","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    thr = int(cfg.get("delay_threshold_minutes", 15))

    def drop_if(reason_name, reason_col):
        if reason_name in reasons and reason_col in df.columns:
            mask = (df["ArrDelayMinutes"].fillna(0) >= thr) & (df[reason_col].fillna(0) > 0)
            n = int(mask.sum())
            df.drop(df[mask].index, inplace=True)
            print(f"[INFO] Excluded {n} flights due to reason '{reason_name}' (col='{reason_col}').")

    drop_if("late_aircraft", "LateAircraftDelay")
    drop_if("security", "SecurityDelay")
    drop_if("carrier", "CarrierDelay")
    # add more if desired: drop_if("nas","NASDelay")

    return df

def keep_only_weather_codes(df, cfg):
    """
    If config.enable_weather_codes_only == true, drop numeric weather features and keep only codes:
      - origin_weathercode, dest_weathercode (daily)
      - origin_dep_weathercode, dest_arr_weathercode (hourly)
    """
    if not cfg.get("enable_weather_codes_only"):
        return df

    keep_cols = set([
        "origin_weathercode", "dest_weathercode",
        "origin_dep_weathercode", "dest_arr_weathercode"
    ])
    weather_cols = [c for c in df.columns if c.startswith("origin_") or c.startswith("dest_")]
    drop_cols = [c for c in weather_cols if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
        print(f"[INFO] Dropped {len(drop_cols)} weather columns; keeping only weather codes.")
    return df

def filter_first_flight_of_day(df, enable=False):
    """
    Keep only the first scheduled departure of the local day (by carrier & origin).
    Uses dep_dt_local (so run after add_timezone_local_times()).
    """
    if not enable:
        return df
    df = df.copy()
    if "dep_dt_local" not in df.columns or "dep_local_date" not in df.columns:
        return df

    df = df.sort_values(["Reporting_Airline","Origin","dep_dt_local"])
    df["rank_local"] = (
        df.groupby(["Reporting_Airline","Origin","dep_local_date"])
          .cumcount()
          .astype("Int64")
    )
    keep = df["rank_local"] == 0
    kept = int(keep.sum())
    print(f"[INFO] First-flight-of-day filter enabled: keeping {kept} rows.")
    return df.loc[keep].drop(columns=["rank_local"])

# ----------------- main -----------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python build_training_set.py path/to/config.json")
        sys.exit(1)

    cfg_path = _resolve_path(sys.argv[1])
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {sys.argv[1]}")
        for t in _resolve_candidates(sys.argv[1])[:4]:
            print(f"[WARN] tried: {t}")
        sys.exit(2)

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Load raw flights (single path or list) and canonicalize columns
    df = read_input_parquet(cfg["input_flights_path"])
    df = canonicalize_bts_columns(df)
    assert_required(df, REQUIRED_COLS)

    # Ensure numeric early
    for c in ["ArrDel15", "ArrDelay", "ArrDelayMinutes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional filters
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        df["Origin"] = df["Origin"].astype(str).str.upper()
        df["Dest"]   = df["Dest"].astype(str).str.upper()
        df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports)]

    single_airline = cfg.get("single_airline")
    if single_airline:
        df = df[df["Reporting_Airline"] == single_airline]
    elif cfg.get("airlines"):
        df = df[df["Reporting_Airline"].isin(cfg["airlines"])]

    if cfg.get("start_date") and cfg.get("end_date"):
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])
        mask = (df["FlightDate"] >= pd.to_datetime(cfg["start_date"])) & \
               (df["FlightDate"] <= pd.to_datetime(cfg["end_date"]))
        df = df[mask]

    if df.empty:
        print("[ERROR] No rows after filtering.")
        sys.exit(2)

    # Exclude selected delay reasons
    df = apply_delay_reason_filters(df, cfg)

    # ---------- Time zones & local times (STRICT per-airport tz) ----------
    airport_json_raw = cfg.get("airport_json", "data/airport_coords.json")
    airport_json = "data/airport_coords.json"
    print(f"[INFO] airport_json -> {airport_json}  (exists={Path(airport_json).exists()})")

    # pre-check CRS time NAs
    for col in ["CRSDepTime","CRSArrTime"]:
        if col in df.columns:
            na_rate = pd.to_numeric(df[col], errors="coerce").isna().mean()
            print(f"[INFO] {col} NA rate: {na_rate:.3f}")

    df = add_timezone_local_times(df, airport_json=airport_json)

    # Optional: keep only first flight of day per carrier-origin
    df = filter_first_flight_of_day(df, enable=cfg.get("first_flight_of_day_only", False))

    # ---------- Congestion ----------
    df = add_simple_congestion(df)
    df = add_taxi_congestion_rolling(
        df,
        window_days=cfg.get("taxi_window_days", 7),
        bin_minutes=cfg.get("taxi_bin_minutes", 30)
    )

    # ---------- Aircraft age ----------
    if cfg.get("add_aircraft_age", True):
        reg_csv_raw = cfg.get("aircraft_registry_csv", "data/aircraft_registry_clean.csv")
        reg_csv = str(_resolve_path(reg_csv_raw))
        print(f"[INFO] aircraft_registry_csv -> {reg_csv}  (exists={Path(reg_csv).exists()})")
        df = add_aircraft_age(df, reg_csv=reg_csv)

    # ---------- Enriched history baselines ----------
    df = add_history_baselines(
        df,
        carrier_days      = cfg.get("carrier_baseline_days", 7),
        od_days           = cfg.get("od_baseline_days", 7),
        flightnum_days    = cfg.get("flightnum_baseline_days", 14),
        origin_days       = cfg.get("origin_baseline_days", 7),
        dest_days         = cfg.get("dest_baseline_days", 7),
    )

    # ---------- Weather (daily) ----------
    if cfg.get("add_weather", True):
        cache_dir = str(_resolve_path(cfg.get("weather_cache_dir", "data/weather_cache")))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df = add_origin_dest_weather_daily(df, airport_json=airport_json, cache_dir=cache_dir)

    # ---------- Weather (hourly) ----------
    if cfg.get("add_hourly_weather", True):
        cache_dir = str(_resolve_path(cfg.get("weather_cache_dir", "data/weather_cache")))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df = add_origin_dest_weather_hourly(df, airport_json=airport_json, cache_dir=cache_dir)

    # Option: keep only weather codes (drop numeric weather)
    df = keep_only_weather_codes(df, cfg)

    # ---------- Optional balancing ----------
    if cfg.get("balance_on_time_delayed"):
        df = balance_on_time_delayed(
            df,
            seed=cfg.get("balance_seed", 42),
            max_rows=cfg.get("balance_max_rows")
        )

    # ---------- Optional final downsampling ----------
    df = downsample_rows(df, cfg)

    # ---------- Output ----------
    out_path_cfg = cfg.get("output_training_path", "data/processed/train_ready.parquet")
    out_path = _resolve_output_path(out_path_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
