#!/usr/bin/env python3
"""
Build "inbound bank" priors from raw BTS arrivals.

Reads monthly files at: <bts_root>/Year=YYYY/Month=MM/flights.parquet
and outputs a parquet keyed by:
  (Reporting_Airline, Dest, Year, Month, DOW, arr_hour_local)

with:
  sample_size, p_ge15, p_ge30, p_ge45, p_ge60, window_start, window_end

Usage (from repo root):
  python src/features/build_inbound_priors.py --config config/inbound_config.json --verbose

Config JSON example:
{
  "bts_root": "data/raw_bts",
  "airports": ["JFK","LAX","ATL", "..."],
  "airlines": ["AA"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-30",
  "output_parquet": "models/priors/inbound_priors.parquet"
}
"""

from __future__ import annotations
import argparse, json, calendar, re
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

# Keep it simple: run from repo root; bts_root is used as-is (no auto-resolve).
# We handle common BTS schema variants and provide verbose diagnostics.

# Airline name -> IATA code helpers (extend as needed)
NAME_TO_IATA = {
    "American Airlines Inc.": "AA",
    "American Airlines": "AA",
    "Delta Air Lines Inc.": "DL",
    "Delta Air Lines": "DL",
    "United Air Lines Inc.": "UA",
    "United Air Lines": "UA",
    "Southwest Airlines Co.": "WN",
    "Southwest Airlines": "WN",
    "Alaska Airlines Inc.": "AS",
    "Alaska Airlines": "AS",
    "JetBlue Airways": "B6",
    "JetBlue Airways Corporation": "B6",
    "Spirit Air Lines": "NK",
    "Spirit Airlines": "NK",
    "Frontier Airlines Inc.": "F9",
    "Frontier Airlines": "F9",
    "Allegiant Air": "G4",
    "Hawaiian Airlines Inc.": "HA",
    "Hawaiian Airlines": "HA",
}

# ICAO (3-letter) -> IATA (2-letter)
ICAO_TO_IATA = {
    "AAL": "AA",  # American
    "DAL": "DL",  # Delta
    "UAL": "UA",  # United
    "SWA": "WN",  # Southwest
    "ASA": "AS",  # Alaska
    "JBU": "B6",  # JetBlue
    "NKS": "NK",  # Spirit
    "FFT": "F9",  # Frontier
    "AAY": "G4",  # Allegiant
    "HAL": "HA",  # Hawaiian
}

# DOT carrier ID -> IATA (common set; extend if needed)
DOTID_TO_IATA = {
    19805: "AA",  # American
    19790: "DL",  # Delta
    19977: "UA",  # United
    19393: "WN",  # Southwest
    19930: "AS",  # Alaska
    20409: "B6",  # JetBlue
    20436: "NK",  # Spirit
    20416: "F9",  # Frontier
    20368: "G4",  # Allegiant
    20418: "HA",  # Hawaiian
}

# Likely airline columns across BTS variants (strings)
AIRLINE_CODE_CANDIDATES_STR = [
    "IATA_CODE_Reporting_Airline",
    "Reporting_Airline",
    "IATA_CODE_Operating_Airline",
    "IATA_CODE_Marketing_Airline",
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Marketing_Airline",
    "Mkt_Carrier",
    "Op_Carrier",
    "MktCarrier",
    "OpCarrier",
    "OP_UNIQUE_CARRIER",
    "UNIQUE_CARRIER",
    "Carrier",
    "CARRIER",
    "Airline",
]

# Likely airline numeric ID columns (DOT/Carrier IDs)
AIRLINE_CODE_CANDIDATES_NUM = [
    "DOT_ID_Reporting_Airline",
    "DOT_ID_Operating_Airline",
    "DOT_ID_Marketing_Airline",
    "Mkt_Carrier_Airline_ID",
    "Op_Carrier_Airline_ID",
    "CARRIER_ID",
    "AirlineID",
    "OP_CARRIER_AIRLINE_ID",
    "MKT_CARRIER_AIRLINE_ID",
]

# Prefer to keep these columns if they exist
KEEP_COLS = [
    "FlightDate", "Dest", "CRSArrTime",
    "ArrDelayMinutes", "ArrDelay", "ArrDel15", "ArrivalDelayGroups",
    "Cancelled", "Diverted",
] + AIRLINE_CODE_CANDIDATES_STR + AIRLINE_CODE_CANDIDATES_NUM

CODE2_RE = re.compile(r"^[A-Za-z0-9]{2}$")
CODE3_RE = re.compile(r"^[A-Za-z0-9]{3}$")

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _month_in_range(y: int, m: int, start: date, end: date) -> bool:
    first = date(y, m, 1)
    last = date(y, m, calendar.monthrange(y, m)[1])
    return not (last < start or first > end)

def _hhmm_to_hour(x) -> float:
    try:
        s = str(int(x)).zfill(4)
        return int(s[:2])
    except Exception:
        return np.nan

def _dbg(msg: str, df: pd.DataFrame, verbose: bool):
    if verbose:
        print(f"[DBG] {msg}: rows={len(df)}")

def _pick_best_string_code(df: pd.DataFrame, verbose: bool=False) -> pd.Series | None:
    """Try to derive two-letter IATA codes from any string-like airline column."""
    present = [c for c in AIRLINE_CODE_CANDIDATES_STR if c in df.columns]
    if verbose:
        print("[DBG] airline string columns present:", present)
    for col in present:
        s = df[col].astype(str).str.strip()
        if s.empty:
            continue
        mask2 = s.str.match(CODE2_RE)
        mask3 = s.str.match(CODE3_RE)
        share2 = float(mask2.mean()) if len(s) else 0.0
        share3 = float(mask3.mean()) if len(s) else 0.0
        out = pd.Series(pd.NA, index=s.index, dtype="object")
        if share2 >= 0.5:
            out.loc[mask2] = s.loc[mask2].str.upper()
            return out
        if share3 >= 0.5:
            out.loc[mask3] = s.loc[mask3].str.upper().map(ICAO_TO_IATA).fillna(pd.NA)
            return out
        mapped = s.map(NAME_TO_IATA)
        if mapped.notna().mean() >= 0.5:
            return mapped
    return None

def _pick_best_numeric_code(df: pd.DataFrame, verbose: bool=False) -> pd.Series | None:
    """Try to derive two-letter IATA codes from numeric DOT/Carrier ID columns."""
    present = [c for c in AIRLINE_CODE_CANDIDATES_NUM if c in df.columns]
    if verbose:
        print("[DBG] airline numeric columns present:", present)
    for col in present:
        vals = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        mapped = vals.map(DOTID_TO_IATA)
        if mapped.notna().mean() >= 0.5:
            return mapped
    return None

def _canonical_airline_code(df: pd.DataFrame, verbose: bool=False) -> pd.Series:
    """
    Return a Series 'AirlineCode' with two-letter IATA codes when possible.
    Priority:
      1) string columns that look like IATA/ICAO or names
      2) numeric DOT/Carrier IDs
    Fall back to NA if nothing matches.
    """
    s = _pick_best_string_code(df, verbose=verbose)
    if s is not None:
        return s
    s = _pick_best_numeric_code(df, verbose=verbose)
    if s is not None:
        return s
    return pd.Series(pd.NA, index=df.index, dtype="object")

def _load_month(
    parquet_path: Path,
    start: date,
    end: date,
    airlines: list[str] | None,
    airports: list[str] | None,
    verbose: bool = False,
) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    _dbg("read", df, verbose)
    if verbose:
        print("[DBG] columns:", list(df.columns))

    # Date filter
    if "FlightDate" not in df.columns:
        return pd.DataFrame()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df = df[(df["FlightDate"].dt.date >= start) & (df["FlightDate"].dt.date <= end)]
    _dbg("after date range", df, verbose)

    # Airline filter (by canonical IATA code)
    df["AirlineCode"] = _canonical_airline_code(df, verbose=verbose)
    if verbose:
        vc = df["AirlineCode"].value_counts(dropna=False).head(10)
        print("[DBG] AirlineCode top10:", dict(vc))
    if airlines:
        want = [a.strip().upper() for a in airlines]
        df = df[df["AirlineCode"].isin(want)]
        _dbg(f"after airline (AirlineCode in {want})", df, verbose)

    # Airport filter (arrivals into Dest)
    if airports:
        if "Dest" not in df.columns:
            return pd.DataFrame()
        df = df[df["Dest"].isin(airports)]
        _dbg("after airports filter", df, verbose)

    # Completed arrivals only
    for flag in ("Cancelled", "Diverted"):
        if flag in df.columns:
            df = df[df[flag].fillna(0) == 0]
    _dbg("after cancel/divert filter", df, verbose)

    if df.empty:
        return df

    # Calendar keys
    df["Year"] = df["FlightDate"].dt.year.astype("Int64")
    df["Month"] = df["FlightDate"].dt.month.astype("Int64")
    df["DOW"] = df["FlightDate"].dt.dayofweek.astype("Int64")  # Mon=0..Sun=6
    df["arr_hour_local"] = (
        df["CRSArrTime"].apply(_hhmm_to_hour).astype("Int64")
        if "CRSArrTime" in df.columns else pd.Series([pd.NA] * len(df), dtype="Int64")
    )

    # Tail indicators — prefer minutes; fallback to groups/ArrDel15
    have_minutes = False
    if "ArrDelayMinutes" in df.columns:
        df["ArrDelayMinutes"] = pd.to_numeric(df["ArrDelayMinutes"], errors="coerce")
        have_minutes = True
    elif "ArrDelay" in df.columns:
        df["ArrDelayMinutes"] = pd.to_numeric(df["ArrDelay"], errors="coerce")
        have_minutes = True
    else:
        df["ArrDelayMinutes"] = np.nan

    if have_minutes:
        df["_ge15"] = (df["ArrDelayMinutes"] >= 15).astype("int32")
        df["_ge30"] = (df["ArrDelayMinutes"] >= 30).astype("int32")
        df["_ge45"] = (df["ArrDelayMinutes"] >= 45).astype("int32")
        df["_ge60"] = (df["ArrDelayMinutes"] >= 60).astype("int32")
    else:
        if "ArrivalDelayGroups" in df.columns:
            g = pd.to_numeric(df["ArrivalDelayGroups"], errors="coerce").fillna(-1).astype(int)
            df["_ge15"] = (g >= 1).astype("int32")
            df["_ge30"] = (g >= 2).astype("int32")
            df["_ge45"] = (g >= 3).astype("int32")
            df["_ge60"] = (g >= 4).astype("int32")
        elif "ArrDel15" in df.columns:
            a15 = pd.to_numeric(df["ArrDel15"], errors="coerce").fillna(0).astype(int)
            df["_ge15"] = a15
            df["_ge30"] = 0
            df["_ge45"] = 0
            df["_ge60"] = 0
        else:
            return pd.DataFrame()

    # Keys we need  (NOTE: includes Year now)
    key_cols = ["AirlineCode", "Dest", "Year", "Month", "DOW", "arr_hour_local"]
    df = df.dropna(subset=key_cols)
    if df.empty:
        return df

    # Aggregate this month
    df["_n"] = 1
    grp = (
        df.groupby(["AirlineCode","Dest","Year","Month","DOW","arr_hour_local"], as_index=False)
          .agg(sample_size=("_n","sum"),
               _ge15=("_ge15","sum"), _ge30=("_ge30","sum"),
               _ge45=("_ge45","sum"), _ge60=("_ge60","sum"))
    )
    _dbg("after groupby (month)", grp, verbose)
    return grp

def build_priors(config_path: Path, verbose: bool=False) -> pd.DataFrame:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    root = Path(cfg["bts_root"]).expanduser()   # relative to repo root (cwd)
    airports = cfg.get("airports") or []
    airlines = cfg.get("airlines") or []
    start = _parse_date(cfg["start_date"])
    end = _parse_date(cfg["end_date"])
    if verbose:
        print(f"[INFO] bts_root={root}")
        print(f"[INFO] window={start}..{end} airlines={airlines or 'ALL'} airports={airports or 'ALL'}")

    monthly_aggs = []
    y, m = start.year, start.month
    while True:
        if _month_in_range(y, m, start, end):
            p = root / f"Year={y:04d}" / f"Month={m:02d}" / "flights.parquet"
            if p.exists():
                print(f"[INFO] reading {p}")
                grp = _load_month(p, start, end, airlines, airports, verbose=verbose)
                if not grp.empty:
                    monthly_aggs.append(grp)
        # advance month
        if (y > end.year) or (y == end.year and m >= end.month):
            break
        m = 1 if m == 12 else m + 1
        y = y + 1 if m == 1 else y

    if not monthly_aggs:
        raise RuntimeError("No data found for the given config. Check bts_root, dates, airlines, airports.")

    agg = pd.concat(monthly_aggs, ignore_index=True)
    agg = (
        agg.groupby(["AirlineCode","Dest","Year","Month","DOW","arr_hour_local"], as_index=False)
           .sum(numeric_only=True)
    )
    agg["sample_size"] = agg["sample_size"].astype("int32")

    # Convert counts → probabilities
    for k, num in [("p_ge15","_ge15"), ("p_ge30","_ge30"), ("p_ge45","_ge45"), ("p_ge60","_ge60")]:
        agg[k] = (agg[num] / agg["sample_size"]).astype("float32")
        agg.drop(columns=[num], inplace=True)

    # Provenance & final shapes; rename AirlineCode→Reporting_Airline to match the inference loader's expected column name
    agg["window_start"] = pd.Timestamp(start).date().isoformat()
    agg["window_end"]   = pd.Timestamp(end).date().isoformat()
    agg = agg.rename(columns={"AirlineCode": "Reporting_Airline"})
    agg = agg.sort_values(["Reporting_Airline","Dest","Year","Month","DOW","arr_hour_local"]).reset_index(drop=True)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to inbound_config.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    out_path = Path(cfg.get("output_parquet", "../../models/priors/inbound_priors.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pri = build_priors(cfg_path, verbose=args.verbose)
    pri.to_parquet(out_path, index=False)
    print(f"[OK] wrote priors → {out_path}  rows={len(pri)}")

if __name__ == "__main__":
    main()

