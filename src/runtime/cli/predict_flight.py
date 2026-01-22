#!/usr/bin/env python3
# src/runtime/cli/predict_flight.py
"""
Example:
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  export AERODATABOX_API_KEY="..."     # if using AeroDataBox
  export OPENSKY_USERNAME="..."        # optional, if you use OpenSky for other ops
  export OPENSKY_PASSWORD="..."
  export NWS_USER_AGENT="flightright/0.1 (your_email@example.com)"

  python -m runtime.cli.predict_flight \
      --models-dir models/UA_weather_minus_historical \
      --flight UA1536 \
      --date 2026-01-22 \
      --prefer aerodatabox,opensky

"""
# src/runtime/cli/predict_flight.py
from __future__ import annotations
import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

# Schedules + Weather providers
from runtime.providers.flight_source import FlightInfoSource
from runtime.providers.schedules.aerodatabox import AeroDataBoxProvider  # noqa: F401 (import hints)
from runtime.providers.weather.nws import NWSProvider

# ----------------------------
# Constants / Config
# ----------------------------
THRESHOLDS = [15, 30, 45, 60]

# Hard-coded airports CSV (as requested)
AIRPORTS_CSV = "data/meta/airports.csv"

# Candidate feature names (keep in sync with training-time names)
CATS_CAND = [
    "Origin", "Dest", "Reporting_Airline", "OD_pair",
    "origin_weathercode", "dest_weathercode",
    "origin_dep_weathercode", "dest_arr_weathercode",
]
NUMS_CAND = [
    # daily (origin/dest)
    "origin_temperature_2m_max", "origin_temperature_2m_min", "origin_precipitation_sum",
    "origin_windspeed_10m_max", "origin_windgusts_10m_max",
    "dest_temperature_2m_max", "dest_temperature_2m_min", "dest_precipitation_sum",
    "dest_windspeed_10m_max", "dest_windgusts_10m_max",
    # hourly at dep/arr
    "origin_dep_temperature_2m", "origin_dep_windspeed_10m", "origin_dep_windgusts_10m", "origin_dep_precipitation",
    "dest_arr_temperature_2m", "dest_arr_windspeed_10m", "dest_arr_windgusts_10m", "dest_arr_precipitation",
]

# ----------------------------
# Small helpers
# ----------------------------
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(msg, flush=True)

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(pd.notna(s), "Unknown")
    def _fmt(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return "Unknown"
        return str(v)
    return s.map(_fmt)

def _hhmm_to_hour(hhmm: str) -> int:
    try:
        s = str(int(hhmm)).zfill(4)
        return int(s[:2])
    except Exception:
        return 0

def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def enforce_tail_monotone(df: pd.DataFrame, cols_low_to_high: List[str]) -> None:
    arr = df[cols_low_to_high].to_numpy(copy=True)
    arr = np.maximum.accumulate(arr, axis=1)
    df[cols_low_to_high] = arr

def explain_top_reasons(model: CatBoostClassifier, X_row: pd.DataFrame, cat_feats, num_feats, top_k=3):
    pool = Pool(X_row, cat_features=[X_row.columns.get_loc(c) for c in cat_feats])
    contribs = model.get_feature_importance(type="PredictionValuesChange", data=pool)
    pairs = list(zip(X_row.columns, contribs))
    top = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:top_k]

    phrases = []
    for name, val in top:
        direction = "higher risk" if val > 0 else "lower risk"
        if "weathercode" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "wind" in name or "gust" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "precip" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "temperature" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif name in ("Origin", "Dest", "OD_pair"):
            phrases.append(f"route {X_row[name].iloc[0]} → {direction}")
        elif name == "Reporting_Airline":
            phrases.append(f"{name} → {direction}")
        else:
            phrases.append(f"{name} → {direction}")
    return phrases

def _split_flight(s: str) -> Tuple[str, str]:
    # e.g. "UA1536" → ("UA","1536") ; "UA 1536" also ok
    m = re.match(r"^\s*([A-Za-z]{2})\s*0*([0-9]+)\s*$", s)
    if not m:
        raise ValueError(f"Unparseable flight code: {s}")
    return m.group(1).upper(), m.group(2)

def _load_airports_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize expected columns
    # expect headers: IATA, AirportName, City, Country, State, Latitude, Longitude, Timezone
    lower = {c.lower(): c for c in df.columns}
    need = ["iata", "latitude", "longitude", "timezone"]
    for key in need:
        if key not in lower:
            raise RuntimeError(f"airports.csv missing column like {key}")
    return df.rename(columns={lower["iata"]: "IATA",
                              lower["latitude"]: "Latitude",
                              lower["longitude"]: "Longitude",
                              lower["timezone"]: "Timezone"})

def _iata_row(df: pd.DataFrame, iata: str) -> pd.Series:
    r = df[df["IATA"].astype(str).str.upper() == iata.upper()]
    if r.empty:
        raise RuntimeError(f"IATA not found in airports.csv: {iata}")
    return r.iloc[0]

def _fmt(v, unit: str | None = None, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if unit:
        return f"{round(float(v), nd)} {unit}"
    return f"{round(float(v), nd)}"

def _print_weather_block(title: str, w: Dict[str, Any], dep_label: str = "dep"):
    # Daily
    tmax = w.get("temperature_2m_max")
    tmin = w.get("temperature_2m_min")
    psum = w.get("precipitation_sum", 0.0)
    ws   = w.get("windspeed_10m_max")
    wg   = w.get("windgusts_10m_max")
    wcode_day = w.get("wx_code_day", "Unknown")

    # Hourly @ dep/arr
    ht  = w.get("dep_temperature_2m")
    hw  = w.get("dep_windspeed_10m")
    hg  = w.get("dep_windgusts_10m")
    hp  = w.get("dep_precipitation", 0.0)
    wcode_hour = w.get("wx_code_hour", "Unknown")

    vfrom = w.get("valid_from_utc")
    vto   = w.get("valid_to_utc")
    avail = w.get("available")

    print(f"\n--- {title} Weather ({'available' if avail else 'unavailable'}) ---")
    if vfrom or vto:
        print(f"Window (UTC): {vfrom} → {vto}")
    print(f"Daily:  Tmax={_fmt(tmax,'°C')}  Tmin={_fmt(tmin,'°C')}  PrecipSum={_fmt(psum,'mm')}  "
          f"WindMax={_fmt(ws,'m/s')}  GustMax={_fmt(wg,'m/s')}  Code={wcode_day}")
    print(f"Hourly @{dep_label}:  T={_fmt(ht,'°C')}  Wind={_fmt(hw,'m/s')}  Gust={_fmt(hg,'m/s')}  "
          f"Precip={_fmt(hp,'mm')}  Code={wcode_hour}")

# ----------------------------
# Core CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict delay bins for a planned flight using trained ordinal models.")
    ap.add_argument("--models-dir", required=True, help="Directory containing ordinal models (with registry.json).")
    ap.add_argument("--flight", required=True, help="Flight like UA1536 or 'UA 1536'.")
    ap.add_argument("--date", required=True, help="Local departure date YYYY-MM-DD (origin local date).")
    ap.add_argument("--prefer", default="aerodatabox,opensky", help="Comma list of schedule providers preference.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(2)

    airline_iata, flight_number = _split_flight(args.flight)

    # ---------------- SCHEDULE LOOKUP ----------------
    prefer_order = [p.strip().lower() for p in args.prefer.split(",") if p.strip()]
    log(f"[INFO] Schedule providers preference: {prefer_order}")

    src = FlightInfoSource(prefer_order=prefer_order)
    planned = src.get_planned_flight(airline_iata, flight_number, args.date)
    if planned is None:
        print("[ERROR] Could not obtain planned schedule for the flight/date.", file=sys.stderr)
        sys.exit(2)

    log(f"[INFO] Planned: {planned}")

    # origin/dest + HHMM
    origin = planned.origin
    dest   = planned.dest
    dep_hhmm = planned.crs_dep_time or "0000"
    arr_hhmm = planned.crs_arr_time or "0000"
    dep_hour_local = _hhmm_to_hour(dep_hhmm)
    arr_hour_local = _hhmm_to_hour(arr_hhmm)

    # ---------------- WEATHER (NWS) ----------------
    # airports.csv → lat/lon
    ap_df = _load_airports_csv(AIRPORTS_CSV)

    orow = _iata_row(ap_df, origin)
    drow = _iata_row(ap_df, dest)

    o_lat, o_lon = float(orow["Latitude"]), float(orow["Longitude"])
    d_lat, d_lon = float(drow["Latitude"]), float(drow["Longitude"])
    log(f"[INFO] Origin {origin} lat/lon=({o_lat},{o_lon}) dep_hour_local={dep_hour_local}")
    log(f"[INFO] Dest   {dest}  lat/lon=({d_lat},{d_lon}) arr_hour_local={arr_hour_local}")

    nws = NWSProvider(user_agent=os.getenv("NWS_USER_AGENT", "flight-delay-model/0.1 (no-user-agent-set)"))

    w_or = nws.get_features(o_lat, o_lon, args.date, dep_hour_local)
    log(f"[INFO] NWS origin window: {w_or.get('valid_from_utc')} → {w_or.get('valid_to_utc')} (available={w_or.get('available')})")
    # For destination we use the provided date and ARR hour; adjust later if you want to auto-advance date for overnight arrivals.
    w_de = nws.get_features(d_lat, d_lon, args.date, arr_hour_local)
    log(f"[INFO] NWS dest   window: {w_de.get('valid_from_utc')} → {w_de.get('valid_to_utc')} (available={w_de.get('available')})")

    # >>> Weather Summary Prints (restored) <<<
    _print_weather_block(f"{origin} (Origin)", w_or, dep_label="DEP")
    _print_weather_block(f"{dest} (Destination)", w_de, dep_label="ARR")

    # ---------------- BUILD FEATURES ----------------
    feat = {}

    # Categorical
    feat["Origin"] = origin
    feat["Dest"] = dest
    feat["Reporting_Airline"] = airline_iata
    feat["OD_pair"] = f"{origin}_{dest}"

    feat["origin_weathercode"]     = w_or.get("wx_code_day") or "Unknown"
    feat["dest_weathercode"]       = w_de.get("wx_code_day") or "Unknown"
    feat["origin_dep_weathercode"] = w_or.get("wx_code_hour") or "Unknown"
    feat["dest_arr_weathercode"]   = w_de.get("wx_code_hour") or "Unknown"

    # Numeric daily (origin)
    feat["origin_temperature_2m_max"] = _safe_float(w_or.get("temperature_2m_max"))
    feat["origin_temperature_2m_min"] = _safe_float(w_or.get("temperature_2m_min"))
    feat["origin_precipitation_sum"]  = _safe_float(w_or.get("precipitation_sum"), 0.0)
    feat["origin_windspeed_10m_max"]  = _safe_float(w_or.get("windspeed_10m_max"))
    feat["origin_windgusts_10m_max"]  = _safe_float(w_or.get("windgusts_10m_max"))

    # Numeric daily (dest)
    feat["dest_temperature_2m_max"] = _safe_float(w_de.get("temperature_2m_max"))
    feat["dest_temperature_2m_min"] = _safe_float(w_de.get("temperature_2m_min"))
    feat["dest_precipitation_sum"]  = _safe_float(w_de.get("precipitation_sum"), 0.0)
    feat["dest_windspeed_10m_max"]  = _safe_float(w_de.get("windspeed_10m_max"))
    feat["dest_windgusts_10m_max"]  = _safe_float(w_de.get("windgusts_10m_max"))

    # Hourly at dep/arr
    feat["origin_dep_temperature_2m"] = _safe_float(w_or.get("dep_temperature_2m"))
    feat["origin_dep_windspeed_10m"]  = _safe_float(w_or.get("dep_windspeed_10m"))
    feat["origin_dep_windgusts_10m"]  = _safe_float(w_or.get("dep_windgusts_10m"))
    feat["origin_dep_precipitation"]  = _safe_float(w_or.get("dep_precipitation"), 0.0)

    feat["dest_arr_temperature_2m"] = _safe_float(w_de.get("dep_temperature_2m"))
    feat["dest_arr_windspeed_10m"]  = _safe_float(w_de.get("dep_windspeed_10m"))
    feat["dest_arr_windgusts_10m"]  = _safe_float(w_de.get("dep_windgusts_10m"))
    feat["dest_arr_precipitation"]  = _safe_float(w_de.get("dep_precipitation"), 0.0)

    X = pd.DataFrame([feat])

    # Cast types like training
    cat_feats_present = [c for c in CATS_CAND if c in X.columns]
    num_feats_present = [c for c in NUMS_CAND if c in X.columns]

    for c in cat_feats_present:
        X[c] = as_str(X[c])
    for c in num_feats_present:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(0.0)

    if args.verbose:
        print(f"[{_now()} +   0.00s] [prep] features ready: rows={len(X)}, cats={len(cat_feats_present)}, nums={len(num_feats_present)}")
        print("[DEBUG] First row features preview:")
        with pd.option_context("display.max_columns", None):
            print(X.iloc[0])

    # ---------------- MODELS: load registry + ensure feature order ----------------
    reg_path = models_dir / "registry.json"
    if not reg_path.exists():
        print(f"[ERROR] registry.json not found in {models_dir}", file=sys.stderr)
        sys.exit(2)
    with open(reg_path, "r") as f:
        registry = json.load(f)

    log(f"[INFO] Loading models from {models_dir}")

    meta_cat, meta_num = None, None
    try:
        meta_path = models_dir / "thr_15" / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta_cat = meta.get("categorical_features") or []
            meta_num = meta.get("numeric_features") or []
            log(f"[DEBUG] Using trained feature order: cats={len(meta_cat)} nums={len(meta_num)}")
    except Exception:
        pass

    if meta_cat is not None and meta_num is not None:
        for c in meta_cat:
            if c not in X.columns:
                X[c] = "Unknown"
        for c in meta_num:
            if c not in X.columns:
                X[c] = 0.0
        X = X[meta_cat + meta_num]
        cat_feats_use = meta_cat
        num_feats_use = meta_num
    else:
        X = X[cat_feats_present + num_feats_present]
        cat_feats_use = cat_feats_present
        num_feats_use = num_feats_present

    # ---------------- Load calibrated models + raw for explanations ----------------
    calibrators: Dict[int, Any] = {}
    raw_models: Dict[int, CatBoostClassifier] = {}

    for thr in THRESHOLDS:
        info = registry.get(str(thr))
        if not info:
            print(f"[ERROR] Threshold {thr} missing in registry.json", file=sys.stderr)
            sys.exit(2)
        cal = joblib.load(info["cal_path"])
        calibrators[thr] = cal

        m = CatBoostClassifier()
        m.load_model(info["model_path"])
        raw_models[thr] = m

    # Predict calibrated probabilities P(delay >= thr)
    proba = {}
    for thr in THRESHOLDS:
        cal = calibrators[thr]
        p = cal.predict_proba(X)[:, 1]
        proba[thr] = float(np.clip(p[0], 0.0, 1.0))  # scalarize

    out = pd.DataFrame(index=[0])
    out["p_ge_15"] = proba[15]
    out["p_ge_30"] = proba[30]
    out["p_ge_45"] = proba[45]
    out["p_ge_60"] = proba[60]

    # Enforce monotone tails (left→right: 60,45,30,15)
    enforce_tail_monotone(out, ["p_ge_60","p_ge_45","p_ge_30","p_ge_15"])
    out["p_on_time"] = 1.0 - out["p_ge_15"]

    # Percentages for display
    out["on_time_pct"] = (100 * out["p_on_time"]).round(1)
    out["ge15_pct"] = (100 * out["p_ge_15"]).round(1)
    out["ge30_pct"] = (100 * out["p_ge_30"]).round(1)
    out["ge45_pct"] = (100 * out["p_ge_45"]).round(1)
    out["ge60_pct"] = (100 * out["p_ge_60"]).round(1)

    # Explain (≥15 model)
    phrases = explain_top_reasons(raw_models[15], X, cat_feats_use, num_feats_use, top_k=3)
    why = "; ".join(phrases)

    # ---------------- Output ----------------
    print("\n=== Delay Risk (percent) ===")
    r0 = out.iloc[0]
    print(f"On-time (0–14m): {r0.on_time_pct:.1f}%")
    print(f"≥15m: {r0.ge15_pct:.1f}%  | ≥30m: {r0.ge30_pct:.1f}%  | ≥45m: {r0.ge45_pct:.1f}%  | ≥60m: {r0.ge60_pct:.1f}%")

    # Disjoint bins from tails
    p15 = float(r0.p_ge_15); p30 = float(r0.p_ge_30); p45 = float(r0.p_ge_45); p60 = float(r0.p_ge_60)
    b15_29 = max(0.0, p15 - p30)
    b30_44 = max(0.0, p30 - p45)
    b45_59 = max(0.0, p45 - p60)
    b60p   = max(0.0, p60)
    b0_14  = max(0.0, 1.0 - p15)
    print(f"Bins → 0–14: {100*b0_14:.1f}% | 15–29: {100*b15_29:.1f}% | 30–44: {100*b30_44:.1f}% | 45–59: {100*b45_59:.1f}% | 60+: {100*b60p:.1f}%")

    print(f"Why: {why}")

if __name__ == "__main__":
    main()
