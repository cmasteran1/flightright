# build_inbound_priors.py
# Build station "inbound bank" priors from historical arrivals.
# Key: (carrier, dest_station, month, dow, arr_hour_local)
# Stats: sample_size and tail risks for >=15/30/45/60 min


# run with
#python build_inbound_priors.py \
#  -input.parquet
# -output.parquet

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    return df

def _ensure_arr_hour_local(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer precomputed arr_hour_local (from your build_training_set)
    if "arr_hour_local" in df.columns and df["arr_hour_local"].notna().any():
        return df
    # Fallback: derive from CRSArrTime (HHMM) if present (naive)
    def hhmm_to_hour(x):
        try:
            s = str(int(x)).zfill(4)
            return int(s[:2])
        except Exception:
            return np.nan
    if "CRSArrTime" in df.columns:
        df["arr_hour_local"] = df["CRSArrTime"].apply(hhmm_to_hour).astype("Int64")
    else:
        # last resort: use scheduled arrival block if available
        if "ArrTimeBlk" in df.columns:
            # e.g., "0900-0959" -> 9
            df["arr_hour_local"] = (
                df["ArrTimeBlk"].astype(str).str[:2].str.replace(r"\D", "", regex=True)
            ).replace("", np.nan).astype("float").astype("Int64")
        else:
            df["arr_hour_local"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df

def _tail_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure ArrDelayMinutes exists and is numeric
    col = "ArrDelayMinutes" if "ArrDelayMinutes" in df.columns else "ArrDelay"
    if col not in df.columns:
        raise RuntimeError("Expected ArrDelayMinutes or ArrDelay in input.")
    df["ArrDelayMinutes"] = pd.to_numeric(df[col], errors="coerce")
    df["_ge15"] = (df["ArrDelayMinutes"] >= 15).astype("Int8")
    df["_ge30"] = (df["ArrDelayMinutes"] >= 30).astype("Int8")
    df["_ge45"] = (df["ArrDelayMinutes"] >= 45).astype("Int8")
    df["_ge60"] = (df["ArrDelayMinutes"] >= 60).astype("Int8")
    return df

def build_priors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_dt(df)
    df = _ensure_arr_hour_local(df)
    df = _tail_flags(df)

    # core keys
    for c in ("Reporting_Airline", "Dest"):
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in input.")
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str)
    df["Dest"] = df["Dest"].astype(str)

    # calendar keys
    df["Month"] = df["FlightDate"].dt.month.astype("Int64")
    # BTS DayOfWeek is 1=Mon..7=Sun in many dumps; recompute from FlightDate to be safe (Mon=0)
    df["DOW"] = df["FlightDate"].dt.dayofweek.astype("Int64")  # 0..6

    grp_cols = ["Reporting_Airline", "Dest", "Month", "DOW", "arr_hour_local"]
    df = df.dropna(subset=grp_cols)

    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              sample_size=("ArrDelayMinutes", "size"),
              p_ge15=("_ge15", "mean"),
              p_ge30=("_ge30", "mean"),
              p_ge45=("_ge45", "mean"),
              p_ge60=("_ge60", "mean"),
          )
    )
    # Keep as float probabilities
    for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]:
        agg[c] = agg[c].astype(float)

    return agg

def main():
    if len(sys.argv) < 3:
        print("Usage: python build_inbound_priors.py <input_parquet_or_csv> <output_parquet>")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        sys.exit(2)

    # Load
    if in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_parquet(in_path)

    priors = build_priors(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    priors.to_parquet(out_path, index=False)
    print(f"[OK] Wrote priors -> {out_path}  rows={len(priors)}")

if __name__ == "__main__":
    main()
