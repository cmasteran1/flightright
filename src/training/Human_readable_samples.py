#!/usr/bin/env python3
# src/training/make_prediction_samples_readable.py

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BIN_LABELS = [
    "< 15 min",
    "15–30 min",
    "30–45 min",
    "45–60 min",
    "≥ 60 min",
]

def choose_delay_bin(row):
    """
    Convert ordinal P(delay ≥ X) into a single delay bin.
    Uses the first threshold that FAILS.
    """
    if row.get("p_dep_ge_15", 0.0) < 0.5:
        return BIN_LABELS[0]
    if row.get("p_dep_ge_30", 0.0) < 0.5:
        return BIN_LABELS[1]
    if row.get("p_dep_ge_45", 0.0) < 0.5:
        return BIN_LABELS[2]
    if row.get("p_dep_ge_60", 0.0) < 0.5:
        return BIN_LABELS[3]
    return BIN_LABELS[4]

def build_explanations(row):
    reasons = []

    # Weather
    for col in ("origin_weathercode", "origin_dep_weathercode"):
        v = row.get(col)
        if pd.notna(v) and int(v) >= 70:
            reasons.append("adverse weather at origin")

    # Congestion
    cong = row.get("origin_congestion_ratio")
    if pd.notna(cong) and cong > 1.1:
        reasons.append("high airport congestion")

    # Historical delay
    hist = row.get("carrier_depdelay_mean_lastNdays")
    if pd.notna(hist) and hist > 10:
        reasons.append("carrier has recent delays")

    # Weak history support
    if row.get("flightnum_od_low_support") is True:
        reasons.append("limited historical data for this flight")

    if not reasons:
        return "no strong delay signals detected"

    return "; ".join(reasons)

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python make_prediction_samples_readable.py input.parquet output.parquet [output.csv]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    outcsv = Path(sys.argv[3]) if len(sys.argv) == 4 else None

    df = pd.read_parquet(inp)

    # Assign human-readable delay bin
    df["predicted_delay_bin"] = df.apply(choose_delay_bin, axis=1)

    # Attach explanation text
    df["delay_explanation"] = df.apply(build_explanations, axis=1)

    # Optional: collapse probabilities into a compact string
    df["delay_probabilities"] = df.apply(
        lambda r: (
            f"P≥15={r.get('p_dep_ge_15', np.nan):.2f}, "
            f"P≥30={r.get('p_dep_ge_30', np.nan):.2f}, "
            f"P≥45={r.get('p_dep_ge_45', np.nan):.2f}, "
            f"P≥60={r.get('p_dep_ge_60', np.nan):.2f}"
        ),
        axis=1,
    )

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    if outcsv:
        df.to_csv(outcsv, index=False)

    print(f"[OK] wrote human-readable predictions -> {outp}")
    if outcsv:
        print(f"[OK] wrote CSV -> {outcsv}")

if __name__ == "__main__":
    main()
