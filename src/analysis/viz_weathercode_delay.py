# viz_weathercode_delay.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("../data/processed/train_ready.parquet")
print([c for c in df.columns if "weathercode" in c])


WEATHER_CODE_COLS = {
    "origin_weathercode":       "Origin – Daily",
    "dest_weathercode":         "Destination – Daily",
    "origin_dep_weathercode":   "Origin – Hourly (dep hour)",
    "dest_arr_weathercode":     "Destination – Hourly (arr hour)",
}

def build_label(df: pd.DataFrame, threshold: int = 15) -> pd.Series:
    # Prefer ArrDel15 if it exists; else derive from ArrDelayMinutes
    y = None
    if "ArrDel15" in df.columns and df["ArrDel15"].notna().any():
        y = pd.to_numeric(df["ArrDel15"], errors="coerce")
    elif "ArrDelayMinutes" in df.columns:
        y = (pd.to_numeric(df["ArrDelayMinutes"], errors="coerce") >= threshold).astype(float)
    else:
        # fallback: ArrDelay (minutes, can be negative)
        y = (pd.to_numeric(df.get("ArrDelay", np.nan), errors="coerce") >= threshold).astype(float)
    return y

def summarize_delay_by_code(df: pd.DataFrame, code_col: str, y: pd.Series) -> pd.DataFrame:
    # Keep rows where the code exists and label is not NA
    sub = df[[code_col]].copy()
    sub["y"] = y
    sub = sub.dropna(subset=[code_col, "y"])
    # Ensure codes are numeric (Open-Meteo weathercode is typically an int). If not, keep as string.
    try:
        sub[code_col] = pd.to_numeric(sub[code_col], errors="coerce")
    except Exception:
        pass
    sub = sub.dropna(subset=[code_col])

    grp = sub.groupby(code_col, dropna=False)["y"].agg(
        total="count",
        delayed_sum="sum",
        delay_rate="mean",
    ).reset_index().rename(columns={code_col: "weathercode"})
    # Sort X by numeric code, if numeric; else by label
    if np.issubdtype(grp["weathercode"].dtype, np.number):
        grp = grp.sort_values("weathercode")
    else:
        grp = grp.sort_values("weathercode", key=lambda s: s.astype(str))
    return grp

def plot_bar(grp: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(10, 5))
    x = grp["weathercode"].astype(str)
    y = grp["delay_rate"].values
    plt.bar(x, y)
    plt.ylim(0, 1)
    plt.xlabel("Weather code")
    plt.ylabel("Chance of ≥N min delay")
    plt.title(title)
    # Annotate bars with counts to show support
    for i, (xi, yi, n) in enumerate(zip(x, y, grp["total"].values)):
        plt.text(i, yi + 0.01, f"n={int(n)}", ha="center", va="bottom", fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"[OK] saved {outpath}")
    # Also show (handy when running locally)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_path", help="Path to processed parquet with weather codes")
    ap.add_argument("--threshold", type=int, default=15, help="Delay threshold minutes (default 15)")
    args = ap.parse_args()

    p = Path(args.parquet_path)
    df = pd.read_parquet(p)

    # Build label
    y = build_label(df, threshold=args.threshold)

    # Which columns exist in this file?
    have = [c for c in WEATHER_CODE_COLS if c in df.columns]
    if not have:
        raise SystemExit("No weather code columns found. Expected any of: "
                         + ", ".join(WEATHER_CODE_COLS.keys()))

    outdir = p.parent
    for col in have:
        title = f"{WEATHER_CODE_COLS[col]} — delay≥{args.threshold} fraction by code"
        grp = summarize_delay_by_code(df, col, y)
        # Guard: if nothing to plot, skip
        if grp.empty:
            print(f"[WARN] No rows to plot for {col}")
            continue
        outpath = outdir / f"delay_rate_by_{col}.png"
        plot_bar(grp, title, outpath)

if __name__ == "__main__":
    main()
