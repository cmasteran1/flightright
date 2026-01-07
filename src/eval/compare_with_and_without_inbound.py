'''
# WITH inbound priors
python src/inference/predict_delay_bins.py \
  models/ordinal \
  data/processed/inbound_prior_delay.parquet \
  --inbound_priors models/priors/inbound_priors.parquet \
  --hours_back 2 --min_support 50 --explain_max_rows 0 --verbose

# keep a copy
cp models/ordinal/predictions_with_bins.parquet data/predictions/with_inbound.parquet

# WITHOUT inbound priors (omit the flag)
python src/inference/predict_delay_bins.py \
  models/ordinal \
  data/processed/inbound_prior_delay.parquet \
  --explain_max_rows 0 --verbose

# keep a copy
cp models/ordinal/predictions_with_bins.parquet data/predictions/without_inbound.parquet

'''


#!/usr/bin/env python3
# src/eval/compare_with_without_inbound.py
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

THRESHOLDS = [15, 30, 45, 60]
KEYS = ["FlightDate","Reporting_Airline","Flight_Number_Reporting_Airline","Origin","Dest","CRSDepTime"]

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{_now()}] {msg}")

def _make_labels(truth: pd.DataFrame):
    """
    Create y_ge_15/30/45/60 from any available dep delay columns.
    Prefers DepDelayMinutes; falls back to DepDelay; else DepDel15 (BTS flag) for 15m only.
    """
    y = {}
    if "DepDelayMinutes" in truth.columns:
        dep = pd.to_numeric(truth["DepDelayMinutes"], errors="coerce").fillna(0)
        for t in THRESHOLDS:
            y[f"y_ge_{t}"] = (dep >= t).astype(int)
    elif "DepDelay" in truth.columns:
        dep = pd.to_numeric(truth["DepDelay"], errors="coerce").fillna(0)
        for t in THRESHOLDS:
            y[f"y_ge_{t}"] = (dep >= t).astype(int)
    else:
        # Minimal fallback: BTS dep flag for 15 only
        if "DepDel15" in truth.columns:
            y["y_ge_15"] = pd.to_numeric(truth["DepDel15"], errors="coerce").fillna(0).astype(int)
        else:
            raise RuntimeError("No departure delay labels found (need DepDelayMinutes, DepDelay, or DepDel15).")
        # For higher thresholds, try ArrivalDelayMinutes as a last resort (not ideal for dep evaluation).
        if "ArrivalDelayGroups" in truth.columns:
            # 0: (-14..14), 1: (15..29), 2: (30..44), 3: (45..59), 4: (60..74) ...
            grp = pd.to_numeric(truth["ArrivalDelayGroups"], errors="coerce")
            y["y_ge_30"] = (grp >= 2).astype(int)
            y["y_ge_45"] = (grp >= 3).astype(int)
            y["y_ge_60"] = (grp >= 4).astype(int)
        else:
            # If we truly have only DepDel15, evaluate ≥15 only.
            for t in [30,45,60]:
                y[f"y_ge_{t}"] = pd.Series(np.nan, index=truth.index, dtype="float")
    return pd.DataFrame(y)

def _pick_join_keys(df: pd.DataFrame):
    have = [k for k in KEYS if k in df.columns]
    if not have:
        raise RuntimeError(f"No join keys found. Need at least one of: {KEYS}")
    return have

def _metrics_block(y, p, label):
    out = {}
    mask = ~pd.isna(y) & ~pd.isna(p)
    yv = y[mask].astype(int).values
    pv = np.clip(p[mask].astype(float).values, 1e-6, 1-1e-6)
    out["n"] = int(mask.sum())
    if out["n"] == 0:
        out["brier"] = np.nan; out["logloss"] = np.nan; out["auc"] = np.nan
        return pd.Series(out, name=label)
    out["brier"] = float(np.mean((pv - yv)**2))
    try:
        out["logloss"] = float(log_loss(yv, pv))
    except Exception:
        out["logloss"] = np.nan
    try:
        if len(np.unique(yv)) == 1:
            out["auc"] = np.nan
        else:
            out["auc"] = float(roc_auc_score(yv, pv))
    except Exception:
        out["auc"] = np.nan
    return pd.Series(out, name=label)

def _calibration(y, p, bins=10):
    mask = ~pd.isna(y) & ~pd.isna(p)
    yv = y[mask].astype(int).values
    pv = p[mask].astype(float).values
    if len(pv) == 0:
        return pd.DataFrame(columns=["bin","count","pred_mean","obs_rate"])
    q = np.linspace(0,1,bins+1)
    # stable bin edges
    edges = np.quantile(pv, q)
    # prevent identical edges causing empty bins
    edges = np.unique(np.clip(edges, 0, 1))
    if len(edges) < 3:  # not enough variation
        edges = np.array([0, 0.5, 1.0])
    idx = np.clip(np.searchsorted(edges, pv, side="right")-1, 0, len(edges)-2)
    df = pd.DataFrame({"bin": idx, "y": yv, "p": pv})
    g = df.groupby("bin").agg(count=("y","size"), pred_mean=("p","mean"), obs_rate=("y","mean")).reset_index()
    g["edge_left"] = edges[g["bin"]]
    g["edge_right"] = edges[g["bin"]+1]
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="Parquet used as scoring input (contains true dep delay columns).")
    ap.add_argument("--pred_with", required=True, help="Predictions parquet generated WITH inbound priors.")
    ap.add_argument("--pred_without", required=True, help="Predictions parquet generated WITHOUT inbound priors.")
    ap.add_argument("--outdir", default="data/eval", help="Directory to write evaluation CSVs.")
    ap.add_argument("--bins", type=int, default=10, help="Calibration bin count.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    log("loading truth / predictions")
    truth = pd.read_parquet(args.truth)
    with_inb = pd.read_parquet(args.pred_with)
    wo_inb  = pd.read_parquet(args.pred_without)

    # Derive labels from truth
    ydf = _make_labels(truth)

    # Decide join keys (use intersection of keys present)
    keys_truth = _pick_join_keys(truth)
    keys_pred  = _pick_join_keys(with_inb)
    join_keys = [k for k in keys_truth if k in keys_pred]
    if not join_keys:
        raise RuntimeError("No common join keys between truth and predictions.")

    log(f"joining on keys: {join_keys}")
    # Cut down truth to keys + labels only (avoid accidental dup columns)
    truth_join = pd.concat([truth[join_keys], ydf], axis=1)

    # Join with pred files
    w = with_inb.merge(truth_join, on=join_keys, how="inner")
    u = wo_inb.merge(truth_join, on=join_keys, how="inner")
    if len(w) != len(u):
        log(f"[WARN] different join sizes: with={len(w)} without={len(u)}. Using inner join on intersection of both.")
        common = with_inb[join_keys].merge(wo_inb[join_keys], on=join_keys, how="inner")
        w = with_inb.merge(common, on=join_keys, how="inner").merge(truth_join, on=join_keys, how="inner")
        u = wo_inb.merge(common, on=join_keys, how="inner").merge(truth_join, on=join_keys, how="inner")

    log(f"eval rows: {len(w)}")

    # Metric tables
    rows = []
    calib_tables = []

    for t in THRESHOLDS:
        pcol_w = "p_ge_{}".format(t)
        pcol_u = "p_ge_{}".format(t)
        # Column names in your pred parquet (from predict_delay_bins.py)
        # with-inbound file stores p_ge_*; without-inbound also stores p_ge_*.

        ycol = f"y_ge_{t}"
        if ycol not in w.columns:
            log(f"[WARN] {ycol} not present (skipping metrics for ≥{t}).")
            continue

        # Overall metrics
        rows.append(_metrics_block(w[ycol], w[pcol_w], f"with_ge{t}"))
        rows.append(_metrics_block(u[ycol], u[pcol_u], f"without_ge{t}"))

        # Calibration tables (overall)
        cw = _calibration(w[ycol], w[pcol_w], bins=args.bins)
        cw["which"] = f"with_ge{t}"
        calib_tables.append(cw)

        cu = _calibration(u[ycol], u[pcol_u], bins=args.bins)
        cu["which"] = f"without_ge{t}"
        calib_tables.append(cu)

        # Segment by inbound_level_used (only exists in 'with' file)
        if "inbound_level_used" in w.columns:
            seg = (w
                   .assign(seg=w["inbound_level_used"].fillna("NONE"))
                   .groupby("seg"))
            seg_rows = []
            for seg_name, g in seg:
                seg_rows.append(_metrics_block(g[ycol], g[pcol_w], f"with_ge{t}__seg={seg_name}"))
            rows.extend(seg_rows)

        # Segment by inbound_low_support
        if "inbound_low_support" in w.columns:
            seg2 = (w
                    .assign(seg=w["inbound_low_support"].fillna(False).astype(bool))
                    .groupby("seg"))
            seg_rows2 = []
            for seg_name, g in seg2:
                label = f"with_ge{t}__low_support={seg_name}"
                seg_rows2.append(_metrics_block(g[ycol], g[pcol_w], label))
            rows.extend(seg_rows2)

    metrics = pd.DataFrame(rows)
    # Diff table (with - without) for same threshold
    def _row_for(thr, prefix):
        m = metrics.loc[metrics.index == f"{prefix}_ge{thr}"]
        return m.iloc[0] if len(m) else pd.Series({"brier":np.nan,"logloss":np.nan,"auc":np.nan,"n":0})
    deltas = []
    for t in THRESHOLDS:
        r_with = _row_for(t, "with")
        r_without = _row_for(t, "without")
        deltas.append(pd.Series({
            "thr": t,
            "delta_brier": r_with.get("brier", np.nan) - r_without.get("brier", np.nan),
            "delta_logloss": r_with.get("logloss", np.nan) - r_without.get("logloss", np.nan),
            "delta_auc": r_with.get("auc", np.nan) - r_without.get("auc", np.nan),
            "n_with": r_with.get("n", 0),
            "n_without": r_without.get("n", 0),
        }))
    deltas = pd.DataFrame(deltas)

    # Write outputs
    metrics_path = outdir / "metrics_overall_and_segments.csv"
    calib_path   = outdir / "calibration_tables.csv"
    deltas_path  = outdir / "deltas_with_minus_without.csv"

    metrics.to_csv(metrics_path, index=True)
    pd.concat(calib_tables, ignore_index=True).to_csv(calib_path, index=False)
    deltas.to_csv(deltas_path, index=False)

    log(f"[OK] wrote {metrics_path}")
    log(f"[OK] wrote {calib_path}")
    log(f"[OK] wrote {deltas_path}")

    # Console summary
    print("\n=== Overall deltas (with − without) ===")
    print(deltas.to_string(index=False))

    print("\nTip: negative delta_brier / delta_logloss and positive delta_auc mean the inbound layer helped.")
    print("You can also slice metrics_overall_and_segments.csv by inbound_level_used to see strongest lift where priors used = HOUR or shrunk.")

if __name__ == "__main__":
    main()
