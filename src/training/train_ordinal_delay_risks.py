# src/training/train_ordinal_delay_risks.py

"""
 python3.9 src/training/train_ordinal_delay_risks.py --outdir models/ordinal --prefix UA_ data/processed/UA_top50_23-24_weekout.parquet
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

from catboost import CatBoostClassifier, Pool

# --------- helper utils ---------
def as_str(series):
    s = series.astype("object")
    s = s.fillna("Unknown")
    return s.map(lambda v: "Unknown" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))

def best_threshold_youden(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def _with_prefix(prefix: str, name: str) -> str:
    """Return '<prefix>_<name>' if prefix provided, else 'name'."""
    return f"{prefix}_{name}" if prefix else name

# Candidate features (same idea as your binary script; keep intersection)
CATS_CAND = [
    "Origin","Dest","Reporting_Airline","OD_pair",
    #"Aircraft_Age_Bucket",
    "origin_weathercode","dest_weathercode",
    "origin_dep_weathercode","dest_arr_weathercode",
]
NUMS_CAND = [
    #"dep_count_origin_bin","arr_count_dest_trail_2h","carrier_flights_prior_day",
    #"origin_taxiout_avg_7d_bin","dest_taxiin_avg_7d_bin",
    #"origin_taxiout_avg_14d_bin","dest_taxiin_avg_14d_bin",
    #"origin_taxiout_avg_28d_bin","dest_taxiin_avg_28d_bin",
    #"carrier_delay_7d_mean","od_delay_7d_mean","flightnum_delay_14d_mean",
    #"origin_delay_7d_mean","dest_delay_7d_mean",
    "origin_temperature_2m_max","origin_temperature_2m_min","origin_precipitation_sum",
    "origin_windspeed_10m_max","origin_windgusts_10m_max",
    "dest_temperature_2m_max","dest_temperature_2m_min","dest_precipitation_sum",
    "dest_windspeed_10m_max","dest_windgusts_10m_max",
    #"origin_dep_temperature_2m","origin_dep_windspeed_10m","origin_dep_windgusts_10m","origin_dep_precipitation",
    #"dest_arr_temperature_2m","dest_arr_windspeed_10m","dest_arr_windgusts_10m","dest_arr_precipitation",
    # optional lagged dest wx features if you added them:
    #"dest_arr_wx_max_code_prev_2h","dest_arr_wx_any_gt3_prev_2h",
    #"dest_arr_precip_sum_prev_3h","dest_arr_wind_max_prev_3h","dest_arr_gust_max_prev_3h",
]

THRESHOLDS = [15, 30, 45, 60]  # minutes

def make_labels(df, thr):
    # Prefer ArrDelayMinutes if present, else derive from ArrDelay
    if "ArrDelayMinutes" in df.columns:
        d = pd.to_numeric(df["ArrDelayMinutes"], errors="coerce").fillna(0)
    else:
        d = pd.to_numeric(df["ArrDelay"], errors="coerce").fillna(0)
    return (d >= thr).astype(int)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_ready", type=str, help="Path to train_ready.parquet")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output dir for models (default: <repo>/models/ordinal)")
    ap.add_argument("--prefix", type=str, default="",
                    help="Optional filename prefix for saved artifacts (e.g., 'aa_week1'). "
                         "Filenames become '<prefix>_catboost_thrXX.cbm', etc.")
    return ap.parse_args()

def main():
    args = parse_args()
    INPUT = Path(args.train_ready)
    OUTDIR = Path(args.outdir) if args.outdir else Path(__file__).resolve().parents[2] / "models" / "ordinal"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip()

    print(f"[INFO] train_ready={INPUT}")
    print(f"[INFO] outdir={OUTDIR}")
    print(f"[INFO] prefix={'<none>' if not prefix else prefix}")

    df = pd.read_parquet(INPUT)

    # OD_pair if missing
    if "OD_pair" not in df.columns and {"Origin","Dest"}.issubset(df.columns):
        df["OD_pair"] = df["Origin"].astype(str).str.upper() + "_" + df["Dest"].astype(str).str.upper()

    # feature lists present
    cat_feats = [c for c in CATS_CAND if c in df.columns]
    num_feats = [c for c in NUMS_CAND if c in df.columns]
    if not cat_feats and not num_feats:
        print("[ERROR] No usable features")
        sys.exit(2)

    X = df[cat_feats + num_feats].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Common split for all thresholds (to keep comparability)
    # Build base label for ≥15 so we can stratify (if not, fall back to non-stratified)
    base_y = make_labels(df, 15)
    try:
        X_train, X_temp, idx_train, idx_temp = train_test_split(
            X, np.arange(len(X)), test_size=0.40, random_state=42, stratify=base_y
        )
        # temp split
        base_y_temp = base_y[idx_temp]
        X_val, X_test, idx_val, idx_test = train_test_split(
            X_temp, idx_temp, test_size=0.50, random_state=42, stratify=base_y_temp
        )
    except Exception:
        # Fallback if stratify fails
        X_train, X_temp, idx_train, idx_temp = train_test_split(
            X, np.arange(len(X)), test_size=0.40, random_state=42
        )
        X_val, X_test, idx_val, idx_test = train_test_split(
            X_temp, idx_temp, test_size=0.50, random_state=42
        )

    # category indices for catboost
    cat_idx = [X.columns.get_loc(c) for c in cat_feats]

    # Train one calibrated model per threshold
    registry = {}
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for thr in THRESHOLDS:
        print(f"\n[INFO] === Training threshold ≥{thr} ===")
        y = make_labels(df, thr)
        y_train = y[idx_train]
        y_val   = y[idx_val]
        y_test  = y[idx_test]

        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        val_pool   = Pool(X_val, y_val, cat_features=cat_idx)

        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            iterations=2000,
            random_seed=42,
            verbose=200,
            od_type="Iter",
            od_wait=100,
        )
        clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Calibrate on validation
        calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
        calibrator.fit(X_val, y_val)

        # Eval
        val_proba  = calibrator.predict_proba(X_val)[:, 1]
        test_proba = calibrator.predict_proba(X_test)[:, 1]
        auc_val  = roc_auc_score(y_val,  val_proba)
        auc_test = roc_auc_score(y_test, test_proba)
        print(f"[thr>={thr:02d}] AUC val={auc_val:.3f} test={auc_test:.3f}")

        # Save
        thr_dir = OUTDIR / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        model_fname = _with_prefix(prefix, f"catboost_thr{thr}.cbm")
        cal_fname   = _with_prefix(prefix, f"calibrated_thr{thr}.joblib")
        meta_fname  = "meta.json"

        clf.save_model(str(thr_dir / model_fname))
        joblib.dump(calibrator, thr_dir / cal_fname)

        # Keep feature names (for inference/explanations)
        meta = {
            "threshold": thr,
            "categorical_features": cat_feats,
            "numeric_features": num_feats,
            "input_parquet": str(INPUT),
            "val_auc": auc_val,
            "test_auc": auc_test,
            "created_at": created_at,
            "prefix": prefix
        }
        with open(thr_dir / meta_fname, "w") as f:
            json.dump(meta, f, indent=2)

        registry[thr] = {
            "model_path": str(thr_dir / model_fname),
            "cal_path": str(thr_dir / cal_fname),
            "prefix": prefix,
            "thr_dir": str(thr_dir),
        }

    with open(OUTDIR / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print("\n[OK] Trained & saved ordinal risk models at:", OUTDIR)
    if prefix:
        print(f"[OK] Artifact filenames are prefixed with '{prefix}_'")

if __name__ == "__main__":
    main()
