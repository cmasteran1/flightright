# src/training/train_catboost_calibrated.py
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
import joblib
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

# ------------------------------ utils ------------------------------

def best_threshold_youden(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def as_str(series):
    # Force true strings (CatBoost categorical must be str or int; not floats/NaN)
    s = series.astype("object")
    s = s.fillna("Unknown")
    return s.map(
        lambda v: "Unknown"
        if pd.isna(v)
        else str(int(v)) if isinstance(v, float) and v.is_integer()
        else str(v)
    )

def reliability_table(y_true, p_pred, n_bins=10):
    y_true = pd.Series(y_true).reset_index(drop=True)
    p_pred = pd.Series(p_pred).reset_index(drop=True).clip(0, 1)

    bins = pd.interval_range(start=0.0, end=1.0, periods=n_bins, closed="right")
    cuts = pd.cut(p_pred, bins, include_lowest=True)
    out = (
        pd.DataFrame({"y": y_true, "p": p_pred, "bin": cuts})
        .groupby("bin", observed=True)
        .agg(count=("y", "size"), mean_pred=("p", "mean"), empirical_pos_rate=("y", "mean"))
        .reset_index()
    )
    # Extract numeric bin edges for plotting
    out["bin_left"]  = out["bin"].apply(lambda iv: iv.left if pd.notna(iv) else np.nan)
    out["bin_right"] = out["bin"].apply(lambda iv: iv.right if pd.notna(iv) else np.nan)
    out["bin_mid"]   = (out["bin_left"].astype(float) + out["bin_right"].astype(float)) / 2.0
    return out

def plot_reliability(tag, y_true, p_pred, outdir: Path, n_bins=10):
    tab = reliability_table(y_true, p_pred, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect")
    ax.plot(tab["mean_pred"], tab["empirical_pos_rate"], marker="o", linewidth=1.5, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability curve ({tag})")
    ax.legend(loc="best")
    fig.tight_layout()

    png_path = outdir / f"calibration_{tag}.png"
    csv_path = outdir / f"calibration_{tag}_table.csv"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    tab.drop(columns=["bin"], errors="ignore").to_csv(csv_path, index=False)
    print(f"[SAVE] reliability plot -> {png_path}")
    print(f"[SAVE] reliability table -> {csv_path}")

# ------------------------------ main ------------------------------

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python train_catboost_calibrated.py path/to/train_ready.parquet [path/to/config.json]")
        sys.exit(1)

    INPUT = Path(sys.argv[1])
    OUTDIR = Path(__file__).resolve().parents[2] / "models"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Optional analysis config
    cfg = {
        "snow_codes": [71, 73, 75, 77, 85, 86],
        "reliability_bins": 10,
        "fp_delay_min": 10,
        "fn_delay_min": 30,
    }
    if len(sys.argv) == 3:
        cfg_path = Path(sys.argv[2])
        with open(cfg_path, "r") as f:
            cfg.update(json.load(f))

    SNOW_CODES = set(cfg.get("snow_codes", []))
    REL_BINS = int(cfg.get("reliability_bins", 10))
    FP_MIN = int(cfg.get("fp_delay_min", 10))
    FN_MIN = int(cfg.get("fn_delay_min", 30))

    df = pd.read_parquet(INPUT)

    # ---- Label: delay ≥ 15 min (binary) ----
    # Prefer ArrDel15 if present, otherwise derive from ArrDelay
    if "ArrDel15" in df.columns and df["ArrDel15"].notna().any():
        y = pd.to_numeric(df["ArrDel15"], errors="coerce").fillna(0).astype(int)
    else:
        y = (pd.to_numeric(df["ArrDelay"], errors="coerce") >= 15).astype(int)

    # ---- Build OD_pair if not present ----
    if "OD_pair" not in df.columns and {"Origin", "Dest"}.issubset(df.columns):
        df["OD_pair"] = df["Origin"].astype(str).str.upper() + "_" + df["Dest"].astype(str).str.upper()

    # ---- Candidate feature lists (we’ll keep intersection with df) ----
    # Categorical candidates
    CATS_CAND = [
        "Origin", "Dest", "Reporting_Airline", "OD_pair",
        "season", "Aircraft_Age_Bucket",

        # daily
        "origin_weathercode", "dest_weathercode",

        # hourly aligned codes
        "origin_dep_weathercode", "dest_arr_weathercode",

        # NEW: composite/binned
        #"dest_arr_wx_stressed",
        #"dest_arr_minutes_since_wx_gt3_6h_bin",
    ]

    # Numeric candidates
    NUMS_CAND = [
        # calendar/schedule
        #"day_of_week", "month", "CRSDepHour", "CRSArrHour",
        #"is_morning_dep", "is_morning_arr", "is_evening_dep", "is_evening_arr",
        #"is_night_dep", "is_night_arr",

        # congestion
        "dep_count_origin_bin", "arr_count_dest_trail_2h", "carrier_flights_prior_day",

        # NEW: recent origin departures
        #"origin_dep_count_trail_60m", "origin_dep_count_trail_120m",
        #"origin_dep_count_carrier_trail_120m",

        # taxi baselines
        "origin_taxiout_avg_7d_bin", "dest_taxiin_avg_7d_bin",
        "origin_taxiout_avg_14d_bin", "dest_taxiin_avg_14d_bin",
        #"origin_taxiout_avg_28d_bin", "dest_taxiin_avg_28d_bin",

        # history baselines
        "carrier_delay_7d_mean", "od_delay_7d_mean", "flightnum_delay_14d_mean",
        "origin_delay_7d_mean", "dest_delay_7d_mean",

        # daily weather (origin/dest)
        #"origin_temperature_2m_max", "origin_temperature_2m_min", "origin_precipitation_sum",
        #"origin_windspeed_10m_max", "origin_windgusts_10m_max",
        #"dest_temperature_2m_max", "dest_temperature_2m_min", "dest_precipitation_sum",
        #"dest_windspeed_10m_max", "dest_windgusts_10m_max",

        # hourly aligned continuous (origin dep / dest arr)
        #"origin_dep_temperature_2m", "origin_dep_windspeed_10m", "origin_dep_windgusts_10m", "origin_dep_precipitation",
        #"dest_arr_temperature_2m", "dest_arr_windspeed_10m", "dest_arr_windgusts_10m", "dest_arr_precipitation",

        # NEW: destination recent-wx windows
       # "dest_arr_wx_max_code_prev_1h", "dest_arr_wx_max_code_prev_2h", "dest_arr_wx_max_code_prev_4h",
        #"dest_arr_precip_sum_prev_1h", "dest_arr_precip_sum_prev_3h", "dest_arr_precip_sum_prev_4h",
        #"dest_arr_wind_max_prev_1h", "dest_arr_wind_max_prev_3h", "dest_arr_wind_max_prev_4h",
        #"dest_arr_gust_max_prev_1h", "dest_arr_gust_max_prev_3h", "dest_arr_gust_max_prev_4h",
        #"dest_arr_minutes_since_wx_gt3_6h",
    ]

    # Keep only features that exist in data
    cat_feats = [c for c in CATS_CAND if c in df.columns]
    num_feats = [c for c in NUMS_CAND if c in df.columns]

    # Coerce booleans to 0/1 numerics if present
    for b in ["is_morning_dep","is_morning_arr","is_evening_dep","is_evening_arr","is_night_dep","is_night_arr",
              "dest_arr_wx_any_gt3_prev_2h"]:
        if b in df.columns:
            df[b] = pd.to_numeric(df[b], errors="coerce").astype("Int64")

    # Prepare X
    X = df[cat_feats + num_feats].copy()

    # Cast categoricals to clean strings
    for c in cat_feats:
        X[c] = as_str(X[c])

    # Numeric fill (median)
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Drop any remaining rows with missing label
    mask_valid = y.notna()
    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)
    # Keep original row indices so we can recover test rows later
    idx = pd.Series(np.arange(len(X)), index=X.index)

    # Shuffle all in sync
    X, y, idx = shuffle(X, y, idx, random_state=42)

    print(f"[INFO] Using {len(cat_feats)} categorical, {len(num_feats)} numeric features.")
    if len(cat_feats) == 0 and len(num_feats) == 0:
        print("[ERROR] No usable features found.")
        sys.exit(2)

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, idx, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # CatBoost pools
    cat_idx = [X.columns.get_loc(c) for c in cat_feats]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool   = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool  = Pool(X_test, y_test, cat_features=cat_idx)

    # ---- Train base CatBoost ----
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
        class_weights=None,
    )
    clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # ---- Calibrate probabilities on validation (isotonic) ----
    calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
    calibrator.fit(X_val, y_val)

    # ---- Threshold via Youden on calibrated validation probs ----
    val_proba_cal = calibrator.predict_proba(X_val)[:, 1]
    th, tpr, fpr = best_threshold_youden(y_val, val_proba_cal)

    # ---- Evaluate on test using calibrated probs ----
    proba_test_cal = calibrator.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test_cal >= th).astype(int)

    print(f"Chosen threshold (Youden's J): {th:.3f}  |  TPR: {tpr:.3f}  FPR: {fpr:.3f}")
    print("Classification report @ chosen threshold:")
    print(classification_report(y_test, y_pred_test))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print(f"AUC (val)={roc_auc_score(y_val, val_proba_cal):.3f}  AUC (test)={roc_auc_score(y_test, proba_test_cal):.3f}")

    # ---- Feature importances (all features) ----
    importances = clf.get_feature_importance(train_pool, type="FeatureImportance")
    imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}) \
             .sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nAll features by CatBoost importance (sorted):")
    with pd.option_context("display.max_rows", None):
        print(imp_df)

    # ---- Reliability / Calibration plots ----
    # overall
    plot_reliability("overall", y_test, proba_test_cal, OUTDIR, n_bins=REL_BINS)

    # snow-only slice (any origin/dest code in SNOW_CODES on daily *or* hourly)
    def _in_snow(row):
        for col in ("origin_weathercode","dest_weathercode","origin_dep_weathercode","dest_arr_weathercode"):
            if col in df.columns:
                v = row.get(col)
                if pd.notna(v) and int(v) in SNOW_CODES:
                    return True
        return False

    test_idx = X_test.index
    snow_mask = df.loc[test_idx].apply(_in_snow, axis=1).astype(bool).values
    print(f"[INFO] Snow-only slice size in test: {snow_mask.sum()} / {len(test_idx)}")
    if snow_mask.any():
        plot_reliability("snow_only", y_test[snow_mask], proba_test_cal[snow_mask], OUTDIR, n_bins=REL_BINS)

    # ---- Error analysis (close misses) ----
    # False positives that still had >= FP_MIN minutes delay
    # False negatives that had >= FN_MIN minutes delay
    # --- Build df_test to inspect errors / delays ---
    df_full = pd.read_parquet(INPUT)  # original, unshuffled data
    df_test = df_full.iloc[idx_test.astype(int)].copy()

    # Ensure numeric ArrDelayMinutes
    df_test["ArrDelayMinutes"] = pd.to_numeric(df_test.get("ArrDelayMinutes"), errors="coerce")

    # Preds & threshold already computed:
    #   test_proba_cal, y_pred_test, th

    # False negatives overall
    fn_mask = (y_test == 1) & (y_pred_test == 0)

    # Restrict to "material" misses: ArrDelayMinutes >= 30
    fn30_mask = fn_mask & (df_test["ArrDelayMinutes"] >= 30)

    # Report rates
    fn30_rate = float(fn30_mask.mean()) if len(fn30_mask) else 0.0
    print("\n[Error analysis]")
    print(f"- False negatives with ≥30 min delay (share of test): {fn30_rate:.3f}")

    # Save samples: only FN ≥30
    N_FP = 200
    N_FN = 200
    fp_mask = (y_test == 0) & (y_pred_test == 1)

    samples_fp = df_test.loc[fp_mask].sample(n=min(N_FP, int(fp_mask.sum())),
                                             random_state=42) if fp_mask.any() else df_test.iloc[0:0]
    samples_fn = df_test.loc[fn30_mask].sample(n=min(N_FN, int(fn30_mask.sum())),
                                               random_state=42) if fn30_mask.any() else df_test.iloc[0:0]

    samples_fp.to_csv(OUTDIR / "samples_false_positives.csv", index=False)
    samples_fn.to_csv(OUTDIR / "samples_false_negatives.csv", index=False)

    print(f"[SAVE] false positives sample -> {OUTDIR / 'samples_false_positives.csv'}")
    print(f"[SAVE] false negatives sample -> {OUTDIR / 'samples_false_negatives.csv'}")

    # ---- Save artifacts ----
    # 1) raw CatBoost model
    cat_model_path = OUTDIR / "catboost_delay15_base.cbm"
    clf.save_model(str(cat_model_path))

    # 2) calibrated sklearn wrapper + metadata
    joblib.dump(calibrator, OUTDIR / "catboost_delay15_calibrated.joblib")
    meta = {
        "threshold_youden": th,
        "tpr_at_threshold": tpr,
        "fpr_at_threshold": fpr,
        "categorical_features": cat_feats,
        "numeric_features": num_feats,
        "input_parquet": str(INPUT),
        "config_used": cfg,
    }
    with open(OUTDIR / "catboost_delay15_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    imp_df.to_csv(OUTDIR / "catboost_delay15_feature_importances.csv", index=False)
    print("\nSaved:")
    print(" - base model         ->", cat_model_path)
    print(" - calibrated model   ->", OUTDIR / "catboost_delay15_calibrated.joblib")
    print(" - meta (threshold)   ->", OUTDIR / "catboost_delay15_meta.json")
    print(" - importances (csv)  ->", OUTDIR / "catboost_delay15_feature_importances.csv")
    print(" - reliability plots  -> calibration_overall.png", end="")
    if snow_mask.any():
        print(", calibration_snow_only.png")
    else:
        print()
    print(" - reliability tables -> calibration_overall_table.csv", end="")
    if snow_mask.any():
        print(", calibration_snow_only_table.csv")
    else:
        print()
    print(" - error samples      -> samples_false_positives.csv, samples_false_negatives.csv")

if __name__ == "__main__":
    main()
