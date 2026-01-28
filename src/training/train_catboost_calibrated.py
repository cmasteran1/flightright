#!/usr/bin/env python3
# src/training/train_catboost_calibrated.py
import os, sys, json, time, signal, faulthandler
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import joblib

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

# -------- optional memory reporting --------
try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

PRINT_MEM = os.getenv("PRINT_MEM", "0") == "1"
FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))  # 0 disables

_t0 = time.time()
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

_last_step_t = time.time()
def log(msg, *, step=False):
    """Timestamped logging with elapsed since start and since last step."""
    global _last_step_t
    now = time.time()
    total = now - _t0
    delta = now - _last_step_t
    if step:
        _last_step_t = now
    print(f"[{_now()} +{total:7.2f}s Δ{delta:6.2f}s] {msg}", flush=True)

def log_mem(tag=""):
    if not PRINT_MEM or not _HAVE_PSUTIL:
        return
    p = psutil.Process()
    rss = p.memory_info().rss / (1024**2)
    log(f"[mem] {tag} RSS={rss:.1f} MB")

def enable_periodic_traces():
    if TRACE_EVERY_SEC > 0:
        try:
            faulthandler.dump_traceback_later(TRACE_EVERY_SEC, repeat=True)
            log(f"[trace] periodic faulthandler enabled every {TRACE_EVERY_SEC}s")
        except Exception as e:
            log(f"[trace] failed to enable periodic traces: {e}")

def enable_usr1_trace():
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        log("[trace] SIGUSR1 handler registered (kill -USR1 <pid> to dump stacks)")
    except Exception as e:
        log(f"[trace] failed to register SIGUSR1 handler: {e}")

# ------------------------------ utils ------------------------------

def best_threshold_youden(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def as_str(series):
    s = series.astype("object")
    s = s.fillna("Unknown").astype("object")
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
    log(f"[SAVE] reliability plot -> {png_path}")
    log(f"[SAVE] reliability table -> {csv_path}")

# ------------------------------ main ------------------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    if len(sys.argv) not in (3, 4):
        print("Usage: python train_catboost_calibrated.py path/to/train_ready.parquet path/to/outdir [path/to/config.json]")
        sys.exit(1)

    INPUT = Path(sys.argv[1])
    OUTDIR = Path(sys.argv[2])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Optional config
    cfg = {
        "snow_codes": [71, 73, 75, 77, 85, 86],
        "reliability_bins": 10,
        "fp_delay_min": 10,
        "fn_delay_min": 30,
        "thresholds": [15, 30, 45, 60],
        "iterations": 2000,
        "depth": 8,
        "od_wait": 100,
    }
    if len(sys.argv) == 4:
        with open(Path(sys.argv[3]), "r") as f:
            cfg.update(json.load(f))

    if FAST:
        cfg["iterations"] = 800
        cfg["depth"] = 6
        cfg["od_wait"] = 80
        log("[FAST] Using lighter CatBoost settings")

    SNOW_CODES = set(cfg.get("snow_codes", []))
    REL_BINS = int(cfg.get("reliability_bins", 10))
    THRESHOLDS = list(cfg.get("thresholds", [15, 30, 45, 60]))

    log(f"Loading data: {INPUT}")
    df = pd.read_parquet(INPUT)
    log(f"Rows={len(df)} Cols={len(df.columns)}", step=True)
    log_mem("after load")

    # Build OD_pair if needed
    if "OD_pair" not in df.columns and {"Origin", "Dest"}.issubset(df.columns):
        df["OD_pair"] = df["Origin"].astype(str).str.upper() + "_" + df["Dest"].astype(str).str.upper()

    # Feature sets (keep intersection)
    CATS_CAND = [
        "Origin","Dest","Reporting_Airline","OD_pair",
        "origin_weathercode","dest_weathercode",
        "origin_dep_weathercode","dest_arr_weathercode",
    ]
    NUMS_CAND = [
        "day_of_week","month","CRSDepHour","CRSArrHour",
        "is_morning_dep","is_morning_arr","is_evening_dep","is_evening_arr",
        "is_night_dep","is_night_arr",
        "dep_count_origin_bin","arr_count_dest_trail_2h","carrier_flights_prior_day",
        #"origin_dep_count_trail_60m","origin_dep_count_trail_120m","origin_dep_count_carrier_trail_120m",
        "carrier_delay_7d_mean","od_delay_7d_mean","flightnum_delay_14d_mean",
        "origin_delay_7d_mean","dest_delay_7d_mean",
        "origin_temperature_2m_max","origin_temperature_2m_min","origin_precipitation_sum",
        "origin_windspeed_10m_max","origin_windgusts_10m_max",
        "dest_temperature_2m_max","dest_temperature_2m_min","dest_precipitation_sum",
        "dest_windspeed_10m_max","dest_windgusts_10m_max",
        "origin_dep_temperature_2m","origin_dep_windspeed_10m","origin_dep_windgusts_10m","origin_dep_precipitation",
        "dest_arr_temperature_2m","dest_arr_windspeed_10m","dest_arr_windgusts_10m","dest_arr_precipitation",
    ]
    cat_feats = [c for c in CATS_CAND if c in df.columns]
    num_feats = [c for c in NUMS_CAND if c in df.columns]
    log(f"[features] cats={len(cat_feats)} nums={len(num_feats)}", step=True)

    # Prepare X (NO global shuffle here)
    X = df[cat_feats + num_feats].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Base labels for stratification (≥15)
    def make_labels(df_in, thr):
        if "ArrDelayMinutes" in df_in.columns and df_in["ArrDelayMinutes"].notna().any():
            d = pd.to_numeric(df_in["ArrDelayMinutes"], errors="coerce").fillna(0)
        else:
            d = pd.to_numeric(df_in.get("ArrDelay"), errors="coerce").fillna(0)
        return (d >= thr).astype(int)

    base_y = make_labels(df, 15)

    # Create splits on indices, stratified by base_y
    n = len(X)
    all_idx = np.arange(n)
    log("[split] stratified index splits on ≥15", step=True)
    idx_train, idx_temp, y_train_base, y_temp_base = train_test_split(
        all_idx, base_y.values, test_size=0.40, random_state=42, stratify=base_y
    )
    idx_val, idx_test, y_val_base, y_test_base = train_test_split(
        idx_temp, y_temp_base, test_size=0.50, random_state=42, stratify=y_temp_base
    )
    log(f"[split] sizes train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # Build pools once per threshold from these consistent index sets
    cat_idx = [X.columns.get_loc(c) for c in cat_feats]

    registry = {}
    for thr in THRESHOLDS:
        log("="*19 + f" Training threshold ≥{thr} minutes " + "="*19, step=True)
        thr_dir = OUTDIR / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        y_all = make_labels(df, thr).values  # aligned to original row order
        y_train = y_all[idx_train]
        y_val   = y_all[idx_val]
        y_test  = y_all[idx_test]

        X_train = X.iloc[idx_train]
        X_val_  = X.iloc[idx_val]
        X_test  = X.iloc[idx_test]

        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        val_pool   = Pool(X_val_,  y_val,   cat_features=cat_idx)
        test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

        iters = int(cfg.get("iterations", 2000))
        depth = int(cfg.get("depth", 8))
        od_wait = int(cfg.get("od_wait", 100))

        log(f"[cb] fit start (iter={iters} depth={depth} od_wait={od_wait})")
        log_mem("before fit")
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=0.05,
            depth=depth,
            l2_leaf_reg=3.0,
            iterations=iters,
            random_seed=42,
            verbose=200,
            od_type="Iter",     # early stopping
            od_wait=od_wait,    # patience
            thread_count=-1,
        )
        clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
        log("[cb] fit done")

        # Calibrate on validation
        log("[cal] isotonic calibration (val) start")
        calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
        calibrator.fit(X_val_, y_val)
        log("[cal] done")

        # Eval / Youden on val
        log("[eval] computing calibrated probabilities + Youden threshold")
        val_proba_cal = calibrator.predict_proba(X_val_)[:, 1]
        th, tpr, fpr = best_threshold_youden(y_val, val_proba_cal)

        proba_test_cal = calibrator.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test_cal >= th).astype(int)

        log(f"[thr≥{thr}] Youden threshold: {th:.3f}  |  TPR: {tpr:.3f}  FPR: {fpr:.3f}")
        log(f"[thr≥{thr}] AUC (val)={roc_auc_score(y_val, val_proba_cal):.3f}  AUC (test)={roc_auc_score(y_test, proba_test_cal):.3f}")

        print(f"[thr≥{thr}] Classification report @ Youden:")
        print(classification_report(y_test, y_pred_test), flush=True)
        print(f"[thr≥{thr}] Confusion matrix:\n{confusion_matrix(y_test, y_pred_test)}", flush=True)

        # Feature importances
        log("[feat] computing feature importances")
        importances = clf.get_feature_importance(train_pool, type="FeatureImportance")
        imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}) \
                 .sort_values("importance", ascending=False).reset_index(drop=True)
        with pd.option_context("display.max_rows", None):
            print(f"[thr≥{thr}] Feature importances (sorted):", flush=True)
            print(imp_df, flush=True)

        # Reliability plots/tables
        if not NO_PLOTS:
            try:
                log("[plot] reliability overall start")
                plot_reliability(f"thr{thr}_overall", y_test, proba_test_cal, thr_dir, n_bins=REL_BINS)
            except Exception as e:
                log(f"[plot][WARN] overall plot failed: {e}")

            try:
                # snow-only slice (build on test indices)
                test_rows = df.iloc[idx_test].copy()
                def _in_snow_row(r):
                    for col in ("origin_weathercode","dest_weathercode","origin_dep_weathercode","dest_arr_weathercode"):
                        if col in test_rows.columns:
                            v = r.get(col)
                            try:
                                if pd.notna(v) and int(v) in SNOW_CODES:
                                    return True
                            except Exception:
                                pass
                    return False
                snow_mask = test_rows.apply(_in_snow_row, axis=1).astype(bool).values
                log(f"[plot] snow-only slice size: {snow_mask.sum()} / {len(test_rows)}")
                if snow_mask.any():
                    plot_reliability(f"thr{thr}_snow_only", y_test[snow_mask], proba_test_cal[snow_mask], thr_dir, n_bins=REL_BINS)
            except Exception as e:
                log(f"[plot][WARN] snow-only plot failed: {e}")

        # Save artifacts
        log("[save] artifacts start")
        cat_model_path = thr_dir / f"catboost_thr{thr}.cbm"
        clf.save_model(str(cat_model_path))
        joblib.dump(calibrator, thr_dir / f"calibrated_thr{thr}.joblib")

        meta = {
            "threshold": thr,
            "threshold_youden": th,
            "tpr_at_threshold": tpr,
            "fpr_at_threshold": fpr,
            "categorical_features": cat_feats,
            "numeric_features": num_feats,
            "input_parquet": str(INPUT),
            "config_used": cfg,
        }
        with open(thr_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        imp_df.to_csv(thr_dir / f"feature_importances_thr{thr}.csv", index=False)
        log("[save] artifacts done")

        # registry entry
        registry[thr] = {
            "model_path": str(cat_model_path),
            "cal_path": str(thr_dir / f"calibrated_thr{thr}.joblib"),
        }

    with open(OUTDIR / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    log(f"[OK] Trained & saved ordinal risk models at: {OUTDIR}")
    log_mem("final")

if __name__ == "__main__":
    main()
