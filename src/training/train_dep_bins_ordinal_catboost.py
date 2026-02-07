#!/usr/bin/env python3
# src/training/train_dep_bins_ordinal_catboost.py
#
# Ordinal/binned departure-delay probability model via calibrated P(delay >= thr).
#
# Run from REPO_ROOT:
#   python src/training/train_dep_bins_ordinal_catboost.py \
#       data/processed/features_dep_WN_50_23-25_unbalanced.parquet \
#       data/models/dep_bins_WN \
#       data/models/dep_bins_WN_config.json
#
# Notes:
# - You can train each threshold on its OWN balanced feature parquet (recommended) by
#   setting cfg["per_threshold_train_paths"].
# - Calibration/evaluation can be done on an unbalanced eval parquet (recommended):
#     cfg["eval_features_path"] = "..._eval_unbalanced.parquet"
# - This version fits calibrators on an UNBALANCED calibration split derived from
#   cfg["eval_features_path"] when provided.
#
import os, sys, json, time, signal, faulthandler
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, log_loss, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool


FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))

_t0 = time.time()
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

_last_step_t = time.time()
def log(msg, *, step=False):
    global _last_step_t
    now = time.time()
    total = now - _t0
    delta = now - _last_step_t
    if step:
        _last_step_t = now
    print(f"[{_now()} +{total:7.2f}s Δ{delta:6.2f}s] {msg}", flush=True)

def enable_periodic_traces():
    if TRACE_EVERY_SEC > 0:
        faulthandler.dump_traceback_later(TRACE_EVERY_SEC, repeat=True)
        log(f"[trace] periodic faulthandler enabled every {TRACE_EVERY_SEC}s")

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

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(~s.isna(), "Unknown").astype("object")
    return s.map(lambda v: "Unknown" if pd.isna(v) else str(v))

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

def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
    """
    p_ge shape = (n, k) for thresholds increasing.
    Enforce p_ge[:,0] >= p_ge[:,1] >= ... by cumulative min.
    """
    p = p_ge.copy()
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j-1])
    return p

def ge_to_bins(p_ge15, p_ge30, p_ge45, p_ge60):
    # ensure monotone
    p_ge = enforce_monotone_ge_probs(np.vstack([p_ge15, p_ge30, p_ge45, p_ge60]).T)
    p15, p30, p45, p60 = p_ge[:, 0], p_ge[:, 1], p_ge[:, 2], p_ge[:, 3]

    p_lt15  = 1.0 - p15
    p_15_30 = np.maximum(0.0, p15 - p30)
    p_30_45 = np.maximum(0.0, p30 - p45)
    p_45_60 = np.maximum(0.0, p45 - p60)
    p_ge60  = np.maximum(0.0, p60)

    P = np.vstack([p_lt15, p_15_30, p_30_45, p_45_60, p_ge60]).T
    # renormalize small numeric drift
    Z = P.sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    return P / Z

def true_bin_from_delay(dep_delay_min: pd.Series) -> pd.Series:
    d = pd.to_numeric(dep_delay_min, errors="coerce").fillna(0.0)
    bins = pd.cut(
        d,
        bins=[-np.inf, 15, 30, 45, 60, np.inf],
        labels=["< 15 min", "15–30 min", "30–45 min", "45–60 min", "≥ 60 min"],
        right=False
    )
    return bins.astype("object")

def make_reason_row(row) -> str:
    reasons = []
    # simple, controllable heuristics
    try:
        if pd.notna(row.get("flightnum_od_low_support")) and int(row["flightnum_od_low_support"]) == 1:
            reasons.append("limited flight-number history")
    except Exception:
        pass

    try:
        v = row.get("flightnum_od_depdelay_mean_lastN")
        if pd.notna(v) and float(v) >= 10:
            reasons.append("this flight/route often departs late")
    except Exception:
        pass

    try:
        v = row.get("carrier_depdelay_mean_lastNdays")
        if pd.notna(v) and float(v) >= 10:
            reasons.append("carrier delays elevated recently")
    except Exception:
        pass

    try:
        v = row.get("carrier_origin_depdelay_mean_lastNdays")
        if pd.notna(v) and float(v) >= 10:
            reasons.append("origin station running late recently")
    except Exception:
        pass

    try:
        w = row.get("origin_daily_weathercode")
        if pd.notna(w):
            reasons.append("weather that day")
    except Exception:
        pass

    if not reasons:
        return "no strong delay signals detected"
    return "; ".join(reasons[:3])

def fmt_probs(p_ge15, p_ge30, p_ge45, p_ge60) -> str:
    # avoid the NaN strings you were seeing: coerce + format
    def f(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.3f}"
    return f"P≥15={f(p_ge15)}, P≥30={f(p_ge30)}, P≥45={f(p_ge45)}, P≥60={f(p_ge60)}"

def _split_unbalanced_for_cal_and_eval(df_unbal: pd.DataFrame, *, cal_size: float, seed: int):
    """
    Split an unbalanced dataframe into (calibration, eval) without forcing class balance.
    We stratify on y>=15 only to keep both classes present in smaller samples.
    """
    if len(df_unbal) < 20:
        return df_unbal.copy(), df_unbal.copy()

    d = pd.to_numeric(df_unbal.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
    y15 = (d >= 15).astype(int).values
    idx = np.arange(len(df_unbal))

    cal_size = float(cal_size)
    cal_size = min(max(cal_size, 0.05), 0.95)

    idx_cal, idx_eval, _, _ = train_test_split(
        idx, y15,
        test_size=(1.0 - cal_size),
        random_state=int(seed),
        stratify=y15
    )
    df_cal = df_unbal.iloc[idx_cal].reset_index(drop=True)
    df_eval = df_unbal.iloc[idx_eval].reset_index(drop=True)
    return df_cal, df_eval


# ------------------------------ main ------------------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    if len(sys.argv) not in (3, 4):
        print("Usage: python train_dep_bins_ordinal_catboost.py path/to/features_unbalanced.parquet path/to/outdir [path/to/config.json]")
        sys.exit(1)

    INPUT = Path(sys.argv[1])
    OUTDIR = Path(sys.argv[2])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = {
        "thresholds": [15, 30, 45, 60],
        "reliability_bins": 10,
        "iterations": 4000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "od_wait": 200,
        "random_seed": 42,

        # You control feature list here (explicit, editable)
        "categorical_features": [
            "Origin","Dest","od_pair","Reporting_Airline",
            "dep_dow","sched_dep_hour","is_holiday","aircraft_type",
            "origin_daily_weathercode","origin_dep_hour_weathercode",
        ],
        "numeric_features": [
            "origin_temp_max_K","origin_temp_min_K","origin_daily_precip_sum_mm",
            "origin_daily_windspeed_max_kmh",
            "origin_dep_temp_K","origin_dep_precip_mm","origin_dep_windspeed_kmh",
            "flightnum_od_depdelay_mean_lastN","flightnum_od_support_count_lastNd",
            "flightnum_od_low_support",
            "carrier_depdelay_mean_lastNdays","carrier_origin_depdelay_mean_lastNdays",
            "turn_time_hours",
        ],

        # Unbalanced eval parquet path (recommended). We'll split it into:
        #   - unbalanced calibration (fit isotonic here)
        #   - unbalanced evaluation (compute bins here)
        "eval_features_path": "",

        # Optional: train each threshold on a different balanced features parquet:
        # {"15": "...parquet", "30": "...parquet", ...}
        "per_threshold_train_paths": {},

        # For per-threshold training, we still split balanced parquet into
        # train / early-stop / balanced sanity test.
        "split": {"test_size": 0.20, "val_size": 0.20},  # used only when per_threshold_train_paths is NOT provided

        # Fraction of the unbalanced eval parquet used for calibration.
        # Remaining fraction used for unbalanced evaluation.
        "unbalanced_cal_size": 0.50,

        "max_rows": None,
    }

    if len(sys.argv) == 4:
        with open(Path(sys.argv[3]), "r") as f:
            cfg.update(json.load(f))

    if FAST:
        cfg["iterations"] = min(int(cfg.get("iterations", 4000)), 1200)
        cfg["depth"] = min(int(cfg.get("depth", 8)), 6)
        cfg["od_wait"] = min(int(cfg.get("od_wait", 200)), 120)
        cfg["learning_rate"] = max(float(cfg.get("learning_rate", 0.03)), 0.05)
        log("[FAST] using lighter CatBoost settings")

    THRESHOLDS = [int(x) for x in cfg.get("thresholds", [15, 30, 45, 60])]
    REL_BINS = int(cfg.get("reliability_bins", 10))
    SEED = int(cfg.get("random_seed", 42))

    log(f"Loading data: {INPUT}")
    df_all = pd.read_parquet(INPUT)
    if cfg.get("max_rows"):
        df_all = df_all.sample(n=int(cfg["max_rows"]), random_state=SEED).reset_index(drop=True)
    log(f"Rows={len(df_all)} Cols={len(df_all.columns)}", step=True)

    # pick features by intersection (explicit control still comes from config lists)
    cat_feats = [c for c in cfg["categorical_features"] if c in df_all.columns]
    num_feats = [c for c in cfg["numeric_features"] if c in df_all.columns]
    dropped_missing = (len(cfg["categorical_features"]) - len(cat_feats)) + (len(cfg["numeric_features"]) - len(num_feats))
    log(f"[features] cats={len(cat_feats)} nums={len(num_feats)} dropped_missing={dropped_missing}", step=True)

    # resolved feature list artifact
    (OUTDIR / "resolved_features.json").write_text(
        json.dumps({"categorical": cat_feats, "numeric": num_feats}, indent=2)
    )
    log(f"[SAVE] resolved features -> {OUTDIR / 'resolved_features.json'}")

    def build_X(df: pd.DataFrame) -> pd.DataFrame:
        X = df[cat_feats + num_feats].copy()
        for c in cat_feats:
            X[c] = as_str(X[c])
        for c in num_feats:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().any():
                med = X[c].median()
                X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
        return X

    def make_y(df: pd.DataFrame, thr: int) -> np.ndarray:
        col = f"y_dep_ge{thr}"
        if col in df.columns:
            return df[col].astype(int).values
        d = pd.to_numeric(df.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
        return (d >= int(thr)).astype(int).values

    # ------------------------------
    # Unbalanced calibration + unbalanced evaluation source:
    # USE cfg["eval_features_path"] (your JSON already has it)
    # ------------------------------
    eval_path = str(cfg.get("eval_features_path") or "").strip()
    if not eval_path:
        raise ValueError("cfg['eval_features_path'] must be set to an unbalanced parquet for calibration/eval.")

    df_unbal = pd.read_parquet(eval_path)
    log(f"[unbalanced] loaded eval_features_path={eval_path} rows={len(df_unbal)}", step=True)

    cal_frac = float(cfg.get("unbalanced_cal_size", 0.50))
    df_cal_unbal, df_eval_unbal = _split_unbalanced_for_cal_and_eval(df_unbal, cal_size=cal_frac, seed=SEED)
    unbal_source = f"eval_features_path:{eval_path}"
    log(f"[unbalanced] split -> cal={len(df_cal_unbal)} eval={len(df_eval_unbal)} (cal_size={cal_frac:.2f})", step=True)

    # train sources per-threshold
    per_thr_paths = cfg.get("per_threshold_train_paths") or {}
    per_thr_paths = {str(k): str(v) for k, v in per_thr_paths.items()}

    # If no per-threshold train paths, we do a single split from df_all
    if not per_thr_paths:
        base_y = make_y(df_all, 15)
        idx = np.arange(len(df_all))
        test_size = float(cfg.get("split", {}).get("test_size", 0.20))
        val_size = float(cfg.get("split", {}).get("val_size", 0.20))

        log("[split] stratified splits on dep>=15 (single INPUT mode)", step=True)
        idx_train, idx_test, _, _ = train_test_split(
            idx, base_y, test_size=test_size, random_state=SEED, stratify=base_y
        )
        rel_val = val_size / max(1e-9, (1.0 - test_size))
        idx_train, idx_val, _, _ = train_test_split(
            idx_train, base_y[idx_train], test_size=rel_val, random_state=SEED, stratify=base_y[idx_train]
        )
        log(f"[split] sizes train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

        df_train_base = df_all.iloc[idx_train].reset_index(drop=True)
        df_es_val     = df_all.iloc[idx_val].reset_index(drop=True)   # early stopping
        df_test_base  = df_all.iloc[idx_test].reset_index(drop=True)  # balanced-ish sanity
    else:
        df_train_base = df_es_val = df_test_base = None  # unused

    registry = {}
    cat_idx = [i for i, c in enumerate(cat_feats)]  # column indices in X

    for thr in THRESHOLDS:
        log("="*18 + f" Train dep >= {thr} " + "="*18, step=True)
        thr_dir = OUTDIR / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        # choose train set
        if per_thr_paths:
            p = per_thr_paths.get(str(thr))
            if not p:
                raise ValueError(f"per_threshold_train_paths missing threshold {thr}")
            df_tr = pd.read_parquet(p)

            base_y = make_y(df_tr, thr)
            idx = np.arange(len(df_tr))

            idx_train, idx_temp, _, _ = train_test_split(
                idx, base_y, test_size=0.40, random_state=SEED, stratify=base_y
            )
            idx_es, idx_test, _, _ = train_test_split(
                idx_temp, base_y[idx_temp], test_size=0.50, random_state=SEED, stratify=base_y[idx_temp]
            )

            df_train = df_tr.iloc[idx_train].reset_index(drop=True)
            df_es    = df_tr.iloc[idx_es].reset_index(drop=True)     # early stopping
            df_test  = df_tr.iloc[idx_test].reset_index(drop=True)   # balanced sanity test

            log(f"[data] per-threshold train={p} sizes train={len(df_train)} earlystop={len(df_es)} test={len(df_test)}")
        else:
            df_train, df_es, df_test = df_train_base, df_es_val, df_test_base

        # Unbalanced calibration/eval are shared across thresholds (labels differ per thr)
        df_cal_thr  = df_cal_unbal
        df_eval_thr = df_eval_unbal

        X_train = build_X(df_train)
        y_train = make_y(df_train, thr)

        X_es = build_X(df_es)
        y_es = make_y(df_es, thr)

        X_test_bal = build_X(df_test)
        y_test_bal = make_y(df_test, thr)

        X_cal = build_X(df_cal_thr)
        y_cal = make_y(df_cal_thr, thr)

        X_eval = build_X(df_eval_thr)
        y_eval = make_y(df_eval_thr, thr)

        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        es_pool    = Pool(X_es,    y_es,    cat_features=cat_idx)

        iters = int(cfg.get("iterations", 4000))
        depth = int(cfg.get("depth", 8))
        lr = float(cfg.get("learning_rate", 0.03))
        l2 = float(cfg.get("l2_leaf_reg", 5.0))
        od_wait = int(cfg.get("od_wait", 200))

        log(f"[cb] fit start (iter={iters} depth={depth} lr={lr} l2={l2} od_wait={od_wait})")
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2,
            iterations=iters,
            random_seed=SEED,
            verbose=200,
            od_type="Iter",
            od_wait=od_wait,
            thread_count=-1,
        )
        clf.fit(train_pool, eval_set=es_pool, use_best_model=True)
        log("[cb] fit done")

        # CALIBRATE ON UNBALANCED CAL SPLIT (from eval_features_path)
        log(f"[cal] isotonic calibration on UNBALANCED cal split (source={unbal_source}) size={len(df_cal_thr)}")
        calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
        calibrator.fit(X_cal, y_cal)
        log("[cal] done")

        # Choose Youden threshold on UNBALANCED cal (calibrated probs)
        cal_proba = calibrator.predict_proba(X_cal)[:, 1]
        th, tpr, fpr = best_threshold_youden(y_cal, cal_proba)
        auc_cal = roc_auc_score(y_cal, cal_proba) if len(np.unique(y_cal)) > 1 else float("nan")

        # Balanced sanity test (from training parquet split)
        test_bal_proba = calibrator.predict_proba(X_test_bal)[:, 1]
        auc_test_bal = roc_auc_score(y_test_bal, test_bal_proba) if len(np.unique(y_test_bal)) > 1 else float("nan")
        y_pred_bal = (test_bal_proba >= th).astype(int)

        # Unbalanced eval (held-out from eval_features_path)
        eval_proba = calibrator.predict_proba(X_eval)[:, 1]
        auc_eval = roc_auc_score(y_eval, eval_proba) if len(np.unique(y_eval)) > 1 else float("nan")
        y_pred_eval = (eval_proba >= th).astype(int)

        log(f"[thr≥{thr}] Youden threshold (from UNBAL cal)={th:.3f} | TPR={tpr:.3f} FPR={fpr:.3f}")
        log(f"[thr≥{thr}] AUC cal_unbal={auc_cal:.3f} | AUC test_bal_sanity={auc_test_bal:.3f} | AUC eval_unbal={auc_eval:.3f}")

        print(f"[thr≥{thr}] Classification report on BALANCED sanity test @ Youden(from unbal cal):", flush=True)
        print(classification_report(y_test_bal, y_pred_bal), flush=True)
        print(f"[thr≥{thr}] Confusion matrix (balanced sanity):\n{confusion_matrix(y_test_bal, y_pred_bal)}", flush=True)

        print(f"[thr≥{thr}] Classification report on UNBALANCED eval @ Youden(from unbal cal):", flush=True)
        print(classification_report(y_eval, y_pred_eval), flush=True)
        print(f"[thr≥{thr}] Confusion matrix (unbalanced eval):\n{confusion_matrix(y_eval, y_pred_eval)}", flush=True)

        if not NO_PLOTS:
            try:
                plot_reliability(f"dep_ge{thr}_unbal_cal", y_cal, cal_proba, thr_dir, n_bins=REL_BINS)
            except Exception as e:
                log(f"[plot][WARN] reliability (unbal cal) failed: {e}")
            try:
                plot_reliability(f"dep_ge{thr}_unbal_eval", y_eval, eval_proba, thr_dir, n_bins=REL_BINS)
            except Exception as e:
                log(f"[plot][WARN] reliability (unbal eval) failed: {e}")

        # save artifacts
        cat_model_path = thr_dir / f"catboost_dep_ge{thr}.cbm"
        clf.save_model(str(cat_model_path))
        joblib.dump(calibrator, thr_dir / f"calibrated_dep_ge{thr}.joblib")

        meta = {
            "threshold": thr,
            "threshold_youden": th,
            "tpr_at_threshold": tpr,
            "fpr_at_threshold": fpr,
            "categorical_features": cat_feats,
            "numeric_features": num_feats,
            "input_features_parquet": str(INPUT),
            "train_source": per_thr_paths.get(str(thr), str(INPUT)),
            "unbalanced_source": unbal_source,
            "unbalanced_cal_rows": int(len(df_cal_thr)),
            "unbalanced_eval_rows": int(len(df_eval_thr)),
            "unbalanced_cal_fraction": float(cal_frac),
            "config_used": cfg,
            "auc_cal_unbalanced": auc_cal,
            "auc_test_balanced_sanity": auc_test_bal,
            "auc_eval_unbalanced": auc_eval,
        }
        with open(thr_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        registry[thr] = {
            "model_path": str(cat_model_path),
            "cal_path": str(thr_dir / f"calibrated_dep_ge{thr}.joblib"),
        }

    # ------------------------------
    # Bin distribution evaluation + prediction samples (on UNBALANCED eval split)
    # ------------------------------
    df_eval = df_eval_unbal.copy().reset_index(drop=True)

    X_eval = build_X(df_eval)
    dep_delay = pd.to_numeric(df_eval.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
    df_eval = df_eval.copy()
    df_eval["true_bin"] = true_bin_from_delay(dep_delay)

    # load calibrators and compute p_ge for eval
    p_ge = {}
    for thr in THRESHOLDS:
        cal = joblib.load(registry[thr]["cal_path"])
        p_ge[thr] = cal.predict_proba(X_eval)[:, 1].astype(float)

    # bin probabilities
    P = ge_to_bins(p_ge[15], p_ge[30], p_ge[45], p_ge[60])
    bin_labels = ["< 15 min", "15–30 min", "30–45 min", "45–60 min", "≥ 60 min"]
    pred_idx = np.argmax(P, axis=1)
    df_eval["pred_bin"] = pd.Series([bin_labels[i] for i in pred_idx], index=df_eval.index)

    # eval metrics
    y_true_mc = pd.Categorical(df_eval["true_bin"], categories=bin_labels, ordered=True).codes
    y_pred_mc = pred_idx
    try:
        ll = log_loss(y_true_mc, P, labels=list(range(len(bin_labels))))
    except Exception:
        ll = float("nan")
    acc = accuracy_score(y_true_mc, y_pred_mc)
    log(f"[BINS] UNBAL eval logloss={ll:.4f}  acc={acc:.4f}")

    # explanations / probability strings
    df_eval["reason"] = df_eval.apply(make_reason_row, axis=1)
    df_eval["p_lt15"]  = P[:, 0]
    df_eval["p_15_30"] = P[:, 1]
    df_eval["p_30_45"] = P[:, 2]
    df_eval["p_45_60"] = P[:, 3]
    df_eval["p_ge60"]  = P[:, 4]
    df_eval["p_ge15"] = p_ge[15]
    df_eval["p_ge30"] = p_ge[30]
    df_eval["p_ge45"] = p_ge[45]
    df_eval["p_ge60_raw"] = p_ge[60]  # raw p>=60 before bin conversion

    df_eval["predicted_delay_bin"] = df_eval["pred_bin"]
    df_eval["delay_explanation"] = df_eval["reason"]
    df_eval["delay_probabilities"] = [
        fmt_probs(a, b, c, d) for a, b, c, d in zip(df_eval["p_ge15"], df_eval["p_ge30"], df_eval["p_ge45"], df_eval["p_ge60_raw"])
    ]

    # save samples (from UNBAL eval)
    sample_n = int(cfg.get("prediction_samples_n", 500))
    sample = df_eval.sample(n=min(sample_n, len(df_eval)), random_state=SEED).reset_index(drop=True)
    sample_path = OUTDIR / "prediction_samples.parquet"
    sample.to_parquet(sample_path, index=False)
    log(f"[SAVE] prediction samples -> {sample_path}")

    with open(OUTDIR / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    log(f"[SAVE] registry -> {OUTDIR / 'registry.json'}")

    bins_meta = {
        "bin_labels": bin_labels,
        "thresholds": THRESHOLDS,
        "eval_logloss": ll,
        "eval_acc": acc,
        "eval_features_path": str(cfg.get("eval_features_path") or ""),
        "unbalanced_source": unbal_source,
        "unbalanced_cal_rows": int(len(df_cal_unbal)),
        "unbalanced_eval_rows": int(len(df_eval_unbal)),
        "unbalanced_cal_fraction": float(cal_frac),
    }
    with open(OUTDIR / "bins_meta.json", "w") as f:
        json.dump(bins_meta, f, indent=2)
    log(f"[SAVE] bins meta -> {OUTDIR / 'bins_meta.json'}")

    log("[OK] finished training dep bin distribution models")

if __name__ == "__main__":
    main()
