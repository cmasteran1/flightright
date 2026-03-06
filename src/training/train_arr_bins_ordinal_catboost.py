#!/usr/bin/env python3
# src/training/train_arr_bins_ordinal_catboost.py
#
# Ordinal/binned ARRIVAL-delay probability model via calibrated P(delay >= thr).
#
# Pattern B (production-safe, mirrors train_dep_bins_ordinal_catboost.py):
# - DO NOT joblib/pickle any custom Python classes.
# - Save a single deployable JOBLIB that is a plain dict:
#   {
#     "artifact_type": "arr_delay_bins_bundle_v1",
#     "thresholds": [...],
#     "bin_labels": [...],
#     "categorical_features": [...],
#     "numeric_features": [...],
#     "feature_order": [...],
#     "calibrators": {thr: CalibratedClassifierCV, ...},  # OK to pickle sklearn objects
#     "registry": {thr: {"model_path": "...cbm", "cal_path": "...joblib"}, ...},
#     "versions": {...},
#     "created_utc": "...",
#   }
#
# Runtime/inference code loads this dict and implements:
# - enforce_monotone_ge_probs
# - ge_to_bins
#
# Run (config-only):
#   python src/training/train_arr_bins_ordinal_catboost.py data/models/arr_bins_WN_config.json
#
import os
import sys
import json
import time
import signal
import faulthandler
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone as dt_timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV

import joblib
from catboost import CatBoostClassifier, Pool


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))

_t0 = time.time()
_last_step_t = time.time()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, *, step: bool = False) -> None:
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


def _as_data_path(p: Path) -> Path:
    return p if p.is_absolute() else (DATA_ROOT / p).resolve()


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _looks_like_json(p: Path) -> bool:
    return p.suffix.lower() == ".json"


def _resolve_run_args(argv) -> Tuple[Path, Path, Optional[Path]]:
    """
    Supported invocations:
      1) script CONFIG.json
      2) script INPUT.parquet OUTDIR
      3) script INPUT.parquet OUTDIR CONFIG.json
    Returns: (INPUT, OUTDIR, cfg_path_or_None)
    """
    if len(argv) == 2:
        p = Path(argv[1])
        if not _looks_like_json(p):
            raise SystemExit("Config-only mode requires a .json file: python ... train_arr_bins_...py path/to/config.json")
        cfg = _load_json(p)
        in_path = cfg.get("input_features_path") or cfg.get("input_features_unbalanced_path")
        outdir = cfg.get("outdir") or cfg.get("output_dir")
        if not in_path or not outdir:
            raise SystemExit("Config-only mode requires cfg['input_features_path'] (or input_features_unbalanced_path) and cfg['outdir'].")
        return Path(in_path), Path(outdir), p

    if len(argv) == 3:
        return Path(argv[1]), Path(argv[2]), None

    if len(argv) == 4:
        return Path(argv[1]), Path(argv[2]), Path(argv[3])

    raise SystemExit(
        "Usage:\n"
        "  python train_arr_bins_ordinal_catboost.py CONFIG.json\n"
        "  python train_arr_bins_ordinal_catboost.py features_unbalanced.parquet outdir\n"
        "  python train_arr_bins_ordinal_catboost.py features_unbalanced.parquet outdir config.json"
    )


def _safe_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import sklearn
        versions["sklearn"] = getattr(sklearn, "__version__", "unknown")
    except Exception:
        versions["sklearn"] = "unknown"
    try:
        import catboost
        versions["catboost"] = getattr(catboost, "__version__", "unknown")
    except Exception:
        versions["catboost"] = "unknown"
    try:
        versions["numpy"] = getattr(np, "__version__", "unknown")
    except Exception:
        versions["numpy"] = "unknown"
    try:
        versions["pandas"] = getattr(pd, "__version__", "unknown")
    except Exception:
        versions["pandas"] = "unknown"
    try:
        versions["joblib"] = getattr(joblib, "__version__", "unknown")
    except Exception:
        versions["joblib"] = "unknown"
    return versions


# ------------------------------ ordinal helpers ------------------------------

def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
    p = p_ge.copy()
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j - 1])
    return p


def ge_to_bins(p_ge15, p_ge30, p_ge45, p_ge60) -> np.ndarray:
    p_ge = enforce_monotone_ge_probs(np.vstack([p_ge15, p_ge30, p_ge45, p_ge60]).T)
    p15, p30, p45, p60 = p_ge[:, 0], p_ge[:, 1], p_ge[:, 2], p_ge[:, 3]
    p_lt15 = 1.0 - p15
    p_15_30 = np.maximum(0.0, p15 - p30)
    p_30_45 = np.maximum(0.0, p30 - p45)
    p_45_60 = np.maximum(0.0, p45 - p60)
    p_ge60 = np.maximum(0.0, p60)
    P = np.vstack([p_lt15, p_15_30, p_30_45, p_45_60, p_ge60]).T
    Z = P.sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    return P / Z


# ------------------------------ calibration diagnostics ------------------------------

def reliability_table(y_true, p_pred, n_bins=10) -> pd.DataFrame:
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
    out["bin_left"] = out["bin"].apply(lambda iv: iv.left if pd.notna(iv) else np.nan)
    out["bin_right"] = out["bin"].apply(lambda iv: iv.right if pd.notna(iv) else np.nan)
    out["bin_mid"] = (out["bin_left"].astype(float) + out["bin_right"].astype(float)) / 2.0
    return out


def plot_reliability(tag, y_true, p_pred, outdir: Path, n_bins=10):
    tab = reliability_table(y_true, p_pred, n_bins=n_bins)
    csv_path = outdir / f"calibration_{tag}_table.csv"
    tab.drop(columns=["bin"], errors="ignore").to_csv(csv_path, index=False)
    log(f"[SAVE] reliability table -> {csv_path}")

    if NO_PLOTS:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[WARN] matplotlib not available for plots: {e}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect")
    ax.plot(tab["mean_pred"], tab["empirical_pos_rate"], marker="o", linewidth=1.5, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability curve ({tag})")
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = outdir / f"calibration_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log(f"[SAVE] reliability plot -> {png_path}")


def best_threshold_youden(y_true, y_proba) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])


# ------------------------------ data splits (mirrors dep pattern) ------------------------------

def _time_eval_split(df: pd.DataFrame, *, date_col: str, last_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if date_col not in df.columns:
        raise RuntimeError(f"Missing {date_col} for time-based eval split.")
    fd = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = fd.max() - pd.Timedelta(days=int(last_days))
    is_eval = fd >= cutoff
    return df.loc[~is_eval].reset_index(drop=True), df.loc[is_eval].reset_index(drop=True)


def _split_train_for_calibration(df_train: pd.DataFrame, y_col: str, *, cal_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibration comes ONLY from the training period (so eval stays "future" clean).
    Stratify on the target label for that threshold.
    """
    cal_size = float(cal_size)
    cal_size = min(max(cal_size, 0.05), 0.95)

    y = pd.to_numeric(df_train[y_col], errors="coerce").fillna(0).astype(int).values
    idx = np.arange(len(df_train))

    if len(df_train) < 50 or len(np.unique(y)) < 2:
        # too small / degenerate
        return df_train.copy(), df_train.copy()

    idx_fit, idx_cal, _, _ = train_test_split(
        idx, y, test_size=cal_size, random_state=int(seed), stratify=y
    )
    df_fit = df_train.iloc[idx_fit].reset_index(drop=True)
    df_cal = df_train.iloc[idx_cal].reset_index(drop=True)
    return df_fit, df_cal


# ------------------------------ feature handling ------------------------------

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(~pd.isna(s), "Unknown")
    s = s.map(lambda v: "Unknown" if str(v).strip().lower() in ("", "nan", "none") else str(v))
    return s.astype("object")


def _resolve_feature_lists(cfg: dict, df_all: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    feat_cfg = cfg.get("features") or {}
    cat_feats = [str(x) for x in (feat_cfg.get("categorical") or [])]
    num_feats = [str(x) for x in (feat_cfg.get("numeric") or [])]
    requested = cat_feats + num_feats

    present = [c for c in requested if c in df_all.columns]

    # Exclusions: keep in parquet for validation, but never train on them
    train_cfg = cfg.get("training") or {}
    exclude_prefixes = tuple(train_cfg.get("exclude_feature_prefixes") or ["y_dep_ge"])
    exclude_cols = set(train_cfg.get("exclude_feature_columns") or [])

    present = [
        c for c in present
        if (c not in exclude_cols) and (not any(c.startswith(p) for p in exclude_prefixes))
    ]

    cat_feats = [c for c in cat_feats if c in present]
    num_feats = [c for c in num_feats if c in present]

    if not present:
        raise RuntimeError("No requested features exist after exclusions. Check cfg['features'].")

    return present, cat_feats, num_feats


def _make_pool(df: pd.DataFrame, feature_order: List[str], cat_feats: List[str], y: Optional[pd.Series] = None) -> Pool:
    X = df[feature_order].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    if y is None:
        return Pool(X, cat_features=cat_feats)
    return Pool(X, y.astype(int), cat_features=cat_feats)


def _load_balanced_or_sample(cfg: dict, thr: int, df_train_period: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors dep trainer intent: use balanced dataset per threshold if present,
    else sample from training period to max_rows with stratification.
    """
    ycol = f"y_arr_ge{thr}"
    if ycol not in df_train_period.columns:
        raise RuntimeError(f"Missing label column in training data: {ycol}")

    train_cfg = cfg.get("training") or {}
    max_rows = int(train_cfg.get("max_train_rows", 20000))
    seed = int(train_cfg.get("seed", 1337))

    # Candidate balanced templates
    fb = cfg.get("feature_balance") or {}
    tpl = fb.get("output_template") or cfg.get("output_features_balanced_path_template")

    cand = None
    if tpl:
        # support both {thr} and {target} styles
        try:
            rel = tpl.format(thr=thr, target="arr")
        except Exception:
            rel = tpl
        cand = _as_data_path(Path(rel))

    if cand is not None and cand.exists():
        log(f"[DATA] Using balanced training file thr={thr}: {cand}")
        df = pd.read_parquet(cand)
    else:
        log(f"[DATA] No balanced file for thr={thr}; sampling from unbalanced training period.")
        df = df_train_period.copy()

    df = df.dropna(subset=[ycol]).reset_index(drop=True)

    if max_rows > 0 and len(df) > max_rows:
        y = df[ycol].astype(int).values
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng = np.random.default_rng(seed + thr)

        # preserve class balance in the source df (balanced files likely ~50/50 already)
        pos_keep = int(round(max_rows * (len(pos) / max(len(df), 1))))
        pos_keep = max(1, min(pos_keep, len(pos)))
        neg_keep = max(1, min(max_rows - pos_keep, len(neg)))

        pick = np.concatenate([
            rng.choice(pos, size=pos_keep, replace=False),
            rng.choice(neg, size=neg_keep, replace=False),
        ])
        df = df.iloc[pick].sample(frac=1.0, random_state=seed + thr).reset_index(drop=True)

    log(f"[DATA] thr={thr} train_rows={len(df):,} pos_frac={df[ycol].mean():.3f}")
    return df


# ------------------------------ main ------------------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    INPUT, OUTDIR, CFG_PATH = _resolve_run_args(sys.argv)
    cfg = _load_json(CFG_PATH) if CFG_PATH is not None else {}

    in_path = _as_data_path(INPUT)
    outdir = _as_data_path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = cfg.get("thresholds", [15, 30, 45, 60])
    thresholds = [int(x) for x in thresholds]
    if len(thresholds) != 4:
        raise SystemExit("This trainer expects exactly 4 thresholds (e.g., 15/30/45/60).")

    # Eval is future data (time-based)
    eval_cfg = cfg.get("eval") or {}
    eval_last_days = int(eval_cfg.get("last_days", 90))
    cal_size = float(eval_cfg.get("calibration_frac_from_train", 0.15))
    seed = int((cfg.get("training") or {}).get("seed", 1337))

    # CatBoost params (FAST_TRAIN supported like dep trainer)
    cb_cfg = cfg.get("catboost") or {}
    iterations = int(cb_cfg.get("iterations", 1200))
    depth = int(cb_cfg.get("depth", 8))
    lr = float(cb_cfg.get("learning_rate", 0.08))
    if FAST:
        iterations = min(iterations, 350)
        depth = min(depth, 8)
        lr = max(lr, 0.12)
        log("[FAST_TRAIN] enabled: reducing iterations / boosting speed")

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=float(cb_cfg.get("l2_leaf_reg", 3.0)),
        random_seed=int(cb_cfg.get("seed", seed)),
        verbose=int(cb_cfg.get("verbose", 200)),
        thread_count=int(cb_cfg.get("thread_count", -1)),
    )

    log(f"[LOAD] reading features: {in_path}", step=True)
    df_all = pd.read_parquet(in_path)

    # Resolve features (exclude y_dep_ge* etc.)
    feature_order, cat_feats, num_feats = _resolve_feature_lists(cfg, df_all)
    log(f"[FEATURES] using {len(feature_order)} features ({len(cat_feats)} categorical, {len(num_feats)} numeric)", step=True)

    # Ensure labels exist
    for thr in thresholds:
        ycol = f"y_arr_ge{thr}"
        if ycol not in df_all.columns:
            raise SystemExit(f"Missing label column: {ycol}")

    # Split train period vs future eval
    df_train_period, df_eval = _time_eval_split(df_all, date_col="FlightDate", last_days=eval_last_days)
    log(f"[SPLIT] train_period={len(df_train_period):,} eval_future={len(df_eval):,} last_days={eval_last_days}", step=True)

    # Eval pools shared by threshold (labels differ, but X is same)
    X_eval_df = df_eval[feature_order].copy()
    for c in cat_feats:
        X_eval_df[c] = as_str(X_eval_df[c])

    models: Dict[int, CatBoostClassifier] = {}
    calibrators: Dict[int, Any] = {}
    registry: Dict[int, Dict[str, str]] = {}
    metrics: Dict[str, Any] = {
        "artifact_type": "arr_train_metrics_v1",
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "thresholds": thresholds,
        "eval_last_days": eval_last_days,
        "n_eval": int(len(df_eval)),
        "per_threshold": {},
        "versions": _safe_versions(),
    }

    for thr in thresholds:
        ycol = f"y_arr_ge{thr}"

        # Train df: balanced file if exists; else sample to max_train_rows
        df_thr = _load_balanced_or_sample(cfg, thr, df_train_period)

        # Calibration split from training period (keeps eval pristine)
        df_fit, df_cal = _split_train_for_calibration(df_thr, ycol, cal_size=cal_size, seed=seed + thr)
        log(f"[CAL] thr={thr} fit={len(df_fit):,} cal={len(df_cal):,} cal_frac={cal_size}", step=True)

        pool_fit = _make_pool(df_fit, feature_order, cat_feats, y=df_fit[ycol])
        pool_cal = _make_pool(df_cal, feature_order, cat_feats, y=df_cal[ycol])

        model = CatBoostClassifier(**params)
        log(f"[TRAIN] thr={thr} starting CatBoost fit...", step=True)
        model.fit(pool_fit)

        # Save model to .cbm (like dep trainer)
        model_path = outdir / f"arr_thr{thr}.cbm"
        model.save_model(str(model_path))
        log(f"[SAVE] model -> {model_path}")

        # Calibrate using held-out CAL subset (not eval)
        # (sklearn warning about cv='prefit' is fine; mirrors your current usage)
        base = model
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")

        # Build calibration matrix the same way we built training/eval matrices:
        X_cal = df_cal[feature_order].copy()
        for c in cat_feats:
            X_cal[c] = as_str(X_cal[c])

        y_cal = df_cal[ycol].astype(int).values
        cal.fit(X_cal, y_cal)

        cal_path = outdir / f"arr_thr{thr}_calibrator.joblib"
        joblib.dump(cal, cal_path)
        log(f"[SAVE] calibrator -> {cal_path}")

        # Evaluate on FUTURE eval set
        y_eval = df_eval[ycol].astype(int).values
        p_eval = cal.predict_proba(X_eval_df)[:, 1].astype(float)

        # metrics
        auc = float(roc_auc_score(y_eval, p_eval)) if len(np.unique(y_eval)) == 2 else float("nan")
        ap = float(average_precision_score(y_eval, p_eval)) if len(np.unique(y_eval)) == 2 else float("nan")
        ll = float(log_loss(y_eval, np.clip(p_eval, 1e-6, 1 - 1e-6)))
        brier = float(brier_score_loss(y_eval, p_eval))
        acc = float(accuracy_score(y_eval, (p_eval >= 0.5).astype(int)))
        youden_thr, youden_tpr, youden_fpr = best_threshold_youden(y_eval, p_eval)
        cm_05 = confusion_matrix(y_eval, (p_eval >= 0.5).astype(int)).tolist()
        cm_y = confusion_matrix(y_eval, (p_eval >= youden_thr).astype(int)).tolist()

        metrics["per_threshold"][str(thr)] = {
            "train_rows": int(len(df_thr)),
            "fit_rows": int(len(df_fit)),
            "cal_rows": int(len(df_cal)),
            "eval_rows": int(len(df_eval)),
            "pos_frac_train": float(df_thr[ycol].mean()),
            "pos_frac_eval": float(float(np.mean(y_eval))),
            "auc": auc,
            "ap": ap,
            "logloss": ll,
            "brier": brier,
            "acc_at_0.5": acc,
            "youden_threshold": float(youden_thr),
            "youden_tpr": float(youden_tpr),
            "youden_fpr": float(youden_fpr),
            "confusion_at_0.5": cm_05,
            "confusion_at_youden": cm_y,
        }

        log(f"[EVAL thr={thr}] AUC={auc:.4f} AP={ap:.4f} LogLoss={ll:.4f} Brier={brier:.4f} Acc@0.5={acc:.4f}")

        # Reliability outputs (table + optional plot)
        plot_reliability(f"arr_thr{thr}", y_eval, p_eval, outdir, n_bins=10)

        # Keep in-memory refs for bundling
        models[thr] = model
        calibrators[thr] = cal
        registry[thr] = {"model_path": str(model_path), "cal_path": str(cal_path)}

    # Write combined eval predictions for bins (like your old trainer, but eval is future-only)
    p_ge = []
    for thr in thresholds:
        cal = calibrators[thr]
        p_ge.append(cal.predict_proba(X_eval_df)[:, 1].astype(float))
    p_ge = enforce_monotone_ge_probs(np.vstack(p_ge).T)
    Pbins = ge_to_bins(p_ge[:, 0], p_ge[:, 1], p_ge[:, 2], p_ge[:, 3])

    df_eval_out = df_eval.copy()
    df_eval_out["p_ge15"] = p_ge[:, 0]
    df_eval_out["p_ge30"] = p_ge[:, 1]
    df_eval_out["p_ge45"] = p_ge[:, 2]
    df_eval_out["p_ge60"] = p_ge[:, 3]
    df_eval_out["p_lt15"] = Pbins[:, 0]
    df_eval_out["p_15_30"] = Pbins[:, 1]
    df_eval_out["p_30_45"] = Pbins[:, 2]
    df_eval_out["p_45_60"] = Pbins[:, 3]
    df_eval_out["p_ge60_bin"] = Pbins[:, 4]

    eval_out_name = str(cfg.get("eval_output_parquet_name", "arr_eval_with_probs.parquet"))
    eval_out_path = outdir / eval_out_name
    df_eval_out.to_parquet(eval_out_path, index=False)
    log(f"[SAVE] eval predictions -> {eval_out_path}")

    # Save metrics JSON
    metrics_name = str(cfg.get("metrics_output_json_name", "arr_train_metrics.json"))
    (outdir / metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"[SAVE] metrics -> {outdir / metrics_name}")

    # Deploy bundle dict (production-safe)
    bundle_name = str(cfg.get("deploy_bundle_joblib_name", "arr_delay_bins_bundle.joblib")).strip() or "arr_delay_bins_bundle.joblib"
    deploy_path = outdir / bundle_name

    bundle: Dict[str, Any] = {
        "artifact_type": "arr_delay_bins_bundle_v1",
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "thresholds": thresholds,
        "bin_labels": ["<15", "15-30", "30-45", "45-60", ">=60"],
        "categorical_features": cat_feats,
        "numeric_features": num_feats,
        "feature_order": feature_order,
        "registry": registry,
        "calibrators": calibrators,
        "versions": _safe_versions(),
        "preprocess": {"categorical_na_value": "Unknown"},
    }

    joblib.dump(bundle, deploy_path)
    log(f"[SAVE] deploy bundle -> {deploy_path}")
    log("[DONE] training complete.", step=True)


if __name__ == "__main__":
    main()