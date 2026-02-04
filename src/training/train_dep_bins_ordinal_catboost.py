#!/usr/bin/env python3
# src/training/train_dep_bins_ordinal_catboost.py
#
# Ordinal (>=15, >=30, >=45, >=60) calibrated binary models -> 5 mutually-exclusive bins:
#   <15, 15-30, 30-45, 45-60, >=60
#
# IMPORTANT: Feature selection is EXPLICIT via config JSON. No guessing.
"""
python src/training/train_dep_bins_ordinal_catboost.py \
  data/processed/WN_50_23-24.parquet \
  models/dep_bins_WN_50_23-24_weather+_standard \
  models/WN_50_23-24_weather+_standard.json

"""
import os, sys, json, time, signal, faulthandler
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

# -------- env toggles --------
FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))  # 0 disables

# -------- logging --------
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

# -------------------- utils --------------------

def best_threshold_youden(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.fillna("Unknown").astype("object")
    return s.map(
        lambda v: "Unknown"
        if pd.isna(v)
        else str(int(v)) if isinstance(v, float) and float(v).is_integer()
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

def clip01(x):
    return np.clip(x, 0.0, 1.0)

BIN_KEYS = ["p_lt15", "p_15_30", "p_30_45", "p_45_60", "p_ge60"]
BIN_HUMAN = {
    "p_lt15": "< 15 min",
    "p_15_30": "15–30 min",
    "p_30_45": "30–45 min",
    "p_45_60": "45–60 min",
    "p_ge60": "≥ 60 min",
}

def make_bins_from_delay(dep_delay_minutes: pd.Series) -> np.ndarray:
    x = pd.to_numeric(dep_delay_minutes, errors="coerce").fillna(0.0).to_numpy()
    out = np.zeros(len(x), dtype=int)
    out[(x >= 15) & (x < 30)] = 1
    out[(x >= 30) & (x < 45)] = 2
    out[(x >= 45) & (x < 60)] = 3
    out[(x >= 60)] = 4
    return out

def compute_bin_probs(p_ge: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    # expects keys 15,30,45,60
    p15 = clip01(p_ge[15])
    p30 = clip01(p_ge[30])
    p45 = clip01(p_ge[45])
    p60 = clip01(p_ge[60])

    # enforce monotone per-row: p15 >= p30 >= p45 >= p60
    p30 = np.minimum(p30, p15)
    p45 = np.minimum(p45, p30)
    p60 = np.minimum(p60, p45)

    out = {}
    out["p_lt15"]  = clip01(1.0 - p15)
    out["p_15_30"] = clip01(p15 - p30)
    out["p_30_45"] = clip01(p30 - p45)
    out["p_45_60"] = clip01(p45 - p60)
    out["p_ge60"]  = clip01(p60)

    s = out["p_lt15"] + out["p_15_30"] + out["p_30_45"] + out["p_45_60"] + out["p_ge60"]
    s = np.where(s <= 0, 1.0, s)
    for k in out:
        out[k] = out[k] / s
    return out

def simple_reason_strings(row: pd.Series, p_ge_row: Dict[int, float], *, max_reasons=3) -> str:
    """
    Small heuristic strings (cheap). You can swap this later for SHAP-based text.
    """
    reasons = []

    # flight history support
    if "flightnum_od_low_support" in row.index and pd.notna(row["flightnum_od_low_support"]):
        if int(row["flightnum_od_low_support"]) == 1:
            reasons.append("limited flight-number history")

    if "flightnum_od_depdelay_mean_lastN" in row.index and pd.notna(row["flightnum_od_depdelay_mean_lastN"]):
        v = float(row["flightnum_od_depdelay_mean_lastN"])
        if v >= 20:
            reasons.append("this flight/route often departs late")

    # carrier baselines
    if "carrier_depdelay_mean_lastNdays" in row.index and pd.notna(row["carrier_depdelay_mean_lastNdays"]):
        v = float(row["carrier_depdelay_mean_lastNdays"])
        if v >= 15:
            reasons.append("carrier delays elevated recently")

    if "carrier_origin_depdelay_mean_lastNdays" in row.index and pd.notna(row["carrier_origin_depdelay_mean_lastNdays"]):
        v = float(row["carrier_origin_depdelay_mean_lastNdays"])
        if v >= 15:
            reasons.append("origin station running late recently")

    # weather
    for wx_col, label in [
        ("origin_dep_hour_weathercode", "weather near departure"),
        ("origin_daily_weathercode", "weather that day"),
    ]:
        if wx_col in row.index and pd.notna(row[wx_col]):
            try:
                code = int(float(row[wx_col]))
                if code not in (0, 1):
                    reasons.append(label)
            except Exception:
                pass

    # congestion
    if "origin_congestion_ratio" in row.index and pd.notna(row["origin_congestion_ratio"]):
        v = float(row["origin_congestion_ratio"])
        if v >= 2.0:
            reasons.append("busy gate environment")

    # long-delay tail
    if p_ge_row.get(60, 0.0) >= 0.25:
        reasons.append("non-trivial risk of long delay")

    if not reasons:
        return "Signals are mixed/weak; model leans on small effects."
    return "; ".join(reasons[:max_reasons])

def resolve_features(
    df: pd.DataFrame,
    cfg: dict,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (cat_feats, num_feats, dropped_feats).
    Enforces explicit control.
    """
    feats = cfg.get("features", {})
    cat = list(feats.get("categorical", []))
    num = list(feats.get("numeric", []))

    strict = bool(cfg.get("strict_features", True))

    requested = cat + num
    missing = [c for c in requested if c not in df.columns]
    if missing and strict:
        raise RuntimeError(
            "Missing requested features (strict_features=true):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    # drop missing if lenient
    cat_ok = [c for c in cat if c in df.columns]
    num_ok = [c for c in num if c in df.columns]
    dropped = [c for c in requested if c not in (cat_ok + num_ok)]

    return cat_ok, num_ok, dropped

# -------------------- main --------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    if len(sys.argv) not in (4,):
        print("Usage: python src/training/train_dep_bins_ordinal_catboost.py path/to/train_ready_dep.parquet path/to/outdir path/to/config.json")
        sys.exit(1)

    INPUT = Path(sys.argv[1])
    OUTDIR = Path(sys.argv[2])
    CFG_PATH = Path(sys.argv[3])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    # Defaults (only training knobs; features come only from cfg.features)
    cfg.setdefault("thresholds", [15, 30, 45, 60])
    cfg.setdefault("reliability_bins", 10)
    cfg.setdefault("iterations", 2000)
    cfg.setdefault("depth", 8)
    cfg.setdefault("od_wait", 120)
    cfg.setdefault("random_seed", 42)
    cfg.setdefault("test_size", 0.20)
    cfg.setdefault("val_size", 0.20)
    cfg.setdefault("save_prediction_samples_n", 2000)
    cfg.setdefault("strict_features", True)
    cfg.setdefault("fill_numeric_na", "median")  # or "zero"

    if FAST:
        cfg["iterations"] = min(int(cfg["iterations"]), 900)
        cfg["depth"] = min(int(cfg["depth"]), 6)
        cfg["od_wait"] = min(int(cfg["od_wait"]), 80)
        log("[FAST] Using lighter CatBoost settings")

    THRESHOLDS = list(cfg["thresholds"])
    REL_BINS = int(cfg["reliability_bins"])
    SEED = int(cfg["random_seed"])

    log(f"Loading data: {INPUT}")
    df = pd.read_parquet(INPUT)
    log(f"Rows={len(df)} Cols={len(df.columns)}", step=True)

    print("time mean")
    print(df.groupby("sched_dep_hour")["DepDelayMinutes"].mean())

    print("sparse or not")
    print(df["flightnum_od_support_count_lastNd"].describe())

    print("low support")
    print(df["flightnum_od_low_support"].value_counts(normalize=True))

    print("delay mean")
    print(df["flightnum_od_depdelay_mean_lastN"].describe())

    if "DepDelayMinutes" not in df.columns:
        raise RuntimeError("DepDelayMinutes missing; dep bin training requires it.")

    cat_feats, num_feats, dropped = resolve_features(df, cfg)
    log(f"[features] cats={len(cat_feats)} nums={len(num_feats)} dropped_missing={len(dropped)}", step=True)
    if dropped:
        log("[features] dropped missing: " + ", ".join(dropped))

    # Save resolved feature list so you can diff runs
    resolved_path = OUTDIR / "resolved_features.json"
    with open(resolved_path, "w") as f:
        json.dump({"categorical": cat_feats, "numeric": num_feats, "dropped_missing": dropped}, f, indent=2)
    log(f"[SAVE] resolved features -> {resolved_path}")

    # Build X
    X = df[cat_feats + num_feats].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            if cfg["fill_numeric_na"] == "zero":
                X[c] = X[c].fillna(0.0)
            else:
                X[c] = X[c].fillna(X[c].median())

    dep = pd.to_numeric(df["DepDelayMinutes"], errors="coerce").fillna(0.0)
    base_y = (dep >= 15).astype(int).values

    all_idx = np.arange(len(X))
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    temp_size = test_size + val_size
    if not (0 < temp_size < 1):
        raise RuntimeError("test_size + val_size must be in (0,1).")

    log("[split] stratified splits on dep>=15", step=True)
    idx_train, idx_temp, y_train_base, y_temp_base = train_test_split(
        all_idx, base_y, test_size=temp_size, random_state=SEED, stratify=base_y
    )
    rel_test = test_size / temp_size
    idx_val, idx_test, y_val_base, y_test_base = train_test_split(
        idx_temp, y_temp_base, test_size=rel_test, random_state=SEED, stratify=y_temp_base
    )
    log(f"[split] sizes train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    cat_idx = [X.columns.get_loc(c) for c in cat_feats]
    X_train = X.iloc[idx_train]
    X_val   = X.iloc[idx_val]
    X_test  = X.iloc[idx_test]

    registry = {}
    p_ge_val: Dict[int, np.ndarray] = {}
    p_ge_test: Dict[int, np.ndarray] = {}

    for thr in THRESHOLDS:
        log("="*18 + f" Train dep >= {thr} " + "="*18, step=True)
        thr_dir = OUTDIR / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)

        y_all = (dep >= thr).astype(int).values
        y_train = y_all[idx_train]
        y_val_y = y_all[idx_val]
        y_test_y = y_all[idx_test]

        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        val_pool   = Pool(X_val,   y_val_y, cat_features=cat_idx)

        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=0.05,
            depth=int(cfg["depth"]),
            l2_leaf_reg=3.0,
            iterations=int(cfg["iterations"]),
            random_seed=SEED,
            verbose=200,
            od_type="Iter",
            od_wait=int(cfg["od_wait"]),
            thread_count=-1,
        )

        log("[cb] fit start")
        clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
        log("[cb] fit done")

        log("[cal] isotonic calibration (val) start")
        calibrator = CalibratedClassifierCV(estimator=clf, cv="prefit", method="isotonic")
        calibrator.fit(X_val, y_val_y)
        log("[cal] done")

        val_p = calibrator.predict_proba(X_val)[:, 1]
        test_p = calibrator.predict_proba(X_test)[:, 1]

        th, tpr, fpr = best_threshold_youden(y_val_y, val_p)
        log(f"[thr≥{thr}] Youden threshold={th:.3f} | TPR={tpr:.3f} FPR={fpr:.3f}")
        log(f"[thr≥{thr}] AUC val={roc_auc_score(y_val_y, val_p):.3f}  AUC test={roc_auc_score(y_test_y, test_p):.3f}")

        y_pred_test = (test_p >= th).astype(int)
        print(f"[thr≥{thr}] Classification report @ Youden:")
        print(classification_report(y_test_y, y_pred_test), flush=True)
        print(f"[thr≥{thr}] Confusion matrix:\n{confusion_matrix(y_test_y, y_pred_test)}", flush=True)

        if not NO_PLOTS:
            try:
                plot_reliability(f"dep_ge{thr}_test", y_test_y, test_p, thr_dir, n_bins=REL_BINS)
            except Exception as e:
                log(f"[plot][WARN] reliability failed: {e}")

        cat_model_path = thr_dir / f"catboost_dep_ge{thr}.cbm"
        clf.save_model(str(cat_model_path))
        cal_path = thr_dir / f"calibrated_dep_ge{thr}.joblib"
        joblib.dump(calibrator, cal_path)

        meta = {
            "target": "dep",
            "threshold": thr,
            "threshold_youden": th,
            "tpr_at_threshold": tpr,
            "fpr_at_threshold": fpr,
            "categorical_features": cat_feats,
            "numeric_features": num_feats,
            "input_parquet": str(INPUT),
            "config_path": str(CFG_PATH),
            "config_used": cfg,
        }
        with open(thr_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        registry[thr] = {
            "model_path": str(cat_model_path),
            "cal_path": str(cal_path),
            "youden_threshold": th,
        }

        p_ge_val[thr] = val_p
        p_ge_test[thr] = test_p

    # Convert to bin probabilities
    probs_val = compute_bin_probs(p_ge_val)
    probs_test = compute_bin_probs(p_ge_test)

    y_bin_all = make_bins_from_delay(dep)
    y_bin_val  = y_bin_all[idx_val]
    y_bin_test = y_bin_all[idx_test]

    P_val  = np.vstack([probs_val[k] for k in BIN_KEYS]).T
    P_test = np.vstack([probs_test[k] for k in BIN_KEYS]).T

    ll_val  = log_loss(y_bin_val,  P_val,  labels=list(range(5)))
    ll_test = log_loss(y_bin_test, P_test, labels=list(range(5)))
    acc_test = (np.argmax(P_test, axis=1) == y_bin_test).mean()

    log(f"[BINS] val logloss={ll_val:.4f}")
    log(f"[BINS] test logloss={ll_test:.4f}  acc={acc_test:.4f}", step=True)

    # Save registry
    with open(OUTDIR / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    log(f"[SAVE] registry -> {OUTDIR / 'registry.json'}")

    # Save sample predictions + reasons
    n_samp = min(int(cfg["save_prediction_samples_n"]), len(idx_test))
    rs = np.random.RandomState(SEED)
    samp_idx = rs.choice(idx_test, size=n_samp, replace=False)

    test_index_to_pos = {int(ix): pos for pos, ix in enumerate(idx_test)}

    rows = []
    for ix in samp_idx:
        row = df.iloc[int(ix)]
        pos = test_index_to_pos[int(ix)]
        p_ge_row = {thr: float(p_ge_test[thr][pos]) for thr in THRESHOLDS}

        # per-row bin probs with monotone enforcement
        p15 = clip01(p_ge_row[15])
        p30 = min(clip01(p_ge_row[30]), p15)
        p45 = min(clip01(p_ge_row[45]), p30)
        p60 = min(clip01(p_ge_row[60]), p45)
        pb = {
            "p_lt15":  float(clip01(1 - p15)),
            "p_15_30": float(clip01(p15 - p30)),
            "p_30_45": float(clip01(p30 - p45)),
            "p_45_60": float(clip01(p45 - p60)),
            "p_ge60":  float(clip01(p60)),
        }
        s = sum(pb.values()) or 1.0
        for k in pb:
            pb[k] /= s

        pred_key = max(pb, key=lambda k: pb[k])
        reason = simple_reason_strings(row, p_ge_row)

        rows.append({
            "Origin": row.get("Origin"),
            "Dest": row.get("Dest"),
            "Reporting_Airline": row.get("Reporting_Airline"),
            "FlightDate": str(row.get("FlightDate")),
            "DepDelayMinutes": float(pd.to_numeric(row.get("DepDelayMinutes"), errors="coerce") or 0.0),
            "true_bin": BIN_HUMAN[BIN_KEYS[y_bin_all[int(ix)]]],
            "pred_bin": BIN_HUMAN[pred_key],
            "reason": reason,
            **pb,
            "p_ge15": float(p15),
            "p_ge30": float(p30),
            "p_ge45": float(p45),
            "p_ge60": float(p60),
        })

    sample_df = pd.DataFrame(rows)
    sample_path = OUTDIR / "prediction_samples.parquet"
    sample_df.to_parquet(sample_path, index=False)
    log(f"[SAVE] prediction samples -> {sample_path}")

    bins_meta = {
        "bin_keys": BIN_KEYS,
        "bin_human": BIN_HUMAN,
        "construction": {
            "p_lt15": "1 - p_ge15",
            "p_15_30": "p_ge15 - p_ge30",
            "p_30_45": "p_ge30 - p_ge45",
            "p_45_60": "p_ge45 - p_ge60",
            "p_ge60": "p_ge60",
            "monotone_enforced": True,
            "renormalized": True
        }
    }
    with open(OUTDIR / "bins_meta.json", "w") as f:
        json.dump(bins_meta, f, indent=2)
    log(f"[SAVE] bins meta -> {OUTDIR / 'bins_meta.json'}")

    log("[OK] finished training dep bin distribution models")


if __name__ == "__main__":
    main()
