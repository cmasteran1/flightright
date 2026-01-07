"""
python src/inference/predict_delay_bins.py \
  models/ordinal \
  data/processed/inbound_prior_delay.parquet \
  --inbound_priors models/priors/inbound_priors.parquet \
  --hours_back 2 \
  --min_support 50 \
  --verbose \
  --debug_head_csv data/predictions/preds_head.csv

"""

"""
python src/inference/predict_delay_bins.py \
  models/ordinal \
  path/to/flights_to_score.parquet \
  --verbose
"""
#!/usr/bin/env python3
# src/inference/predict_delay_bins.py
import sys, json, argparse, time
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from pandas.api.types import is_string_dtype
from pandas.api.types import CategoricalDtype as _CategoricalDtype

# ------------------------------
# Config
# ------------------------------
CATS_CAND = [
    "Origin", "Dest", "Reporting_Airline", "OD_pair",
    "Aircraft_Age_Bucket",
    "origin_weathercode", "dest_weathercode",
    "origin_dep_weathercode", "dest_arr_weathercode",
]
NUMS_CAND = [
    "dep_count_origin_bin", "arr_count_dest_trail_2h", "carrier_flights_prior_day",
    "origin_taxiout_avg_7d_bin", "dest_taxiin_avg_7d_bin",
    "origin_taxiout_avg_14d_bin", "dest_taxiin_avg_14d_bin",
    "origin_taxiout_avg_28d_bin", "dest_taxiin_avg_28d_bin",
    "carrier_delay_7d_mean", "od_delay_7d_mean", "flightnum_delay_14d_mean",
    "origin_delay_7d_mean", "dest_delay_7d_mean",
    "origin_temperature_2m_max", "origin_temperature_2m_min", "origin_precipitation_sum",
    "origin_windspeed_10m_max", "origin_windgusts_10m_max",
    "dest_temperature_2m_max", "dest_temperature_2m_min", "dest_precipitation_sum",
    "dest_windspeed_10m_max", "dest_windgusts_10m_max",
    "origin_dep_temperature_2m", "origin_dep_windspeed_10m", "origin_dep_windgusts_10m", "origin_dep_precipitation",
    "dest_arr_temperature_2m", "dest_arr_windspeed_10m", "dest_arr_windgusts_10m", "dest_arr_precipitation",
    "dest_arr_wx_max_code_prev_2h", "dest_arr_wx_any_gt3_prev_2h",
    "dest_arr_precip_sum_prev_3h", "dest_arr_wind_max_prev_3h", "dest_arr_gust_max_prev_3h",
]
THRESHOLDS = [15, 30, 45, 60]

# ------------------------------
# Logging helpers
# ------------------------------
_t0_global = time.time()
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, verbose: bool = True):
    if verbose:
        dt = time.time() - _t0_global
        print(f"[{_now()} +{dt:7.2f}s] {msg}")

# ------------------------------
# Helpers
# ------------------------------
def as_str(series: pd.Series) -> pd.Series:
    """
    Convert a Series to strings without triggering future downcasting warnings.
    '1.0' -> '1' for integer-like floats. Missing -> 'Unknown'.
    """
    s = series.copy()
    if is_string_dtype(s) or isinstance(s.dtype, _CategoricalDtype):
        s = s.astype("string")
        s = s.fillna("Unknown")
        return s
    else:
        s = s.astype(object)
        s = s.where(s.notna(), "Unknown")

        def _fmt(v):
            if v is pd.NA or v is None:
                return "Unknown"
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, float):
                if np.isnan(v):
                    return "Unknown"
                return str(int(v)) if float(v).is_integer() else str(v)
            return str(v)

        return s.map(_fmt)

def load_registry(models_root: Path):
    with open(models_root / "registry.json", "r") as f:
        return json.load(f)

def prep_features(df: pd.DataFrame, verbose: bool = True):
    # Ensure OD_pair
    if "OD_pair" not in df.columns and {"Origin", "Dest"}.issubset(df.columns):
        df["OD_pair"] = df["Origin"].astype(str).str.upper() + "_" + df["Dest"].astype(str).str.upper()

    cat_feats = [c for c in CATS_CAND if c in df.columns]
    num_feats = [c for c in NUMS_CAND if c in df.columns]
    X = df[cat_feats + num_feats].copy()

    # Cast cats
    for c in cat_feats:
        X[c] = as_str(X[c])
    # Cast nums
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    log(f"[prep] features ready: rows={len(X)}, cats={len(cat_feats)}, nums={len(num_feats)}", verbose)
    return X, cat_feats, num_feats

def explain_top_reasons(model: CatBoostClassifier, X_row: pd.DataFrame, cat_feats, num_feats, top_k=3):
    pool = Pool(X_row, cat_features=[X_row.columns.get_loc(c) for c in cat_feats])
    contribs = model.get_feature_importance(type="PredictionValuesChange", data=pool)
    pairs = list(zip(X_row.columns, contribs))
    top = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:top_k]

    phrases = []
    for name, val in top:
        direction = "higher risk" if val > 0 else "lower risk"
        if "weathercode" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "wind" in name or "gust" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "precip" in name:
            phrases.append(f"{name.replace('_', ' ')} → {direction}")
        elif "delay_" in name:
            phrases.append(f"recent delay baseline ({name}) → {direction}")
        elif "taxi" in name:
            phrases.append(f"taxi congestion ({name}) → {direction}")
        elif name in ("Origin", "Dest", "OD_pair"):
            phrases.append(f"route {X_row[name].iloc[0]} → {direction}")
        else:
            phrases.append(f"{name} → {direction}")
    return phrases

def _hhmm_to_hour(x):
    try:
        s = str(int(x)).zfill(4)
        return int(s[:2])
    except Exception:
        return np.nan

def enforce_tail_monotone(df: pd.DataFrame, cols_low_to_high: List[str]) -> None:
    """
    Enforce non-decreasing tails left→right.
    cols_low_to_high should be ordered [p_ge_60, p_ge_45, p_ge_30, p_ge_15].
    Operates in-place.
    """
    arr = df[cols_low_to_high].to_numpy(copy=True)
    arr = np.maximum.accumulate(arr, axis=1)
    df[cols_low_to_high] = arr

# ------------------------------
# Vectorized Inbound Priors
# ------------------------------
class InboundPriorsVec:
    """
    Expects priors parquet with:
      Reporting_Airline, Dest, Year, Month, DOW, arr_hour_local,
      sample_size, p_ge15, p_ge30, p_ge45, p_ge60
    Provides vectorized computation for many (carrier, origin, year, month, dow, dep_hour) keys.
    """
    BASE_KEYS = ["Reporting_Airline","Dest","Year","Month","DOW","arr_hour_local"]

    def __init__(self, priors_parquet: Path):
        pri = pd.read_parquet(priors_parquet).copy()

        required = set(self.BASE_KEYS + ["sample_size","p_ge15","p_ge30","p_ge45","p_ge60"])
        missing = required - set(pri.columns)
        if missing:
            raise RuntimeError(f"Inbound priors parquet missing columns: {missing}")

        # types
        pri["Reporting_Airline"] = pri["Reporting_Airline"].astype(str)
        pri["Dest"] = pri["Dest"].astype(str)
        for c in ("Year","Month","DOW","arr_hour_local"):
            pri[c] = pd.to_numeric(pri[c], errors="coerce").astype("Int64")
        for c in ("p_ge15","p_ge30","p_ge45","p_ge60"):
            pri[c] = pd.to_numeric(pri[c], errors="coerce").astype(float)
        pri["sample_size"] = pd.to_numeric(pri["sample_size"], errors="coerce").fillna(0).astype(int)

        # exact table
        self.hour_df = pri[self.BASE_KEYS + ["p_ge15","p_ge30","p_ge45","p_ge60","sample_size"]].copy()

        # backoff tables (within-year first; then cross-year)
        self.lvl1 = (pri.groupby(["Reporting_Airline","Dest","Year","Month","DOW"], as_index=False)
                        .agg(p_ge15=("p_ge15","mean"), p_ge30=("p_ge30","mean"),
                             p_ge45=("p_ge45","mean"), p_ge60=("p_ge60","mean"),
                             sample_size=("sample_size","sum")))
        self.lvl2 = (pri.groupby(["Reporting_Airline","Dest","Year","Month"], as_index=False)
                        .agg(p_ge15=("p_ge15","mean"), p_ge30=("p_ge30","mean"),
                             p_ge45=("p_ge45","mean"), p_ge60=("p_ge60","mean"),
                             sample_size=("sample_size","sum")))
        self.lvl3 = (pri.groupby(["Reporting_Airline","Dest","Year"], as_index=False)
                        .agg(p_ge15=("p_ge15","mean"), p_ge30=("p_ge30","mean"),
                             p_ge45=("p_ge45","mean"), p_ge60=("p_ge60","mean"),
                             sample_size=("sample_size","sum")))
        self.lvl4 = (pri.groupby(["Reporting_Airline","Dest"], as_index=False)
                        .agg(p_ge15=("p_ge15","mean"), p_ge30=("p_ge30","mean"),
                             p_ge45=("p_ge45","mean"), p_ge60=("p_ge60","mean"),
                             sample_size=("sample_size","sum")))

    @staticmethod
    def _mod24(x):
        return ((x % 24) + 24) % 24

    def compute_for_keys(
        self,
        keys_df: pd.DataFrame,
        hours_back: int = 2,
        min_support: int = 50,
        shrink_alpha: int = 50,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        keys_df cols:
          Reporting_Airline, Origin(as dep station), Year, Month, DOW, dep_hour_local
        Returns dataframe with the same index as keys_df, including:
          pI_ge15/30/45/60, inbound_low_support, inbound_level_used
        """
        t0 = time.time()
        log(f"[priors] unique keys to score: {len(keys_df)}", verbose)

        # Build long-form requested hours for averaging
        long_frames = []
        for h in range(1, hours_back + 1):
            tmp = keys_df.copy()
            tmp["arr_hour_local"] = self._mod24(tmp["dep_hour_local"].astype(int) - h)
            tmp["h_part"] = h
            long_frames.append(tmp)
        req_long = pd.concat(long_frames, ignore_index=True)
        # map Origin -> Dest for arrivals
        req_long = req_long.rename(columns={"Origin": "Dest"})

        # Exact hour-level merge
        merge_cols = ["Reporting_Airline","Dest","Year","Month","DOW","arr_hour_local"]
        exact = req_long.merge(self.hour_df, on=merge_cols, how="left", copy=False)

        # Aggregate across the requested hours per base key
        agg_cols = ["Reporting_Airline","Dest","Year","Month","DOW","dep_hour_local"]
        exact["hit"] = exact["p_ge15"].notna().astype(int)
        hour_agg = (exact.groupby(agg_cols, as_index=False)
                         .agg(p_ge15=("p_ge15","mean"),
                              p_ge30=("p_ge30","mean"),
                              p_ge45=("p_ge45","mean"),
                              p_ge60=("p_ge60","mean"),
                              sample_size=("sample_size","sum"),
                              h_hits=("hit","sum")))

        # Base frame aligned to keys_df index
        base = keys_df.copy()
        base = base.rename(columns={"Origin":"Dest"})
        base["_idx"] = base.index
        base = base.merge(hour_agg, on=agg_cols, how="left")
        base.set_index("_idx", inplace=True)

        base["inbound_level_used"] = np.where(base["h_hits"].fillna(0) > 0, "HOUR", "NONE")
        base["sample_size"] = base["sample_size"].fillna(0).astype(int)

        def _backfill_level(base_mask, level_df, on_cols, label):
            """Merge using preserved index and assign back by aligned indices."""
            if not base_mask.any():
                return
            tmp = base.loc[base_mask, on_cols].copy()
            tmp["_idx"] = tmp.index
            m = tmp.merge(level_df, on=on_cols, how="left")
            m.set_index("_idx", inplace=True)
            has = m["p_ge15"].notna()
            if has.any():
                idxs = m.index[has]
                for c in ["p_ge15","p_ge30","p_ge45","p_ge60","sample_size"]:
                    base.loc[idxs, c] = m.loc[idxs, c].values
                base.loc[idxs, "inbound_level_used"] = label

        # Backoffs LVL1→LVL2→LVL3→LVL4 for those with NONE
        mask_none = base["inbound_level_used"].eq("NONE")
        _backfill_level(mask_none, self.lvl1,
                        ["Reporting_Airline","Dest","Year","Month","DOW"], "LVL1")
        mask_none = base["inbound_level_used"].eq("NONE")
        _backfill_level(mask_none, self.lvl2,
                        ["Reporting_Airline","Dest","Year","Month"], "LVL2")
        mask_none = base["inbound_level_used"].eq("NONE")
        _backfill_level(mask_none, self.lvl3,
                        ["Reporting_Airline","Dest","Year"], "LVL3")
        mask_none = base["inbound_level_used"].eq("NONE")
        _backfill_level(mask_none, self.lvl4,
                        ["Reporting_Airline","Dest"], "LVL4")

        for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]:
            base[c] = base[c].fillna(0.0)
        base["sample_size"] = base["sample_size"].fillna(0).astype(int)

        # Shrink hour-level low support toward best available backoff
        hour_mask = base["inbound_level_used"].eq("HOUR") & (base["sample_size"] < int(min_support))
        if hour_mask.any():
            # Build bk_* by trying LVL1→LVL2→LVL3→LVL4
            bk = pd.DataFrame(index=base.index)
            for c in ["bk_p15","bk_p30","bk_p45","bk_p60"]:
                bk[c] = np.nan

            def _fill_bk(mask, level_df, on_cols):
                if not mask.any():
                    return
                tmp = base.loc[mask, on_cols].copy()
                tmp["_idx"] = tmp.index
                m = tmp.merge(level_df, on=on_cols, how="left")
                m.set_index("_idx", inplace=True)
                has = m["p_ge15"].notna()
                if has.any():
                    idxs = m.index[has]
                    bk.loc[idxs, "bk_p15"] = m.loc[idxs, "p_ge15"].values
                    bk.loc[idxs, "bk_p30"] = m.loc[idxs, "p_ge30"].values
                    bk.loc[idxs, "bk_p45"] = m.loc[idxs, "p_ge45"].values
                    bk.loc[idxs, "bk_p60"] = m.loc[idxs, "p_ge60"].values

            need = hour_mask & bk["bk_p15"].isna()
            _fill_bk(need, self.lvl1, ["Reporting_Airline","Dest","Year","Month","DOW"])
            need = hour_mask & bk["bk_p15"].isna()
            _fill_bk(need, self.lvl2, ["Reporting_Airline","Dest","Year","Month"])
            need = hour_mask & bk["bk_p15"].isna()
            _fill_bk(need, self.lvl3, ["Reporting_Airline","Dest","Year"])
            need = hour_mask & bk["bk_p15"].isna()
            _fill_bk(need, self.lvl4, ["Reporting_Airline","Dest"])

            # Shrink
            n = base.loc[hour_mask, "sample_size"].astype(float)
            alpha = float(shrink_alpha)
            denom = (n + alpha).replace(0, 1.0)

            for (c_out, c_bk) in [("p_ge15","bk_p15"),("p_ge30","bk_p30"),
                                  ("p_ge45","bk_p45"),("p_ge60","bk_p60")]:
                base.loc[hour_mask, c_out] = (
                    (n * base.loc[hour_mask, c_out].astype(float) + alpha * bk.loc[hour_mask, c_bk].fillna(0.0))
                    / denom
                )
            base.loc[hour_mask, "inbound_level_used"] = f"HOUR→LVL1_SHRUNK(alpha={int(shrink_alpha)})"

        # Final low_support flag after shrink decision
        base["inbound_low_support"] = base["sample_size"] < int(min_support)

        # Return only needed columns, keyed by original keys_df index
        out = base[["p_ge15","p_ge30","p_ge45","p_ge60","inbound_low_support","inbound_level_used"]].copy()
        out = out.rename(columns={"p_ge15":"pI_ge15","p_ge30":"pI_ge30","p_ge45":"pI_ge45","p_ge60":"pI_ge60"})
        # Ensure we return aligned to keys_df index
        out = out.reindex(keys_df.index)
        log(f"[priors] computed in {time.time()-t0:.2f}s", verbose)
        return out

# ------------------------------
# Combine locals with inbound
# ------------------------------
def enforce_tail_monotone(df: pd.DataFrame, cols_low_to_high: List[str]) -> None:
    arr = df[cols_low_to_high].to_numpy(copy=True)
    arr = np.maximum.accumulate(arr, axis=1)
    df[cols_low_to_high] = arr

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models_root", type=str, help="Path to models/ordinal")
    ap.add_argument("input_parquet", type=str, help="Path to flights_to_score.parquet")
    ap.add_argument("--inbound_priors", type=str, default=None, help="Path to models/priors/inbound_priors.parquet")
    ap.add_argument("--hours_back", type=int, default=2, help="Prior hours to average for inbound arrivals")
    ap.add_argument("--min_support", type=int, default=50, help="Low-support threshold for priors")
    ap.add_argument("--shrink_alpha", type=int, default=50, help="Strength of backoff prior when low support")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress logs")
    ap.add_argument("--debug_head_csv", type=str, default=None, help="Optional path to write a small CSV preview")
    ap.add_argument("--explain_max_rows", type=int, default=2000, help="Explain at most this many rows (0 to skip)")
    args = ap.parse_args()

    MODELS_ROOT = Path(args.models_root)
    INPUT = Path(args.input_parquet)
    verbose = args.verbose

    log("starting predict_delay_bins", verbose)
    log(f"models_root={MODELS_ROOT}", verbose)
    log(f"input_parquet={INPUT}", verbose)
    if args.inbound_priors:
        log(f"inbound_priors={args.inbound_priors}  hours_back={args.hours_back}  min_support={args.min_support}", verbose)
        log(f"shrink_alpha={args.shrink_alpha}", verbose)
    else:
        log("inbound_priors=None (skipping propagation layer)", verbose)

    # Registry + models
    log("loading registry.json", verbose)
    registry = load_registry(MODELS_ROOT)
    log(f"registry thresh keys: {sorted(registry.keys(), key=lambda x:int(x))}", verbose)

    # Read input
    log("reading input parquet", verbose)
    df = pd.read_parquet(INPUT)
    log(f"input rows={len(df)} cols={len(df.columns)}", verbose)

    # Feature prep
    log("prepping features", verbose)
    X, cat_feats, num_feats = prep_features(df, verbose=verbose)

    # Load calibrated estimators + raw models for explanations
    calibrators = {}
    raw_models = {}
    for thr in THRESHOLDS:
        info = registry[str(thr)]
        log(f"loading calibrator for ≥{thr}: {info['cal_path']}", verbose)
        cal = joblib.load(info["cal_path"])
        calibrators[thr] = cal

        log(f"loading raw model for ≥{thr}: {info['model_path']}", verbose)
        m = CatBoostClassifier()
        m.load_model(info["model_path"])
        raw_models[thr] = m

    # Predict probabilities P(delay >= thr)
    log("predicting calibrated probabilities", verbose)
    proba = {}
    for thr in THRESHOLDS:
        t1 = time.time()
        p = calibrators[thr].predict_proba(X)[:, 1]
        proba[thr] = np.clip(p, 0, 1)
        log(f"  ≥{thr}: done in {time.time()-t1:.2f}s", verbose)

    out = pd.DataFrame(index=df.index)
    out["p_ge_15_local"] = proba[15]
    out["p_ge_30_local"] = proba[30]
    out["p_ge_45_local"] = proba[45]
    out["p_ge_60_local"] = proba[60]

    # Enforce monotone on locals (left→right: 60,45,30,15)
    enforce_tail_monotone(out, ["p_ge_60_local","p_ge_45_local","p_ge_30_local","p_ge_15_local"])
    out["p_on_time_local"] = 1.0 - out["p_ge_15_local"]
    log("local tails computed (with monotone fix)", verbose)

    # ------------------------------
    # Optional inbound combination (vectorized)
    # ------------------------------
    if args.inbound_priors:
        log("loading inbound priors parquet", verbose)
        pri = InboundPriorsVec(Path(args.inbound_priors))

        # Calendar cols
        log("deriving calendar keys for priors lookup", verbose)
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["Year"] = df["FlightDate"].dt.year.astype("Int64")
        df["Month"] = df["FlightDate"].dt.month.astype("Int64")
        df["DOW"] = df["FlightDate"].dt.dayofweek.astype("Int64")  # 0..6

        if "dep_hour_local" in df.columns and df["dep_hour_local"].notna().any():
            df["dep_hour_local"] = pd.to_numeric(df["dep_hour_local"], errors="coerce").astype("Int64")
        else:
            df["dep_hour_local"] = df.get("CRSDepTime", pd.Series(index=df.index, dtype="float"))
            df["dep_hour_local"] = df["dep_hour_local"].apply(_hhmm_to_hour).astype("Int64")

        key_cols = ["Reporting_Airline","Origin","Year","Month","DOW","dep_hour_local"]
        keys_df = df[key_cols].copy()
        keys_df.index = df.index  # preserve row mapping

        # Compute once per unique key
        uniq_keys = keys_df.drop_duplicates()
        log(f"vectorized inbound computation on {len(uniq_keys)} unique keys", verbose)
        inbound_keys = pri.compute_for_keys(
            uniq_keys, hours_back=args.hours_back,
            min_support=args.min_support, shrink_alpha=args.shrink_alpha,
            verbose=verbose
        )

        # Map back to all rows (align by index)
        inbound_all = inbound_keys.reindex(keys_df.index)

        # Attach inbound tails to out
        out[["pI_ge15","pI_ge30","pI_ge45","pI_ge60"]] = inbound_all[["pI_ge15","pI_ge30","pI_ge45","pI_ge60"]].to_numpy()
        out[["inbound_low_support","inbound_level_used"]] = inbound_all[["inbound_low_support","inbound_level_used"]].to_numpy()

        # Summaries
        log("inbound levels distribution:", verbose)
        try:
            lvl_counts = out["inbound_level_used"].value_counts(dropna=False)
            print(lvl_counts.to_string())
        except Exception:
            pass
        low_sup_rate = float(np.mean(out["inbound_low_support"])) if len(out) else 0.0
        log(f"inbound low_support rate: {100*low_sup_rate:.2f}%", verbose)

        # Combine locals + inbound per row (independence)
        for thr, pI_col, pL_col, pC_col in [
            (15,"pI_ge15","p_ge_15_local","p_ge_15"),
            (30,"pI_ge30","p_ge_30_local","p_ge_30"),
            (45,"pI_ge45","p_ge_45_local","p_ge_45"),
            (60,"pI_ge60","p_ge_60_local","p_ge_60"),
        ]:
            pI = out[pI_col].fillna(0.0).astype(float)
            pL = out[pL_col].fillna(0.0).astype(float)
            out[pC_col] = 1.0 - (1.0 - pL) * (1.0 - pI)

        # Final on-time
        out["p_on_time"] = 1.0 - out["p_ge_15"]
    else:
        # No priors → use locals for final
        out["p_on_time"] = out["p_on_time_local"]
        out["p_ge_15"] = out["p_ge_15_local"]
        out["p_ge_30"] = out["p_ge_30_local"]
        out["p_ge_45"] = out["p_ge_45_local"]
        out["p_ge_60"] = out["p_ge_60_local"]

    # Enforce monotonicity on final tails (and recompute complement)
    enforce_tail_monotone(out, ["p_ge_60","p_ge_45","p_ge_30","p_ge_15"])
    out["p_on_time"] = 1.0 - out["p_ge_15"]
    log("final tails (post monotone) computed", verbose)

    # Percentages for display
    out["on_time_pct"] = (100 * out["p_on_time"]).round(1)
    out["ge15_pct"] = (100 * out["p_ge_15"]).round(1)
    out["ge30_pct"] = (100 * out["p_ge_30"]).round(1)
    out["ge45_pct"] = (100 * out["p_ge_45"]).round(1)
    out["ge60_pct"] = (100 * out["p_ge_60"]).round(1)

    # Explanations: cap how many we compute (fast)
    max_rows = max(0, int(args.explain_max_rows))
    do_explain = max_rows > 0
    log(f"generating explanations from ≥15 model (limit={max_rows})", verbose)
    whys = [""] * len(X)
    if do_explain:
        thr15_model = raw_models[15]
        n_explain = min(max_rows, len(X))
        for i in range(n_explain):
            if i % 20000 == 0 and i > 0:
                log(f"  explained {i}/{n_explain} rows...", verbose)
            row = X.iloc[[i]]
            phrases = explain_top_reasons(thr15_model, row, cat_feats, num_feats, top_k=3)
            if "pI_ge15" in out.columns:
                pI15 = float(out.at[out.index[i], "pI_ge15"])
                inbound_msg = "inbound propagation risk elevated" if pI15 >= 0.20 else "inbound propagation risk low"
                phrases.append(inbound_msg)
                if "inbound_low_support" in out.columns and bool(out.at[out.index[i], "inbound_low_support"]):
                    phrases.append("low historical support for inbound priors")
            whys[i] = "; ".join(phrases)
    out["why_short"] = whys

    # Attach identifiers
    keys = [c for c in ["FlightDate","Reporting_Airline","Flight_Number_Reporting_Airline","Origin","Dest","CRSDepTime"]
            if c in df.columns]
    cols_order = ["on_time_pct","ge15_pct","ge30_pct","ge45_pct","ge60_pct","why_short"]
    diag_cols = ["p_ge_15","p_ge_30","p_ge_45","p_ge_60","p_on_time",
                 "p_ge_15_local","p_ge_30_local","p_ge_45_local","p_ge_60_local","p_on_time_local"]
    if "pI_ge15" in out.columns:
        diag_cols += ["pI_ge15","pI_ge30","pI_ge45","pI_ge60","inbound_low_support","inbound_level_used"]

    result = pd.concat([df[keys], out[cols_order + diag_cols]], axis=1)

    # Optional quick preview
    if args.debug_head_csv:
        try:
            head_n = min(200, len(result))
            Path(args.debug_head_csv).parent.mkdir(parents=True, exist_ok=True)
            result.head(head_n).to_csv(args.debug_head_csv, index=False)
            log(f"wrote debug head CSV → {args.debug_head_csv} (rows={head_n})", verbose)
        except Exception as e:
            log(f"[WARN] failed to write debug_head_csv: {e}", verbose)

    # Save
    OUT = MODELS_ROOT / "predictions_with_bins.parquet"
    log(f"saving parquet → {OUT}", verbose)
    result.to_parquet(OUT, index=False)
    log(f"[OK] wrote {OUT} (rows={len(result)})", verbose)

    # -------- Example print with checks --------
    if len(result) > 0:
        r0 = result.iloc[0]
        print("\nExample:")
        print(f"On-time (0–14m): {r0.on_time_pct:.1f}%")
        print(f"≥15m: {r0.ge15_pct:.1f}%  | ≥30m: {r0.ge30_pct:.1f}%  | ≥45m: {r0.ge45_pct:.1f}%  | ≥60m: {r0.ge60_pct:.1f}%")

        # Raw tails for numeric checks
        p15 = float(r0.p_ge_15); p30 = float(r0.p_ge_30); p45 = float(r0.p_ge_45); p60 = float(r0.p_ge_60); pon = float(r0.p_on_time)

        # Disjoint bins from tails
        b15_29 = max(0.0, p15 - p30)
        b30_44 = max(0.0, p30 - p45)
        b45_59 = max(0.0, p45 - p60)
        b60p   = max(0.0, p60)
        b0_14  = max(0.0, 1.0 - p15)
        print(f"Bins → 0–14: {100*b0_14:.1f}% | 15–29: {100*b15_29:.1f}% | 30–44: {100*b30_44:.1f}% | 45–59: {100*b45_59:.1f}% | 60+: {100*b60p:.1f}%")

        ok_monotone = (p15 >= p30) and (p30 >= p45) and (p45 >= p60)
        ok_complement = abs((pon + p15) - 1.0) < 1e-6
        if not ok_monotone:
            print("[WARN] Tails not monotone for example row (should be p15>=p30>=p45>=p60).")
            print(f"       Raw tails: p15={p15:.4f} p30={p30:.4f} p45={p45:.4f} p60={p60:.4f}")
        if not ok_complement:
            print("[WARN] Complement check failed for example row: p_on_time + p_ge_15 != 1.0")
            print(f"       p_on_time={pon:.6f}, p_ge_15={p15:.6f}, sum={pon+p15:.6f}")

        print(f"Why: {r0.why_short}")

    log("finished predict_delay_bins", verbose)

if __name__ == "__main__":
    main()
