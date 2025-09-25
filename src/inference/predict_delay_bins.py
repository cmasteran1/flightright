# src/inference/predict_delay_bins.py
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool

# Same feature lists as training
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


def as_str(series):
    s = series.astype("object").fillna("Unknown")
    return s.map(
        lambda v: "Unknown" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))


def load_registry(models_root):
    with open(models_root / "registry.json", "r") as f:
        return json.load(f)


def prep_features(df):
    # Ensure OD_pair
    if "OD_pair" not in df.columns and {"Origin", "Dest"}.issubset(df.columns):
        df["OD_pair"] = df["Origin"].astype(str).str.upper() + "_" + df["Dest"].astype(str).str.upper()

    cat_feats = [c for c in CATS_CAND if c in df.columns]
    num_feats = [c for c in NUMS_CAND if c in df.columns]
    X = df[cat_feats + num_feats].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X, cat_feats, num_feats


def explain_top_reasons(model: CatBoostClassifier, X_row: pd.DataFrame, cat_feats, num_feats, top_k=3):
    # SHAP-like per-feature contributions from CatBoost
    pool = Pool(X_row, cat_features=[X_row.columns.get_loc(c) for c in cat_feats])
    contribs = model.get_feature_importance(type="PredictionValuesChange", data=pool)
    # contribs is array shape (n_features,)
    pairs = list(zip(X_row.columns, contribs))
    # sort by absolute contribution descending
    top = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:top_k]

    # Map to short phrases
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


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_delay_bins.py path/to/models/ordinal path/to/flights_to_score.parquet")
        sys.exit(1)

    MODELS_ROOT = Path(sys.argv[1])
    INPUT = Path(sys.argv[2])

    registry = load_registry(MODELS_ROOT)
    df = pd.read_parquet(INPUT)
    X, cat_feats, num_feats = prep_features(df)

    # Load calibrated estimators
    calibrators = {}
    raw_models = {}
    for thr in THRESHOLDS:
        info = registry[str(thr)]
        cal = joblib.load(info["cal_path"])
        calibrators[thr] = cal
        # Load raw model for explanations
        m = CatBoostClassifier()
        m.load_model(info["model_path"])
        raw_models[thr] = m

    # Predict probabilities P(delay >= thr)
    proba = {}
    for thr in THRESHOLDS:
        p = calibrators[thr].predict_proba(X)[:, 1]
        # Clip to [0,1] and enforce weak monotonicity in post (optional)
        proba[thr] = np.clip(p, 0, 1)

    out = pd.DataFrame(index=df.index)
    out["p_ge_15"] = proba[15]
    out["p_ge_30"] = proba[30]
    out["p_ge_45"] = proba[45]
    out["p_ge_60"] = proba[60]
    # On-time = 1 - p_ge_15
    out["p_on_time"] = 1.0 - out["p_ge_15"]

    # (Optional) enforce monotone: p_ge_60 <= p_ge_45 <= p_ge_30 <= p_ge_15
    out[["p_ge_60", "p_ge_45", "p_ge_30", "p_ge_15"]] = (
        out[["p_ge_60", "p_ge_45", "p_ge_30", "p_ge_15"]].cummax(axis=1).iloc[:, ::-1]
    )

    # Format user-facing bins (percentages)
    out["on_time_pct"] = (100 * out["p_on_time"]).round(1)
    out["ge15_pct"] = (100 * out["p_ge_15"]).round(1)
    out["ge30_pct"] = (100 * out["p_ge_30"]).round(1)
    out["ge45_pct"] = (100 * out["p_ge_45"]).round(1)
    out["ge60_pct"] = (100 * out["p_ge_60"]).round(1)

    # Simple “why”: from the ≥15 model’s local feature contributions
    whys = []
    thr15_model = raw_models[15]
    for i in range(len(X)):
        row = X.iloc[[i]]
        phrases = explain_top_reasons(thr15_model, row, cat_feats, num_feats, top_k=3)
        whys.append("; ".join(phrases))
    out["why_short"] = whys

    # Attach key identifiers for display
    keys = [c for c in
            ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"] if
            c in df.columns]
    result = pd.concat([df[keys], out[["on_time_pct", "ge15_pct", "ge30_pct", "ge45_pct", "ge60_pct", "why_short"]]],
                       axis=1)

    # Save
    OUT = MODELS_ROOT / "predictions_with_bins.parquet"
    result.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT}")
    # Example row print
    if len(result) > 0:
        r0 = result.iloc[0]
        print("\nExample:")
        print(f"On-time (0–14m): {r0.on_time_pct:.1f}%")
        print(
            f"≥15m: {r0.ge15_pct:.1f}%  | ≥30m: {r0.ge30_pct:.1f}%  | ≥45m: {r0.ge45_pct:.1f}%  | ≥60m: {r0.ge60_pct:.1f}%")
        print(f"Why: {r0.why_short}")


if __name__ == "__main__":
    main()
