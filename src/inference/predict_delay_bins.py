# predict_delay_bins.py


#python predict_delay_bins.py \
#  --model_dir ../../models/ordinal \
#  --input ../../data/processed/train_ready.parquet \
#  --output ../../data/predictions/delay_bins.parquet \
#  --output_csv ../../data/predictions/delay_bins_head.csv \
#  --inbound_priors ../../models/priors/inbound_priors.parquet \
#  --hours_back 2 \
#  --min_support 50 \
#  --explain

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from joblib import load as joblib_load

# Silence the deprecation by using isinstance with CategoricalDtype
from pandas.api.types import CategoricalDtype as _CatDtype

BIN_EDGES = [0, 15, 30, 45, 60, 10**9]
BIN_LABELS = ["0-14", "15-29", "30-44", "45-59", "60+"]

def _load_registry(model_dir: Path):
    reg_path = model_dir / "registry.json"
    with open(reg_path, "r") as f:
        reg = json.load(f)
    # figure feature list from any CatBoost model
    any_thr = sorted(reg.keys(), key=lambda x: int(x))[0]
    model_path = Path(reg[any_thr]["model_path"])
    cb = CatBoostClassifier()
    cb.load_model(str(model_path))
    feature_names = cb.feature_names_
    print(f"[INFO] Feature count (from CatBoost): {len(feature_names)}")
    # load all calibrated wrappers
    cals = {}
    for thr in sorted(reg.keys(), key=lambda x: int(x)):
        cal_path = Path(reg[thr]["cal_path"])
        cal = joblib_load(cal_path)
        print(f"[INFO] Loaded calibrated ≥{thr} model: {cal_path}")
        cals[int(thr)] = cal
    return feature_names, cals

def _prep_frame(df: pd.DataFrame, feature_names):
    X = df.copy()
    # Make sure categorical-looking columns are strings
    for colname in feature_names:
        if colname not in X.columns:
            X[colname] = np.nan
    X = X[feature_names]
    for c in X.columns:
        col = X[c]
        dt = col.dtype
        if str(dt) in ("object", "string") or isinstance(dt, _CatDtype):
            X[c] = col.fillna("Unknown").astype(str)
        else:
            X[c] = pd.to_numeric(col, errors="coerce")
    return X

# ---------- inbound priors helpers ----------

class InboundPriors:
    def __init__(self, df: pd.DataFrame):
        # Expect columns: Reporting_Airline, Dest, Month, DOW, arr_hour_local,
        # p_ge15,p_ge30,p_ge45,p_ge60, sample_size
        self.df = df.copy()
        self._mk_indexes()

    def _mk_indexes(self):
        self.df["Reporting_Airline"] = self.df["Reporting_Airline"].astype(str)
        self.df["Dest"] = self.df["Dest"].astype(str)
        self.df["Month"] = self.df["Month"].astype(int)
        self.df["DOW"] = self.df["DOW"].astype(int)
        self.df["arr_hour_local"] = self.df["arr_hour_local"].astype(int)

        self.idx_cols = ["Reporting_Airline","Dest","Month","DOW","arr_hour_local"]
        self.df.set_index(self.idx_cols, inplace=True, drop=False)

        # Backoffs
        self.df_lvl1 = self.df.reset_index(drop=True).set_index(["Reporting_Airline","Dest","Month","DOW"], drop=False)
        self.df_lvl2 = self.df.reset_index(drop=True).set_index(["Reporting_Airline","Dest","Month"], drop=False)
        self.df_lvl3 = self.df.reset_index(drop=True).set_index(["Reporting_Airline","Dest"], drop=False)

    def _lookup_hour(self, key5):
        try:
            return self.df.loc[key5]
        except KeyError:
            return None

    def _avg_two_hours(self, keys5):
        # average probabilities and sum sample_size across available hours
        rows = [self._lookup_hour(k) for k in keys5]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        cat = pd.concat(rows, axis=1).T  # small
        out = {}
        for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]:
            out[c] = float(cat[c].mean())
        out["sample_size"] = int(cat["sample_size"].sum())
        return out

    def get_tail_probs(self, carrier, dest, month, dow, dep_hour, hours_back=2, min_support=50):
        # arrivals into 'dest' in the prior hours_back hours:
        hours = [int((dep_hour - h) % 24) for h in range(1, hours_back+1)]
        keys5 = [(carrier, dest, int(month), int(dow), h) for h in hours]
        out = self._avg_two_hours(keys5)
        level_used = "HOUR"

        # Backoffs if missing
        if out is None:
            # mean over all hours for (C,Dest,Month,DOW)
            try:
                rows = self.df_lvl1.loc[(carrier, dest, int(month), int(dow))]
                if isinstance(rows, pd.DataFrame):
                    cat = rows
                else:
                    cat = rows.to_frame().T
                out = {c: float(cat[c].mean()) for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]}
                out["sample_size"] = int(cat["sample_size"].sum())
                level_used = "LVL1"
            except KeyError:
                pass
        if out is None:
            try:
                rows = self.df_lvl2.loc[(carrier, dest, int(month))]
                if isinstance(rows, pd.DataFrame):
                    cat = rows
                else:
                    cat = rows.to_frame().T
                out = {c: float(cat[c].mean()) for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]}
                out["sample_size"] = int(cat["sample_size"].sum())
                level_used = "LVL2"
            except KeyError:
                pass
        if out is None:
            try:
                rows = self.df_lvl3.loc[(carrier, dest)]
                if isinstance(rows, pd.DataFrame):
                    cat = rows
                else:
                    cat = rows.to_frame().T
                out = {c: float(cat[c].mean()) for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]}
                out["sample_size"] = int(cat["sample_size"].sum())
                level_used = "LVL3"
            except KeyError:
                pass

        if out is None:
            # final fallback: zero inbound pressure
            out = {c: 0.0 for c in ["p_ge15","p_ge30","p_ge45","p_ge60"]}
            out["sample_size"] = 0
            level_used = "NONE"

        out["level_used"] = level_used
        out["low_support"] = bool(out["sample_size"] < int(min_support))
        return out

def _combine_ordinals(local, inbound):
    # Combine tail probs independently at each threshold
    comb = {}
    for thr_key in ["p_ge15","p_ge30","p_ge45","p_ge60"]:
        pL = float(local[thr_key])
        pI = float(inbound[thr_key])
        comb[thr_key] = 1.0 - (1.0 - pL) * (1.0 - pI)
    # Convert to disjoint bin probs
    p0_14 = 1.0 - comb["p_ge15"]
    p15_29 = comb["p_ge15"] - comb["p_ge30"]
    p30_44 = comb["p_ge30"] - comb["p_ge45"]
    p45_59 = comb["p_ge45"] - comb["p_ge60"]
    p60p   = comb["p_ge60"]
    # Guard numerical drift
    probs = np.maximum(0.0, np.array([p0_14,p15_29,p30_44,p45_59,p60p]))
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    return {
        "p_on_time_0_14": probs[0],
        "p_15_29": probs[1],
        "p_30_44": probs[2],
        "p_45_59": probs[3],
        "p_60_plus": probs[4],
        # keep tails too
        "p_ge15": comb["p_ge15"],
        "p_ge30": comb["p_ge30"],
        "p_ge45": comb["p_ge45"],
        "p_ge60": comb["p_ge60"],
    }

def _most_likely_bin_row(row):
    probs = [row["p_on_time_0_14"], row["p_15_29"], row["p_30_44"], row["p_45_59"], row["p_60_plus"]]
    return BIN_LABELS[int(np.argmax(probs))]

def _exp_from_probs(row):
    # crude expected delay using bin midpoints (8,22,37,52,75)
    mids = np.array([8,22,37,52,75], dtype=float)
    probs = np.array([row["p_on_time_0_14"], row["p_15_29"], row["p_30_44"], row["p_45_59"], row["p_60_plus"]], dtype=float)
    return float(np.dot(mids, probs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--explain", action="store_true")
    # NEW: inbound priors
    ap.add_argument("--inbound_priors", default=None, help="Parquet from build_inbound_priors.py")
    ap.add_argument("--hours_back", type=int, default=2, help="How many prior hours to average (default 2)")
    ap.add_argument("--min_support", type=int, default=50, help="Low-support threshold for priors")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    feature_names, calibrated = _load_registry(model_dir)

    # Load input
    in_path = Path(args.input)
    df = pd.read_parquet(in_path)

    # Make sure we have keys for priors: Month, DOW, dep_hour_local, Origin
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["Month"] = df["FlightDate"].dt.month.astype("Int64")
    df["DOW"] = df["FlightDate"].dt.dayofweek.astype("Int64")  # 0..6

    # dep hour local: prefer existing; else derive from CRSDepTime
    if "dep_hour_local" in df.columns and df["dep_hour_local"].notna().any():
        dep_hour = df["dep_hour_local"].astype("Int64")
    else:
        def hhmm_to_hour(x):
            try:
                s = str(int(x)).zfill(4)
                return int(s[:2])
            except Exception:
                return np.nan
        dep_hour = df["CRSDepTime"].apply(hhmm_to_hour).astype("Int64")
    df["dep_hour_local"] = dep_hour

    # Prepare features and local calibrated tail probs
    X = _prep_frame(df, feature_names)
    local = {}
    for thr, cal in calibrated.items():
        p = cal.predict_proba(X)[:, 1]
        local[thr] = p
    # Pack local tails as a frame for combination
    local_df = pd.DataFrame({
        "pL_ge15": local.get(15, np.zeros(len(df))),
        "pL_ge30": local.get(30, np.zeros(len(df))),
        "pL_ge45": local.get(45, np.zeros(len(df))),
        "pL_ge60": local.get(60, np.zeros(len(df))),
    })

    # Optional inbound priors
    pri = None
    if args.inbound_priors:
        pri_df = pd.read_parquet(args.inbound_priors)
        pri = InboundPriors(pri_df)

    # Combine per row
    out_rows = []
    low_support_flags = []
    inbound_levels = []
    pI_cols = {"pI_ge15": [], "pI_ge30": [], "pI_ge45": [], "pI_ge60": []}

    for i, r in df.iterrows():
        # Local tails
        loc = {
            "p_ge15": float(local_df.at[i, "pL_ge15"]),
            "p_ge30": float(local_df.at[i, "pL_ge30"]),
            "p_ge45": float(local_df.at[i, "pL_ge45"]),
            "p_ge60": float(local_df.at[i, "pL_ge60"]),
        }

        # Inbound tails (arrivals into ORIGIN station before this departure)
        if pri is not None:
            carrier = str(r["Reporting_Airline"])
            dest_station = str(r["Origin"])
            month = int(r["Month"]) if pd.notna(r["Month"]) else 1
            dow   = int(r["DOW"]) if pd.notna(r["DOW"]) else 0
            dep_h = int(r["dep_hour_local"]) if pd.notna(r["dep_hour_local"]) else 0

            inbound = pri.get_tail_probs(
                carrier=carrier,
                dest=dest_station,
                month=month,
                dow=dow,
                dep_hour=dep_h,
                hours_back=args.hours_back,
                min_support=args.min_support
            )
            pI = {
                "p_ge15": inbound["p_ge15"],
                "p_ge30": inbound["p_ge30"],
                "p_ge45": inbound["p_ge45"],
                "p_ge60": inbound["p_ge60"],
            }
            low_support_flags.append(inbound["low_support"])
            inbound_levels.append(inbound["level_used"])
            for k,v in zip(["pI_ge15","pI_ge30","pI_ge45","pI_ge60"], [pI["p_ge15"],pI["p_ge30"],pI["p_ge45"],pI["p_ge60"]]):
                pI_cols[k].append(v)
        else:
            pI = {k: 0.0 for k in ["p_ge15","p_ge30","p_ge45","p_ge60"]}
            low_support_flags.append(False)
            inbound_levels.append("OFF")
            for k in ["pI_ge15","pI_ge30","pI_ge45","pI_ge60"]:
                pI_cols[k].append(0.0)

        comb = _combine_ordinals(loc, pI)
        out_rows.append(comb)

    prob_df = pd.DataFrame(out_rows, index=df.index)
    for k, vals in pI_cols.items():
        prob_df[k] = vals
    prob_df["inbound_low_support"] = low_support_flags
    prob_df["inbound_level_used"] = inbound_levels

    # Most likely bin + expected minutes
    prob_df["most_likely_bin"] = prob_df.apply(_most_likely_bin_row, axis=1)
    prob_df["delay_minutes_expected"] = prob_df.apply(_exp_from_probs, axis=1)

    # Build 'why'
    why_msgs = []
    if args.explain:
        for i, r in df.iterrows():
            msgs = []
            # Basic drivers you already used in earlier version:
            # (Keep it simple here; you can reuse your richer logic.)
            if float(prob_df.at[i, "pI_ge15"]) > 0.2:
                msgs.append("inbound propagation risk elevated")
            else:
                msgs.append("inbound propagation risk low")
            if prob_df.at[i, "inbound_low_support"]:
                msgs.append("low historical support for inbound priors")
            why_msgs.append("; ".join(msgs))
    else:
        why_msgs = [""] * len(df)
    prob_df["why"] = why_msgs

    # Select output columns similar to your earlier CSV
    keep_cols = [
        "FlightDate","Reporting_Airline","Flight_Number_Reporting_Airline","Origin","Dest",
        "p_on_time_0_14","p_15_29","p_30_44","p_45_59","p_60_plus",
        "most_likely_bin","delay_minutes_expected","why",
        "p_ge15","p_ge30","p_ge45","p_ge60",
        "pI_ge15","pI_ge30","pI_ge45","pI_ge60",
        "inbound_low_support","inbound_level_used"
    ]
    out = pd.concat([df, prob_df], axis=1)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[keep_cols].to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out)}")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        out[keep_cols].head(200).to_csv(csv_path, index=False)
        print(f"[OK] wrote head CSV {csv_path}")

if __name__ == "__main__":
    main()

