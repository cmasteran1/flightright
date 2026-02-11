#!/usr/bin/env python3
# src/fetch_prune/features_dep.py
#
# Run from REPO_ROOT:
#   python src/fetch_prune/features_dep.py data/dep_arr_config.json
#
import sys
import json
from pathlib import Path
from datetime import date, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()


def _abspath(p: str, *, base: str = "repo") -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if base == "repo":
        return (REPO_ROOT / pp).resolve()
    if base == "data":
        return (DATA_ROOT / pp).resolve()
    raise ValueError("base must be 'repo' or 'data'")


def _format_path_template(p: str, *, target: str, thr: Optional[int] = None) -> str:
    if thr is None:
        return p.format(target=target)
    return p.format(target=target, thr=thr)


# ---------- holidays ----------
def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    d = d + timedelta(days=shift)
    d = d + timedelta(weeks=n - 1)
    return d


def thanksgiving_day(year: int) -> date:
    return _nth_weekday_of_month(year, 11, weekday=3, n=4)


def add_holiday_flag(df: pd.DataFrame, thanksgiving_window: int = 2, christmas_window: int = 3) -> pd.DataFrame:
    df = df.copy()
    base = pd.to_datetime(df["FlightDate"], errors="coerce")
    years = sorted(pd.unique(base.dt.year.dropna()))
    tg = {int(y): thanksgiving_day(int(y)) for y in years}
    xmas = {int(y): date(int(y), 12, 25) for y in years}

    def is_near(d: date, center: date, win: int) -> bool:
        return abs((d - center).days) <= win

    flags = []
    for ts in base:
        if pd.isna(ts):
            flags.append(0)
            continue
        d = ts.date()
        y = d.year
        f = 0
        if y in tg and is_near(d, tg[y], thanksgiving_window):
            f = 1
        if y in xmas and is_near(d, xmas[y], christmas_window):
            f = 1
        flags.append(f)

    df["is_holiday"] = pd.Series(flags, index=df.index).astype("Int8")
    return df


# ---------- weather unit conversions ----------
def add_weather_kelvin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "origin_temperature_2m_max" in df.columns:
        df["origin_temp_max_K"] = pd.to_numeric(df["origin_temperature_2m_max"], errors="coerce") + 273.15
    if "origin_temperature_2m_min" in df.columns:
        df["origin_temp_min_K"] = pd.to_numeric(df["origin_temperature_2m_min"], errors="coerce") + 273.15

    if "origin_precipitation_sum" in df.columns:
        df["origin_daily_precip_sum_mm"] = pd.to_numeric(df["origin_precipitation_sum"], errors="coerce")
    if "origin_windspeed_10m_max" in df.columns:
        df["origin_daily_windspeed_max_kmh"] = pd.to_numeric(df["origin_windspeed_10m_max"], errors="coerce")

    if "origin_dep_temperature_2m" in df.columns:
        df["origin_dep_temp_K"] = pd.to_numeric(df["origin_dep_temperature_2m"], errors="coerce") + 273.15
    if "origin_dep_precipitation" in df.columns:
        df["origin_dep_precip_mm"] = pd.to_numeric(df["origin_dep_precipitation"], errors="coerce")
    if "origin_dep_windspeed_10m" in df.columns:
        df["origin_dep_windspeed_kmh"] = pd.to_numeric(df["origin_dep_windspeed_10m"], errors="coerce")

    return df


# ---------- flight-number OD history features (vectorized, from HISTORY POOL) ----------
def add_recent_flightnum_od_mean_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    n_recent: int = 20,
    n_days: int = 14,
    low_support_leq: int = 2,
) -> pd.DataFrame:
    df = df.copy()

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]

    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()

    hist = hist.copy()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in hist.columns:
            hist[c] = hist[c].astype(str).str.upper()
    hist["Flight_Number_Reporting_Airline"] = pd.to_numeric(hist.get("Flight_Number_Reporting_Airline"), errors="coerce")
    hist["FlightDate"] = pd.to_datetime(hist["FlightDate"], errors="coerce").dt.normalize()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    need_cols = ["FlightDate"] + key + ["DepDelayMinutes"]
    df_small = df[need_cols].copy()
    hist_small = hist[need_cols].copy()

    df_small["_is_target"] = True
    hist_small["_is_target"] = False

    all_rows = pd.concat([hist_small, df_small], ignore_index=True)
    all_rows = all_rows.dropna(subset=["FlightDate"] + key).copy()
    all_rows = all_rows.sort_values(key + ["FlightDate"]).reset_index(drop=True)

    def _roll_mean_lastN(g: pd.DataFrame) -> pd.Series:
        s = g["DepDelayMinutes"]
        out = s.shift(1).rolling(window=int(n_recent), min_periods=1).mean()
        out.index = g.index
        return out

    all_rows["flightnum_od_depdelay_mean_lastN"] = (
        all_rows.groupby(key, sort=False, group_keys=False).apply(_roll_mean_lastN)
    )

    def _roll_support_lastNd(g: pd.DataFrame) -> pd.Series:
        tmp = pd.DataFrame({"FlightDate": g["FlightDate"].values}, index=g.index)
        tmp["ind"] = 1.0
        s = tmp.set_index("FlightDate")["ind"]
        rolled = s.shift(1).rolling(f"{int(n_days)}D", min_periods=0).sum()
        tmp2 = tmp.copy()
        tmp2["support"] = rolled.to_numpy()
        out = pd.Series(tmp2["support"].values, index=tmp2.index)
        return out

    all_rows["flightnum_od_support_count_lastNd"] = (
        all_rows.groupby(key, sort=False, group_keys=False).apply(_roll_support_lastNd)
    )

    supp = pd.to_numeric(all_rows["flightnum_od_support_count_lastNd"], errors="coerce").fillna(0).astype(np.int32)
    all_rows["flightnum_od_support_count_lastNd"] = supp.astype(np.int16)

    all_rows["flightnum_od_low_support"] = (
        all_rows["flightnum_od_support_count_lastNd"] <= int(low_support_leq)
    ).astype(np.int8)

    feats = all_rows[all_rows["_is_target"]][
        ["FlightDate"] + key + [
            "flightnum_od_depdelay_mean_lastN",
            "flightnum_od_support_count_lastNd",
            "flightnum_od_low_support",
        ]
    ].copy()

    df = df.merge(feats, on=["FlightDate"] + key, how="left")

    df["flightnum_od_support_count_lastNd"] = (
        pd.to_numeric(df["flightnum_od_support_count_lastNd"], errors="coerce")
        .fillna(0)
        .astype("Int16")
    )
    df["flightnum_od_low_support"] = (
        pd.to_numeric(df["flightnum_od_low_support"], errors="coerce")
        .fillna(1)
        .astype("Int8")
    )

    return df


def add_carrier_dep_delay_baselines_from_history(df: pd.DataFrame, hist: pd.DataFrame, n_days: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str).str.upper()
    df["Origin"] = df["Origin"].astype(str).str.upper()

    hist = hist.copy()
    hist["FlightDate"] = pd.to_datetime(hist["FlightDate"], errors="coerce").dt.normalize()
    hist["Reporting_Airline"] = hist["Reporting_Airline"].astype(str).str.upper()
    hist["Origin"] = hist["Origin"].astype(str).str.upper()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    carr = hist.dropna(subset=["Reporting_Airline", "FlightDate", "DepDelayMinutes"]).copy()
    carr_daily = (
        carr.groupby(["Reporting_Airline", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )
    carr_daily["carrier_depdelay_mean_lastNdays"] = (
        carr_daily.groupby("Reporting_Airline")["DepDelayMinutes"]
        .apply(lambda s: s.shift(1).rolling(window=int(n_days), min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        carr_daily[["Reporting_Airline", "FlightDate", "carrier_depdelay_mean_lastNdays"]],
        on=["Reporting_Airline", "FlightDate"],
        how="left",
    )

    co = carr.dropna(subset=["Origin"]).copy()
    co_daily = (
        co.groupby(["Reporting_Airline", "Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "Origin", "FlightDate"])
    )
    co_daily["carrier_origin_depdelay_mean_lastNdays"] = (
        co_daily.groupby(["Reporting_Airline", "Origin"])["DepDelayMinutes"]
        .apply(lambda s: s.shift(1).rolling(window=int(n_days), min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    df = df.merge(
        co_daily[["Reporting_Airline", "Origin", "FlightDate", "carrier_origin_depdelay_mean_lastNdays"]],
        on=["Reporting_Airline", "Origin", "FlightDate"],
        how="left",
    )
    return df


def add_turn_time_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")

    df["dep_dt_utc"] = pd.to_datetime(df.get("dep_dt_local"), errors="coerce", utc=True)
    df["arr_dt_utc"] = pd.to_datetime(df.get("arr_dt_local"), errors="coerce", utc=True)

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline"]
    df = df.sort_values(key + ["dep_dt_utc"])

    df["prev_arr_dt_utc"] = df.groupby(key, sort=False)["arr_dt_utc"].shift(1)
    df["prev_dest"] = df.groupby(key, sort=False)["Dest"].shift(1)

    ok = df["prev_dest"].astype(str).str.upper() == df["Origin"].astype(str).str.upper()
    delta_hours = (df["dep_dt_utc"] - df["prev_arr_dt_utc"]).dt.total_seconds() / 3600.0
    df["turn_time_hours"] = np.where(ok, delta_hours, np.nan)
    return df


def add_dep_labels_for_thresholds(df: pd.DataFrame, thresholds: List[int], delay_col: str = "DepDelayMinutes") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_numeric(df.get(delay_col), errors="coerce").fillna(0.0)
    for thr in thresholds:
        df[f"y_dep_ge{int(thr)}"] = (d >= int(thr)).astype("Int8")
    if "y_dep15" not in df.columns and 15 in thresholds:
        df["y_dep15"] = df["y_dep_ge15"]
    return df


def balance_to_pos_frac(df: pd.DataFrame, label_col: str, pos_frac: float, seed: int, max_rows: Optional[int]) -> pd.DataFrame:
    df = df.copy()
    y = df[label_col].astype(int)
    pos = df[y == 1]
    neg = df[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        print(f"[WARN] Cannot balance {label_col}: one class empty.")
        return df

    target = float(pos_frac)
    if not (0.0 < target < 1.0):
        raise ValueError("pos_frac must be in (0,1)")

    a_max = len(pos)
    b_needed_for_a_max = int(round(a_max * (1 - target) / target))
    if b_needed_for_a_max <= len(neg):
        a = a_max
        b = b_needed_for_a_max
        limiting = "pos"
    else:
        b_max = len(neg)
        a_needed_for_b_max = int(round(b_max * target / (1 - target)))
        a = min(a_needed_for_b_max, a_max)
        b = b_max
        limiting = "neg"

    pos_s = pos.sample(n=a, random_state=seed)
    neg_s = neg.sample(n=b, random_state=seed)
    out = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if max_rows is not None and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)

    achieved = out[label_col].mean()
    print(f"[INFO] Balanced {label_col} to pos_frac≈{achieved:.3f} (target={target:.3f}, limiting={limiting}) rows={len(out)}")
    return out


# ------------------------------ main ------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python features_dep.py data/dep_arr_config.json")
        sys.exit(1)

    cfg_path = _abspath(sys.argv[1], base="repo")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    target = (cfg.get("target") or "dep").lower()
    if target != "dep":
        raise ValueError("features_dep.py expects cfg.target == 'dep'")

    in_enriched = cfg.get("output_enriched_unbalanced_path")
    if not in_enriched:
        raise KeyError("cfg.output_enriched_unbalanced_path is required (written by prepare_dataset.py)")
    in_path = _abspath(_format_path_template(in_enriched, target=target), base="data")
    if not in_path.exists():
        raise FileNotFoundError(f"Enriched parquet not found: {in_path}")

    print(f"[INFO] reading enriched -> {in_path}")
    df = pd.read_parquet(in_path)

    hist_path_cfg = cfg.get("history_output_path", "intermediate/history_pool_dep.parquet")
    hist_path = _abspath(_format_path_template(hist_path_cfg, target=target), base="data")
    if not hist_path.exists():
        raise FileNotFoundError(
            f"History pool parquet not found: {hist_path}\n"
            "Run prepare_dataset.py with history_output_path enabled, or set cfg.history_output_path correctly."
        )
    hist = pd.read_parquet(hist_path)

    df["Origin"] = df["Origin"].astype(str).str.upper()
    df["Dest"] = df["Dest"].astype(str).str.upper()
    df["od_pair"] = df["Origin"] + "_" + df["Dest"]

    if "dep_dt_local" in df.columns:
        dts = pd.to_datetime(df["dep_dt_local"], errors="coerce")
        df["dep_dow"] = dts.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = dts.dt.hour.astype("Int8")
    else:
        fd = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["dep_dow"] = fd.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").floordiv(100).astype("Int8")

    feat_cfg = cfg.get("features_dep") or {}
    hw = (feat_cfg.get("holiday_windows") or {})
    df = add_holiday_flag(
        df,
        thanksgiving_window=int(hw.get("thanksgiving", 2)),
        christmas_window=int(hw.get("christmas", 3)),
    )

    if "aircraft_type" not in df.columns:
        df["aircraft_type"] = "Unknown"

    if "origin_weathercode" in df.columns:
        df["origin_daily_weathercode"] = pd.to_numeric(df["origin_weathercode"], errors="coerce").astype("Int64").astype("object")
    if "origin_dep_weathercode" in df.columns:
        df["origin_dep_hour_weathercode"] = pd.to_numeric(df["origin_dep_weathercode"], errors="coerce").astype("Int64").astype("object")

    df = add_weather_kelvin(df)

    n_recent = int(feat_cfg.get("n_recent_flightnum_od", 20))
    n_days_hist = int(feat_cfg.get("n_days_flightnum_od", 14))
    low_support_leq = int(feat_cfg.get("low_support_leq", 2))

    df = add_recent_flightnum_od_mean_from_history(
        df, hist,
        n_recent=n_recent,
        n_days=n_days_hist,
        low_support_leq=low_support_leq,
    )

    n_days_carrier = int(feat_cfg.get("n_days_carrier_network", 14))
    df = add_carrier_dep_delay_baselines_from_history(df, hist, n_days=n_days_carrier)

    df = add_turn_time_hours(df)

    thresholds = (cfg.get("balance", {}) or {}).get("thresholds", [15, 30, 45, 60])
    thresholds = [int(x) for x in thresholds]
    delay_col = (cfg.get("balance", {}) or {}).get("delay_col", "DepDelayMinutes")
    df = add_dep_labels_for_thresholds(df, thresholds, delay_col=delay_col)

    # default is now DATA_ROOT-relative
    out_feat_cfg = cfg.get("output_features_unbalanced_path", "processed/features_dep_unbalanced.parquet")
    out_feat_cfg = _format_path_template(out_feat_cfg, target=target)
    out_feat_path = _abspath(out_feat_cfg, base="data")
    out_feat_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_feat_path, index=False)
    print(f"[OK] wrote features unbalanced rows={len(df)} -> {out_feat_path}")

    fb = cfg.get("feature_balance") or {}
    if not bool(fb.get("enabled", True)):
        return

    eval_cfg = fb.get("eval") or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    train_pool = df

    if eval_enabled:
        by = (eval_cfg.get("by") or "last_days").lower()
        if by != "last_days":
            raise ValueError("feature_balance.eval.by currently supports only 'last_days'")
        last_days = int(eval_cfg.get("last_days", 90))
        fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
        cutoff = fd.max() - pd.Timedelta(days=last_days)
        eval_mask = fd > cutoff

        eval_df = df[eval_mask].copy()
        train_pool = df[~eval_mask].copy()

        # default is now DATA_ROOT-relative
        eval_out = eval_cfg.get("output_path", "processed/eval_dep_unbalanced.parquet")
        eval_out = _format_path_template(eval_out, target=target)
        eval_path = _abspath(eval_out, base="data")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_parquet(eval_path, index=False)

        print(f"[INFO] feature eval last_days={last_days} cutoff>{cutoff.date()} rows={len(eval_df)} train_pool rows={len(train_pool)}")
        print(f"[OK] wrote feature eval unbalanced -> {eval_path}")

    thr_list = [int(x) for x in (fb.get("thresholds") or thresholds)]
    pos_frac = float(fb.get("pos_frac", 0.50))
    seed = int(fb.get("seed", 42))
    max_rows = fb.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None

    # default is now DATA_ROOT-relative
    out_tmpl = fb.get("output_template", "processed/train_{target}_ge{thr}_balanced.parquet")

    for thr in thr_list:
        label_col = f"y_dep_ge{thr}"
        if label_col not in train_pool.columns:
            raise KeyError(f"Missing label column {label_col} in features dataframe")

        bal_df = balance_to_pos_frac(train_pool, label_col=label_col, pos_frac=pos_frac, seed=seed, max_rows=max_rows)

        out_p = _format_path_template(out_tmpl, target=target, thr=thr)
        out_p = _abspath(out_p, base="data")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        bal_df.to_parquet(out_p, index=False)
        print(f"[OK] wrote balanced feature train thr={thr} -> {out_p}")


if __name__ == "__main__":
    main()
