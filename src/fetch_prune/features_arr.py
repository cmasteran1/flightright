#!/usr/bin/env python3
# src/fetch_prune/features_arr.py
#
# (same header as before; only the ETA congestion computation was made bulletproof)

import sys
import json
import gc
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

MEAN_WINDOWS = [7, 14]
MEDIAN_WINDOWS = [7]
THRESHOLDS_DEFAULT = [15, 30, 45, 60]


def _abspath(p: str, *, base: str = "repo") -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if base == "repo":
        return (REPO_ROOT / pp).resolve()
    if base == "data":
        return (DATA_ROOT / pp).resolve()
    raise ValueError("base must be 'repo' or 'data'")


def _ensure_base_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    for col in ["Origin", "Dest", "Reporting_Airline", "DepTimeBlk"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    if "Flight_Number_Reporting_Airline" in df.columns:
        df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df["Flight_Number_Reporting_Airline"], errors="coerce")

    for c in ["CRSDepTime", "CRSArrTime", "WheelsOff", "AirTime", "ArrDelayMinutes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Origin", "Dest", "Reporting_Airline", "DepTimeBlk"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def _hhmm_series_to_minutes(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = ~np.isnan(x)
    if not np.any(m):
        return out
    xi = x[m].astype(np.int64)
    hh = xi // 100
    mm = xi % 100
    ok = (hh >= 0) & (hh <= 47) & (mm >= 0) & (mm <= 59)
    mins = hh * 60 + mm
    tmp = np.full(xi.shape, np.nan, dtype=float)
    tmp[ok] = mins[ok].astype(float)
    out[m] = tmp
    return out


def _minutes_diff_wrap_vec(actual_min: np.ndarray, sched_min: np.ndarray) -> np.ndarray:
    d = actual_min - sched_min
    m = ~np.isnan(d)
    d2 = d.copy()
    d2[m & (d2 < -720)] += 1440.0
    d2[m & (d2 > 720)] -= 1440.0
    return d2


def _prefilter_hist(hist: pd.DataFrame, keys_df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    keys_df = keys_df[key_cols].drop_duplicates()
    return hist.merge(keys_df, on=key_cols, how="inner")


def _attach_asof(df: pd.DataFrame, stats_tbl: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    st = stats_tbl.copy()

    df2["FlightDate"] = pd.to_datetime(df2["FlightDate"], errors="coerce")
    st["FlightDate"] = pd.to_datetime(st["FlightDate"], errors="coerce")

    for k in keys:
        if k not in df2.columns or k not in st.columns:
            continue

        dfd = df2[k].dtype
        std = st[k].dtype

        if isinstance(dfd, CategoricalDtype):
            st[k] = st[k].astype(str)
            st[k] = pd.Categorical(st[k], categories=df2[k].cat.categories)
        elif isinstance(std, CategoricalDtype):
            df2[k] = df2[k].astype(str)
            df2[k] = pd.Categorical(df2[k], categories=st[k].cat.categories)
        else:
            if dfd != std:
                df2[k] = df2[k].astype(str)
                st[k] = st[k].astype(str)

    mask_left = df2["FlightDate"].notna()
    left = df2.loc[mask_left].copy()
    left_nat = df2.loc[~mask_left].copy()

    mask_right = st["FlightDate"].notna()
    right = st.loc[mask_right].copy()

    sort_cols = ["FlightDate"] + keys
    left = left.sort_values(sort_cols, kind="mergesort")
    right = right.sort_values(sort_cols, kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        on="FlightDate",
        by=keys,
        direction="backward",
        allow_exact_matches=True,
    )

    out = pd.concat([merged, left_nat], axis=0)
    out = out.loc[df.index]
    return out


def _rolling_tbl_mean_count(hist: pd.DataFrame, keys: List[str], value_col: str, window_days: int, prefix: str) -> pd.DataFrame:
    h = hist.dropna(subset=["FlightDate"]).copy()
    h = h.dropna(subset=keys, how="any").copy()
    h[value_col] = pd.to_numeric(h[value_col], errors="coerce")
    h = h.dropna(subset=[value_col]).copy()
    h = h.sort_values(["FlightDate"] + keys, kind="mergesort")

    g = h.set_index("FlightDate").groupby(keys, sort=False, observed=True)[value_col]
    rw = g.rolling(f"{window_days}D", closed="left")

    mean_s = rw.mean()
    cnt_s = rw.count()

    out = pd.concat([mean_s, cnt_s], axis=1).reset_index()
    out.columns = keys + ["FlightDate", f"{prefix}mean_{window_days}d", f"{prefix}count_{window_days}d"]
    out = out.sort_values(["FlightDate"] + keys, kind="mergesort")
    return out


def _rolling_tbl_median(hist: pd.DataFrame, keys: List[str], value_col: str, window_days: int, prefix: str) -> pd.DataFrame:
    h = hist.dropna(subset=["FlightDate"]).copy()
    h = h.dropna(subset=keys, how="any").copy()
    h[value_col] = pd.to_numeric(h[value_col], errors="coerce")
    h = h.dropna(subset=[value_col]).copy()
    h = h.sort_values(["FlightDate"] + keys, kind="mergesort")

    g = h.set_index("FlightDate").groupby(keys, sort=False, observed=True)[value_col]
    rw = g.rolling(f"{window_days}D", closed="left")
    med_s = rw.median()

    out = med_s.reset_index()
    out = out.rename(columns={value_col: f"{prefix}median_{window_days}d"})
    out = out.sort_values(["FlightDate"] + keys, kind="mergesort")
    return out


def add_route_arrival_delay_stats(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist, df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_mean_{w}d": f"arrdelay_mean_{w}d_fn_od", f"arrdelay_count_{w}d": f"arrdelay_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    tbl = _rolling_tbl_median(hist_fn, keys_fn, "ArrDelayMinutes", 7, prefix="arrdelay_").rename(
        columns={"arrdelay_median_7d": "arrdelay_median_7d_fn_od"}
    )
    df = _attach_asof(df, tbl, keys_fn)
    del tbl, hist_fn
    gc.collect()

    keys_car = ["Reporting_Airline", "Origin", "Dest"]
    hist_car = _prefilter_hist(hist, df, keys_car)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_car, keys_car, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_mean_{w}d": f"arrdelay_mean_{w}d_car_od", f"arrdelay_count_{w}d": f"arrdelay_n_{w}d_car_od"}
        )
        df = _attach_asof(df, tbl, keys_car)
        del tbl
        gc.collect()

    tbl = _rolling_tbl_median(hist_car, keys_car, "ArrDelayMinutes", 7, prefix="arrdelay_").rename(
        columns={"arrdelay_median_7d": "arrdelay_median_7d_car_od"}
    )
    df = _attach_asof(df, tbl, keys_car)
    del tbl, hist_car
    gc.collect()

    return df


def add_recent_airtime_stats(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist.dropna(subset=["AirTime"]), df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "AirTime", w, prefix="airtime_").rename(
            columns={f"airtime_mean_{w}d": f"airtime_mean_{w}d_fn_od", f"airtime_count_{w}d": f"airtime_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    del hist_fn
    gc.collect()

    keys_car = ["Reporting_Airline", "Origin", "Dest"]
    hist_car = _prefilter_hist(hist.dropna(subset=["AirTime"]), df, keys_car)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_car, keys_car, "AirTime", w, prefix="airtime_").rename(
            columns={f"airtime_mean_{w}d": f"airtime_mean_{w}d_car_od", f"airtime_count_{w}d": f"airtime_n_{w}d_car_od"}
        )
        df = _attach_asof(df, tbl, keys_car)
        del tbl
        gc.collect()

    del hist_car
    gc.collect()

    return df


def add_recent_wheels_off_slip_stats(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    if "DepTimeBlk" not in df.columns:
        hh = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").fillna(-1).astype(int)
        hh = (hh // 100).clip(0, 23)
        df["DepTimeBlk"] = hh.map(lambda x: f"{x:02d}00-{x:02d}59").astype("category")

    hist2 = hist.copy()
    if "DepTimeBlk" not in hist2.columns:
        hh = pd.to_numeric(hist2.get("CRSDepTime"), errors="coerce").fillna(-1).astype(int)
        hh = (hh // 100).clip(0, 23)
        hist2["DepTimeBlk"] = hh.map(lambda x: f"{x:02d}00-{x:02d}59").astype("category")

    crs_min = _hhmm_series_to_minutes(hist2["CRSDepTime"])
    wo_min = _hhmm_series_to_minutes(hist2["WheelsOff"])
    hist2["wo_slip_min"] = _minutes_diff_wrap_vec(wo_min, crs_min)
    hist2 = hist2.dropna(subset=["wo_slip_min"]).copy()

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist2, df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "wo_slip_min", w, prefix="wo_slip_").rename(
            columns={f"wo_slip_mean_{w}d": f"wo_slip_mean_{w}d_fn_od", f"wo_slip_count_{w}d": f"wo_slip_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    del hist_fn
    gc.collect()

    keys_blk = ["Origin", "DepTimeBlk"]
    hist_blk = _prefilter_hist(hist2, df, keys_blk)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_blk, keys_blk, "wo_slip_min", w, prefix="wo_slip_").rename(
            columns={f"wo_slip_mean_{w}d": f"wo_slip_mean_{w}d_origin_blk", f"wo_slip_count_{w}d": f"wo_slip_n_{w}d_origin_blk"}
        )
        df = _attach_asof(df, tbl, keys_blk)
        del tbl
        gc.collect()

    del hist_blk, hist2
    gc.collect()
    return df


def add_projected_arrival_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Congestion counts in ±60 minutes around:
      - scheduled arrival time (arr_dt_local)
      - ETA-shifted arrival time computed in ns:
            eta_ns = arr_ns + shift_minutes * 60e9
    This avoids tz-aware datetime casting pitfalls.
    """
    df = _ensure_base_types(df).copy()

    if "arr_dt_local" not in df.columns:
        raise KeyError("Projected arrival congestion requires arr_dt_local in enriched parquet.")

    arr_dt = pd.to_datetime(df["arr_dt_local"], errors="coerce")
    df["_arr_local_date"] = arr_dt.dt.date

    arr_ns_all = arr_dt.astype("int64").to_numpy()
    valid = arr_ns_all != np.iinfo("int64").min

    # shift source: 14d carrier OD mean, else 7d carrier OD mean, else 0
    if "arrdelay_mean_14d_car_od" in df.columns:
        shift_min_all = pd.to_numeric(df["arrdelay_mean_14d_car_od"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "arrdelay_mean_7d_car_od" in df.columns:
        shift_min_all = pd.to_numeric(df["arrdelay_mean_7d_car_od"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        shift_min_all = np.zeros(len(df), dtype=float)

    # ns shift (int64) — round to nearest minute in ns
    shift_ns_valid = np.rint(shift_min_all[valid] * 60.0 * 1_000_000_000.0).astype(np.int64)

    df["dest_arrivals_pm60_sched"] = 0
    df["dest_airline_arrivals_pm60_sched"] = 0
    df["dest_arrivals_pm60_eta"] = 0
    df["dest_airline_arrivals_pm60_eta"] = 0

    window_ns = int(pd.Timedelta(minutes=60).value)

    work = df.loc[valid, ["Dest", "Reporting_Airline", "_arr_local_date"]].copy()
    work["idx"] = work.index
    work["arr_ns"] = arr_ns_all[valid]
    work["eta_ns"] = work["arr_ns"].to_numpy(dtype=np.int64, copy=False) + shift_ns_valid

    # (Dest, date)
    for (dest, d0), g in work.groupby(["Dest", "_arr_local_date"], sort=False):
        times = np.sort(g["arr_ns"].to_numpy(dtype=np.int64, copy=False))

        q = g["arr_ns"].to_numpy(dtype=np.int64, copy=False)
        left = np.searchsorted(times, q - window_ns, side="left")
        right = np.searchsorted(times, q + window_ns, side="right")
        counts = (right - left - 1).astype(np.int32)
        counts[counts < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_arrivals_pm60_sched"] = counts

        q2 = g["eta_ns"].to_numpy(dtype=np.int64, copy=False)
        left2 = np.searchsorted(times, q2 - window_ns, side="left")
        right2 = np.searchsorted(times, q2 + window_ns, side="right")
        counts2 = (right2 - left2 - 1).astype(np.int32)
        counts2[counts2 < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_arrivals_pm60_eta"] = counts2

    # (Dest, airline, date)
    for (dest, carr, d0), g in work.groupby(["Dest", "Reporting_Airline", "_arr_local_date"], sort=False):
        times = np.sort(g["arr_ns"].to_numpy(dtype=np.int64, copy=False))

        q = g["arr_ns"].to_numpy(dtype=np.int64, copy=False)
        left = np.searchsorted(times, q - window_ns, side="left")
        right = np.searchsorted(times, q + window_ns, side="right")
        counts = (right - left - 1).astype(np.int32)
        counts[counts < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_airline_arrivals_pm60_sched"] = counts

        q2 = g["eta_ns"].to_numpy(dtype=np.int64, copy=False)
        left2 = np.searchsorted(times, q2 - window_ns, side="left")
        right2 = np.searchsorted(times, q2 + window_ns, side="right")
        counts2 = (right2 - left2 - 1).astype(np.int32)
        counts2[counts2 < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_airline_arrivals_pm60_eta"] = counts2

    df = df.drop(columns=["_arr_local_date"], errors="ignore")

    for c in ["dest_arrivals_pm60_sched", "dest_airline_arrivals_pm60_sched", "dest_arrivals_pm60_eta", "dest_airline_arrivals_pm60_eta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int32")

    return df


def _normalize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "FlightDate" in x.columns:
        fd = pd.to_datetime(x["FlightDate"], errors="coerce")
        x["FlightDate"] = fd.dt.normalize()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in x.columns:
            x[c] = x[c].astype(str).str.upper().str.strip()
    for c in ["CRSDepTime", "CRSArrTime"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("Int64")
    if "Flight_Number_Reporting_Airline" in x.columns:
        x["Flight_Number_Reporting_Airline"] = pd.to_numeric(x["Flight_Number_Reporting_Airline"], errors="coerce").astype("Int64")
    return x


def merge_dep_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dep_feat_path_raw = cfg.get("dep_features_unbalanced_path")
    if not dep_feat_path_raw:
        print("[INFO] dep_features_unbalanced_path not set; skipping dep-feature merge.")
        return df

    dep_feat_path = _abspath(dep_feat_path_raw, base="repo")
    if not dep_feat_path.exists():
        print(f"[WARN] dep features file not found; skipping dep-feature merge: {dep_feat_path}")
        return df

    print(f"[INFO] Merging dep features from: {dep_feat_path}")

    dep = pd.read_parquet(dep_feat_path)
    dep = _normalize_join_keys(dep)
    df_norm = _normalize_join_keys(df)

    join_keys = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"]
    join_keys = [k for k in join_keys if (k in df_norm.columns and k in dep.columns)]
    if not join_keys:
        raise RuntimeError("No common join keys found to merge dep features into arrival features.")

    dep_cols = [c for c in dep.columns if (c not in df_norm.columns) or (c in join_keys)]
    dep = dep[dep_cols].copy()

    merged = df_norm.merge(dep, on=join_keys, how="left", suffixes=("", "_depdup"))
    dup_cols = [c for c in merged.columns if c.endswith("_depdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    del dep, df_norm
    gc.collect()
    return merged


def add_arrival_labels(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df.get("ArrDelayMinutes"), errors="coerce")
    for thr in thresholds:
        df[f"y_arr_ge{thr}"] = (y >= float(thr)).astype("Int8")
    return df


def main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    target = str(cfg.get("target", "dep")).strip().lower()
    if target != "arr":
        raise ValueError("features_arr.py expects cfg['target'] == 'arr'")

    thresholds = [int(x) for x in cfg.get("thresholds", THRESHOLDS_DEFAULT)]

    enriched_path = _abspath(cfg["output_enriched_unbalanced_path"], base="repo")
    history_path = _abspath(cfg["history_output_path"], base="repo")
    out_path = _abspath(cfg["output_features_unbalanced_path"], base="repo")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading enriched: {enriched_path}")
    need_df = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "CRSArrTime", "arr_dt_local", "DepTimeBlk", "ArrDelayMinutes",
    ]
    df = pd.read_parquet(enriched_path, columns=need_df)
    df = _ensure_base_types(df)

    print(f"[INFO] Reading history pool: {history_path}")
    need_hist = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "ArrDelayMinutes", "AirTime", "WheelsOff",
    ]
    hist = pd.read_parquet(history_path, columns=need_hist)
    hist = _ensure_base_types(hist)

    df_start = df["FlightDate"].min()
    df_end = df["FlightDate"].max()
    if pd.notna(df_start) and pd.notna(df_end):
        hist = hist[(hist["FlightDate"] >= (df_start - pd.Timedelta(days=14))) & (hist["FlightDate"] <= df_end)].copy()
        gc.collect()

    df = add_route_arrival_delay_stats(df, hist); gc.collect()
    df = add_recent_airtime_stats(df, hist); gc.collect()
    df = add_recent_wheels_off_slip_stats(df, hist); gc.collect()
    df = add_projected_arrival_congestion(df); gc.collect()
    df = merge_dep_features(df, cfg); gc.collect()
    df = add_arrival_labels(df, thresholds)

    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote arrival features rows={len(df):,} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/fetch_prune/features_arr.py data/arr_config.json")
    main(sys.argv[1])