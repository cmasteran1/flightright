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
from typing import Optional, List, Tuple, Dict
from collections import deque

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


# ------------------------------ parquet read helpers (column projection) ------------------------------

def _parquet_columns(path: Path) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(str(path))
        return list(pf.schema.names)
    except Exception:
        return None


def _read_parquet_projected(path: Path, want_cols: List[str]) -> pd.DataFrame:
    cols = _parquet_columns(path)
    if cols is None:
        return pd.read_parquet(path)

    keep = [c for c in want_cols if c in cols]
    if not keep:
        return pd.read_parquet(path)

    return pd.read_parquet(path, columns=keep)


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

    def is_near(d0: date, center: date, win: int) -> bool:
        return abs((d0 - center).days) <= win

    flags = []
    for ts in base:
        if pd.isna(ts):
            flags.append(0)
            continue
        d0 = ts.date()
        y = d0.year
        f = 0
        if y in tg and is_near(d0, tg[y], thanksgiving_window):
            f = 1
        if y in xmas and is_near(d0, xmas[y], christmas_window):
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


# ---------- helper: ensure we have a usable scheduled-departure datetime ----------
def _ensure_dep_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dep_dt_local" in df.columns:
        df["dep_dt_local"] = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    else:
        fd = pd.to_datetime(df.get("FlightDate"), errors="coerce")
        hhmm = pd.to_numeric(df.get("CRSDepTime"), errors="coerce")

        def _mk(fd0, t0):
            if pd.isna(fd0) or pd.isna(t0):
                return pd.NaT
            try:
                s = str(int(t0)).zfill(4)
                hh = int(s[:2])
                mm = int(s[2:])
                return pd.Timestamp(fd0.date()) + pd.Timedelta(hours=hh, minutes=mm)
            except Exception:
                return pd.NaT

        df["dep_dt_local"] = [_mk(a, b) for a, b in zip(fd.tolist(), hhmm.tolist())]

    dts = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    df["dep_local_date"] = dts.dt.date
    return df


# ---------- congestion features (±3 hours around scheduled dep), from FULL HISTORY ----------

def _counts_within_window_for_queries(sorted_pool_ns: np.ndarray, sorted_query_ns: np.ndarray, window_ns: int) -> np.ndarray:
    n_pool = len(sorted_pool_ns)
    out = np.zeros(len(sorted_query_ns), dtype=np.int32)
    left = 0
    right = 0
    for i, t in enumerate(sorted_query_ns):
        lo = t - window_ns
        hi = t + window_ns
        while left < n_pool and sorted_pool_ns[left] < lo:
            left += 1
        if right < left:
            right = left
        while right < n_pool and sorted_pool_ns[right] <= hi:
            right += 1
        out[i] = max(0, right - left)
    return out


def _build_congestion_time_dicts_from_history(
    hist: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, object], np.ndarray], Dict[Tuple[str, str, object], np.ndarray]]:
    h = hist.copy()
    h["Origin"] = h.get("Origin", "").astype(str).str.upper().str.strip()
    h["Reporting_Airline"] = h.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    if "dep_dt_local" not in h.columns:
        raise KeyError("History pool must include dep_dt_local for congestion-from-history. Re-run prepare_dataset.py to rebuild history pool with dep_dt_local kept.")

    dep_local = pd.to_datetime(h["dep_dt_local"], errors="coerce")
    dep_utc = pd.to_datetime(dep_local, errors="coerce", utc=True)
    dep_ns = dep_utc.astype("int64")
    valid = dep_ns != np.iinfo("int64").min

    if "dep_local_date" in h.columns:
        dep_local_date = pd.Series(h["dep_local_date"])
    else:
        dep_local_date = dep_local.dt.date

    h2 = pd.DataFrame(
        {
            "Origin": h.loc[valid, "Origin"].values,
            "Reporting_Airline": h.loc[valid, "Reporting_Airline"].values,
            "dep_local_date": pd.Series(dep_local_date.loc[valid].values),
            "dep_ns": dep_ns[valid],
        }
    )

    h2 = h2.sort_values(["Origin", "dep_local_date", "dep_ns"], kind="mergesort")

    times_by_origin_date: Dict[Tuple[str, object], np.ndarray] = {}
    for (o, d0), g in h2.groupby(["Origin", "dep_local_date"], sort=False):
        times_by_origin_date[(o, d0)] = g["dep_ns"].to_numpy(dtype=np.int64, copy=False)

    h3 = h2.sort_values(["Origin", "Reporting_Airline", "dep_local_date", "dep_ns"], kind="mergesort")
    times_by_origin_airline_date: Dict[Tuple[str, str, object], np.ndarray] = {}
    for (o, carr, d0), g in h3.groupby(["Origin", "Reporting_Airline", "dep_local_date"], sort=False):
        times_by_origin_airline_date[(o, carr, d0)] = g["dep_ns"].to_numpy(dtype=np.int64, copy=False)

    return times_by_origin_date, times_by_origin_airline_date


def add_congestion_3h_features_from_history(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_dep_dt(df)

    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    dep_local = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    dep_utc = pd.to_datetime(dep_local, errors="coerce", utc=True)
    dep_ns = dep_utc.astype("int64")
    valid = dep_ns != np.iinfo("int64").min
    if "dep_local_date" not in df.columns:
        df["dep_local_date"] = dep_local.dt.date

    df["origin_congestion_3h_total"] = pd.Series([0] * len(df), index=df.index, dtype="Int16")
    df["origin_airline_congestion_3h_total"] = pd.Series([0] * len(df), index=df.index, dtype="Int16")
    if not valid.any():
        return df

    WINDOW_NS = int(pd.Timedelta(hours=3).value)
    times_by_origin_date, times_by_origin_airline_date = _build_congestion_time_dicts_from_history(hist)

    work = pd.DataFrame(
        {
            "idx": df.index[valid],
            "Origin": df.loc[valid, "Origin"].values,
            "Reporting_Airline": df.loc[valid, "Reporting_Airline"].values,
            "dep_local_date": pd.Series(df.loc[valid, "dep_local_date"].values),
            "dep_ns": dep_ns[valid],
        }
    )

    for (o, d0), g in work.groupby(["Origin", "dep_local_date"], sort=False):
        pool = times_by_origin_date.get((o, d0))
        if pool is None or len(pool) == 0:
            continue
        g2 = g.sort_values("dep_ns", kind="mergesort")
        q = g2["dep_ns"].to_numpy(dtype=np.int64, copy=False)
        counts = _counts_within_window_for_queries(pool, q, WINDOW_NS)
        counts = np.maximum(0, counts - 1)
        counts = np.clip(counts, 0, np.iinfo(np.int16).max).astype(np.int16)
        df.loc[g2["idx"].to_numpy(), "origin_congestion_3h_total"] = pd.Series(counts, index=g2["idx"].to_numpy()).astype("Int16")

    for (o, carr, d0), g in work.groupby(["Origin", "Reporting_Airline", "dep_local_date"], sort=False):
        pool = times_by_origin_airline_date.get((o, carr, d0))
        if pool is None or len(pool) == 0:
            continue
        g2 = g.sort_values("dep_ns", kind="mergesort")
        q = g2["dep_ns"].to_numpy(dtype=np.int64, copy=False)
        counts = _counts_within_window_for_queries(pool, q, WINDOW_NS)
        counts = np.maximum(0, counts - 1)
        counts = np.clip(counts, 0, np.iinfo(np.int16).max).astype(np.int16)
        df.loc[g2["idx"].to_numpy(), "origin_airline_congestion_3h_total"] = pd.Series(counts, index=g2["idx"].to_numpy()).astype("Int16")

    return df


# ---------- tail + flight-number temporal features ----------
def add_tail_and_flightnum_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_dep_dt(df)

    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Dest"] = df.get("Dest", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    tail = df.get("Tail_Number", "").astype(str).str.upper().str.strip()
    bad_tail = tail.isin(["", "NAN", "NONE", "NULL", "UNKNOWN", "UNKN", "NA"])
    tail = tail.where(~bad_tail, "")
    df["Tail_Number"] = tail

    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")

    dep_utc = pd.to_datetime(df.get("dep_dt_local"), errors="coerce", utc=True)
    arr_utc = pd.to_datetime(df.get("arr_dt_local"), errors="coerce", utc=True) if "arr_dt_local" in df.columns else pd.to_datetime(pd.Series([pd.NaT] * len(df)), utc=True)

    df["_dep_utc"] = dep_utc
    df["_arr_utc"] = arr_utc

    dep_local_date = df.get("dep_local_date")
    if dep_local_date is None:
        dts = pd.to_datetime(df.get("dep_dt_local"), errors="coerce")
        dep_local_date = dts.dt.date
        df["dep_local_date"] = dep_local_date

    valid_tail = df["Tail_Number"].astype(str).str.len() > 0
    valid_dep = df["_dep_utc"].notna()
    valid_day = pd.Series(dep_local_date).notna()

    df["tail_leg_num_day"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int16")
    mask_leg = valid_tail & valid_dep & valid_day
    if mask_leg.any():
        key = ["Tail_Number", "dep_local_date"]
        order = df.loc[mask_leg].sort_values(key + ["_dep_utc"]).copy()
        order["tail_leg_num_day"] = order.groupby(key, sort=False).cumcount() + 1
        df.loc[order.index, "tail_leg_num_day"] = pd.to_numeric(order["tail_leg_num_day"], errors="coerce").astype("Int16")

    df["flightnum_hours_since_first_departure_today"] = np.nan
    valid_fn = df["Flight_Number_Reporting_Airline"].notna() & valid_dep & valid_day
    if valid_fn.any():
        key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "dep_local_date"]
        first_dep = df.loc[valid_fn].groupby(key, sort=False)["_dep_utc"].transform("min")
        hours = (df.loc[valid_fn, "_dep_utc"] - first_dep).dt.total_seconds() / 3600.0
        df.loc[valid_fn, "flightnum_hours_since_first_departure_today"] = hours.astype(float)

    df["has_recent_arrival_turn_5h"] = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index).astype("Int8")

    ok_rows = valid_tail & valid_dep
    if ok_rows.any():
        w = df.loc[ok_rows, ["Tail_Number", "Origin", "Dest", "_dep_utc", "_arr_utc"]].copy()
        w = w.sort_values(["Tail_Number", "_dep_utc"])

        w["prev_arr_utc"] = w.groupby("Tail_Number", sort=False)["_arr_utc"].shift(1)
        w["prev_dest"] = w.groupby("Tail_Number", sort=False)["Dest"].shift(1)

        gap_h = (w["_dep_utc"] - w["prev_arr_utc"]).dt.total_seconds() / 3600.0
        cond = (
            w["prev_arr_utc"].notna()
            & w["prev_dest"].notna()
            & (w["prev_dest"].astype(str).str.upper() == w["Origin"].astype(str).str.upper())
            & (gap_h >= 0.0)
            & (gap_h <= 5.0)
        )

        df.loc[w.index, "has_recent_arrival_turn_5h"] = pd.Series(cond.astype(np.int8).values, index=w.index).astype("Int8")

    df = df.drop(columns=["_dep_utc", "_arr_utc"], errors="ignore")
    return df


# ---------- NEW: flight-number OD rolling means over last {7,14,21,28} days ----------
def add_flightnum_od_depdelay_means_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
    *,
    n_recent: int = 20,
    low_support_leq: int = 2,
) -> pd.DataFrame:
    """
    Produces features:
      - flightnum_od_depdelay_mean_last{W}  for W in windows_days (calendar-day windows)
      - flightnum_od_support_count_last{Wmax}d
      - flightnum_od_low_support_last{Wmax}d

    IMPORTANT:
      - uses strictly PRIOR observations (no leakage) by processing in time order and only
        adding current-row delay AFTER computing features for that row.
      - windowed means are over the last W calendar days (not last W flights).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df
    wmax = int(max(windows))

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]

    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")
    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()

    hist = hist.copy()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in hist.columns:
            hist[c] = hist[c].astype(str).str.upper().str.strip()
    hist["Flight_Number_Reporting_Airline"] = pd.to_numeric(hist.get("Flight_Number_Reporting_Airline"), errors="coerce")
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    need_df = ["FlightDate"] + key + ["DepDelayMinutes", "dep_dt_local"]
    need_hist = ["FlightDate"] + key + ["DepDelayMinutes", "dep_dt_local"]
    need_df = [c for c in need_df if c in df.columns]
    need_hist = [c for c in need_hist if c in hist.columns]

    df_small = df[need_df].copy()
    hist_small = hist[need_hist].copy()

    df_small = df_small.dropna(subset=["FlightDate"] + key).copy()
    hist_small = hist_small.dropna(subset=["FlightDate"] + key).copy()

    def _time_i8(frame: pd.DataFrame) -> np.ndarray:
        if "dep_dt_local" in frame.columns:
            t = pd.to_datetime(frame["dep_dt_local"], errors="coerce", utc=True)
            out = t.astype("int64")
            out = np.where(out == np.iinfo("int64").min, 0, out)
            return out.astype(np.int64, copy=False)
        return np.zeros(len(frame), dtype=np.int64)

    df_small["_t"] = _time_i8(df_small)
    hist_small["_t"] = _time_i8(hist_small)

    sort_cols = key + ["FlightDate", "_t"]
    df_small = df_small.sort_values(sort_cols, kind="mergesort")
    hist_small = hist_small.sort_values(sort_cols, kind="mergesort")

    # output columns
    out_means = {w: pd.Series(np.nan, index=df.index, dtype="float64") for w in windows}
    out_supp = pd.Series(np.zeros(len(df), dtype=np.int16), index=df.index, dtype="Int16")
    out_low = pd.Series(np.ones(len(df), dtype=np.int8), index=df.index, dtype="Int8")

    # build grouped history arrays
    hist_groups: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not hist_small.empty:
        h_dates = hist_small["FlightDate"].to_numpy()
        h_date_int = h_dates.astype("datetime64[D]").astype(np.int32)
        h_t = hist_small["_t"].to_numpy(dtype=np.int64, copy=False)
        h_delay = pd.to_numeric(hist_small["DepDelayMinutes"], errors="coerce").to_numpy(dtype=np.float32, copy=False)

        h_key_df = hist_small[key]
        h_key_vals = [h_key_df[c].to_numpy() for c in key]
        n = len(hist_small)

        if n > 0:
            starts = [0]
            for i in range(1, n):
                changed = False
                for arr in h_key_vals:
                    if arr[i] != arr[i - 1]:
                        changed = True
                        break
                if changed:
                    starts.append(i)
            starts.append(n)

            def _key_at(i: int) -> Tuple:
                return tuple(arr[i] for arr in h_key_vals)

            for a, b in zip(starts[:-1], starts[1:]):
                k = _key_at(a)
                hist_groups[k] = (h_date_int[a:b], h_t[a:b], h_delay[a:b])

    if df_small.empty:
        for w in windows:
            df[f"flightnum_od_depdelay_mean_last{w}"] = out_means[w]
        df[f"flightnum_od_support_count_last{wmax}d"] = out_supp
        df[f"flightnum_od_low_support_last{wmax}d"] = out_low
        return df

    d_dates = df_small["FlightDate"].to_numpy()
    d_date_int = d_dates.astype("datetime64[D]").astype(np.int32)
    d_t = df_small["_t"].to_numpy(dtype=np.int64, copy=False)
    d_delay = pd.to_numeric(df_small.get("DepDelayMinutes"), errors="coerce").to_numpy(dtype=np.float32, copy=False)
    d_idx = df_small.index.to_numpy()

    d_key_df = df_small[key]
    d_key_vals = [d_key_df[c].to_numpy() for c in key]
    ndf = len(df_small)

    starts = [0]
    for i in range(1, ndf):
        changed = False
        for arr in d_key_vals:
            if arr[i] != arr[i - 1]:
                changed = True
                break
        if changed:
            starts.append(i)
    starts.append(ndf)

    def _df_key_at(i: int) -> Tuple:
        return tuple(arr[i] for arr in d_key_vals)

    for a, b in zip(starts[:-1], starts[1:]):
        k = _df_key_at(a)

        df_dates_i = d_date_int[a:b]
        df_t_i = d_t[a:b]
        df_delay_i = d_delay[a:b]
        df_idx_i = d_idx[a:b]

        h = hist_groups.get(k)
        if h is None:
            h_dates_i = np.empty(0, dtype=np.int32)
            h_t_i = np.empty(0, dtype=np.int64)
            h_delay_i = np.empty(0, dtype=np.float32)
        else:
            h_dates_i, h_t_i, h_delay_i = h

        i_hist = 0
        i_df = 0
        n_hist = len(h_dates_i)
        n_df_g = len(df_dates_i)

        # Maintain:
        # - lastN flights deque for "recent flight history" (not requested as feature but useful if you later want it)
        # - window deques for each W in days: store (date_int, delay) for last W calendar days
        lastN = deque(maxlen=int(n_recent))
        win_deques = {w: deque() for w in windows}
        win_sums = {w: 0.0 for w in windows}
        win_counts = {w: 0 for w in windows}

        def _evict(win: int, current_date_int: int):
            cutoff = current_date_int - int(win)
            dq = win_deques[win]
            while dq and dq[0][0] < cutoff:
                d0, v0 = dq.popleft()
                win_sums[win] -= float(v0)
                win_counts[win] -= 1

        def _evict_all(current_date_int: int):
            for w in windows:
                _evict(w, current_date_int)

        def _push_to_all(date_int: int, delayv: float):
            for w in windows:
                # store the observation at its FlightDate
                win_deques[w].append((date_int, float(delayv)))
                win_sums[w] += float(delayv)
                win_counts[w] += 1

        while i_df < n_df_g:
            next_df_date = int(df_dates_i[i_df])
            next_df_t = int(df_t_i[i_df])

            # add all HISTORY events strictly before this df row (date, then time)
            while i_hist < n_hist:
                hd = int(h_dates_i[i_hist])
                ht = int(h_t_i[i_hist])
                if (hd < next_df_date) or (hd == next_df_date and ht < next_df_t):
                    delayv = float(h_delay_i[i_hist])
                    if np.isfinite(delayv):
                        lastN.append(delayv)
                        _push_to_all(hd, delayv)
                    i_hist += 1
                else:
                    break

            # evict old observations per window
            _evict_all(next_df_date)

            # compute features for this df row (based on prior obs only)
            for w in windows:
                if win_counts[w] > 0:
                    out_means[w].loc[df_idx_i[i_df]] = float(win_sums[w] / max(1, win_counts[w]))
                else:
                    out_means[w].loc[df_idx_i[i_df]] = np.nan

            # support + low-support based on longest window
            supp = int(win_counts[wmax])
            out_supp.loc[df_idx_i[i_df]] = np.int16(min(supp, np.iinfo(np.int16).max))
            out_low.loc[df_idx_i[i_df]] = np.int8(1 if supp <= int(low_support_leq) else 0)

            # now add THIS row’s delay into the windows for future rows (still no leakage)
            dv = float(df_delay_i[i_df]) if np.isfinite(df_delay_i[i_df]) else np.nan
            if np.isfinite(dv):
                lastN.append(dv)
                _push_to_all(next_df_date, dv)

            i_df += 1

    for w in windows:
        df[f"flightnum_od_depdelay_mean_last{w}"] = pd.to_numeric(out_means[w], errors="coerce")

    df[f"flightnum_od_support_count_last{wmax}d"] = pd.to_numeric(out_supp, errors="coerce").fillna(0).astype("Int16")
    df[f"flightnum_od_low_support_last{wmax}d"] = pd.to_numeric(out_low, errors="coerce").fillna(1).astype("Int8")
    return df


# ---------- NEW: carrier + carrier-origin rolling means over multiple windows ----------
def add_carrier_dep_delay_baselines_from_history_multi(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Adds:
      carrier_depdelay_mean_last{W}
      carrier_origin_depdelay_mean_last{W}
    where W is in windows_days.

    Uses daily carrier (and carrier-origin) mean dep delay, then rolling over W days, shifted by 1 day (no leakage).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str).str.upper().str.strip()
    df["Origin"] = df["Origin"].astype(str).str.upper().str.strip()

    hist = hist.copy()
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["Reporting_Airline"] = hist.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    carr = hist.dropna(subset=["Reporting_Airline", "FlightDate", "DepDelayMinutes"]).copy()
    carr_daily = (
        carr.groupby(["Reporting_Airline", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )

    for w in windows:
        carr_daily[f"carrier_depdelay_mean_last{w}"] = (
            carr_daily.groupby("Reporting_Airline")["DepDelayMinutes"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    keep_cols = ["Reporting_Airline", "FlightDate"] + [f"carrier_depdelay_mean_last{w}" for w in windows]
    df = df.merge(carr_daily[keep_cols], on=["Reporting_Airline", "FlightDate"], how="left")

    co = carr.dropna(subset=["Origin"]).copy()
    co_daily = (
        co.groupby(["Reporting_Airline", "Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "Origin", "FlightDate"])
    )

    for w in windows:
        co_daily[f"carrier_origin_depdelay_mean_last{w}"] = (
            co_daily.groupby(["Reporting_Airline", "Origin"])["DepDelayMinutes"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    keep_cols = ["Reporting_Airline", "Origin", "FlightDate"] + [f"carrier_origin_depdelay_mean_last{w}" for w in windows]
    df = df.merge(co_daily[keep_cols], on=["Reporting_Airline", "Origin", "FlightDate"], how="left")

    return df


# ---------- NEW: origin-airport rolling means over multiple windows ----------
def add_origin_dep_delay_baselines_from_history_multi(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Adds:
      origin_depdelay_mean_last{W}
    where W is in windows_days.

    Uses daily origin mean dep delay (all carriers), rolling over W days, shifted by 1 day (no leakage).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Origin"] = df["Origin"].astype(str).str.upper().str.strip()

    hist = hist.copy()
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    o = hist.dropna(subset=["Origin", "FlightDate", "DepDelayMinutes"]).copy()
    o_daily = (
        o.groupby(["Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Origin", "FlightDate"])
    )

    for w in windows:
        o_daily[f"origin_depdelay_mean_last{w}"] = (
            o_daily.groupby("Origin")["DepDelayMinutes"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    keep_cols = ["Origin", "FlightDate"] + [f"origin_depdelay_mean_last{w}" for w in windows]
    df = df.merge(o_daily[keep_cols], on=["Origin", "FlightDate"], how="left")
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

    df_cols_want = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "CRSDepTime",
        "dep_dt_local",
        "arr_dt_local",
        "DepDelayMinutes",
        "Tail_Number",
        "aircraft_type",
        "origin_temperature_2m_max",
        "origin_temperature_2m_min",
        "origin_precipitation_sum",
        "origin_windspeed_10m_max",
        "origin_weathercode",
        "origin_dep_temperature_2m",
        "origin_dep_precipitation",
        "origin_dep_windspeed_10m",
        "origin_dep_weathercode",
    ]

    print(f"[INFO] reading enriched -> {in_path}")
    df = _read_parquet_projected(in_path, df_cols_want)

    hist_path_cfg = cfg.get("history_output_path", "intermediate/history_pool_dep.parquet")
    hist_path = _abspath(_format_path_template(hist_path_cfg, target=target), base="data")
    if not hist_path.exists():
        raise FileNotFoundError(
            f"History pool parquet not found: {hist_path}\n"
            "Run prepare_dataset.py with history_output_path enabled, or set cfg.history_output_path correctly."
        )

    hist_cols_want = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "DepDelayMinutes",
        "CRSDepTime",
        "dep_dt_local",
        "dep_local_date",
    ]
    hist = _read_parquet_projected(hist_path, hist_cols_want)

    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce")
    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Dest"] = df.get("Dest", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce")
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["Dest"] = hist.get("Dest", "").astype(str).str.upper().str.strip()
    hist["Reporting_Airline"] = hist.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

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

    df = add_congestion_3h_features_from_history(df, hist)
    df = add_tail_and_flightnum_time_features(df)

    windows_days = feat_cfg.get("rolling_windows_days") or [7, 14, 21, 28]
    windows_days = [int(x) for x in windows_days]

    n_recent = int(feat_cfg.get("n_recent_flightnum_od", 20))
    low_support_leq = int(feat_cfg.get("low_support_leq", 2))

    df = add_flightnum_od_depdelay_means_from_history(
        df,
        hist,
        windows_days=windows_days,
        n_recent=n_recent,
        low_support_leq=low_support_leq,
    )

    df = add_carrier_dep_delay_baselines_from_history_multi(df, hist, windows_days=windows_days)
    df = add_origin_dep_delay_baselines_from_history_multi(df, hist, windows_days=windows_days)

    df = add_turn_time_hours(df)

    thresholds = (cfg.get("balance", {}) or {}).get("thresholds", [15, 30, 45, 60])
    thresholds = [int(x) for x in thresholds]
    delay_col = (cfg.get("balance", {}) or {}).get("delay_col", "DepDelayMinutes")
    df = add_dep_labels_for_thresholds(df, thresholds, delay_col=delay_col)

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
