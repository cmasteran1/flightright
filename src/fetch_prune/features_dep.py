# features_dep.py
import sys
import json
from pathlib import Path
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd


# ----------------- paths (assume run from REPO_ROOT) -----------------
def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (Path.cwd() / pp).resolve()

def _resolve_output_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (Path.cwd() / pp).resolve()

def _format_path_template(p: str, target: str) -> str:
    return p.format(target=target) if "{target}" in p else p


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
    if "dep_local_date" in df.columns and df["dep_local_date"].notna().any():
        base = pd.to_datetime(df["dep_local_date"])
    else:
        base = pd.to_datetime(df["FlightDate"])

    years = sorted(pd.unique(base.dt.year.dropna()))
    tg = {y: thanksgiving_day(int(y)) for y in years}
    xmas = {y: date(int(y), 12, 25) for y in years}

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
        if is_near(d, tg[y], thanksgiving_window):
            f = 1
        if is_near(d, xmas[y], christmas_window):
            f = 1
        flags.append(f)

    df["is_holiday"] = pd.Series(flags, index=df.index).astype("Int8")
    return df


# ---------- congestion counts (origin only) ----------
def add_origin_congestion_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["origin_dep_count_trail_60m"] = np.nan
    df["origin_arr_count_trail_60m"] = np.nan

    if "dep_dt_local" in df.columns:
        mask = df["dep_dt_local"].notna()
        gcols = ["Origin", "dep_local_date"]
        for (_, _), idx in df[mask].sort_values(gcols + ["dep_dt_local"]).groupby(gcols).groups.items():
            mins = df.loc[idx, "dep_dt_local"].apply(lambda x: x.hour * 60 + x.minute).to_numpy()
            cum = np.arange(1, len(mins) + 1)
            lower = np.searchsorted(mins, mins - 60, side="left")
            win = cum - np.where(lower > 0, cum[lower - 1], 0)
            df.loc[idx, "origin_dep_count_trail_60m"] = win

    if "arr_dt_local" in df.columns:
        mask = df["arr_dt_local"].notna()
        tmp = df.loc[mask, ["Dest", "arr_local_date", "arr_dt_local"]].copy()
        tmp = tmp.sort_values(["Dest", "arr_local_date", "arr_dt_local"])
        tmp["arr_minutes"] = tmp["arr_dt_local"].apply(lambda x: x.hour * 60 + x.minute).to_numpy()
        tmp["arr_count_trail_60m"] = np.nan

        for (_, _), idx in tmp.groupby(["Dest", "arr_local_date"]).groups.items():
            mins = tmp.loc[idx, "arr_minutes"].to_numpy()
            cum = np.arange(1, len(mins) + 1)
            lower = np.searchsorted(mins, mins - 60, side="left")
            win = cum - np.where(lower > 0, cum[lower - 1], 0)
            tmp.loc[idx, "arr_count_trail_60m"] = win

        tmp = tmp[["Dest", "arr_local_date", "arr_dt_local", "arr_count_trail_60m"]]
        df = df.merge(
            tmp.rename(
                columns={
                    "Dest": "Origin",
                    "arr_local_date": "dep_local_date",
                    "arr_dt_local": "dep_dt_local",
                    "arr_count_trail_60m": "origin_arr_count_trail_60m",
                }
            ),
            on=["Origin", "dep_local_date", "dep_dt_local"],
            how="left",
        )

    return df


# ---------- gates join ----------
def add_gates_and_congestion_ratio(df: pd.DataFrame, gates_csv: str) -> pd.DataFrame:
    df = df.copy()
    p = Path(gates_csv) if gates_csv else None
    if not p or not p.exists():
        df["origin_gates"] = np.nan
        df["origin_congestion_ratio"] = np.nan
        return df

    gates = pd.read_csv(p, dtype={"IATA": str, "gates": float})
    gates["IATA"] = gates["IATA"].str.upper()
    gates = gates.rename(columns={"IATA": "Origin", "gates": "origin_gates"})
    df["Origin"] = df["Origin"].astype(str).str.upper()

    df = df.merge(gates[["Origin", "origin_gates"]], on="Origin", how="left")
    num = (
        pd.to_numeric(df.get("origin_dep_count_trail_60m"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("origin_arr_count_trail_60m"), errors="coerce").fillna(0)
    )
    den = pd.to_numeric(df["origin_gates"], errors="coerce")
    df["origin_congestion_ratio"] = np.where(den > 0, num / den, np.nan)
    return df


# ---------- history-based features from pool ----------
def add_flightnum_od_recent_mean_and_support(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    n_recent: int = 20,
    support_days: int = 14,
    low_support_threshold: int = 2,
) -> pd.DataFrame:
    """
    Uses HISTORY POOL (unbalanced) to compute for each row in df:
      - flightnum_od_depdelay_mean_lastN  (rolling mean of previous N flights for same carrier+flightnum+OD)
      - flightnum_od_support_count_lastNd (count of prior flights in last support_days days, excluding current day)
      - flightnum_od_low_support          (1 if count <= low_support_threshold else 0)

    Subtlety:
      - We operate on FlightDate (not dep_dt_local) per your note that a flight number+OD is effectively once per day.
      - No leakage: "prior" excludes same-day by shifting / using strictly earlier dates.
    """
    out = df.copy()

    # Normalize keys + types
    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        out[c] = out[c].astype(str).str.upper()
    out["Flight_Number_Reporting_Airline"] = pd.to_numeric(out.get("Flight_Number_Reporting_Airline"), errors="coerce")
    out["FlightDate"] = pd.to_datetime(out["FlightDate"], errors="coerce")

    h = hist.copy()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in h.columns:
            h[c] = h[c].astype(str).str.upper()
    h["Flight_Number_Reporting_Airline"] = pd.to_numeric(h.get("Flight_Number_Reporting_Airline"), errors="coerce")
    h["FlightDate"] = pd.to_datetime(h["FlightDate"], errors="coerce")
    h["DepDelayMinutes"] = pd.to_numeric(h.get("DepDelayMinutes"), errors="coerce")

    # Daily aggregation (one record per key per day)
    # If there are occasional duplicates, average them.
    hd = (
        h.dropna(subset=key + ["FlightDate"])
         .groupby(key + ["FlightDate"], as_index=False)
         .agg(depdelay_mean_day=("DepDelayMinutes", "mean"),
              depdelay_count_day=("DepDelayMinutes", "size"))
         .sort_values(key + ["FlightDate"])
    )

    # Rolling mean of previous N flights (across days)
    def _roll_prevN(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=n_recent, min_periods=3).mean()

    hd["flightnum_od_depdelay_mean_lastN"] = (
        hd.groupby(key, sort=False)["depdelay_mean_day"]
          .apply(_roll_prevN)
          .reset_index(level=key, drop=True)
    )

    # Rolling support count over last support_days (count of flights, excluding current day)
    # Here we roll over DAILY counts with a time-based window.
    # Implementation: per group, set index=FlightDate and do shifted rolling sum.
    parts = []
    for _, g in hd.groupby(key, sort=False):
        gg = g.sort_values("FlightDate").copy()
        gg = gg.set_index("FlightDate")
        # sum of counts in prior window (exclude current day)
        prior_counts = gg["depdelay_count_day"].shift(1)
        support = prior_counts.rolling(f"{support_days}D", min_periods=1).sum()
        gg["flightnum_od_support_count_lastNd"] = support.values
        gg = gg.reset_index()
        parts.append(gg)

    hd2 = pd.concat(parts, ignore_index=True) if parts else hd.copy()
    hd2["flightnum_od_low_support"] = (
        (pd.to_numeric(hd2["flightnum_od_support_count_lastNd"], errors="coerce").fillna(0) <= low_support_threshold)
        .astype("Int8")
    )

    # Merge onto df by key + FlightDate
    out = out.merge(
        hd2[key + ["FlightDate", "flightnum_od_depdelay_mean_lastN", "flightnum_od_support_count_lastNd", "flightnum_od_low_support"]],
        on=key + ["FlightDate"],
        how="left",
    )

    return out


def add_carrier_dep_delay_baselines_from_history(df: pd.DataFrame, hist: pd.DataFrame, n_days: int = 14) -> pd.DataFrame:
    """
    From history pool:
      - carrier_depdelay_mean_lastNdays         (by Reporting_Airline, daily mean, shift(1), rolling window)
      - carrier_origin_depdelay_mean_lastNdays  (by Reporting_Airline+Origin, daily mean, shift(1), rolling window)
    """
    out = df.copy()
    out["FlightDate"] = pd.to_datetime(out["FlightDate"], errors="coerce")
    out["Reporting_Airline"] = out["Reporting_Airline"].astype(str)
    out["Origin"] = out["Origin"].astype(str).str.upper()

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h["FlightDate"], errors="coerce")
    h["DepDelayMinutes"] = pd.to_numeric(h.get("DepDelayMinutes"), errors="coerce")
    h["Reporting_Airline"] = h["Reporting_Airline"].astype(str)
    h["Origin"] = h["Origin"].astype(str).str.upper()

    # carrier nationwide
    carr = h.dropna(subset=["Reporting_Airline", "FlightDate"]).copy()
    carr_daily = (
        carr.groupby(["Reporting_Airline", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )
    carr_daily["carrier_depdelay_mean_lastNdays"] = (
        carr_daily.groupby("Reporting_Airline")["DepDelayMinutes"]
        .apply(lambda s: s.shift(1).rolling(window=n_days, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    out = out.merge(
        carr_daily[["Reporting_Airline", "FlightDate", "carrier_depdelay_mean_lastNdays"]],
        on=["Reporting_Airline", "FlightDate"],
        how="left",
    )

    # carrier at origin
    co = h.dropna(subset=["Reporting_Airline", "Origin", "FlightDate"]).copy()
    co_daily = (
        co.groupby(["Reporting_Airline", "Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "Origin", "FlightDate"])
    )
    co_daily["carrier_origin_depdelay_mean_lastNdays"] = (
        co_daily.groupby(["Reporting_Airline", "Origin"])["DepDelayMinutes"]
        .apply(lambda s: s.shift(1).rolling(window=n_days, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    out = out.merge(
        co_daily[["Reporting_Airline", "Origin", "FlightDate", "carrier_origin_depdelay_mean_lastNdays"]],
        on=["Reporting_Airline", "Origin", "FlightDate"],
        how="left",
    )

    return out


# ---------- turn-time proxy (previous leg of same flight number) ----------
def add_turn_time_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")

    df["dep_dt_utc"] = pd.to_datetime(df["dep_dt_local"], errors="coerce", utc=True)
    df["arr_dt_utc"] = pd.to_datetime(df["arr_dt_local"], errors="coerce", utc=True)

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline"]
    df = df.sort_values(key + ["dep_dt_utc"])

    df["prev_arr_dt_utc"] = df.groupby(key, sort=False)["arr_dt_utc"].shift(1)
    df["prev_dest"] = df.groupby(key, sort=False)["Dest"].shift(1)

    ok = df["prev_dest"].astype(str).str.upper() == df["Origin"].astype(str).str.upper()
    delta_hours = (df["dep_dt_utc"] - df["prev_arr_dt_utc"]).dt.total_seconds() / 3600.0
    df["turn_time_hours"] = np.where(ok, delta_hours, np.nan)
    return df


# ---------- weather unit conversions ----------
def add_weather_kelvin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "origin_temperature_2m_max" in df.columns:
        df["origin_temp_max_K"] = pd.to_numeric(df["origin_temperature_2m_max"], errors="coerce") + 273.15
    if "origin_temperature_2m_min" in df.columns:
        df["origin_temp_min_K"] = pd.to_numeric(df["origin_temperature_2m_min"], errors="coerce") + 273.15
    if "origin_dep_temperature_2m" in df.columns:
        df["origin_dep_temp_K"] = pd.to_numeric(df["origin_dep_temperature_2m"], errors="coerce") + 273.15

    if "origin_precipitation_sum" in df.columns:
        df["origin_daily_precip_sum_mm"] = pd.to_numeric(df["origin_precipitation_sum"], errors="coerce")
    if "origin_windspeed_10m_max" in df.columns:
        df["origin_daily_windspeed_max_kmh"] = pd.to_numeric(df["origin_windspeed_10m_max"], errors="coerce")

    if "origin_dep_precipitation" in df.columns:
        df["origin_dep_precip_mm"] = pd.to_numeric(df["origin_dep_precipitation"], errors="coerce")
    if "origin_dep_windspeed_10m" in df.columns:
        df["origin_dep_windspeed_kmh"] = pd.to_numeric(df["origin_dep_windspeed_10m"], errors="coerce")
    if "origin_dep_windgusts_10m" in df.columns:
        df["origin_dep_windgust_kmh"] = pd.to_numeric(df["origin_dep_windgusts_10m"], errors="coerce")

    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/fetch_prune/features_dep.py path/to/config.json")
        sys.exit(1)

    cfg_path = _resolve_path(sys.argv[1])
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    target = (cfg.get("target") or "dep").lower()
    if target != "dep":
        raise ValueError("features_dep.py expects cfg.target == 'dep'")

    # Read intermediate
    inter_cfg = cfg.get("output_intermediate_path", "data/intermediate/flights_enriched_{target}.parquet")
    inter_cfg = _format_path_template(inter_cfg, target)
    in_path = _resolve_path(inter_cfg)
    if not in_path.exists():
        raise FileNotFoundError(f"Intermediate parquet not found: {in_path}")
    df = pd.read_parquet(in_path)

    # Optional: apply airport restriction again (either/both)
    if cfg.get("airports"):
        airports = set(a.upper() for a in cfg["airports"])
        df["Origin"] = df["Origin"].astype(str).str.upper()
        df["Dest"] = df["Dest"].astype(str).str.upper()

        mode = (cfg.get("airport_filter_mode") or "either").lower()
        if mode == "both":
            df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports)]
        else:
            df = df[df["Origin"].isin(airports) | df["Dest"].isin(airports)]

    # Load history pool
    hist_path_cfg = cfg.get("history_output_path", "data/intermediate/history_pool_dep.parquet")
    hist_path = _resolve_path(hist_path_cfg)
    if not hist_path.exists():
        raise FileNotFoundError(
            f"History pool parquet not found: {hist_path}. "
            "Run prepare_dataset.py first (it writes history_output_path)."
        )
    hist = pd.read_parquet(hist_path)

    # Core categoricals
    df["Origin"] = df["Origin"].astype(str).str.upper()
    df["Dest"] = df["Dest"].astype(str).str.upper()
    df["od_pair"] = df["Origin"] + "_" + df["Dest"]

    # dep day-of-week + scheduled dep hour
    if "dep_dt_local" in df.columns:
        dts = pd.to_datetime(df["dep_dt_local"], errors="coerce")
        df["dep_dow"] = dts.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = dts.dt.hour.astype("Int8")
    else:
        fd = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["dep_dow"] = fd.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").floordiv(100).astype("Int8")

    # Holiday
    feat_cfg = cfg.get("features_dep") or {}
    hw = feat_cfg.get("holiday_windows") or {}
    df = add_holiday_flag(
        df,
        thanksgiving_window=int(hw.get("thanksgiving", 2)),
        christmas_window=int(hw.get("christmas", 3)),
    )

    # Aircraft type placeholder (until you add mapping)
    df["aircraft_type"] = "Unknown"

    # Weather codes categorical
    if "origin_weathercode" in df.columns:
        df["origin_daily_weathercode"] = pd.to_numeric(df["origin_weathercode"], errors="coerce").astype("Int64").astype("object")
    if "origin_dep_weathercode" in df.columns:
        df["origin_dep_hour_weathercode"] = pd.to_numeric(df["origin_dep_weathercode"], errors="coerce").astype("Int64").astype("object")

    # Weather numeric conversions
    df = add_weather_kelvin(df)

    # Congestion + optional gates ratio
    df = add_origin_congestion_counts(df)
    gates_csv = cfg.get("airport_gates_csv")
    if gates_csv:
        df = add_gates_and_congestion_ratio(df, str(_resolve_path(gates_csv)))
    else:
        df["origin_gates"] = np.nan
        df["origin_congestion_ratio"] = np.nan

    # History-based: recent flightnum-OD dep delay mean + support + low-support flag
    n_recent = int(feat_cfg.get("n_recent_flightnum_od", 20))
    support_days = int(feat_cfg.get("support_days_flightnum_od", 14))
    low_thr = int(feat_cfg.get("low_support_threshold", 2))
    df = add_flightnum_od_recent_mean_and_support(
        df, hist, n_recent=n_recent, support_days=support_days, low_support_threshold=low_thr
    )

    # History-based: carrier baselines
    n_days = int(feat_cfg.get("n_days_carrier_network", 14))
    df = add_carrier_dep_delay_baselines_from_history(df, hist, n_days=n_days)

    # Turn time proxy (still computed from the enriched intermediate)
    df = add_turn_time_hours(df)

    # Label (should exist from prepare stage)
    if "y_dep15" not in df.columns:
        df["DepDelayMinutes"] = pd.to_numeric(df.get("DepDelayMinutes"), errors="coerce")
        df["y_dep15"] = (df["DepDelayMinutes"] >= int(cfg.get("dep_delay_threshold_min", 15))).astype("Int8")

    # Final feature selection
    feature_cols = [
        # categoricals
        "Origin",
        "Dest",
        "od_pair",
        "dep_dow",
        "sched_dep_hour",
        "is_holiday",
        "aircraft_type",
        "Reporting_Airline",
        # weather categorical
        "origin_daily_weathercode",
        "origin_dep_hour_weathercode",
        # weather numeric
        "origin_temp_max_K",
        "origin_temp_min_K",
        "origin_daily_precip_sum_mm",
        "origin_dep_temp_K",
        "origin_dep_precip_mm",
        "origin_dep_windspeed_kmh",
        "origin_daily_windspeed_max_kmh",
        # numeric operational
        "origin_congestion_ratio",
        # history features
        "flightnum_od_depdelay_mean_lastN",
        "flightnum_od_support_count_lastNd",
        "flightnum_od_low_support",
        "carrier_depdelay_mean_lastNdays",
        "carrier_origin_depdelay_mean_lastNdays",
        # turn-time
        "turn_time_hours",
    ]

    keep = []
    for c in feature_cols + ["y_dep15", "DepDelayMinutes", "FlightDate", "dep_dt_local", "arr_dt_local"]:
        if c in df.columns and c not in keep:
            keep.append(c)

    out = df[keep].copy()

    out_cfg = cfg.get("output_training_path", "data/processed/train_ready_{target}.parquet")
    out_cfg = _format_path_template(out_cfg, target)
    out_path = _resolve_output_path(out_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[OK] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()

