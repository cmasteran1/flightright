# src/runtime/feature_bridge.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

class FeatureBridge:
    """
    Maps schedule + forecast → model features.
    Flexible: if a requested feature is missing, fill robustly.
    """
    def __init__(self, airports_csv: Path, medians_json: Path | None = None):
        self.airports = pd.read_csv(airports_csv)  # columns: IATA, ICAO, lat, lon, tz
        self.airports["IATA"] = self.airports["IATA"].str.upper()
        self._med = {}
        if medians_json and Path(medians_json).exists():
            self._med = json.loads(Path(medians_json).read_text())

    def airport_row(self, iata: str) -> dict:
        r = self.airports[self.airports["IATA"] == iata.upper()]
        if r.empty:
            raise KeyError(f"Unknown airport: {iata}")
        return r.iloc[0].to_dict()

    def hhmm_to_hour(self, hhmm: str) -> int:
        try:
            s = str(int(hhmm)).zfill(4)
            return int(s[:2])
        except Exception:
            return 0

    def _fill(self, series: pd.Series, name: str) -> pd.Series:
        if series.dtype.kind in "biufc":
            if series.isna().any():
                val = self._med.get(name)
                if val is None:
                    val = float(np.nanmedian(series.values.astype(float)))
                series = series.fillna(val)
        else:
            series = series.astype("object").fillna("Unknown").astype(str)
        return series

    def build_row(self,
                  sched: dict,
                  origin_hourly: pd.DataFrame,
                  dest_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a single-row DataFrame with as many of your model features as possible.
        """
        origin = sched["origin"]
        dest   = sched["dest"]
        dep_hr = self.hhmm_to_hour(sched["crs_dep_time"])
        arr_hr = self.hhmm_to_hour(sched["crs_arr_time"])

        row = {
            "Origin": origin,
            "Dest": dest,
            "Reporting_Airline": sched["airline"],
            "OD_pair": f"{origin}_{dest}",

            # Local-hour slices (origin @dep, dest @arr). Here we pick the
            # exact hour record; you can switch to small windows (e.g., max prev 2h).
            "origin_dep_temperature_2m": self._pick(origin_hourly, dep_hr, "origin_dep_temperature_2m"),
            "origin_dep_windspeed_10m": self._pick(origin_hourly, dep_hr, "origin_dep_windspeed_10m"),
            "origin_dep_windgusts_10m": self._pick(origin_hourly, dep_hr, "origin_dep_windgusts_10m"),
            "origin_dep_precipitation": self._pick(origin_hourly, dep_hr, "origin_dep_precipitation"),
            "origin_dep_weathercode":   self._pick(origin_hourly, dep_hr, "origin_dep_weathercode"),

            "dest_arr_temperature_2m": self._pick(dest_hourly, arr_hr, "origin_dep_temperature_2m"),
            "dest_arr_windspeed_10m": self._pick(dest_hourly, arr_hr, "origin_dep_windspeed_10m"),
            "dest_arr_windgusts_10m": self._pick(dest_hourly, arr_hr, "origin_dep_windgusts_10m"),
            "dest_arr_precipitation": self._pick(dest_hourly, arr_hr, "origin_dep_precipitation"),
            "dest_arr_weathercode":   self._pick(dest_hourly, arr_hr, "origin_dep_weathercode"),
        }

        df = pd.DataFrame([row])

        # Fill robustly
        for c in df.columns:
            df[c] = self._fill(df[c], c)
        return df

    def _pick(self, hourly: pd.DataFrame, hour: int, col: str):
        if hourly is None or hourly.empty:
            return np.nan
        # Expect an integer "ts hour" column
        hcol = hourly["ts"].dt.hour if "ts" in hourly.columns else None
        if hcol is None:
            return np.nan
        sel = hourly[hcol == hour]
        if sel.empty or col not in hourly.columns:
            return np.nan
        return float(sel.iloc[0][col])

from runtime.providers.history.opensky_history import OpenSkyHistory
from runtime.providers.history.aggregates import RecentAggregates

def build_recent_history_features(
        airline_iata: str,
        origin_iata: str,
        dest_iata: str,
        day_start_unix_utc: int,
        iata_icao_map_csv: str = "data/meta/iata_icao_map.csv",
        carrier_csv: str = "src/utils/carrier.csv",
        use_aerodatabox_exact_flight: bool = False,
        flight_number: Optional[str] = None,
        aerodatabox_provider: Optional[object] = None,
) -> dict:
    osh = OpenSkyHistory(iata_icao_airport_csv=iata_icao_map_csv,
                         carrier_csv=carrier_csv)
    agg = RecentAggregates(opensky=osh,
                           aerodatabox=aerodatabox_provider if use_aerodatabox_exact_flight else None)
    exact = (airline_iata, flight_number) if (use_aerodatabox_exact_flight and flight_number) else None
    return agg.compute(airline_iata, origin_iata, dest_iata, day_start_unix_utc, exact_flightnum_mean=exact)
