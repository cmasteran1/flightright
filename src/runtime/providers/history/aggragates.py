# src/runtime/providers/history/aggregates.py
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

import requests
import pandas as pd

from .opensky_history import OpenSkyHistory  # your existing proxy feature source

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _to_utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@dataclass
class RecentAggregates:
    """
    Aggregates builder for recent-history features.

    REQUIRED (must succeed or caller should error out):
      - carrier_flights_prior_day (expected scheduled departures for airline at ORIGIN, prior day)

    OPTIONAL (best-effort; fall back to 0.0 on failure):
      - carrier_delay_7d_mean
      - od_delay_7d_mean
      - flightnum_delay_14d_mean
      - origin_delay_7d_mean
      - dest_delay_7d_mean
    """
    opensky: Optional[OpenSkyHistory] = None
    aerodatabox: Optional[Any] = None  # instance of AeroDataBoxProvider or None
    iata_icao_airport_csv: str = "data/meta/iata_icao_map.csv"

    def compute(
        self,
        airline_iata: str,
        origin_iata: str,
        dest_iata: str,
        day_start_unix_utc: int,
        exact_flightnum_mean: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {
            "carrier_flights_prior_day": 0.0,
            "carrier_delay_7d_mean":     0.0,
            "od_delay_7d_mean":          0.0,
            "flightnum_delay_14d_mean":  0.0,
            "origin_delay_7d_mean":      0.0,
            "dest_delay_7d_mean":        0.0,
        }

        # --- REQUIRED: expected carrier flights prior day (schedule-based, AeroDataBox) ---
        v = self.expected_carrier_departures_prior_day(airline_iata, origin_iata, day_start_unix_utc)
        # Treat <=0 as failure as well (your preference)
        if v is None or v <= 0:
            raise RuntimeError(
                f"carrier_flights_prior_day unavailable or zero "
                f"(airline={airline_iata}, origin={origin_iata})"
            )
        feats["carrier_flights_prior_day"] = float(v)

        # --- OPTIONALS (best-effort) ---
        try:
            if self.opensky is not None:
                feats["carrier_delay_7d_mean"] = float(self.opensky.mean_arr_delay_for_carrier_last_ndays(
                    airline_iata, day_start_unix_utc, n_days=7
                ) or 0.0)
                feats["origin_delay_7d_mean"] = float(self.opensky.mean_arr_delay_for_airport_last_ndays(
                    origin_iata, day_start_unix_utc, n_days=7
                ) or 0.0)
                feats["dest_delay_7d_mean"] = float(self.opensky.mean_arr_delay_for_airport_last_ndays(
                    dest_iata, day_start_unix_utc, n_days=7
                ) or 0.0)
                feats["od_delay_7d_mean"] = float(self.opensky.mean_arr_delay_for_od_last_ndays(
                    origin_iata, dest_iata, day_start_unix_utc, n_days=7
                ) or 0.0)
                if exact_flightnum_mean is None:
                    feats["flightnum_delay_14d_mean"] = float(self.opensky.mean_arr_delay_for_flightnum_last_ndays(
                        exact_flightnum_mean[0] if exact_flightnum_mean else airline_iata,
                        exact_flightnum_mean[1] if exact_flightnum_mean else "",
                        origin_iata, dest_iata, day_start_unix_utc, n_days=14
                    ) or 0.0)
        except Exception:
            pass

        # If AEDBX exact flightnum mean is requested, try to override
        try:
            if exact_flightnum_mean and self.aerodatabox is not None:
                aa, fn = exact_flightnum_mean
                m = self._aedbx_flightnum_mean_delay_lookback(aa, fn, day_start_unix_utc, days=14)
                if m is not None:
                    feats["flightnum_delay_14d_mean"] = float(m)
        except Exception:
            pass

        return feats

    def expected_carrier_departures_prior_day(
        self,
        airline_iata: str,
        origin_iata: str,
        day_start_unix_utc: int
    ) -> Optional[int]:
        """
        Count scheduled departures for `airline_iata` at `origin_iata` on the prior local day
        using AeroDataBox airport departures (UTC window prior to `day_start_unix_utc`).

        Returns int count (may be 0 if truly none) or None on hard failure.
        """
        api_key = os.getenv("AERODATABOX_API_KEY", "").strip()
        if not api_key:
            # Treat missing key as hard failure for the REQUIRED feature
            return None

        # Map IATA -> ICAO for airport
        try:
            df_map = _load_csv(self.iata_icao_airport_csv)
            map_lower = {c.lower(): c for c in df_map.columns}
            col_iata = map_lower.get("iata"); col_icao = map_lower.get("icao")
            if not (col_iata and col_icao):
                return None
            row = df_map[df_map[col_iata].astype(str).str.upper() == origin_iata.upper()].head(1)
            if row.empty:
                return None
            origin_icao = str(row.iloc[0][col_icao]).upper()
        except Exception:
            return None

        prior_start = day_start_unix_utc - 24*3600
        prior_end   = day_start_unix_utc - 1

        base = "https://aerodatabox.p.rapidapi.com/flights/airports/icao/{icao}/{f}/{t}"
        url = base.format(icao=origin_icao, f=_to_utc_iso(prior_start), t=_to_utc_iso(prior_end))

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
            "Accept": "application/json",
        }
        params = {
            "withLeg": "true",
            "withCancelled": "true",
            "withCodeshared": "true",
        }

        try:
            r = requests.get(url, headers=headers, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None

        legs = []
        if isinstance(data, dict):
            if "departures" in data:
                legs = data.get("departures") or []
            elif "flights" in data:
                legs = data.get("flights") or []
            else:
                for v in data.values():
                    if isinstance(v, list):
                        legs.extend(v)
        elif isinstance(data, list):
            legs = data

        cnt = 0
        for g in legs:
            al = (g.get("airline") or {}) if isinstance(g, dict) else {}
            iata = (al.get("iata") or "").upper()
            if iata == airline_iata.upper():
                cnt += 1
        return cnt

    def _aedbx_flightnum_mean_delay_lookback(
        self, airline_iata: str, flight_number: str, day_start_unix_utc: int, days: int = 14
    ) -> Optional[float]:
        api_key = os.getenv("AERODATABOX_API_KEY", "").strip()
        if not api_key:
            return None

        from_dt = datetime.fromtimestamp(day_start_unix_utc, tz=timezone.utc).date()
        deltas = []
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
            "Accept": "application/json",
        }
        base = "https://aerodatabox.p.rapidapi.com/flights/number/{airline}{num}/{date}"

        for i in range(1, days + 1):
            d = (from_dt - timedelta(days=i)).isoformat()
            url = base.format(airline=airline_iata, num=flight_number, date=d)
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if not r.ok:
                    continue
                arr = r.json()
                if not isinstance(arr, list):
                    continue
                for leg in arr:
                    dep = (leg.get("departure") or {})
                    arrv = (leg.get("arrival") or {})
                    sch = ((arrv.get("scheduledTime") or {}).get("utc")
                           or arrv.get("scheduledTimeUtc"))
                    pred = ((arrv.get("predictedTime") or {}).get("utc")
                            or arrv.get("predictedTimeUtc")
                            or arrv.get("actualTimeUtc"))
                    if sch and pred:
                        try:
                            t_s = datetime.fromisoformat(str(sch).replace("Z","+00:00"))
                            t_p = datetime.fromisoformat(str(pred).replace("Z","+00:00"))
                            delta_min = (t_p - t_s).total_seconds() / 60.0
                            deltas.append(delta_min)
                        except Exception:
                            pass
            except Exception:
                continue

        if not deltas:
            return None
        return float(sum(deltas) / len(deltas))
