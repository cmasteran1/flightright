#!/usr/bin/env python3
# src/providers/openskydata.py
"""
OpenSky provider (recent/day-of operations, NOT planned schedules).

OpenSky REST (documented behavior):
- arrivals:    GET https://opensky-network.org/api/flights/arrival?airport=KJFK&begin=...&end=...
- departures:  GET https://opensky-network.org/api/flights/departure?airport=KJFK&begin=...&end=...
- aircraft:    GET https://opensky-network.org/api/flights/aircraft?icao24=...&begin=...&end=...

Notes:
- OpenSky requires ICAO *airport* codes for arrivals/departures (e.g., KJFK), not IATA.
- Time range is UNIX epoch seconds. Anonymous users are severely limited; authenticated accounts can query a ~7 day window (varies).
- OpenSky does not provide *future* schedules. We expose `get_planned_flight` but it returns None here.
"""
# src/runtime/providers/openskydata.py
from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple

import time
import requests
import pandas as pd

# Optional: if you have this base, import it; otherwise keep the method names consistent.
try:
    from .schedules.base import PlannedFlight  # noqa: F401  (not used here but keeps type hints consistent)
except Exception:
    PlannedFlight = Any  # fallback type


@dataclass
class OpenSkyData:
    """
    Minimal OpenSky wrapper used by FlightInfoSource.

    Notes:
    - OpenSky does NOT provide reliable *planned* schedules. So
      `get_planned_flight` returns None (FlightInfoSource then tries other providers).
    - We keep departures() here because your earlier test scripts used it.
    - Optionally loads an IATA->ICAO map from CSV for other ops.
    """
    username: Optional[str] = None
    password: Optional[str] = None
    iata_icao_map_path: Optional[str] = None
    timeout: int = 20

    def __post_init__(self):
        # env fallbacks
        if self.username is None:
            self.username = os.getenv("OPENSKY_USERNAME") or None
        if self.password is None:
            self.password = os.getenv("OPENSKY_PASSWORD") or None
        if self.iata_icao_map_path is None:
            self.iata_icao_map_path = os.getenv("IATA_ICAO_MAP") or None

        self._iata_to_icao: Dict[str, str] = {}
        if self.iata_icao_map_path and os.path.exists(self.iata_icao_map_path):
            try:
                df = pd.read_csv(self.iata_icao_map_path)
                # accept either columns: IATA, ICAO  OR  iata, icao
                cols = {c.lower(): c for c in df.columns}
                iata_col = cols.get("iata")
                icao_col = cols.get("icao")
                if iata_col and icao_col:
                    m = (
                        df[[iata_col, icao_col]]
                        .dropna()
                        .astype(str)
                        .apply(lambda s: (s[iata_col].upper().strip(), s[icao_col].upper().strip()), axis=1)
                    )
                    self._iata_to_icao = {k: v for k, v in m if k and v}
            except Exception:
                # best-effort map load; don’t crash
                pass

    # ------------- Schedules interface -------------
    # We surface this method so FlightInfoSource can call it.
    # OpenSky doesn’t give planned schedules, so return None.
    def get_planned_flight(
        self,
        airline: str,
        flight_number: str,
        date_str: str,
        *,
        origin: Optional[str] = None,
        dest: Optional[str] = None,
    ) -> Optional[PlannedFlight]:
        return None

    # ------------- Useful helper for recent traffic -------------
    # A convenience method some of your scripts use.
    # Returns a DataFrame of recent departures for a given IATA within [begin,end] (epoch seconds).
    def get_departures(self, iata: str, begin: int, end: int) -> pd.DataFrame:
        icao = self._iata_to_icao.get(iata.upper())
        if not icao:
            raise RuntimeError(f"[OpenSky] No ICAO mapping for IATA={iata} (set IATA_ICAO_MAP)")

        url = f"https://opensky-network.org/api/flights/departure?airport={icao}&begin={begin}&end={end}"
        auth = (self.username, self.password) if self.username and self.password else None

        r = requests.get(url, auth=auth, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() if r.content else []
        if not isinstance(data, list):
            data = []

        # Normalize to DataFrame
        cols_keep = [
            "icao24", "firstSeen", "lastSeen", "estDepartureAirport", "estArrivalAirport",
            "callsign"
        ]
        rows = []
        for row in data:
            rows.append({k: row.get(k) for k in cols_keep})
        df = pd.DataFrame(rows)

        # Add human times
        for c in ("firstSeen", "lastSeen"):
            if c in df.columns:
                df[c + "_utc"] = pd.to_datetime(df[c], unit="s", utc=True)

        return df
