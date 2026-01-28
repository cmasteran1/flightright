# src/runtime/providers/history/opensky_history.py
from __future__ import annotations
import os, time, math, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

OPENSKY_DEP = "https://opensky-network.org/api/flights/departure"
OPENSKY_ARR = "https://opensky-network.org/api/flights/arrival"

def _now() -> int:
    return int(time.time())

def _read_map_csv(path: str, key: str, val: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df[key].astype(str).str.upper(), df[val].astype(str).str.upper()))

@dataclass
class OpenSkyHistory:
    iata_icao_airport_csv: str
    carrier_csv: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 20

    def __post_init__(self):
        self.iata2icao_airport = _read_map_csv(self.iata_icao_airport_csv, "IATA", "ICAO")
        # Optional IATA→ICAO airline (for callsign parsing, e.g., UAL -> UA)
        self.iata2icao_airline = {}
        self.icao2iata_airline = {}
        if self.carrier_csv and os.path.exists(self.carrier_csv):
            df = pd.read_csv(self.carrier_csv)
            if {"IATA_CODE","ICAO_CODE"}.issubset(df.columns):
                self.iata2icao_airline = dict(zip(df["IATA_CODE"].astype(str).str.upper(),
                                                  df["ICAO_CODE"].astype(str).str.upper()))
                self.icao2iata_airline = {v:k for k,v in self.iata2icao_airline.items()}

    def _airport_icao(self, iata: str) -> Optional[str]:
        return self.iata2icao_airport.get(str(iata).upper())

    def _auth(self):
        u = self.user or os.getenv("OPENSKY_USERNAME")
        p = self.password or os.getenv("OPENSKY_PASSWORD")
        return (u, p) if (u and p) else None

    def _get(self, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        r = requests.get(url, params=params, auth=self._auth(), timeout=self.timeout)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        j = r.json()
        return j if isinstance(j, list) else []

    def departures(self, iata: str, begin_unix: int, end_unix: int) -> pd.DataFrame:
        icao = self._airport_icao(iata)
        if not icao:
            return pd.DataFrame(columns=["icao24","callsign","firstSeen","estArrivalAirport","lastSeen"])
        rows = self._get(OPENSKY_DEP, {"airport": icao, "begin": begin_unix, "end": end_unix})
        return pd.DataFrame.from_records(rows)

    def arrivals(self, iata: str, begin_unix: int, end_unix: int) -> pd.DataFrame:
        icao = self._airport_icao(iata)
        if not icao:
            return pd.DataFrame(columns=["icao24","callsign","firstSeen","estDepartureAirport","lastSeen"])
        rows = self._get(OPENSKY_ARR, {"airport": icao, "begin": begin_unix, "end": end_unix})
        return pd.DataFrame.from_records(rows)

    def callsign_to_airline_iata(self, callsign: Optional[str]) -> Optional[str]:
        if not callsign:
            return None
        cs = str(callsign).strip().upper()
        # US majors use 3-letter ICAO in callsign (UAL1536, AAL123, DAL45, SWA12, ASA, FFT, etc.)
        prefix = "".join([ch for ch in cs if ch.isalpha()])
        if len(prefix) >= 3 and self.icao2iata_airline:
            return self.icao2iata_airline.get(prefix[:3])
        # fallback: sometimes 2-letter IATA present (rare)
        if len(prefix) >= 2:
            two = prefix[:2]
            if two in self.iata2icao_airline:
                return two
        return None

    @staticmethod
    def compute_block_minutes(df: pd.DataFrame) -> pd.Series:
        # lastSeen - firstSeen, seconds → minutes
        s = (pd.to_numeric(df.get("lastSeen"), errors="coerce") -
             pd.to_numeric(df.get("firstSeen"), errors="coerce"))
        return s.fillna(0).clip(lower=0) / 60.0
