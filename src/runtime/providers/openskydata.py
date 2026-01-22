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

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import os
import time
import json
import math
import logging
import requests
import pandas as pd

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _LOG.addHandler(_h)
_LOG.setLevel(logging.INFO)

OPENSKY_BASE = "https://opensky-network.org/api"


def _read_iata_icao_map(csv_path: Path) -> Dict[str, str]:
    """
    Load a simple IATA->ICAO mapping CSV with columns ['IATA','ICAO'].
    Returns dict mapping IATA (upper) -> ICAO (upper). Missing ICAO becomes '' (not included).
    """
    m: Dict[str, str] = {}
    df = pd.read_csv(csv_path)
    if not {"IATA", "ICAO"}.issubset(df.columns):
        raise RuntimeError("iata_icao_map CSV must have columns: IATA, ICAO")
    df["IATA"] = df["IATA"].astype(str).str.upper().str.strip()
    df["ICAO"] = df["ICAO"].astype(str).str.upper().str.strip()
    for _, r in df.iterrows():
        iata, icao = r["IATA"], r["ICAO"]
        if iata and isinstance(iata, str) and isinstance(icao, str) and icao:
            m[iata] = icao
    return m


def _callsign_from_iata(iata_airline: str, flight_number: str) -> str:
    """
    Build a callsign string used commonly in OpenSky results, e.g., 'AAL1234'.
    This is NOT guaranteed for every airline (some callsigns differ from IATA).
    For Big-4 US carriers, this is usually workable:
      AA -> AAL, DL -> DAL, UA -> UAL, WN -> SWA, AS -> ASA, B6 -> JBU, NK -> NKS, F9 -> FFT, etc.
    We provide a minimal map and let callers override via env if needed.
    """
    iata = (iata_airline or "").upper().strip()
    num  = "".join(ch for ch in str(flight_number) if ch.isalnum())
    # Minimal default mapping
    default = {
        "AA": "AAL", "DL": "DAL", "UA": "UAL", "WN": "SWA",
        "AS": "ASA", "B6": "JBU", "NK": "NKS", "F9": "FFT",
        "HA": "HAL", "G4": "AAY"
    }
    # Optional JSON override via env (e.g., {"AA":"AAL","DL":"DAL"})
    override_json = os.environ.get("OPENSKY_IATA_TO_ICAO_AIRLINE_JSON", "").strip()
    table = default
    if override_json:
        try:
            table = {**default, **json.loads(override_json)}
        except Exception:
            _LOG.warning("Failed to parse OPENSKY_IATA_TO_ICAO_AIRLINE_JSON; using defaults.")
    prefix = table.get(iata, iata)  # fallback: use IATA itself
    return f"{prefix}{num}"


@dataclass
class OpenSkyProvider:
    """
    Lightweight client for OpenSky day-of/historical flight info.

    Auth:
      - Basic auth via env:
        OPENSKY_USERNAME, OPENSKY_PASSWORD

    Airport mapping:
      - Use iata_icao_map.csv (IATA, ICAO) to convert for /flights/(arrival|departure).
      - Provide path via constructor or env IATA_ICAO_MAP (optional).
    """
    username: Optional[str] = None
    password: Optional[str] = None
    iata_icao_map_path: Optional[Path] = None

    def __post_init__(self):
        if self.username is None:
            self.username = os.environ.get("OPENSKY_USERNAME", None)
        if self.password is None:
            self.password = os.environ.get("OPENSKY_PASSWORD", None)

        path = self.iata_icao_map_path or os.environ.get("IATA_ICAO_MAP", None)
        self._iata2icao: Dict[str, str] = {}
        if path:
            try:
                self._iata2icao = _read_iata_icao_map(Path(path))
                _LOG.info(f"[OpenSky] Loaded IATA→ICAO map entries: {len(self._iata2icao)}")
            except Exception as e:
                _LOG.warning(f"[OpenSky] Could not read IATA→ICAO map ({path}): {e}")

        self._auth = (self.username, self.password) if (self.username and self.password) else None

    # --------------------------
    # Public: arrivals / departures (historical/day-of)
    # --------------------------
    def get_departures(
        self,
        airport_iata: str,
        begin_epoch: int,
        end_epoch: int,
        limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get recent departures from an airport (IATA input; converted to ICAO).
        """
        icao = self._airport_icao(airport_iata)
        if not icao:
            raise ValueError(f"No ICAO found for IATA '{airport_iata}'. Provide mapping via IATA_ICAO_MAP.")
        url = f"{OPENSKY_BASE}/flights/departure"
        params = {"airport": icao, "begin": int(begin_epoch), "end": int(end_epoch)}
        rows = self._get_json(url, params)
        df = self._rows_to_df(rows)
        if limit_rows:
            df = df.head(int(limit_rows))
        return df

    def get_arrivals(
        self,
        airport_iata: str,
        begin_epoch: int,
        end_epoch: int,
        limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get recent arrivals to an airport (IATA input; converted to ICAO).
        """
        icao = self._airport_icao(airport_iata)
        if not icao:
            raise ValueError(f"No ICAO found for IATA '{airport_iata}'. Provide mapping via IATA_ICAO_MAP.")
        url = f"{OPENSKY_BASE}/flights/arrival"
        params = {"airport": icao, "begin": int(begin_epoch), "end": int(end_epoch)}
        rows = self._get_json(url, params)
        df = self._rows_to_df(rows)
        if limit_rows:
            df = df.head(int(limit_rows))
        return df

    def get_by_callsign(
        self,
        callsign: str,
        begin_epoch: int,
        end_epoch: int
    ) -> pd.DataFrame:
        """
        Try to find flights matching a callsign (e.g. 'AAL1234') by scanning arrivals + departures
        windows across many airports is expensive. Instead, use /flights/aircraft if you know the icao24,
        OR query departures/arrivals for candidate airports.

        Here we provide a convenience method that:
        1) Queries a broad set of departures (if you pass airports via env OPENSKY_SCAN_DEPARTURE_IATA_LIST=JFK,DFW,...)
        2) Filters by callsign
        Return may be empty if outside OpenSky window limits or no match.
        """
        scan = os.environ.get("OPENSKY_SCAN_DEPARTURE_IATA_LIST", "")
        airports = [a.strip().upper() for a in scan.split(",") if a.strip()]
        if not airports:
            _LOG.warning("OPENSKY_SCAN_DEPARTURE_IATA_LIST not set; get_by_callsign() will return empty without airports to scan.")
            return pd.DataFrame()

        frames = []
        for iata in airports:
            try:
                f = self.get_departures(iata, begin_epoch, end_epoch)
                frames.append(f)
            except Exception as e:
                _LOG.warning(f"OpenSky departures fetch failed for {iata}: {e}")

        if not frames:
            return pd.DataFrame()
        all_dep = pd.concat(frames, ignore_index=True)
        if "callsign" in all_dep.columns:
            mask = all_dep["callsign"].astype(str).str.fullmatch(str(callsign).strip(), case=False, na=False)
            return all_dep.loc[mask].copy()
        return pd.DataFrame()

    # --------------------------
    # Planned flight placeholder
    # --------------------------
    def get_planned_flight(
        self,
        airline_iata: str,
        flight_number: str,
        flight_date_local: str
    ) -> Optional[Dict[str, Any]]:
        """
        OpenSky does not provide *future planned* schedules.
        We return None here so the selector can fall back to a planned-capable provider (e.g., AeroDataBox).
        """
        _LOG.info("OpenSky: planned schedule lookup is not supported (returning None).")
        return None

    # --------------------------
    # Internals
    # --------------------------
    def _airport_icao(self, iata: str) -> Optional[str]:
        if not iata:
            return None
        iata = iata.strip().upper()
        return self._iata2icao.get(iata)

    def _get_json(self, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        r = requests.get(url, params=params, auth=self._auth, timeout=30)
        if r.status_code == 429:
            raise RuntimeError("OpenSky rate-limited (HTTP 429). Reduce queries or use smaller windows.")
        if r.status_code >= 400:
            raise RuntimeError(f"OpenSky error {r.status_code}: {r.text}")
        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"OpenSky response not JSON: {e}")
        if not isinstance(data, list):
            # Some errors arrive as dict
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"OpenSky error: {data.get('error')}")
            raise RuntimeError("OpenSky returned unexpected payload (not a list).")
        return data

    @staticmethod
    def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=[
                "icao24","firstSeen","estDepartureAirport","lastSeen","estArrivalAirport",
                "callsign","estDepartureAirportHorizDistance","estDepartureAirportVertDistance",
                "estArrivalAirportHorizDistance","estArrivalAirportVertDistance","departureAirportCandidatesCount",
                "arrivalAirportCandidatesCount"
            ])
        df = pd.DataFrame(rows)
        # Basic types
        for c in ("firstSeen","lastSeen"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Optional: Convert to datetime for convenience
        for c in ("firstSeen","lastSeen"):
            if c in df.columns:
                df[c+"_dt"] = pd.to_datetime(df[c], unit="s", errors="coerce", utc=True)
        # Normalize callsign spacing
        if "callsign" in df.columns:
            df["callsign"] = df["callsign"].astype(str).str.strip()
        return df
