#!/usr/bin/env python3
# src/runtime/providers/flight_source.py
"""
Simple provider selector for flight info.

Providers:
  - AeroDataBox (planned schedules): runtime.providers.schedules.aerodatabox.AeroDataBoxProvider
  - OpenSky (day-of / historical windows): runtime.providers.openskydata.OpenSkyProvider

Usage:
    # Ensure PYTHONPATH points at your repo's src/ directory:
    #   export PYTHONPATH="$PWD/src:$PYTHONPATH"
    #
    # Also set your AeroDataBox RapidAPI key if you plan to use it:
    #   export AERODATABOX_API_KEY="...your key..."

    from runtime.providers.flight_source import FlightInfoSource

    src = FlightInfoSource(
        prefer_order=["aerodatabox", "opensky"],
        iata_icao_map="data/meta/iata_icao_map.csv",
    )
    info = src.get_planned_flight("AA", "1234", "2026-02-05")
    print(info)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import providers with exact names/paths in your tree
from .schedules.aerodatabox import AeroDataBoxProvider  # type: ignore
from .openskydata import OpenSkyProvider  # type: ignore


@dataclass
class FlightInfoSource:
    """
    prefer_order: which providers to try (first success wins).
                  Supported names: "aerodatabox", "opensky"
    iata_icao_map: optional path to IATA→ICAO CSV (used by OpenSky provider)
    """
    prefer_order: List[str] = field(default_factory=lambda: ["aerodatabox", "opensky"])
    iata_icao_map: Optional[str] = None

    def _provider(self, name: str):
        name = name.lower().strip()
        if name == "aerodatabox":
            return AeroDataBoxProvider()
        if name == "opensky":
            kwargs = {}
            if self.iata_icao_map:
                kwargs["iata_icao_map_path"] = Path(self.iata_icao_map)
            return OpenSkyProvider(**kwargs)
        raise ValueError(f"Unknown provider: {name}")

    # --- planned (future) flight schedule ---
    def get_planned_flight(
        self,
        airline_iata: str,
        flight_number: str,
        flight_date_local: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a dict-like PlannedFlight (provider-dependent) or None.
        Tries providers in prefer_order until one returns a result.
        """
        last_err = None
        for name in self.prefer_order:
            prov = self._provider(name)
            if not hasattr(prov, "get_planned_flight"):
                continue
            try:
                out = prov.get_planned_flight(airline_iata, flight_number, flight_date_local)
                if out:
                    if isinstance(out, dict):
                        out["_provider"] = name
                    return out
            except NotImplementedError:
                continue
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return None

    # --- day-of/historical airport windows (OpenSky typically) ---
    def get_departures(self, airport_iata: str, begin_epoch: int, end_epoch: int):
        last_err = None
        for name in self.prefer_order:
            prov = self._provider(name)
            if hasattr(prov, "get_departures"):
                try:
                    return prov.get_departures(airport_iata, begin_epoch, end_epoch)
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("No provider could serve get_departures")

    def get_arrivals(self, airport_iata: str, begin_epoch: int, end_epoch: int):
        last_err = None
        for name in self.prefer_order:
            prov = self._provider(name)
            if hasattr(prov, "get_arrivals"):
                try:
                    return prov.get_arrivals(airport_iata, begin_epoch, end_epoch)
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("No provider could serve get_arrivals")

    # Optional helper (if a provider implements it)
    def get_by_callsign(self, callsign: str, begin_epoch: int, end_epoch: int):
        last_err = None
        for name in self.prefer_order:
            prov = self._provider(name)
            if hasattr(prov, "get_by_callsign"):
                try:
                    return prov.get_by_callsign(callsign, begin_epoch, end_epoch)
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("No provider could serve get_by_callsign")
