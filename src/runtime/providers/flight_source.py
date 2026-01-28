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

# src/runtime/providers/flight_source.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .schedules.aerodatabox import AeroDataBoxProvider
from .schedules.base import PlannedFlight, ScheduleProvider
from .openskydata import OpenSkyData  # may not provide planned schedule

@dataclass
class FlightInfoSource:
    """
    Facade over multiple schedule sources.
    prefer_order: list like ["aerodatabox","opensky"].
    """
    prefer_order: Optional[List[str]] = None

    def __post_init__(self):
        order = (self.prefer_order or ["aerodatabox", "opensky"])
        self.providers: Dict[str, ScheduleProvider] = {}

        for key in order:
            key_l = key.lower().strip()
            if key_l == "aerodatabox":
                self.providers[key_l] = AeroDataBoxProvider()
            elif key_l == "opensky":
                # OpenSky doesn’t support planned schedules reliably; keep for other ops.
                self.providers[key_l] = OpenSkyData()
            # silently ignore unknown strings to keep it simple

        self.order = [k for k in order if k.lower() in self.providers]

    def get_planned_flight(
        self,
        airline: str,
        flight_number: str,
        date_str: str,
        *,
        origin: Optional[str] = None,
        dest: Optional[str] = None
    ) -> Optional[PlannedFlight]:
        """
        Try each provider in order. Pass through origin/dest hints when supported.
        """
        last_err: Optional[Exception] = None
        for key in self.order:
            prov = self.providers[key.lower()]
            try:
                # Prefer calling with origin/dest when the provider accepts them.
                try:
                    out = prov.get_planned_flight(airline, flight_number, date_str, origin=origin, dest=dest)  # type: ignore[arg-type]
                except TypeError:
                    out = prov.get_planned_flight(airline, flight_number, date_str)  # type: ignore[call-arg]

                if out is not None:
                    return out
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return None
