# src/runtime/providers/schedules/base.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, Dict

@dataclass
class PlannedFlight:
    airline: str           # "AA"
    flight_number: str     # "1234"
    flight_date: datetime  # naive date (local to origin later via tz db)
    origin: str            # IATA
    dest: str              # IATA
    crs_dep_time: str      # "HHMM" local planned
    crs_arr_time: str      # "HHMM" local planned
    aircraft: Optional[str] = None
    tail: Optional[str] = None
    extra: Optional[Dict] = None

class ScheduleProvider(Protocol):
    def get_planned_flight(self, airline: str, flight_number: str, date_str: str) -> PlannedFlight:
        """Return planned flight for airline+flight_number on yyyy-mm-dd (carrier local calendar)."""
        ...
