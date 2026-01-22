# src/runtime/providers/schedules/aerodatabox.py
from __future__ import annotations
import os
import requests
from datetime import datetime
from .base import ScheduleProvider, PlannedFlight

class AeroDataBoxProvider(ScheduleProvider):
    """
    Thin wrapper for AeroDataBox (RapidAPI).
    Needs env var: AERODATABOX_API_KEY
    """
    BASE = "https://aerodatabox.p.rapidapi.com/flights/number/{airline}{number}/{date}"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.api_key = os.getenv("AERODATABOX_API_KEY", "").strip()

    @staticmethod
    def _hhmm_from_local_field(local_str: str) -> str:
        """
        local_str like '2026-01-22 07:46-07:00' → '0746'
        """
        try:
            # split date/time and offset
            parts = local_str.strip().split()
            if len(parts) < 2:
                return "0000"
            time_part = parts[1]  # '07:46-07:00' OR '07:46'
            hh_mm = time_part.split("-")[0]  # '07:46'
            hh, mm = hh_mm.split(":")[:2]
            return f"{int(hh):02d}{int(mm):02d}"
        except Exception:
            return "0000"

    def get_planned_flight(self, airline: str, flight_number: str, date_str: str) -> PlannedFlight:
        if not self.api_key:
            raise RuntimeError("AERODATABOX_API_KEY not set")
        url = self.BASE.format(airline=airline, number=flight_number, date=date_str)
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
        }
        r = requests.get(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise RuntimeError("AeroDataBox returned no schedule for that flight/date")

        leg = data[0]  # choose first leg; refine later if you need codeshares
        dep = (leg.get("departure") or {})
        arr = (leg.get("arrival") or {})

        # parse IATA
        origin = ((dep.get("airport") or {}).get("iata") or "").upper()
        dest   = ((arr.get("airport") or {}).get("iata") or "").upper()

        # parse local time strings
        dep_local = ((dep.get("scheduledTime") or {}).get("local") or "")  # e.g. '2026-01-22 07:46-07:00'
        arr_local = ((arr.get("scheduledTime") or {}).get("local") or "")
        crs_dep_time = self._hhmm_from_local_field(dep_local)
        crs_arr_time = self._hhmm_from_local_field(arr_local)

        aircraft = (leg.get("aircraft") or {}).get("model")
        tail     = (leg.get("aircraft") or {}).get("reg")

        return PlannedFlight(
            airline=airline.upper(),
            flight_number=str(flight_number),
            flight_date=datetime.fromisoformat(f"{date_str}"),
            origin=origin,
            dest=dest,
            crs_dep_time=crs_dep_time,
            crs_arr_time=crs_arr_time,
            aircraft=aircraft,
            tail=tail,
            extra={"raw": leg},
        )
