# src/runtime/providers/schedules/aerodatabox.py
from __future__ import annotations
import os, requests
from datetime import datetime
from .base import ScheduleProvider, PlannedFlight

class AeroDataBoxProvider(ScheduleProvider):
    """
    Thin wrapper for AeroDataBox (has a free tier; check their docs).
    Expects env var AERODATABOX_API_KEY.
    """
    BASE = "https://aerodatabox.p.rapidapi.com/flights/number/{airline}{number}/{date}"

    def __init__(self, timeout=15):
        self.timeout = timeout
        self.api_key = os.getenv("AERODATABOX_API_KEY", "")

    def get_planned_flight(self, airline: str, flight_number: str, date_str: str) -> PlannedFlight:
        url = self.BASE.format(airline=airline, number=flight_number, date=date_str)
        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }
        r = requests.get(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Pick 1st matching leg (there may be multiple codeshares)
        if not data:
            raise RuntimeError("No schedule returned")

        leg = data[0]
        # Defensive parsing—provider payloads vary; adjust as needed
        origin = leg.get("departure", {}).get("airport", {}).get("iata", "")
        dest   = leg.get("arrival", {}).get("airport", {}).get("iata", "")
        dep    = leg.get("departure", {}).get("scheduledTimeLocal", "00:00")
        arr    = leg.get("arrival",   {}).get("scheduledTimeLocal", "00:00")

        def hhmm(s: str) -> str:
            # Expect "HH:MM"
            try:
                hh, mm = s.split(":")[:2]
                return f"{int(hh):02d}{int(mm):02d}"
            except Exception:
                return "0000"

        aircraft = (leg.get("aircraft", {}) or {}).get("model", None)
        tail     = (leg.get("aircraft", {}) or {}).get("reg", None)

        return PlannedFlight(
            airline=airline.upper(),
            flight_number=str(flight_number),
            flight_date=datetime.fromisoformat(f"{date_str}"),
            origin=origin.upper(),
            dest=dest.upper(),
            crs_dep_time=hhmm(dep),
            crs_arr_time=hhmm(arr),
            aircraft=aircraft,
            tail=tail,
            extra={"raw": leg},
        )
