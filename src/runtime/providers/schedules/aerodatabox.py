# src/runtime/providers/schedules/aerodatabox.py
from __future__ import annotations
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

from .base import ScheduleProvider, PlannedFlight

class AeroDataBoxProvider(ScheduleProvider):
    """
    Thin wrapper for AeroDataBox (RapidAPI).
    Expects env var AERODATABOX_API_KEY.

    get_planned_flight(..., origin=IATA?, dest=IATA?) will filter the returned
    list of legs and return ONLY the one matching the requested OD (if given).
    """
    BASE = "https://aerodatabox.p.rapidapi.com/flights/number/{airline}{number}/{date}"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.api_key = os.getenv("AERODATABOX_API_KEY", "")

    def _headers(self) -> Dict[str, str]:
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
            "Accept": "application/json",
        }

    @staticmethod
    def _hhmm_from_local(obj: Dict[str, Any]) -> str:
        """
        obj like:
           {"utc":"2026-02-10 06:20Z","local":"2026-02-10 06:20+00:00"}
        Return "HHMM" (local).
        """
        s = ((obj or {}).get("local") or "").strip()
        # "YYYY-MM-DD HH:MM±hh:mm"
        try:
            # Split date+time
            time_part = s.split()[1]
            hh, mm = time_part.split(":")[0:2]
            return f"{int(hh):02d}{int(mm):02d}"
        except Exception:
            return "0000"

    @staticmethod
    def _pick_leg(data: List[Dict[str, Any]], origin: Optional[str], dest: Optional[str]) -> Optional[Dict[str, Any]]:
        if not data:
            return None
        if origin is None and dest is None:
            # Backward-compat: return first
            return data[0]
        o = (origin or "").upper()
        d = (dest or "").upper()

        # First try exact IATA match
        for leg in data:
            dep_iata = (((leg.get("departure") or {}).get("airport") or {}).get("iata") or "").upper()
            arr_iata = (((leg.get("arrival")  or {}).get("airport") or {}).get("iata") or "").upper()
            if dep_iata == o and arr_iata == d:
                return leg

        # If not found, try ICAO fallback (rare)
        for leg in data:
            dep_icao = (((leg.get("departure") or {}).get("airport") or {}).get("icao") or "").upper()
            arr_icao = (((leg.get("arrival")  or {}).get("airport") or {}).get("icao") or "").upper()
            if dep_icao and arr_icao and (o or d):
                # You said to rely on IATA by default, so only use this when needed
                return leg  # last-ditch; better to return something than nothing

        return None

    def get_planned_flight(
        self,
        airline: str,
        flight_number: str,
        date_str: str,
        *,
        origin: Optional[str] = None,
        dest: Optional[str] = None
    ) -> Optional[PlannedFlight]:
        url = self.BASE.format(airline=airline, number=flight_number, date=date_str)
        r = requests.get(url, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None

        leg = self._pick_leg(data, origin, dest)
        if not leg:
            return None

        dep = (leg.get("departure") or {})
        arr = (leg.get("arrival") or {})
        dep_ap = (dep.get("airport") or {})
        arr_ap = (arr.get("airport") or {})

        origin_iata = (dep_ap.get("iata") or "").upper()
        dest_iata   = (arr_ap.get("iata") or "").upper()

        dep_local = self._hhmm_from_local(dep.get("scheduledTime") or {})
        arr_local = self._hhmm_from_local(arr.get("scheduledTime") or {})

        aircraft = ((leg.get("aircraft") or {}).get("model") or None)
        tail     = ((leg.get("aircraft") or {}).get("reg") or None)

        return PlannedFlight(
            airline=airline.upper(),
            flight_number=str(flight_number),
            flight_date=datetime.fromisoformat(f"{date_str}"),
            origin=origin_iata,
            dest=dest_iata,
            crs_dep_time=dep_local,
            crs_arr_time=arr_local,
            aircraft=aircraft,
            tail=tail,
            extra={"raw": leg},
        )
