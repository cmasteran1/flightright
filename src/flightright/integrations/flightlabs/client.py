# src/flightright/integrations/flightlabs/client.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from flightright.core.http import HttpClient


@dataclass(frozen=True)
class FlightLabsClient:
    """
    Minimal wrapper for the two endpoints you need now:
      - Future flights prediction (scheduled flights by airport+date)
      - Flight data by date (history for flight number, max 14d range)
    """
    access_key: str
    http: HttpClient = HttpClient()
    base_url: str = "https://goflightlabs.com"

    def future_flights(
        self,
        iata_code: str,
        day: date,
        *,
        type_: str = "departure",
    ) -> Dict[str, Any]:
        # Docs: /advanced-future-flights?type=departure|arrival&iataCode=...&date=YYYY-MM-DD :contentReference[oaicite:2]{index=2}
        url = f"{self.base_url}/advanced-future-flights"
        params = {
            "access_key": self.access_key,
            "type": type_,
            "iataCode": iata_code,
            "date": day.isoformat(),
        }
        return self.http.get_json(url, params=params)

    def flight_data_by_date_number(
        self,
        flight_number: str,
        date_from: date,
        date_to: date,
    ) -> Dict[str, Any]:
        # Docs: /v2/flight?search_by=number&flight_number=...&date_from=...&date_to=... :contentReference[oaicite:3]{index=3}
        url = f"{self.base_url}/v2/flight"
        params = {
            "access_key": self.access_key,
            "search_by": "number",
            "flight_number": flight_number,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }
        return self.http.get_json(url, params=params)


def normalize_flight_number(reporting_airline_iata: str, flight_number: str | int) -> str:
    """
    FlightLabs examples show formats like "AA 1821" in response, but request uses e.g. "LH811".
    We'll standardize request as "AA1821".
    """
    al = str(reporting_airline_iata).strip().upper()
    fn = str(flight_number).strip()
    fn = fn.replace(" ", "")
    # if user passes "AA 1821" already, strip spaces
    if fn.upper().startswith(al):
        return al + fn[len(al):]
    return al + fn