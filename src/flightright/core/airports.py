# src/flightright/core/airports.py

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class Airport:
    iata: str
    icao: Optional[str]
    name: str
    city: str
    state: str
    country: str
    latitude: float
    longitude: float
    timezone: str


def load_airports_csv(path: str | Path) -> Dict[str, Airport]:
    """
    Loads airports.csv with header:
    IATA,ICAO,AirportName,City,State,Country,Latitude,Longitude,Timezone
    """
    path = Path(path)
    out: Dict[str, Airport] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"IATA", "AirportName", "City", "State", "Country", "Latitude", "Longitude", "Timezone"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"airports.csv missing columns: {sorted(missing)} (found: {reader.fieldnames})")

        for row in reader:
            iata = (row.get("IATA") or "").strip().upper()
            if not iata:
                continue
            icao = (row.get("ICAO") or "").strip().upper() or None
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except Exception:
                continue

            out[iata] = Airport(
                iata=iata,
                icao=icao,
                name=(row.get("AirportName") or "").strip(),
                city=(row.get("City") or "").strip(),
                state=(row.get("State") or "").strip(),
                country=(row.get("Country") or "").strip(),
                latitude=lat,
                longitude=lon,
                timezone=(row.get("Timezone") or "").strip(),
            )
    return out