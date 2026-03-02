from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    rolling_snapshots_dir: Path
    rolling_sqlite_path: Path

    def snapshot_path(self, airport_iata: str, local_date_yyyy_mm_dd: str) -> Path:
        """
        Expected snapshot path:
          ../flightrightdata/data/rolling_snapshots/{IATA}/{YYYY-MM-DD}.json.gz
        """
        airport_iata = airport_iata.upper().strip()
        return self.rolling_snapshots_dir / airport_iata / f"{local_date_yyyy_mm_dd}.json.gz"