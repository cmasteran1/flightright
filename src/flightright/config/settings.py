from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """
    Centralized config. Keep this simple so it ports cleanly to a VM/container.

    Repo layout assumption:
      flightright/          (repo root)
      flightright/src/...   (python package)
      flightrightdata/      (sibling directory)
    """
    repo_root: Path
    data_root: Path

    airports_csv: Path
    tracked_airports_path: Path

    rolling_snapshots_dir: Path
    rolling_sqlite_path: Path

    @staticmethod
    def default() -> "Settings":
        repo_root = Path(__file__).resolve().parents[3]  # .../flightright/
        data_root = (repo_root / ".." / "flightrightdata" / "data").resolve()

        airports_csv = data_root / "meta" / "airports.csv"
        tracked_airports_path = data_root / "meta" / "airport_rankings" / "tracked_airports.txt"

        rolling_snapshots_dir = data_root / "rolling_snapshots"
        rolling_sqlite_path = data_root / "rolling" / "rolling.sqlite"

        return Settings(
            repo_root=repo_root,
            data_root=data_root,
            airports_csv=airports_csv,
            tracked_airports_path=tracked_airports_path,
            rolling_snapshots_dir=rolling_snapshots_dir,
            rolling_sqlite_path=rolling_sqlite_path,
        )