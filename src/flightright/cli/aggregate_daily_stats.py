from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from flightright.config.settings import Settings
from flightright.storage.paths import DataPaths
from flightright.storage.snapshots import read_snapshot_json_gz, iter_snapshot_records
from flightright.storage.sqlite import connect_sqlite, ensure_schema
from flightright.storage.rollups import upsert_daily_rollup


@dataclass
class Agg:
    dep_delay_sum: float = 0.0
    dep_delay_count: int = 0
    dep_flight_count: int = 0


def _iter_airport_date_files(rolling_snapshots_dir: Path, date_yyyy_mm_dd: str) -> Iterable[Tuple[str, Path]]:
    """
    Finds:
      rolling_snapshots/{IATA}/{YYYY-MM-DD}.json.gz
    Returns (IATA, path).
    """
    if not rolling_snapshots_dir.exists():
        return

    for airport_dir in rolling_snapshots_dir.iterdir():
        if not airport_dir.is_dir():
            continue
        airport_iata = airport_dir.name.upper()
        p = airport_dir / f"{date_yyyy_mm_dd}.json.gz"
        if p.exists():
            yield airport_iata, p


def aggregate_one_day(settings: Settings, date_yyyy_mm_dd: str) -> None:
    paths = DataPaths(
        rolling_snapshots_dir=settings.rolling_snapshots_dir,
        rolling_sqlite_path=settings.rolling_sqlite_path,
    )

    conn = connect_sqlite(paths.rolling_sqlite_path)
    try:
        ensure_schema(conn)

        # key: (origin, carrier) -> Agg
        by_origin_carrier: Dict[Tuple[str, str], Agg] = defaultdict(Agg)

        found_any = False
        for origin, file_path in _iter_airport_date_files(paths.rolling_snapshots_dir, date_yyyy_mm_dd):
            found_any = True
            payload = read_snapshot_json_gz(file_path)

            for rec in iter_snapshot_records(payload, default_origin=origin):
                key = (rec.origin, rec.carrier)
                a = by_origin_carrier[key]
                a.dep_flight_count += 1
                if rec.dep_delay_min is not None:
                    a.dep_delay_sum += float(rec.dep_delay_min)
                    a.dep_delay_count += 1

        if not found_any:
            print(f"[aggregate] No snapshot files found for date={date_yyyy_mm_dd} under {paths.rolling_snapshots_dir}")
            return

        # Write one row per (date, origin, carrier)
        for (origin, carrier), a in by_origin_carrier.items():
            upsert_daily_rollup(
                conn=conn,
                date_yyyy_mm_dd=date_yyyy_mm_dd,
                origin=origin,
                carrier=carrier,
                dep_delay_sum=a.dep_delay_sum,
                dep_delay_count=a.dep_delay_count,
                dep_flight_count=a.dep_flight_count,
            )

        conn.commit()
        print(f"[aggregate] Wrote rollups for date={date_yyyy_mm_dd}: {len(by_origin_carrier)} origin-carrier rows")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate airport-day snapshots into rolling sqlite rollups.")
    parser.add_argument("--date", required=True, help="Airport-local date in YYYY-MM-DD (the snapshot day).")
    args = parser.parse_args()

    settings = Settings.default()
    aggregate_one_day(settings, args.date)


if __name__ == "__main__":
    main()