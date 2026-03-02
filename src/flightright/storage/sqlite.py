from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator


def connect_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    We store daily aggregates so rolling windows are fast.

    Tables:
      daily_rollup
        date TEXT (YYYY-MM-DD, airport-local day)
        origin TEXT (IATA)
        carrier TEXT (IATA)
        dep_delay_sum REAL
        dep_delay_count INTEGER
        dep_flight_count INTEGER  (all departures, even if delay unknown)
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_rollup (
          date TEXT NOT NULL,
          origin TEXT NOT NULL,
          carrier TEXT NOT NULL,
          dep_delay_sum REAL NOT NULL,
          dep_delay_count INTEGER NOT NULL,
          dep_flight_count INTEGER NOT NULL,
          PRIMARY KEY (date, origin, carrier)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_rollup_date ON daily_rollup(date);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_rollup_origin ON daily_rollup(origin);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_rollup_carrier ON daily_rollup(carrier);")
    conn.commit()