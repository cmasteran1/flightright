from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RollingResult:
    mean_delay_min: Optional[float]
    support_count: int  # number of flights with a known delay in the window
    flight_count: int   # total flights observed in the window (known + unknown)


def upsert_daily_rollup(
    conn: sqlite3.Connection,
    date_yyyy_mm_dd: str,
    origin: str,
    carrier: str,
    dep_delay_sum: float,
    dep_delay_count: int,
    dep_flight_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO daily_rollup(date, origin, carrier, dep_delay_sum, dep_delay_count, dep_flight_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, origin, carrier) DO UPDATE SET
          dep_delay_sum=excluded.dep_delay_sum,
          dep_delay_count=excluded.dep_delay_count,
          dep_flight_count=excluded.dep_flight_count;
        """,
        (date_yyyy_mm_dd, origin, carrier, float(dep_delay_sum), int(dep_delay_count), int(dep_flight_count)),
    )


def _sum_window(
    conn: sqlite3.Connection,
    date_end_exclusive: str,
    window_days: int,
    origin: Optional[str],
    carrier: Optional[str],
) -> Tuple[float, int, int]:
    """
    Sum dep_delay_sum, dep_delay_count, dep_flight_count over the previous `window_days`
    ending at date_end_exclusive (exclusive). SQLite date math uses TEXT dates.
    """
    where = []
    params = []

    where.append("date < ?")
    params.append(date_end_exclusive)

    where.append("date >= date(?, '-' || ? || ' day')")
    params.append(date_end_exclusive)
    params.append(window_days)

    if origin is not None:
        where.append("origin = ?")
        params.append(origin)

    if carrier is not None:
        where.append("carrier = ?")
        params.append(carrier)

    sql = f"""
      SELECT
        COALESCE(SUM(dep_delay_sum), 0.0),
        COALESCE(SUM(dep_delay_count), 0),
        COALESCE(SUM(dep_flight_count), 0)
      FROM daily_rollup
      WHERE {" AND ".join(where)}
    """
    row = conn.execute(sql, params).fetchone()
    assert row is not None
    return float(row[0]), int(row[1]), int(row[2])


def get_rolling_depdelay_mean(
    conn: sqlite3.Connection,
    date_end_exclusive: str,
    window_days: int,
    *,
    origin: Optional[str] = None,
    carrier: Optional[str] = None,
    min_support: int = 1,
) -> RollingResult:
    """
    Rolling mean delay (minutes) in [date_end_exclusive - window_days, date_end_exclusive),
    filtered by origin and/or carrier.

    We also return:
      support_count = number of flights WITH a known delay
      flight_count   = total flights observed (including unknown delay)
    """
    dep_delay_sum, dep_delay_count, dep_flight_count = _sum_window(
        conn=conn,
        date_end_exclusive=date_end_exclusive,
        window_days=window_days,
        origin=origin,
        carrier=carrier,
    )
    if dep_delay_count < min_support:
        return RollingResult(mean_delay_min=None, support_count=dep_delay_count, flight_count=dep_flight_count)

    return RollingResult(
        mean_delay_min=dep_delay_sum / dep_delay_count if dep_delay_count > 0 else None,
        support_count=dep_delay_count,
        flight_count=dep_flight_count,
    )