# src/flightright/core/time.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class AirportTime:
    """
    Convenience helpers for handling airport-local dates/times reliably.
    """
    tz_name: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)

    def now_local(self) -> datetime:
        return datetime.now(tz=self.tz)

    def to_local(self, dt_utc: datetime) -> datetime:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(self.tz)

    def to_utc(self, dt_local: datetime) -> datetime:
        if dt_local.tzinfo is None:
            dt_local = dt_local.replace(tzinfo=self.tz)
        return dt_local.astimezone(timezone.utc)

    def local_date(self, dt_utc: datetime) -> date:
        return self.to_local(dt_utc).date()


def parse_iso_or_space_datetime(s: str) -> datetime:
    """
    FlightLabs sometimes returns timestamps like:
      - '2026-01-20 14:17Z'
      - '2026-01-20 09:17-05:00'
      - ISO 8601 variants (with/without seconds)

    Returns an aware datetime when possible.
    """
    s = s.strip()
    # Common: "YYYY-MM-DD HH:MMZ"
    if s.endswith("Z") and " " in s:
        # "2026-01-20 14:17Z" -> "2026-01-20T14:17:00+00:00"
        base = s[:-1].replace(" ", "T")
        if len(base) == len("YYYY-MM-DDTHH:MM"):
            base = base + ":00"
        return datetime.fromisoformat(base + "+00:00")

    # Common: "YYYY-MM-DD HH:MM-05:00"
    if " " in s:
        base = s.replace(" ", "T")
        if len(base) == len("YYYY-MM-DDTHH:MM-05:00"):
            # no seconds
            base = base.replace("T", "T")  # noop
        return datetime.fromisoformat(base)

    # ISO
    return datetime.fromisoformat(s)