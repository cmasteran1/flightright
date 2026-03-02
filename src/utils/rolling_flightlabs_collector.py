#!/usr/bin/env python3
"""
rolling_collector.py

Long-running background process that:
- tracks a fixed set of airports
- runs a data pull at ~23:50 LOCAL TIME for each airport
- writes daily JSON snapshots for later rolling-stat aggregation

Current behavior:
- Uses FlightLabs "Flight Schedules API" endpoint:
    https://app.goflightlabs.com/advanced-flights-schedules
  Example (docs):
    https://app.goflightlabs.com/advanced-flights-schedules?access_key=...&iataCode=JFK&type=departure
- Writes raw responses to disk

Debugging upgrades:
- on failures prints URL, status code, content-type
- on JSON parse failure prints a short response-body preview
- startup endpoint test (optional/ON by default) to verify the API works before entering the long loop
- redacts access_key from printed URLs/errors
- enforces endpoint limit <= 1000 (clamps and warns)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests
from zoneinfo import ZoneInfo


# ----------------------------- config -----------------------------

DEFAULT_RUN_MINUTE = 50  # 23:50 local time
DEFAULT_RUN_HOUR = 23

REQUEST_TIMEOUT = 30  # seconds
UTC = ZoneInfo("UTC")

FLIGHTLABS_SCHEDULES_URL = "https://app.goflightlabs.com/advanced-flights-schedules"
FLIGHTLABS_MAX_LIMIT = 1000


# ----------------------------- helpers -----------------------------

def load_airports(path: Path) -> List[str]:
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_timezones(path: Path) -> Dict[str, str]:
    """
    Reads your airport metadata CSV with header:
      IATA,ICAO,AirportName,City,State,Country,Latitude,Longitude,Timezone

    Returns: dict[IATA] = Timezone (IANA string, e.g. America/New_York)
    """
    tz: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"IATA", "Timezone"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Timezone CSV missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            iata = (row.get("IATA") or "").strip().upper()
            zone = (row.get("Timezone") or "").strip()
            if not iata or not zone:
                continue
            tz[iata] = zone

    return tz


def next_run_utc(iata: str, tz_name: str, hour: int, minute: int) -> datetime:
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)

    candidate = now_local.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )

    if candidate <= now_local:
        candidate += timedelta(days=1)

    return candidate.astimezone(UTC)


def _preview(text: str, n: int = 300) -> str:
    t = text.strip()
    if len(t) <= n:
        return t
    return t[:n] + " ...[truncated]"


def _redact_access_key_in_url(url: str) -> str:
    """
    Prevent leaking the API key in logs. Works for typical querystrings:
      ...?access_key=XYZ&...
    """
    if "access_key=" not in url:
        return url
    # crude but reliable enough for logging
    parts = url.split("access_key=")
    prefix = parts[0] + "access_key="
    rest = parts[1]
    if "&" in rest:
        _, tail = rest.split("&", 1)
        return prefix + "[REDACTED]&" + tail
    return prefix + "[REDACTED]"


def _clamp_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    return min(int(limit), FLIGHTLABS_MAX_LIMIT)


# ----------------------------- FlightLabs calls -----------------------------

def fetch_flight_schedules(
    *,
    airport_iata: str,
    access_key: str,
    flight_type: str,
    airline_iata: Optional[str],
    flight_iata: Optional[str],
    limit: Optional[int],
    skip: Optional[int],
) -> dict:
    """
    Call FlightLabs Flight Schedules API.

    Docs show:
      https://app.goflightlabs.com/advanced-flights-schedules?access_key=...&iataCode=JFK&type=departure

    Documented parameters include:
      access_key (required), iataCode (required), type (optional; defaults to arrival),
      airline_iata, flight_iata, limit, skip, etc.

    NOTE: This endpoint enforces limit <= 1000 (observed from API error response).
    """
    params: Dict[str, Any] = {
        "access_key": access_key,
        "iataCode": airport_iata,
        "type": flight_type,  # "departure" or "arrival"
    }

    if airline_iata:
        params["airline_iata"] = airline_iata.strip().upper()
    if flight_iata:
        params["flight_iata"] = flight_iata.strip().upper()

    limit2 = _clamp_limit(limit)
    if limit2 is not None:
        params["limit"] = int(limit2)
    if skip is not None:
        params["skip"] = int(skip)

    r = requests.get(FLIGHTLABS_SCHEDULES_URL, params=params, timeout=REQUEST_TIMEOUT)
    content_type = (r.headers.get("Content-Type") or "").lower()
    safe_url = _redact_access_key_in_url(r.url)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        body = _preview(r.text)
        raise RuntimeError(
            f"HTTP {r.status_code} for {safe_url} | content-type={content_type} | body_preview={body}"
        ) from e

    try:
        return r.json()
    except ValueError as e:
        body = _preview(r.text)
        raise RuntimeError(
            f"Non-JSON response for {safe_url} | HTTP {r.status_code} | content-type={content_type} | body_preview={body}"
        ) from e


# ----------------------------- startup test -----------------------------

def startup_test_endpoints(
    *,
    airports: List[str],
    access_key: str,
    flight_type: str,
    airline_iata: Optional[str],
    flight_iata: Optional[str],
    limit: Optional[int],
    skip: Optional[int],
    sample_n: int,
    per_airport_timeout_s: float,
) -> None:
    """
    Hit the endpoint immediately at startup so you know the credentials/params are valid
    before the script goes into long sleeps.

    - By default tests ALL airports (sample_n=0).
    - Uses a small limit by default (you control via args) to reduce cost/time.
    - Fails fast: first error stops the run with a clear message.
    """
    if not airports:
        raise ValueError("No airports provided; startup test cannot run.")

    test_list = airports if sample_n <= 0 else airports[:sample_n]

    print(f"[TEST] Startup endpoint test: {len(test_list)} airports (type={flight_type})")
    print(f"[TEST] Using limit={limit}, skip={skip}, timeout={per_airport_timeout_s}s per airport")

    # temporarily override global timeout for this phase (without changing REQUEST_TIMEOUT for main loop)
    old_timeout = globals().get("REQUEST_TIMEOUT", 30)
    globals()["REQUEST_TIMEOUT"] = int(max(1, per_airport_timeout_s))

    ok = 0
    try:
        for a in test_list:
            try:
                _ = fetch_flight_schedules(
                    airport_iata=a,
                    access_key=access_key,
                    flight_type=flight_type,
                    airline_iata=airline_iata,
                    flight_iata=flight_iata,
                    limit=limit,
                    skip=skip,
                )
                ok += 1
                print(f"[TEST-OK] {a}")
            except Exception as e:
                print(f"[TEST-FAIL] {a}: {e}")
                raise SystemExit(2)
    finally:
        globals()["REQUEST_TIMEOUT"] = old_timeout

    print(f"[TEST] Startup endpoint test passed: {ok}/{len(test_list)} airports")


# ----------------------------- job scheduling -----------------------------

@dataclass(order=True)
class ScheduledJob:
    run_at_utc: datetime
    airport: str


class Scheduler:
    def __init__(
        self,
        airports: List[str],
        tz_map: Dict[str, str],
        hour: int,
        minute: int,
    ):
        self.tz_map = tz_map
        self.hour = hour
        self.minute = minute
        self.queue: List[ScheduledJob] = []

        for a in airports:
            if a not in tz_map:
                raise ValueError(f"No timezone mapping for airport {a}")
            run_at = next_run_utc(a, tz_map[a], hour, minute)
            heapq.heappush(self.queue, ScheduledJob(run_at, a))

    def next_job(self) -> ScheduledJob:
        return heapq.heappop(self.queue)

    def reschedule(self, airport: str):
        run_at = next_run_utc(
            airport,
            self.tz_map[airport],
            self.hour,
            self.minute,
        )
        heapq.heappush(self.queue, ScheduledJob(run_at, airport))


# ----------------------------- main loop -----------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--airports", required=True, help="tracked_airports.txt")
    p.add_argument(
        "--timezones",
        required=True,
        help="airport metadata CSV containing IATA and Timezone columns",
    )
    p.add_argument("--output-dir", required=True, help="where to write daily snapshots")
    p.add_argument("--access-key", required=True, help="FlightLabs API key")

    # API filters
    p.add_argument(
        "--type",
        dest="flight_type",
        default="departure",
        choices=["departure", "arrival"],
        help='Flight type for schedules ("departure" or "arrival")',
    )
    p.add_argument("--airline-iata", default=None, help="Optional airline IATA filter (e.g. AA)")
    p.add_argument("--flight-iata", default=None, help="Optional flight IATA filter (e.g. AA171)")
    p.add_argument("--limit", type=int, default=500, help="Max flights per call (endpoint enforces <= 1000)")
    p.add_argument("--skip", type=int, default=0, help="Records to skip for pagination (if supported)")

    # startup tests
    p.add_argument(
        "--startup-test",
        action="store_true",
        default=True,
        help="Run endpoint tests immediately on startup (default: ON).",
    )
    p.add_argument(
        "--no-startup-test",
        dest="startup_test",
        action="store_false",
        help="Disable startup endpoint tests.",
    )
    p.add_argument(
        "--startup-test-sample",
        type=int,
        default=0,
        help="If >0, only test the first N airports at startup (0 = test all).",
    )
    p.add_argument(
        "--startup-test-limit",
        type=int,
        default=1,
        help="limit to use during startup tests (kept small to be fast/cheap).",
    )
    p.add_argument(
        "--startup-test-timeout",
        type=float,
        default=10.0,
        help="Per-airport timeout (seconds) during startup tests.",
    )

    # scheduling
    p.add_argument("--hour", type=int, default=DEFAULT_RUN_HOUR)
    p.add_argument("--minute", type=int, default=DEFAULT_RUN_MINUTE)
    args = p.parse_args()

    airports = load_airports(Path(args.airports))
    tz_map = load_timezones(Path(args.timezones))

    # enforce limit <= 1000 (this prevents the exact error you saw)
    if args.limit > FLIGHTLABS_MAX_LIMIT:
        print(f"[WARN] --limit {args.limit} exceeds API max {FLIGHTLABS_MAX_LIMIT}; clamping to {FLIGHTLABS_MAX_LIMIT}")
        args.limit = FLIGHTLABS_MAX_LIMIT

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Startup test before scheduler begins sleeping
    if args.startup_test:
        startup_test_endpoints(
            airports=airports,
            access_key=args.access_key,
            flight_type=args.flight_type,
            airline_iata=args.airline_iata,
            flight_iata=args.flight_iata,
            limit=min(args.startup_test_limit, FLIGHTLABS_MAX_LIMIT),
            skip=0,
            sample_n=args.startup_test_sample,
            per_airport_timeout_s=args.startup_test_timeout,
        )

    scheduler = Scheduler(
        airports=airports,
        tz_map=tz_map,
        hour=args.hour,
        minute=args.minute,
    )

    print(f"[INFO] Tracking {len(airports)} airports")
    print("[INFO] Collector started")

    while True:
        job = scheduler.next_job()
        now_utc = datetime.now(UTC)

        sleep_s = (job.run_at_utc - now_utc).total_seconds()
        if sleep_s > 0:
            time.sleep(sleep_s)

        airport = job.airport
        tz = ZoneInfo(tz_map[airport])
        local_now = datetime.now(tz)
        local_date: date = local_now.date()

        print(f"[RUN] {airport} local_date={local_date} type={args.flight_type}")

        try:
            data = fetch_flight_schedules(
                airport_iata=airport,
                access_key=args.access_key,
                flight_type=args.flight_type,
                airline_iata=args.airline_iata,
                flight_iata=args.flight_iata,
                limit=args.limit,
                skip=args.skip,
            )

            out_dir = out_root / airport
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{local_date.isoformat()}.json"

            with out_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "airport": airport,
                        "local_date": local_date.isoformat(),
                        "fetched_at_utc": datetime.now(UTC).isoformat(),
                        "endpoint": "advanced-flights-schedules",
                        "request": {
                            "type": args.flight_type,
                            "airline_iata": args.airline_iata,
                            "flight_iata": args.flight_iata,
                            "limit": args.limit,
                            "skip": args.skip,
                        },
                        "data": data,
                    },
                    f,
                    indent=2,
                )

            print(f"[OK] wrote {out_file}")

        except Exception as e:
            print(f"[ERROR] {airport}: {e}")

        scheduler.reschedule(airport)


if __name__ == "__main__":
    raise SystemExit(main())