#!/usr/bin/env python3
"""
rolling_collector.py

Long-running background process that:
- tracks a fixed set of airports
- runs data pulls multiple times per LOCAL day per airport
- writes raw JSON snapshots
- on the final run of the day, merges that day's snapshots into a deduped daily union

Endpoint:
- FlightLabs Flight Schedules API:
    https://app.goflightlabs.com/advanced-flights-schedules

Reliability upgrades:
- requests.Session() connection pooling
- separate (connect, read) timeouts
- retry with exponential backoff + jitter for:
    - timeouts / transient RequestException
    - HTTP 429 and 5xx

Security upgrade:
- redact access_key from any logged URLs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import heapq
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from requests import Response
from requests.exceptions import RequestException, ReadTimeout, ConnectTimeout, Timeout
from zoneinfo import ZoneInfo


# ----------------------------- config -----------------------------

DEFAULT_RUN_TIMES = ["08:00", "16:00", "23:50"]  # local times
DEFAULT_FINAL_TIME = "23:50"                     # which local time triggers daily merge/finalize

# Use a tuple timeout: (connect_timeout, read_timeout)
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60
REQUEST_TIMEOUT: Tuple[int, int] = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)

UTC = ZoneInfo("UTC")

# Retry policy
MAX_RETRIES = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
JITTER_S = 0.5

# Small delay between calls to reduce burstiness
INTER_CALL_SLEEP_S = 0.2


# ----------------------------- helpers -----------------------------

def redact_url(url: str, redact_keys: Tuple[str, ...] = ("access_key",)) -> str:
    """
    Return a version of the URL with sensitive query params redacted.
    Example: access_key=... -> access_key=***
    """
    try:
        parts = urlsplit(url)
        q = parse_qsl(parts.query, keep_blank_values=True)
        redacted = []
        for k, v in q:
            if k in redact_keys:
                redacted.append((k, "***"))
            else:
                redacted.append((k, v))
        new_query = urlencode(redacted, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    except Exception:
        # If parsing fails, at least avoid leaking full URL
        return "<redacted-url>"


def load_airports(path: Path) -> List[str]:
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_timezones(path: Path) -> Dict[str, str]:
    """
    Reads airport metadata CSV with header containing at least:
      IATA, Timezone

    Returns: dict[IATA] = Timezone (IANA string)
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


def parse_hhmm(s: str) -> Tuple[int, int]:
    s = s.strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time '{s}', expected HH:MM")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid time '{s}', expected HH:MM 00-23:00-59")
    return hh, mm


def next_run_utc_for_time(tz_name: str, hour: int, minute: int) -> datetime:
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)

    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_local:
        candidate += timedelta(days=1)

    return candidate.astimezone(UTC)


def _preview(text: str, n: int = 300) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[:n] + " ...[truncated]"


def _sleep_backoff(attempt: int) -> None:
    delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
    delay += random.random() * JITTER_S
    time.sleep(delay)


# ----------------------------- HTTP / FlightLabs -----------------------------

def should_retry_http(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def request_with_retries(
    session: requests.Session,
    *,
    method: str,
    url: str,
    params: Dict[str, Any],
    timeout: Tuple[int, int],
) -> Response:
    for attempt in range(0, MAX_RETRIES + 1):
        try:
            r = session.request(method, url, params=params, timeout=timeout)

            # Build redacted URL for logs (never log r.url raw)
            safe_url = redact_url(r.url)

            if should_retry_http(r.status_code):
                body = _preview(r.text)
                print(
                    f"[WARN] HTTP {r.status_code} retryable for {safe_url} "
                    f"(attempt {attempt+1}/{MAX_RETRIES+1}) body_preview={body}"
                )
                if attempt < MAX_RETRIES:
                    _sleep_backoff(attempt + 1)
                    continue
            return r

        except (ReadTimeout, ConnectTimeout, Timeout) as e:
            print(
                f"[WARN] Timeout contacting {redact_url(url)} "
                f"(attempt {attempt+1}/{MAX_RETRIES+1}): {e}"
            )
            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt + 1)
                continue
            raise

        except RequestException as e:
            print(
                f"[WARN] RequestException contacting {redact_url(url)} "
                f"(attempt {attempt+1}/{MAX_RETRIES+1}): {e}"
            )
            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt + 1)
                continue
            raise

    raise RuntimeError("request_with_retries failed unexpectedly")


def fetch_flight_schedules(
    session: requests.Session,
    *,
    airport_iata: str,
    access_key: str,
    flight_type: str,
    airline_iata: Optional[str],
    flight_iata: Optional[str],
    limit: Optional[int],
    skip: Optional[int],
) -> dict:
    url = "https://app.goflightlabs.com/advanced-flights-schedules"

    params: Dict[str, Any] = {
        "access_key": access_key,
        "iataCode": airport_iata,
        "type": flight_type,  # "departure" or "arrival"
    }
    if airline_iata:
        params["airline_iata"] = airline_iata.strip().upper()
    if flight_iata:
        params["flight_iata"] = flight_iata.strip().upper()
    if limit is not None:
        params["limit"] = int(limit)
    if skip is not None:
        params["skip"] = int(skip)

    r = request_with_retries(session, method="GET", url=url, params=params, timeout=REQUEST_TIMEOUT)
    content_type = (r.headers.get("Content-Type") or "").lower()
    safe_url = redact_url(r.url)

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


# ----------------------------- dedupe / daily union -----------------------------

def _get_nested(d: Any, path: List[str]) -> Optional[Any]:
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def flight_dedupe_key(f: Any) -> str:
    if not isinstance(f, dict):
        return "obj:" + hashlib.sha256(json.dumps(f, sort_keys=True, default=str).encode()).hexdigest()

    candidates: List[Optional[str]] = []
    candidates.append(str(f.get("flight_iata") or f.get("flightIata") or f.get("flight") or "") or None)
    candidates.append(str(_get_nested(f, ["flight", "iata"]) or _get_nested(f, ["flight", "iataNumber"]) or "") or None)
    candidates.append(str(f.get("flight_number") or f.get("flightNumber") or "") or None)

    candidates.append(str(f.get("airline_iata") or f.get("airlineIata") or "") or None)
    candidates.append(str(_get_nested(f, ["airline", "iata"]) or "") or None)

    candidates.append(str(_get_nested(f, ["departure", "iataCode"]) or _get_nested(f, ["departure", "iata"]) or "") or None)
    candidates.append(str(_get_nested(f, ["arrival", "iataCode"]) or _get_nested(f, ["arrival", "iata"]) or "") or None)

    candidates.append(str(_get_nested(f, ["departure", "scheduled"]) or _get_nested(f, ["departure", "scheduledTime"]) or "") or None)
    candidates.append(str(_get_nested(f, ["arrival", "scheduled"]) or _get_nested(f, ["arrival", "scheduledTime"]) or "") or None)

    parts = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    if parts:
        return "k:" + "|".join(parts)

    blob = json.dumps(f, sort_keys=True, default=str)
    return "h:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_flights(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("results"), list):
            return payload["results"]
    return []


def merge_daily_snapshots(snapshot_files: List[Path]) -> Dict[str, Any]:
    flights_by_key: Dict[str, Any] = {}
    errors: List[str] = []

    for fp in sorted(snapshot_files):
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            data = raw.get("data", raw)
            flights = extract_flights(data)
            for f in flights:
                k = flight_dedupe_key(f)
                flights_by_key[k] = f
        except Exception as e:
            errors.append(f"{fp.name}: {e}")

    merged = list(flights_by_key.values())
    return {
        "merged_flights": merged,
        "unique_count": len(merged),
        "snapshot_count": len(snapshot_files),
        "snapshot_files": [p.name for p in sorted(snapshot_files)],
        "merge_errors": errors,
    }


# ----------------------------- job scheduling -----------------------------

@dataclass(order=True)
class ScheduledJob:
    run_at_utc: datetime
    airport: str
    hhmm: str
    is_final: bool


class Scheduler:
    def __init__(
        self,
        airports: List[str],
        tz_map: Dict[str, str],
        run_times: List[str],
        final_time: str,
    ):
        self.tz_map = tz_map
        self.run_times = run_times
        self.final_time = final_time
        self.queue: List[ScheduledJob] = []

        for a in airports:
            if a not in tz_map:
                raise ValueError(f"No timezone mapping for airport {a}")
            for hhmm in run_times:
                hh, mm = parse_hhmm(hhmm)
                run_at = next_run_utc_for_time(tz_map[a], hh, mm)
                heapq.heappush(
                    self.queue,
                    ScheduledJob(run_at_utc=run_at, airport=a, hhmm=hhmm, is_final=(hhmm == final_time)),
                )

    def next_job(self) -> ScheduledJob:
        return heapq.heappop(self.queue)

    def reschedule(self, airport: str, hhmm: str, is_final: bool):
        hh, mm = parse_hhmm(hhmm)
        run_at = next_run_utc_for_time(self.tz_map[airport], hh, mm)
        heapq.heappush(self.queue, ScheduledJob(run_at_utc=run_at, airport=airport, hhmm=hhmm, is_final=is_final))


# ----------------------------- main loop -----------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--airports", required=True, help="tracked_airports.txt")
    p.add_argument(
        "--timezones",
        required=True,
        help="airport metadata CSV containing IATA and Timezone columns",
    )
    p.add_argument("--output-dir", required=True, help="where to write snapshots + daily unions")
    p.add_argument("--access-key", required=True, help="FlightLabs API key")

    # API filters
    p.add_argument(
        "--type",
        dest="flight_type",
        default="departure",
        choices=["departure", "arrival"],
    )
    p.add_argument("--airline-iata", default=None, help="Optional airline IATA filter (e.g. AA)")
    p.add_argument("--flight-iata", default=None, help="Optional flight IATA filter (e.g. AA171)")
    p.add_argument("--limit", type=int, default=500, help="Max flights per call (if supported)")
    p.add_argument("--skip", type=int, default=0, help="Records to skip for pagination (if supported)")

    # schedule
    p.add_argument(
        "--run-times",
        default=",".join(DEFAULT_RUN_TIMES),
        help='Comma-separated local times HH:MM, e.g. "08:00,16:00,23:50"',
    )
    p.add_argument(
        "--final-time",
        default=DEFAULT_FINAL_TIME,
        help='Which HH:MM in --run-times counts as the final daily merge time (default "23:50")',
    )

    args = p.parse_args()

    run_times = [t.strip() for t in args.run_times.split(",") if t.strip()]
    if not run_times:
        raise ValueError("No run times specified")
    for t in run_times:
        parse_hhmm(t)
    if args.final_time not in run_times:
        raise ValueError(f"--final-time {args.final_time} must be one of --run-times: {run_times}")

    airports = load_airports(Path(args.airports))
    tz_map = load_timezones(Path(args.timezones))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scheduler = Scheduler(
        airports=airports,
        tz_map=tz_map,
        run_times=run_times,
        final_time=args.final_time,
    )

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "rolling_collector/1.0"})

    print(f"[INFO] Tracking {len(airports)} airports")
    print(f"[INFO] Run times (local): {run_times} | final={args.final_time}")
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
        local_date = local_now.date()

        print(f"[RUN] {airport} local_date={local_date} hhmm={job.hhmm} type={args.flight_type} final={job.is_final}")

        try:
            if INTER_CALL_SLEEP_S > 0:
                time.sleep(INTER_CALL_SLEEP_S)

            data = fetch_flight_schedules(
                session,
                airport_iata=airport,
                access_key=args.access_key,
                flight_type=args.flight_type,
                airline_iata=args.airline_iata,
                flight_iata=args.flight_iata,
                limit=args.limit,
                skip=args.skip,
            )

            base = out_root / airport
            snap_dir = base / "snapshots" / local_date.isoformat()
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_file = snap_dir / f"{job.hhmm.replace(':', '')}.json"

            with snap_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "airport": airport,
                        "local_date": local_date.isoformat(),
                        "hhmm_local": job.hhmm,
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

            print(f"[OK] wrote snapshot {snap_file}")

            if job.is_final:
                daily_dir = base / "daily"
                daily_dir.mkdir(parents=True, exist_ok=True)
                daily_file = daily_dir / f"{local_date.isoformat()}.json"

                snapshot_files = sorted(snap_dir.glob("*.json"))
                merged = merge_daily_snapshots(snapshot_files)

                with daily_file.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "airport": airport,
                            "local_date": local_date.isoformat(),
                            "finalized_at_utc": datetime.now(UTC).isoformat(),
                            "source_snapshot_dir": str(snap_dir),
                            **merged,
                        },
                        f,
                        indent=2,
                    )

                print(f"[OK] wrote daily union {daily_file} (unique={merged['unique_count']} from {merged['snapshot_count']} snapshots)")

        except Exception as e:
            print(f"[ERROR] {airport}: {e}")

        scheduler.reschedule(airport, job.hhmm, job.is_final)


if __name__ == "__main__":
    raise SystemExit(main())