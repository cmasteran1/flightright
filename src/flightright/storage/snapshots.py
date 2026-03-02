from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class SnapshotRecord:
    """
    A normalized per-flight record pulled from the airport schedule snapshot.
    This is intentionally minimal: just what we need to build rollups and congestion.
    """
    origin: str
    carrier: str
    dep_delay_min: Optional[float]  # None if unknown/missing


def read_snapshot_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_snapshot_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def parse_departure_delay_minutes(f: Dict[str, Any]) -> Optional[float]:
    """
    FlightLabs-like airport departures response often includes:
      - dep_delayed (minutes)
      - delayed (minutes) (sometimes overall)
    We'll prefer dep_delayed if present.

    Returns None if delay can't be determined.
    """
    if "dep_delayed" in f:
        v = _safe_float(f.get("dep_delayed"))
        if v is not None:
            return v

    # fallback: "delayed" sometimes present
    if "delayed" in f:
        v = _safe_float(f.get("delayed"))
        if v is not None:
            return v

    # could compute from timestamps if you trust them; keep conservative for now
    return None


def iter_snapshot_records(snapshot_payload: Dict[str, Any], default_origin: str) -> Iterable[SnapshotRecord]:
    """
    snapshot_payload is expected to look like:
      {"success": true, "type":"departure", "data":[{...flight...}, ...]}

    We treat origin as the airport we collected for (default_origin),
    because some payloads include dep_iata anyway, but we don't want to trust
    weirdness in vendor data.
    """
    data = snapshot_payload.get("data", [])
    if not isinstance(data, list):
        return

    origin = default_origin.upper().strip()

    for f in data:
        if not isinstance(f, dict):
            continue
        carrier = (f.get("airline_iata") or "").upper().strip()
        if not carrier:
            continue

        dep_delay = parse_departure_delay_minutes(f)
        yield SnapshotRecord(origin=origin, carrier=carrier, dep_delay_min=dep_delay)