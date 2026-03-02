from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FeatureAvailability:
    """
    Booleans for whether a feature-group is available.
    The selector uses these to pick a model tier.
    """
    has_hourly_weather: bool
    has_daily_weather: bool
    has_rollup_history: bool          # carrier/origin/carrier_origin rollups
    has_flightnum_od_history: bool     # on-demand, but could fail for some reason
    has_congestion: bool              # needs airport-day schedule snapshot
    has_tail_inference: bool          # needs flight chain reconstruction


@dataclass(frozen=True)
class ModelChoice:
    name: str  # "full" | "daily_only_weather" | "limited_history" | "fallback"
    reason: str


@dataclass(frozen=True)
class FlightRequest:
    """
    Minimal request object for feature building.
    """
    origin: str
    dest: str
    carrier: str
    flight_number: str
    dep_local_date: str  # YYYY-MM-DD at origin local time
    sched_dep_hour: int  # 0-23