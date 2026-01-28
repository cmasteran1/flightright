# src/runtime/providers/history/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HistoryProvider:
    """Interface for recent-history aggregations used at prediction time."""
    def recent_metrics(self, *, airline_iata: str, flight_number: str, origin: str, dest: str,
                       asof_local_date: str) -> Dict[str, Any]:
        """
        Compute recent aggregations as of `asof_local_date` (YYYY-MM-DD, origin local).
        Must return ALL of:
          - carrier_flights_prior_day (int)
          - carrier_delay_7d_mean (float minutes)
          - od_delay_7d_mean (float minutes)
          - flightnum_delay_14d_mean (float minutes)
          - origin_delay_7d_mean (float minutes)
          - dest_delay_7d_mean (float minutes)
        Raise RuntimeError if any are not computable.
        """
        raise NotImplementedError
