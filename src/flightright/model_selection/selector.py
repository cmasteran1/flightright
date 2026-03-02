# src/flightright/model_selection/selector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    categorical_features: List[str]
    numeric_features: List[str]


def _missing(features: Dict[str, object], required: Iterable[str]) -> List[str]:
    missing = []
    for k in required:
        if k not in features or features[k] is None:
            missing.append(k)
    return missing


def pick_model(
    *,
    features: Dict[str, object],
    full_model: ModelSpec,
    daily_only_weather_model: ModelSpec,
    limited_history_model: ModelSpec,
) -> Tuple[ModelSpec, List[str]]:
    """
    Selection rules (your intent):
      - Prefer full model when hourly weather + (eventually) carrier/airport history exist.
      - Fall back to daily-only-weather model if hourly weather missing.
      - Fall back to limited-history model if carrier/airport history missing (but hourly weather present).
    For now, since you can't test long-term history yet, this will still work—full model will be missing
    carrier/origin rolling fields and selector will choose limited_history_model.
    """
    full_req = set(full_model.categorical_features + full_model.numeric_features)
    daily_req = set(daily_only_weather_model.categorical_features + daily_only_weather_model.numeric_features)
    limited_req = set(limited_history_model.categorical_features + limited_history_model.numeric_features)

    full_missing = _missing(features, full_req)
    if not full_missing:
        return full_model, []

    # If hourly weather fields missing, try daily-only-weather model first
    hourly_weather_keys = {"origin_dep_temp_K", "origin_dep_precip_mm", "origin_dep_windspeed_kmh", "origin_dep_hour_weathercode"}
    if any(k in full_req and (k not in features or features[k] is None) for k in hourly_weather_keys):
        daily_missing = _missing(features, daily_req)
        if not daily_missing:
            return daily_only_weather_model, []
        # if even daily is missing, fall back to the "best we can" (limited) and report missing
        limited_missing = _missing(features, limited_req)
        return limited_history_model, sorted(set(daily_missing) | set(limited_missing))

    # Otherwise hourly weather exists, but history likely missing -> try limited_history_model
    limited_missing = _missing(features, limited_req)
    if not limited_missing:
        return limited_history_model, []

    # Nothing fully satisfied; pick the one with least missing.
    daily_missing = _missing(features, daily_req)
    choices = [
        (full_model, len(full_missing), full_missing),
        (limited_history_model, len(limited_missing), limited_missing),
        (daily_only_weather_model, len(daily_missing), daily_missing),
    ]
    choices.sort(key=lambda x: x[1])
    return choices[0][0], choices[0][2]