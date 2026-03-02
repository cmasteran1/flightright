from __future__ import annotations

from flightright.core.schemas import FeatureAvailability, ModelChoice


def choose_model(av: FeatureAvailability) -> ModelChoice:
    """
    Your current tiers:

    FULL model requires:
      - daily weather
      - hourly weather
      - rollup history (carrier/origin/carrier_origin)
      - flightnum_od_history (on-demand)
      - congestion
      - tail inference

    DAILY-ONLY-WEATHER model:
      - daily weather
      - rollup history
      - flightnum_od_history
      - congestion
      - tail inference
      (NO hourly weather required)

    LIMITED-HISTORY model:
      - daily weather
      - hourly weather
      - flightnum_od_history
      - tail inference
      (NO congestion, NO rollup history)

    fallback: anything else
    """
    if (
        av.has_daily_weather
        and av.has_hourly_weather
        and av.has_rollup_history
        and av.has_flightnum_od_history
        and av.has_congestion
        and av.has_tail_inference
    ):
        return ModelChoice("full", "All feature groups available")

    if (
        av.has_daily_weather
        and av.has_rollup_history
        and av.has_flightnum_od_history
        and av.has_congestion
        and av.has_tail_inference
    ):
        return ModelChoice("daily_only_weather", "Hourly weather unavailable; using daily-only weather model")

    if (
        av.has_daily_weather
        and av.has_hourly_weather
        and av.has_flightnum_od_history
        and av.has_tail_inference
    ):
        return ModelChoice("limited_history", "Rollup history and/or congestion unavailable; using limited-history model")

    return ModelChoice("fallback", "Insufficient features for tiered models")