#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # py3.9/3.10

# Order of precedence:
# 1) Environment variables
# 2) config/providers.toml (if present)
# 3) config/providers.example.toml (fallback for non-secret defaults)

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_TOML = ROOT / "config" / "providers.toml"
EXAMPLE_TOML   = ROOT / "config" / "providers.example.toml"

def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """shallow merge dict b into a"""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out

class Settings:
    def __init__(self):
        cfg = _merge(_read_toml(EXAMPLE_TOML), _read_toml(PROVIDERS_TOML))

        def env_or(cfg_path: list[str], env_name: str, default: Optional[str] = None) -> Optional[str]:
            # Read from env first
            if env_name in os.environ and os.environ[env_name]:
                return os.environ[env_name]
            # then from toml nested dict
            node = cfg
            for key in cfg_path:
                if not isinstance(node, dict) or key not in node:
                    return default
                node = node[key]
            val = node if isinstance(node, str) and node else None
            return val or default

        # Flight providers
        self.AERODATABOX_API_KEY  = env_or(["AeroDataBox","API_KEY"], "AERODATABOX_API_KEY")
        self.AVIATIONSTACK_API_KEY = env_or(["AviationStack","API_KEY"], "AVIATIONSTACK_API_KEY")
        self.AIRLABS_API_KEY      = env_or(["AirLabs","API_KEY"], "AIRLABS_API_KEY")
        self.OPEN_SKY_USERNAME    = env_or(["OpenSky","USERNAME"], "OPEN_SKY_USERNAME")
        self.OPEN_SKY_PASSWORD    = env_or(["OpenSky","PASSWORD"], "OPEN_SKY_PASSWORD")

        # Weather
        self.TOMORROWIO_API_KEY   = env_or(["TomorrowIO","API_KEY"], "TOMORROWIO_API_KEY")
        self.VISUALCROSSING_API_KEY = env_or(["VisualCrossing","API_KEY"], "VISUALCROSSING_API_KEY")
        self.NWS_USER_AGENT       = env_or(["NWS","USER_AGENT"], "NWS_USER_AGENT", "you@example.com (your_app)")

        # RapidAPI (optional)
        self.RAPIDAPI_API_KEY     = env_or(["RapidAPI","API_KEY"], "RAPIDAPI_API_KEY")

        # Non-secret defaults
        def _int(default: int, path: list[str], env_name: str) -> int:
            v = os.environ.get(env_name)
            if v and v.isdigit():
                return int(v)
            node = cfg
            for k in path:
                node = node.get(k, {})
            if isinstance(node, int):
                return node
            return default

        self.REQUEST_TIMEOUT_SEC  = _int(30, ["Defaults","REQUEST_TIMEOUT_SEC"], "REQUEST_TIMEOUT_SEC")
        self.RETRY_MAX            = _int(3,  ["Defaults","RETRY_MAX"], "RETRY_MAX")

    # Convenience: provider-ready headers
    def headers_for(self, provider: str) -> Dict[str, str]:
        p = provider.lower()
        if p == "nws":
            # REQUIRED: Identify yourself per NWS policy
            return {"User-Agent": self.NWS_USER_AGENT, "Accept": "application/geo+json"}
        if p == "aerodatabox":
            # AeroDataBox typically uses `X-RapidAPI-Key` if accessed via RapidAPI, OR plain `x-rapidapi-key`
            # If you use their direct API, check their docs (may be Authorization bearer)
            if self.AERODATABOX_API_KEY:
                return {"Authorization": f"Bearer {self.AERODATABOX_API_KEY}"}
        if p == "tomorrowio":
            # Usually passed as query param, but header form works with some clients
            if self.TOMORROWIO_API_KEY:
                return {"Authorization": f"Bearer {self.TOMORROWIO_API_KEY}"}
        if p == "rapidapi" and self.RAPIDAPI_API_KEY:
            return {"X-RapidAPI-Key": self.RAPIDAPI_API_KEY}
        return {}

settings = Settings()
