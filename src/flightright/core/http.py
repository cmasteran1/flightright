# src/flightright/core/http.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class HttpClient:
    timeout_s: int = 30

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        try:
            return r.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Non-JSON response from {r.url}: {r.text[:500]}") from e