# src/flightright/config/env.py

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    """
    Load .env from repo root if present.
    Safe to call multiple times.
    """
    # Walk upward to find .env (repo-root safe)
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break


load_env()


def require(name: str) -> str:
    if name not in os.environ:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return os.environ[name]


def optional(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


# ---- Secrets ----
FLIGHTLABS_ACCESS_KEY = require("FLIGHTLABS_ACCESS_KEY")