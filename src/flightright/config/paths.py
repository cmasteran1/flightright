# src/flightright/config/paths.py

from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """
    Find the flightright repo root by walking upward until we see a 'src' folder.
    Assumes this file lives at: <repo>/src/flightright/config/paths.py
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "src").exists():
            return parent
    # Fallback: assume typical layout
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _find_repo_root()

# Your desired layout:
# <parent_of_repo>/flightrightdata   (sibling of repo)
DATA_ROOT = REPO_ROOT.parent / "flightrightdata"

# Metadata
AIRPORTS_CSV = DATA_ROOT / "data" / "meta" / "airports.csv"

# Rolling airport schedules written by collector
ROLLING_SNAPSHOTS_DIR = DATA_ROOT / "data" / "rolling_snapshots"
ROLLING_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_paths() -> None:
    """
    Helpful early failure if your data repo isn't where we expect.
    """
    if not DATA_ROOT.exists():
        raise RuntimeError(
            f"DATA_ROOT does not exist: {DATA_ROOT}\n"
            f"Expected flightrightdata to live next to repo root: {REPO_ROOT}"
        )
    if not AIRPORTS_CSV.exists():
        raise RuntimeError(
            f"AIRPORTS_CSV not found: {AIRPORTS_CSV}\n"
            "Expected flightrightdata/data/meta/airports.csv"
        )