#!/usr/bin/env python3
"""
Build data/airport_gates.csv by scraping gate counts from Wikipedia.

Approach (best-effort):
  1) Read airport list from data/airport_coords.json (keys).
  2) For each IATA, use MediaWiki API to find the best matching airport page.
  3) Fetch the page extract (plain text) and parse for "total of X gates" and similar patterns.
  4) Write CSV: IATA,gates,source_url,method,confidence,notes

Caveats:
  - Some airports do not have a "total gates" statement on Wikipedia.
  - Some pages list concourse gates separately; totals may be absent.
  - Regex extraction can be wrong; you should review low-confidence rows.
"""

import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AIRPORT_JSON = REPO_ROOT / "data" / "airport_coords.json"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "airport_gates.csv"

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "flightright-airport-gates/0.1 (contact: local-script)"


# ---------- HTTP with backoff ----------
def get_with_backoff(url: str, params: dict, max_tries: int = 6) -> requests.Response:
    delay = 1.0
    backoff = 1.7
    last_exc = None

    for _ in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(60, delay * backoff)
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay = min(60, delay * backoff)

    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed.")


# ---------- Wikipedia helpers ----------
def wiki_search(query: str, limit: int = 5) -> List[dict]:
    r = get_with_backoff(
        WIKI_API,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
        },
    )
    return r.json().get("query", {}).get("search", [])


def wiki_get_extract_and_url(pageid: int) -> Tuple[str, str, str]:
    r = get_with_backoff(
        WIKI_API,
        {
            "action": "query",
            "prop": "extracts|info",
            "pageids": str(pageid),
            "explaintext": 1,
            "inprop": "url",
            "format": "json",
        },
    )
    pages = r.json().get("query", {}).get("pages", {})
    p = pages.get(str(pageid), {})
    extract = p.get("extract", "") or ""
    fullurl = p.get("fullurl", "") or ""
    title = p.get("title", "") or ""
    return extract, fullurl, title


# ---------- parsing ----------
@dataclass
class GateParseResult:
    gates: Optional[int]
    method: str
    confidence: float
    notes: str


# Prioritized patterns: most explicit -> least explicit
PATTERNS: List[Tuple[str, str, float]] = [
    # "with a total of 51 gates"
    (r"\btotal of\s+(\d{1,4})\s+gates\b", "total_of_gates", 0.95),
    # "a total 51 gates" (missing "of")
    (r"\ba total\s+(\d{1,4})\s+gates\b", "a_total_gates", 0.90),
    # "has 51 gates" or "has ... with 51 gates"
    (r"\bhas\s+(\d{1,4})\s+gates\b", "has_gates", 0.75),
    # "with 51 gates"
    (r"\bwith\s+(\d{1,4})\s+gates\b", "with_gates", 0.60),
]


def parse_gates_from_text(text: str) -> GateParseResult:
    t = " ".join(text.split())  # normalize whitespace

    matches = []
    for pat, method, base_conf in PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            try:
                n = int(m.group(1))
                if 1 <= n <= 2000:
                    matches.append((n, method, base_conf, m.group(0)))
            except Exception:
                pass

    if not matches:
        return GateParseResult(None, "no_match", 0.0, "No gate-count pattern matched")

    # If multiple, prefer highest confidence; if tie, prefer largest count (often totals > concourses)
    matches.sort(key=lambda x: (x[2], x[0]), reverse=True)
    n, method, conf, snippet = matches[0]

    # If there are conflicting numbers, reduce confidence a bit and note it
    uniq = sorted({m[0] for m in matches})
    notes = f"matched snippet: '{snippet}'"
    if len(uniq) > 1:
        conf = max(0.4, conf - 0.15)
        notes += f" | multiple candidates: {uniq}"

    return GateParseResult(n, method, conf, notes)


def choose_best_wikipedia_page(iata: str) -> Optional[Tuple[int, str]]:
    """
    Try a few search queries; choose the best-looking result.
    Returns (pageid, title) or None.
    """
    iata = iata.upper().strip()

    queries = [
        f"{iata} airport",
        f'"{iata}" airport',
        f"{iata} International Airport",
        f"{iata} Airport Wikipedia",
    ]

    results = []
    for q in queries:
        results.extend(wiki_search(q, limit=5))

    if not results:
        return None

    # Score results: prefer titles containing "Airport" and not "List of"
    def score(item: dict) -> float:
        title = (item.get("title") or "").lower()
        s = 0.0
        if "airport" in title:
            s += 2.0
        if "international airport" in title:
            s += 0.5
        if title.startswith("list of") or "list of" in title:
            s -= 2.0
        # mild preference for exact iata appearing in snippet/title text
        snippet = (item.get("snippet") or "").lower()
        if iata.lower() in title:
            s += 0.3
        if iata.lower() in snippet:
            s += 0.2
        return s

    results.sort(key=score, reverse=True)
    best = results[0]
    pageid = best.get("pageid")
    title = best.get("title") or ""
    if not pageid:
        return None
    return int(pageid), title


def load_airports_from_airport_coords_json(path: Path) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "airport_coords" in data:
        data = data["airport_coords"]
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected dict in {path}")
    airports = sorted({str(k).upper() for k in data.keys()})
    return airports


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["IATA", "gates", "source_url", "wiki_title", "method", "confidence", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    airport_json = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_AIRPORT_JSON
    out_csv = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_OUT_CSV

    if not airport_json.exists():
        raise FileNotFoundError(f"airport_coords.json not found: {airport_json}")

    airports = load_airports_from_airport_coords_json(airport_json)
    print(f"[INFO] Loaded {len(airports)} airports from {airport_json}")

    rows = []
    ok = 0
    miss = 0

    for i, ap in enumerate(airports, start=1):
        ap = ap.upper()
        print(f"[{i}/{len(airports)}] {ap} ...", end=" ", flush=True)

        best = choose_best_wikipedia_page(ap)
        if not best:
            rows.append(
                {
                    "IATA": ap,
                    "gates": "",
                    "source_url": "",
                    "wiki_title": "",
                    "method": "no_page",
                    "confidence": 0.0,
                    "notes": "No Wikipedia page found via search",
                }
            )
            miss += 1
            print("no_page")
            continue

        pageid, _ = best
        extract, url, title = wiki_get_extract_and_url(pageid)
        if not extract.strip():
            rows.append(
                {
                    "IATA": ap,
                    "gates": "",
                    "source_url": url,
                    "wiki_title": title,
                    "method": "empty_extract",
                    "confidence": 0.0,
                    "notes": "Empty Wikipedia extract",
                }
            )
            miss += 1
            print("empty_extract")
            continue

        parsed = parse_gates_from_text(extract)
        if parsed.gates is None:
            rows.append(
                {
                    "IATA": ap,
                    "gates": "",
                    "source_url": url,
                    "wiki_title": title,
                    "method": parsed.method,
                    "confidence": parsed.confidence,
                    "notes": parsed.notes,
                }
            )
            miss += 1
            print("no_match")
        else:
            rows.append(
                {
                    "IATA": ap,
                    "gates": int(parsed.gates),
                    "source_url": url,
                    "wiki_title": title,
                    "method": parsed.method,
                    "confidence": round(float(parsed.confidence), 3),
                    "notes": parsed.notes,
                }
            )
            ok += 1
            print(f"{parsed.gates} ({parsed.method}, conf={parsed.confidence:.2f})")

    write_csv(rows, out_csv)
    print(f"[OK] wrote {len(rows)} rows -> {out_csv}")
    print(f"[INFO] parsed gates for {ok} airports; missing/unknown for {miss} airports")
    print("[TIP] Review low-confidence rows (confidence < 0.75) and fill missing gates manually where needed.")


if __name__ == "__main__":
    main()
