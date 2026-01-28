# src/runtime/providers/schedules/aerodatabox_history.py
from __future__ import annotations
import os, json, time, hashlib, pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

# ---- RapidAPI AeroDataBox endpoints (adjust here if they change) ----
NUMBER_DAY_URL        = "https://aerodatabox.p.rapidapi.com/flights/number/{airline}{number}/{date}"
AIRPORT_DEPS_DAY_URL  = "https://aerodatabox.p.rapidapi.com/flights/airports/icao/{icao}/{date}/departures"
AIRPORT_ARRS_DAY_URL  = "https://aerodatabox.p.rapidapi.com/flights/airports/icao/{icao}/{date}/arrivals"

def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = s.replace(" ", "T")
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _arrival_delay_minutes(leg: Dict[str, Any]) -> Optional[float]:
    arr = leg.get("arrival") or {}
    sched = arr.get("scheduledTime") or {}
    sched_local = sched.get("local") or sched.get("utc")
    # prefer actual, then predicted
    actual = (arr.get("actualTime") or {}) or {}
    actual_local = actual.get("local") or actual.get("utc")
    if not actual_local:
        pred = arr.get("predictedTime") or {}
        actual_local = pred.get("local") or pred.get("utc")
    t_sched = _parse_iso(sched_local)
    t_actual = _parse_iso(actual_local)
    if not t_sched or not t_actual:
        return None
    return float((t_actual - t_sched).total_seconds() / 60.0)

def _keyify(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

@dataclass
class AeroDataBoxHistory:
    api_key: str
    api_host: str = "aerodatabox.p.rapidapi.com"
    timeout: int = 20
    cache_dir: str = ".cache/aerodb"
    cache_ttl_sec: int = 1800  # 30 minutes

    def _headers(self) -> Dict[str, str]:
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.api_host,
            "Accept": "application/json",
        }

    # ----------------- Tiny JSON cache -----------------
    def _get_json_cached(self, url: str) -> Any:
        root = pathlib.Path(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        f = root / f"{_keyify(url)}.json"

        if f.exists() and (time.time() - f.stat().st_mtime) < self.cache_ttl_sec:
            try:
                with open(f, "r") as fh:
                    return json.load(fh)
            except Exception:
                pass  # fall through to refetch

        r = requests.get(url, headers=self._headers(), timeout=self.timeout)
        if r.status_code == 404:
            f.write_text("[]")
            return []
        r.raise_for_status()
        data = r.json()
        with open(f, "w") as fh:
            json.dump(data, fh)
        return data

    def _fetch_all_pages(self, base_url: str) -> List[Dict[str, Any]]:
        """Fetch all pages for airport-day endpoints. Supports both array and
        {items: [...], page: n, pages: N} shapes."""
        out: List[Dict[str, Any]] = []
        page = 1
        while True:
            url = f"{base_url}?page={page}"
            data = self._get_json_cached(url)
            if isinstance(data, dict) and "items" in data:
                items = data.get("items") or []
                out.extend(items)
                pages = int(data.get("pages") or 1)
                if page >= pages or not items:
                    break
            elif isinstance(data, list):
                # Some tiers return a plain list (no pagination)
                out.extend(data)
                break
            else:
                break
            page += 1
            if page > 10:  # hard stop to avoid cost explosions if API lies
                break
        return out

    def _fetch_airport_day(self, icao: str, date_str: str, kind: str) -> List[Dict[str, Any]]:
        if kind == "departures":
            base = AIRPORT_DEPS_DAY_URL.format(icao=icao.upper(), date=date_str)
        else:
            base = AIRPORT_ARRS_DAY_URL.format(icao=icao.upper(), date=date_str)
        return self._fetch_all_pages(base)

    # ----------------- Fetch helpers -----------------
    def _fetch_number_day(self, airline: str, number: str, date_str: str) -> List[Dict[str, Any]]:
        url = NUMBER_DAY_URL.format(airline=airline.upper(), number=number, date=date_str)
        return self._get_json_cached(url) or []

    def _fetch_airport_day(self, icao: str, date_str: str, kind: str) -> List[Dict[str, Any]]:
        if kind == "departures":
            url = AIRPORT_DEPS_DAY_URL.format(icao=icao.upper(), date=date_str)
        else:
            url = AIRPORT_ARRS_DAY_URL.format(icao=icao.upper(), date=date_str)
        return self._get_json_cached(url) or []

    # ----------------- Roll-ups -----------------
    def get_rollups(
        self,
        *,
        airline_iata: str,
        flight_number: str,
        origin_iata: str,
        dest_iata: str,
        origin_icao: str,
        dest_icao: str,
        origin_local_date: str,
    ) -> Dict[str, float]:
        """Compute all required metrics with minimal API calls and robust fallbacks."""
        # Anchor date (local at origin)
        try:
            y, m, d = [int(x) for x in origin_local_date.split("-")]
            anchor = datetime(y, m, d)
        except Exception as e:
            raise RuntimeError(f"Bad origin_local_date: {origin_local_date} ({e})")

        # ---- 14-day flight-number loop (re-used for 7-day features) ----
        delays_14: List[float] = []
        delays_7_for_carrier_proxy: List[float] = []
        delays_7_for_od: List[float] = []
        delays_7_arriving_dest_from_fn: List[float] = []  # fallback for dest_delay_7d_mean

        for i in range(1, 15):  # 1..14 prior calendar days
            day = (anchor - timedelta(days=i)).date().isoformat()
            legs = self._fetch_number_day(airline_iata, flight_number, day)
            if not legs:
                continue
            for leg in legs:
                al = ((leg.get("airline") or {}).get("iata") or "").upper()
                dep = ((leg.get("departure") or {}).get("airport") or {}).get("iata", "")
                arr = ((leg.get("arrival")  or {}).get("airport") or {}).get("iata", "")
                if al != airline_iata.upper():
                    continue
                dmin = _arrival_delay_minutes(leg)
                if dmin is None:
                    continue

                # 14d mean for this flight number
                delays_14.append(dmin)

                # Last-7 day buckets
                if i <= 7:
                    delays_7_for_carrier_proxy.append(dmin)
                    if dep and arr and dep.upper() == origin_iata.upper() and arr.upper() == dest_iata.upper():
                        delays_7_for_od.append(dmin)
                    if arr and arr.upper() == dest_iata.upper():
                        delays_7_arriving_dest_from_fn.append(dmin)

        def _mean(a: List[float]) -> Optional[float]:
            return None if not a else float(sum(a) / len(a))

        flightnum_delay_14d_mean = _mean(delays_14)
        carrier_delay_7d_mean    = _mean(delays_7_for_carrier_proxy)  # proxy: same flight number (cheap)
        od_delay_7d_mean         = _mean(delays_7_for_od)

        # ---- carrier_flights_prior_day (calendar day) ----
        # Tier 1: origin departures for that carrier on prior day
        prior_day = (anchor - timedelta(days=1)).date().isoformat()
        cfpd_count: Optional[float] = None

        deps = self._fetch_airport_day(origin_icao, prior_day, kind="departures")
        if deps:
            cnt = 0
            for leg in deps:
                al = ((leg.get("airline") or {}).get("iata") or "").upper()
                if al == airline_iata.upper():
                    cnt += 1
            if cnt > 0:
                cfpd_count = float(cnt)

        # Tier 2 (fallback): destination arrivals for that carrier on prior day
        if cfpd_count is None:
            arrs = self._fetch_airport_day(dest_icao, prior_day, kind="arrivals")
            if arrs:
                cnt = 0
                for leg in arrs:
                    al = ((leg.get("airline") or {}).get("iata") or "").upper()
                    if al == airline_iata.upper():
                        cnt += 1
                if cnt > 0:
                    cfpd_count = float(cnt)

        # Tier 3 (last-resort proxy): count this flight-number’s legs on the prior day
        # (Costs nothing extra; keeps pipeline usable in free/low-tiers.)
        if cfpd_count is None:
            legs_prior = self._fetch_number_day(airline_iata, flight_number, prior_day)
            if legs_prior:
                cfpd_count = float(len(legs_prior))

        # ---- dest_delay_7d_mean (true airport-wide first, then proxy) ----
        dest_delays_7: List[float] = []
        for i in range(1, 8):
            day = (anchor - timedelta(days=i)).date().isoformat()
            arrs = self._fetch_airport_day(dest_icao, day, kind="arrivals")
            for leg in arrs:
                dmin = _arrival_delay_minutes(leg)
                if dmin is not None:
                    dest_delays_7.append(dmin)

        dest_delay_7d_mean = _mean(dest_delays_7)
        if dest_delay_7d_mean is None:
            # Fallback proxy: use all occurrences of this flight number that ARRIVED at dest in last 7d
            dest_delay_7d_mean = _mean(delays_7_arriving_dest_from_fn)

        # ---- origin_delay_7d_mean (prefer arrivals into origin; else proxy) ----
        origin_arrivals_7: List[float] = []
        for i in range(1, 8):
            day = (anchor - timedelta(days=i)).date().isoformat()
            arrs = self._fetch_airport_day(origin_icao, day, kind="arrivals")
            for leg in arrs:
                dmin = _arrival_delay_minutes(leg)
                if dmin is not None:
                    origin_arrivals_7.append(dmin)
        origin_delay_7d_mean = _mean(origin_arrivals_7)
        if origin_delay_7d_mean is None:
            origin_delay_7d_mean = _mean(delays_7_for_carrier_proxy)

        # ---- Validate (hard-fail only if *all* tiers failed) ----
        out = {
            "carrier_flights_prior_day": cfpd_count,            # may be proxy if airport endpoints blocked
            "carrier_delay_7d_mean":    carrier_delay_7d_mean,  # proxy by same flight number
            "od_delay_7d_mean":         od_delay_7d_mean,       # exact OD from same flight number
            "flightnum_delay_14d_mean": flightnum_delay_14d_mean,
            "origin_delay_7d_mean":     origin_delay_7d_mean,   # airport arrivals preferred
            "dest_delay_7d_mean":       dest_delay_7d_mean,     # airport arrivals preferred
        }

        missing = [k for k, v in out.items() if v is None]
        if missing:
            raise RuntimeError(
                "AeroDataBoxHistory could not compute required metrics (after fallbacks): "
                + ", ".join(missing)
            )

        return {k: float(v) for k, v in out.items()}
