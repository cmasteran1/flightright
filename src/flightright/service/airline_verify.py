from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None
    PlaywrightTimeoutError = Exception


# ----------------------------------------------------------------------
# Result model
# ----------------------------------------------------------------------
@dataclass
class AirlineVerificationResult:
    exists: bool
    airline_source: str
    flight_number: str
    origin: str
    dest: str
    dep_date: str

    scheduled_departure_local: Optional[str] = None
    scheduled_arrival_local: Optional[str] = None
    scheduled_departure_utc: Optional[str] = None
    scheduled_arrival_utc: Optional[str] = None

    notes: List[str] = field(default_factory=list)
    raw_text_excerpt: Optional[str] = None


# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------
def verify_flight_on_airline_site(
    *,
    airline_iata: str,
    flight_number: str,
    origin: str,
    dest: str,
    dep_date: date,
    airport_timezones: Dict[str, str],
) -> AirlineVerificationResult:
    """
    Verify a future flight directly on an airline website.

    Current implementation priority:
      1) Southwest (WN) real scraping via Playwright
      2) AA / DL / UA placeholder fallbacks for later extension
    """
    airline_iata = (airline_iata or "").strip().upper()
    flight_number = str(flight_number).strip()
    origin = (origin or "").strip().upper()
    dest = (dest or "").strip().upper()

    logger.warning(
        "verify_flight_on_airline_site called airline=%s flight=%s origin=%s dest=%s dep_date=%s",
        airline_iata,
        flight_number,
        origin,
        dest,
        dep_date.isoformat(),
    )

    if airline_iata == "WN":
        return _verify_southwest(
            flight_number=flight_number,
            origin=origin,
            dest=dest,
            dep_date=dep_date,
            airport_timezones=airport_timezones,
        )

    if airline_iata == "AA":
        return _not_implemented_result(
            airline_source="american",
            airline_iata=airline_iata,
            flight_number=flight_number,
            origin=origin,
            dest=dest,
            dep_date=dep_date,
            note="American verifier not implemented yet.",
        )

    if airline_iata == "DL":
        return _not_implemented_result(
            airline_source="delta",
            airline_iata=airline_iata,
            flight_number=flight_number,
            origin=origin,
            dest=dest,
            dep_date=dep_date,
            note="Delta verifier not implemented yet.",
        )

    if airline_iata == "UA":
        return _not_implemented_result(
            airline_source="united",
            airline_iata=airline_iata,
            flight_number=flight_number,
            origin=origin,
            dest=dest,
            dep_date=dep_date,
            note="United verifier not implemented yet.",
        )

    return _not_implemented_result(
        airline_source="unsupported",
        airline_iata=airline_iata,
        flight_number=flight_number,
        origin=origin,
        dest=dest,
        dep_date=dep_date,
        note=f"No airline-site verifier implemented for {airline_iata}.",
    )


# ----------------------------------------------------------------------
# Southwest real scraper
# ----------------------------------------------------------------------
def _verify_southwest(
    *,
    flight_number: str,
    origin: str,
    dest: str,
    dep_date: date,
    airport_timezones: Dict[str, str],
) -> AirlineVerificationResult:
    logger.warning(
        "_verify_southwest called flight=%s origin=%s dest=%s dep_date=%s",
        flight_number,
        origin,
        dest,
        dep_date.isoformat(),
    )

    result = AirlineVerificationResult(
        exists=False,
        airline_source="southwest",
        flight_number=f"WN{flight_number}",
        origin=origin,
        dest=dest,
        dep_date=dep_date.isoformat(),
        notes=[],
    )

    if sync_playwright is None:
        result.notes.append("Playwright is not installed/importable in this runtime.")
        return result

    dep_tz = airport_timezones.get(origin)
    arr_tz = airport_timezones.get(dest)

    if not dep_tz:
        result.notes.append(f"Missing timezone for origin airport {origin}.")
        return result
    if not arr_tz:
        result.notes.append(f"Missing timezone for destination airport {dest}.")
        return result

    flight_status_url = "https://www.southwest.com/air/flight-status/"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 1200},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/136.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = context.new_page()

            result.notes.append(f"Opening {flight_status_url}")
            page.goto(flight_status_url, wait_until="domcontentloaded", timeout=60000)

            _dismiss_common_cookie_or_modal_ui(page, result.notes)

            filled = _southwest_fill_flight_status_form(
                page=page,
                flight_number=flight_number,
                origin=origin,
                dest=dest,
                dep_date=dep_date,
                notes=result.notes,
            )
            if not filled:
                result.notes.append("Could not confidently fill Southwest flight-status form.")
                result.raw_text_excerpt = _safe_page_excerpt(page)
                browser.close()
                return result

            _southwest_submit(page=page, notes=result.notes)

            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                result.notes.append("networkidle wait timed out after submit; continuing with DOM parse.")

            try:
                page.wait_for_timeout(2500)
            except Exception:
                pass

            body_text = _extract_page_text(page)
            result.raw_text_excerpt = body_text[:4000] if body_text else None

            found = _southwest_result_indicates_match(
                text=body_text,
                flight_number=flight_number,
                origin=origin,
                dest=dest,
            )
            if not found:
                result.notes.append("No convincing Southwest result match found in returned page text.")
                browser.close()
                return result

            dep_local, arr_local = _extract_southwest_times_from_text(body_text)
            if dep_local:
                result.scheduled_departure_local = dep_local
            if arr_local:
                result.scheduled_arrival_local = arr_local

            if dep_local:
                dep_dt_utc = _local_time_to_utc_iso(
                    dep_date=dep_date,
                    hhmm_ampm=dep_local,
                    tz_name=dep_tz,
                )
                result.scheduled_departure_utc = dep_dt_utc

            if arr_local:
                arr_dt_utc = _local_time_to_utc_iso_with_rollover(
                    dep_date=dep_date,
                    dep_time_ampm=dep_local,
                    arr_time_ampm=arr_local,
                    arr_tz_name=arr_tz,
                )
                result.scheduled_arrival_utc = arr_dt_utc

            result.exists = True
            result.notes.append("Southwest verifier found a matching result.")

            browser.close()
            return result

    except PlaywrightTimeoutError as e:
        result.notes.append(f"Playwright timeout: {e}")
        return result
    except Exception as e:
        logger.exception("Southwest verification failed")
        result.notes.append(f"Southwest verifier exception: {type(e).__name__}: {e}")
        return result


def _southwest_fill_flight_status_form(
    *,
    page,
    flight_number: str,
    origin: str,
    dest: str,
    dep_date: date,
    notes: List[str],
) -> bool:
    """
    Best-effort form fill for Southwest's official flight-status UI.

    We try multiple selectors because the exact DOM can drift.
    """
    date_strs = [
        dep_date.strftime("%m/%d/%Y"),
        dep_date.strftime("%m/%d/%y"),
        dep_date.strftime("%m-%d-%Y"),
    ]

    # Try route-based fields first.
    origin_selectors = [
        'input[name*="origin"]',
        'input[id*="origin"]',
        'input[aria-label*="Origin"]',
        'input[placeholder*="From"]',
    ]
    dest_selectors = [
        'input[name*="destination"]',
        'input[id*="destination"]',
        'input[aria-label*="Destination"]',
        'input[placeholder*="To"]',
    ]
    flightnum_selectors = [
        'input[name*="flightNumber"]',
        'input[id*="flightNumber"]',
        'input[aria-label*="Flight number"]',
        'input[placeholder*="Flight number"]',
    ]
    date_selectors = [
        'input[name*="date"]',
        'input[id*="date"]',
        'input[aria-label*="Date"]',
        'input[placeholder*="Date"]',
    ]

    route_filled = False
    if _fill_first(page, origin_selectors, origin, notes):
        route_filled = True
    if _fill_first(page, dest_selectors, dest, notes):
        route_filled = True

    # Prefer flight number field if available.
    flightnum_filled = _fill_first(page, flightnum_selectors, flight_number, notes)

    date_filled = False
    for ds in date_strs:
        if _fill_first(page, date_selectors, ds, notes):
            date_filled = True
            break

    # Sometimes date pickers resist direct fill; try JS value set.
    if not date_filled:
        for selector in date_selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() > 0:
                    locator.evaluate(
                        """(el, value) => {
                            el.value = value;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        }""",
                        date_strs[0],
                    )
                    notes.append(f"Set date via JS on selector: {selector}")
                    date_filled = True
                    break
            except Exception:
                continue

    notes.append(
        f"Southwest form fill summary route_filled={route_filled} "
        f"flightnum_filled={flightnum_filled} date_filled={date_filled}"
    )

    return (route_filled or flightnum_filled) and date_filled


def _southwest_submit(*, page, notes: List[str]) -> None:
    submit_selectors = [
        'button[type="submit"]',
        'button:has-text("Search")',
        'button:has-text("Check status")',
        'button:has-text("Check Status")',
        'input[type="submit"]',
    ]

    for selector in submit_selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() > 0:
                loc.click(timeout=5000)
                notes.append(f"Clicked submit selector: {selector}")
                return
        except Exception:
            continue

    notes.append("Could not find submit button; attempting Enter key.")
    try:
        page.keyboard.press("Enter")
    except Exception as e:
        notes.append(f"Enter key submit failed: {e}")


def _southwest_result_indicates_match(
    *,
    text: str,
    flight_number: str,
    origin: str,
    dest: str,
) -> bool:
    if not text:
        return False

    norm = _normalize_space(text).upper()

    flight_patterns = [
        f"SOUTHWEST {flight_number}",
        f"FLIGHT {flight_number}",
        f"WN {flight_number}",
        f"WN{flight_number}",
    ]
    has_flight = any(p in norm for p in flight_patterns)

    has_route_codes = origin.upper() in norm and dest.upper() in norm

    city_arrow_patterns = [
        f"{origin.upper()} TO {dest.upper()}",
        f"{origin.upper()} - {dest.upper()}",
    ]
    has_route_phrase = any(p in norm for p in city_arrow_patterns)

    return has_flight and (has_route_codes or has_route_phrase)


def _extract_southwest_times_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Very forgiving text parser.

    We look for time pairs first, then fallback to first 2 plausible times.
    """
    if not text:
        return None, None

    norm = _normalize_space(text)

    # Common pair patterns like "6:15 AM - 8:40 AM"
    pair_patterns = [
        r'(\d{1,2}:\d{2}\s?(?:AM|PM))\s*[-–]\s*(\d{1,2}:\d{2}\s?(?:AM|PM))',
        r'Departs?\s*(\d{1,2}:\d{2}\s?(?:AM|PM)).{0,60}?Arrives?\s*(\d{1,2}:\d{2}\s?(?:AM|PM))',
        r'Departure\s*(\d{1,2}:\d{2}\s?(?:AM|PM)).{0,60}?Arrival\s*(\d{1,2}:\d{2}\s?(?:AM|PM))',
    ]
    for pat in pair_patterns:
        m = re.search(pat, norm, flags=re.IGNORECASE)
        if m:
            return _clean_ampm_time(m.group(1)), _clean_ampm_time(m.group(2))

    all_times = re.findall(r'\b\d{1,2}:\d{2}\s?(?:AM|PM)\b', norm, flags=re.IGNORECASE)
    cleaned = [_clean_ampm_time(t) for t in all_times]
    cleaned = [t for t in cleaned if t]

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    if len(deduped) >= 2:
        return deduped[0], deduped[1]

    if len(deduped) == 1:
        return deduped[0], None

    return None, None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _not_implemented_result(
    *,
    airline_source: str,
    airline_iata: str,
    flight_number: str,
    origin: str,
    dest: str,
    dep_date: date,
    note: str,
) -> AirlineVerificationResult:
    return AirlineVerificationResult(
        exists=False,
        airline_source=airline_source,
        flight_number=f"{airline_iata}{flight_number}",
        origin=origin,
        dest=dest,
        dep_date=dep_date.isoformat(),
        notes=[note],
    )


def _dismiss_common_cookie_or_modal_ui(page, notes: List[str]) -> None:
    selectors = [
        'button:has-text("Accept")',
        'button:has-text("I Accept")',
        'button:has-text("Agree")',
        'button:has-text("Continue")',
        'button[aria-label*="close"]',
        'button:has-text("Close")',
    ]
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() > 0 and loc.is_visible():
                loc.click(timeout=2000)
                notes.append(f"Dismissed modal/cookie UI with selector: {selector}")
        except Exception:
            continue


def _fill_first(page, selectors: List[str], value: str, notes: List[str]) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            loc.click(timeout=2000)
            try:
                loc.fill("")
            except Exception:
                pass
            loc.fill(value, timeout=3000)
            notes.append(f"Filled selector {selector} with value {value}")
            return True
        except Exception:
            continue
    return False


def _extract_page_text(page) -> str:
    try:
        body = page.locator("body").inner_text(timeout=5000)
        return _normalize_space(body)
    except Exception:
        return ""


def _safe_page_excerpt(page) -> str:
    text = _extract_page_text(page)
    return text[:4000] if text else ""


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _clean_ampm_time(s: str) -> Optional[str]:
    if not s:
        return None
    s = _normalize_space(s).upper()
    s = s.replace("A M", "AM").replace("P M", "PM")
    m = re.match(r'^(\d{1,2}):(\d{2})\s?(AM|PM)$', s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ampm = m.group(3)
    if not (1 <= hh <= 12 and 0 <= mm <= 59):
        return None
    return f"{hh}:{mm:02d} {ampm}"


def _local_time_to_utc_iso(
    *,
    dep_date: date,
    hhmm_ampm: str,
    tz_name: str,
) -> Optional[str]:
    try:
        from zoneinfo import ZoneInfo

        local_dt = datetime.strptime(
            f"{dep_date.isoformat()} {hhmm_ampm}",
            "%Y-%m-%d %I:%M %p",
        ).replace(tzinfo=ZoneInfo(tz_name))
        return local_dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _local_time_to_utc_iso_with_rollover(
    *,
    dep_date: date,
    dep_time_ampm: Optional[str],
    arr_time_ampm: str,
    arr_tz_name: str,
) -> Optional[str]:
    """
    Convert arrival local time to UTC, allowing next-day rollover when
    arrival appears earlier than departure in local clock terms.
    """
    try:
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        arr_local = datetime.strptime(
            f"{dep_date.isoformat()} {arr_time_ampm}",
            "%Y-%m-%d %I:%M %p",
        ).replace(tzinfo=ZoneInfo(arr_tz_name))

        if dep_time_ampm:
            dep_naive = datetime.strptime(
                f"{dep_date.isoformat()} {dep_time_ampm}",
                "%Y-%m-%d %I:%M %p",
            )
            arr_naive = datetime.strptime(
                f"{dep_date.isoformat()} {arr_time_ampm}",
                "%Y-%m-%d %I:%M %p",
            )
            if arr_naive < dep_naive:
                arr_local = arr_local + timedelta(days=1)

        return arr_local.astimezone(timezone.utc).isoformat()
    except Exception:
        return None