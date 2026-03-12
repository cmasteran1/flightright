// ---------- helpers ----------
function roundMinutes(x){
  if (x === null || x === undefined) return null;
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  return Math.round(n);
}

function roundProb(x){
  if (x === null || x === undefined) return null;
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 1000) / 1000;
}

function clamp01(x){
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function normalizeSeverity(x){
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.min(4, n)) / 4;
}

function fmtPct(p){
  const n = clamp01(p);
  return (Math.round(n * 1000) / 10).toFixed(1) + "%";
}

function safeText(s){
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
  }[c]));
}

function normalizeForLookup(s){
  return String(s ?? "")
    .toLowerCase()
    .replace(/[()\-/,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractCodeFromText(value, codeLength){
  const upper = String(value ?? "").toUpperCase();
  const direct = upper.match(new RegExp(`\\b([A-Z0-9]{${codeLength}})\\b`));
  return direct ? direct[1] : null;
}

function normalizePredictError(status, data, fallbackMessage){
  const raw =
    String(
      data?.user_message ||
      data?.detail?.user_message ||
      data?.detail ||
      data?.error ||
      fallbackMessage ||
      ""
    ).trim();

  const errorCode = data?.error_code || data?.detail?.error_code || "";
  const warning = data?.warning || data?.detail?.warning || "";
  const userTitle = data?.user_title || data?.detail?.user_title;
  const userMessage = data?.user_message || data?.detail?.user_message;
  const userAction = data?.user_action || data?.detail?.user_action;
  const needsScheduleInputs = Boolean(
    data?.needs_schedule_inputs || data?.detail?.needs_schedule_inputs
  );

  const lower = raw.toLowerCase();

  if (errorCode === "flight_not_found_needs_schedule_times" || needsScheduleInputs) {
    return {
      title: userTitle || "We couldn’t confirm that flight automatically",
      message: userMessage || "We could not find this flight automatically, but it may still exist.",
      action: userAction || "Enter the local scheduled departure and arrival times from your itinerary, then try again.",
      warning: warning || "Flight was not found automatically, but it may still exist.",
      needsScheduleInputs: true
    };
  }

  if (userTitle || userMessage || userAction) {
    return {
      title: userTitle || "We couldn’t run this prediction",
      message: userMessage || "We were not able to generate a delay estimate for this flight.",
      action: userAction || "Please double-check the flight details and try again."
    };
  }

  if (status === 400) {
    if (lower.includes("same-day") || lower.includes("same day") || lower.includes("today")) {
      return {
        title: "Prediction not available for today",
        message: "This tool works best for flights scheduled for tomorrow or later, and could not produce a reliable prediction for this date.",
        action: "Try checking the same flight for a later date."
      };
    }

    if (lower.includes("flight not found") || lower.includes("not found")) {
      return {
        title: "We couldn’t find that flight automatically",
        message: "The flight details did not match a scheduled flight in our automatic lookup. The flight may still exist.",
        action: "Check the airline code, flight number, origin, destination, and date. If you know the scheduled times, enter them and try again."
      };
    }

    return {
      title: "Please check the flight details",
      message: "We couldn’t run the prediction because some of the flight information did not match what the system expected.",
      action: "Make sure the airline, flight number, airports, and date are correct, then try again."
    };
  }

  if (status === 404) {
    return {
      title: "We couldn’t find data for this flight",
      message: "The system could not find the flight or the supporting data needed to make a prediction.",
      action: "Verify the flight details or try a different date."
    };
  }

  if (status === 422) {
    return {
      title: "Not enough information to make a prediction",
      message: raw || "We found the request, but there was not enough valid data to generate a reliable result.",
      action: "Try a different date or check back later."
    };
  }

  if (status === 502 || status === 503 || status === 504) {
    return {
      title: "Temporary service problem",
      message: "FlightRight had trouble reaching the prediction service.",
      action: "Please try again in a minute."
    };
  }

  return {
    title: "We couldn’t run this prediction",
    message: raw || "We were not able to generate a delay estimate for this flight.",
    action: "Please try again. If the problem continues, double-check the flight details."
  };
}

function renderPredictError(err){
  const title = safeText(err?.title || "We couldn’t run this prediction");
  const message = safeText(err?.message || "We were not able to generate a delay estimate.");
  const action = safeText(err?.action || "Please try again.");

  return `
    <div class="empty" role="alert" aria-live="polite">
      <div style="font-weight:700; font-size:1.05rem; margin-bottom:8px;">${title}</div>
      <div style="margin-bottom:8px;">${message}</div>
      <div class="small muted">${action}</div>
    </div>
  `;
}

function todayISO(){
  const d = new Date();
  const tz = d.getTimezoneOffset() * 60000;
  return new Date(d - tz).toISOString().slice(0,10);
}

function addDaysISO(n){
  const d = new Date();
  d.setDate(d.getDate() + n);
  const tz = d.getTimezoneOffset() * 60000;
  return new Date(d - tz).toISOString().slice(0,10);
}

function wxIconSvg(kind){
  const base = `
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  `;
  const end = `</svg>`;
  if (kind === "clear") {
    return base + `
      <circle cx="12" cy="12" r="4.5" stroke="rgba(255,255,255,.9)" stroke-width="1.6"/>
      <path d="M12 1.8v3.2M12 19v3.2M1.8 12h3.2M19 12h3.2M4.2 4.2l2.3 2.3M17.5 17.5l2.3 2.3M19.8 4.2l-2.3 2.3M6.5 17.5l-2.3 2.3"
        stroke="rgba(255,255,255,.75)" stroke-width="1.6" stroke-linecap="round"/>
    ` + end;
  }
  if (kind === "cloud") {
    return base + `
      <path d="M7.5 18.2h9.2a3.6 3.6 0 0 0 .4-7.2A5.3 5.3 0 0 0 6.7 9.2a3.7 3.7 0 0 0 .8 9z"
        stroke="rgba(255,255,255,.85)" stroke-width="1.6" stroke-linejoin="round"/>
    ` + end;
  }
  if (kind === "rain") {
    return base + `
      <path d="M7.2 16.8h9.6a3.2 3.2 0 0 0 .3-6.4A4.9 4.9 0 0 0 7 9.1a3.4 3.4 0 0 0 .2 7.7z"
        stroke="rgba(255,255,255,.85)" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9 19.2l-.9 2.2M12 19.2l-.9 2.2M15 19.2l-.9 2.2"
        stroke="rgba(124,92,255,.95)" stroke-width="1.6" stroke-linecap="round"/>
    ` + end;
  }
  if (kind === "storm") {
    return base + `
      <path d="M7.2 15.9h9.6a3.2 3.2 0 0 0 .3-6.4A4.9 4.9 0 0 0 7 8.2a3.4 3.4 0 0 0 .2 7.7z"
        stroke="rgba(255,255,255,.85)" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M12.2 16.2l-1.3 3h2.2l-1.4 3.2"
        stroke="rgba(247,201,72,.95)" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
    ` + end;
  }
  if (kind === "snow") {
    return base + `
      <path d="M7.2 16.8h9.6a3.2 3.2 0 0 0 .3-6.4A4.9 4.9 0 0 0 7 9.1a3.4 3.4 0 0 0 .2 7.7z"
        stroke="rgba(255,255,255,.85)" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9.5 19.3h0M12 20.2h0M14.5 19.3h0"
        stroke="rgba(56,211,159,.95)" stroke-width="4" stroke-linecap="round"/>
    ` + end;
  }
  if (kind === "fog") {
    return base + `
      <path d="M6.2 12.2h11.6M5 15.2h14M7.4 18.2h9.2"
        stroke="rgba(255,255,255,.75)" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M8 10.2h8"
        stroke="rgba(255,255,255,.5)" stroke-width="1.6" stroke-linecap="round"/>
    ` + end;
  }
  return base + `
    <path d="M12 7v6" stroke="rgba(255,255,255,.75)" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M12 17.2h.01" stroke="rgba(255,255,255,.75)" stroke-width="4" stroke-linecap="round"/>
  ` + end;
}

function wxKindFromCode(code){
  const c = Number(code);
  if (!Number.isFinite(c)) return "unknown";
  if (c === 0) return "clear";
  if ([1,2,3].includes(c)) return "cloud";
  if ([45,48].includes(c)) return "fog";
  if ((c >= 51 && c <= 67) || (c >= 80 && c <= 82)) return "rain";
  if (c >= 71 && c <= 77) return "snow";
  if (c >= 95) return "storm";
  return "cloud";
}

function bandFromRisk100(score100){
  const s = Number(score100);
  if (!Number.isFinite(s)) return { name: "Unknown", cls: "bandCaution" };
  if (s < 20) return { name: "Safe", cls: "bandSafe" };
  if (s < 40) return { name: "Mild", cls: "bandCaution" };
  if (s < 60) return { name: "Caution", cls: "bandRisk" };
  return { name: "High risk", cls: "bandHigh" };
}

function computeRiskScoreFromBuckets(labels, probs){
  if (!Array.isArray(labels) || !Array.isArray(probs) || labels.length !== probs.length) {
    return null;
  }

  const weightMap = {
    "< 15 min": 0.00,
    "15–30 min": 0.25,
    "15-30 min": 0.25,
    "30–45 min": 0.50,
    "30-45 min": 0.50,
    "45–60 min": 0.75,
    "45-60 min": 0.75,
    "≥ 60 min": 1.00,
    ">= 60 min": 1.00
  };

  let total = 0;
  let probSum = 0;

  for (let i = 0; i < labels.length; i++) {
    const label = String(labels[i] ?? "").trim();
    const p = clamp01(probs[i]);
    const w = weightMap[label];
    if (!Number.isFinite(w)) continue;
    total += p * w;
    probSum += p;
  }

  if (probSum <= 0) return null;
  return Math.round(total * 100);
}

function normalizeIata(x){
  return String(x ?? "").trim().toUpperCase();
}

function normalizeFlightnum(x){
  return String(x ?? "").trim();
}

function extractRequestedDepartureLocal(resp, input){
  function isIsoLocalDateTime(value) {
    return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(value);
  }

  function isDateOnly(value) {
    return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value);
  }

  function isTimeOnly(value) {
    return typeof value === "string" && /^\d{2}:\d{2}(:\d{2})?$/.test(value);
  }

  function recursiveCollect(obj, matches, seen = new Set()) {
    if (!obj || typeof obj !== "object") return;
    if (seen.has(obj)) return;
    seen.add(obj);

    for (const [key, value] of Object.entries(obj)) {
      const lowerKey = key.toLowerCase();

      if (
        lowerKey.includes("scheduled_departure_local") ||
        lowerKey.includes("sched_departure_local") ||
        lowerKey.includes("departure_local") ||
        lowerKey.includes("dep_local")
      ) {
        if (isIsoLocalDateTime(value)) matches.iso.push(value);
      }

      if (
        lowerKey.includes("sched_dep_time_24h") ||
        lowerKey.includes("scheduled_departure_time") ||
        lowerKey.includes("departure_time_local") ||
        lowerKey === "dep_time" ||
        lowerKey === "departure_time"
      ) {
        if (isTimeOnly(value)) matches.time.push(value);
      }

      if (lowerKey === "date" || lowerKey.endsWith("_date")) {
        if (isDateOnly(value)) matches.date.push(value);
      }

      if (value && typeof value === "object") {
        recursiveCollect(value, matches, seen);
      }
    }
  }

  function utcToLocalIso(utcValue, timeZone) {
    if (!utcValue || !timeZone) return null;

    const d = new Date(utcValue);
    if (Number.isNaN(d.getTime())) return null;

    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23"
    }).formatToParts(d);

    const get = (type) => parts.find((p) => p.type === type)?.value;
    const year = get("year");
    const month = get("month");
    const day = get("day");
    const hour = get("hour");
    const minute = get("minute");
    const second = get("second");

    if (!year || !month || !day || !hour || !minute || !second) return null;
    return `${year}-${month}-${day}T${hour}:${minute}:${second}`;
  }

  const directCandidates = [
    resp?.flight?.scheduled_departure_local,
    resp?.flight?.sched_departure_local,
    resp?.verification?.scheduled_departure_local,
    resp?.verification?.sched_departure_local,
    resp?.resolved_flight?.scheduled_departure_local,
    resp?.resolved_flight?.sched_departure_local,
    resp?.tabs?.flight?.scheduled_departure_local,
    resp?.tabs?.flight?.sched_departure_local,
    resp?.scheduled_departure_local,
    resp?.sched_departure_local,
    resp?.departure_local,
    resp?.dep_local
  ];

  for (const c of directCandidates) {
    if (isIsoLocalDateTime(c)) {
      return c.length === 16 ? `${c}:00` : c;
    }
  }

  const matches = { iso: [], date: [], time: [] };
  recursiveCollect(resp, matches);

  if (matches.iso.length) {
    const value = matches.iso[0];
    return value.length === 16 ? `${value}:00` : value;
  }

  const originCode =
    resp?.request?.Origin ||
    resp?.request?.origin ||
    input?.origin ||
    null;

  const originTz =
    originCode && lookupState?.airports?.[originCode]
      ? lookupState.airports[originCode].timezone
      : null;

  const utcCandidate =
    resp?.request?.scheduled_departure_utc ||
    resp?.request?.sched_departure_utc ||
    null;

  const localFromUtc = utcToLocalIso(utcCandidate, originTz);
  if (localFromUtc) {
    return localFromUtc;
  }

  const dateCandidates = [
    ...matches.date,
    resp?.request?.flight_date,
    resp?.date,
    resp?.flight?.date,
    resp?.verification?.date,
    resp?.resolved_flight?.date,
    input?.date
  ].filter(isDateOnly);

  const timeCandidates = [
    ...matches.time,
    resp?.sched_dep_time_24h,
    resp?.flight?.sched_dep_time_24h,
    resp?.verification?.sched_dep_time_24h,
    resp?.resolved_flight?.sched_dep_time_24h,
    input?.sched_dep_time_24h
  ].filter(isTimeOnly);

  const resolvedDate = dateCandidates[0] || null;
  const resolvedTime = timeCandidates[0] || null;

  if (resolvedDate && resolvedTime) {
    const hhmmss = resolvedTime.length === 5 ? `${resolvedTime}:00` : resolvedTime;
    return `${resolvedDate}T${hhmmss}`;
  }

  return null;
}
function extractPredictionPills(predictionPayload){
  const p = predictionPayload?.prediction || predictionPayload || {};
  const labels = p.bin_labels ?? p.labels ?? null;
  const probs = p.bin_proba ?? p.bin_probs ?? p.probs ?? null;

  if (Array.isArray(labels) && Array.isArray(probs) && labels.length === probs.length) {
    return labels.map((label, idx) => ({
      label: String(label),
      value: fmtPct(probs[idx])
    }));
  }

  return [];
}

function formatMoney(value, currency){
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${safeText(currency || "USD")} ${Math.round(n)}`;
}

function formatStops(value){
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "Nonstop";
  if (n === 1) return "1 stop";
  return `${n} stops`;
}
function formatDurationMinutes(value){
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";

  const hours = Math.floor(n / 60);
  const mins = n % 60;

  if (hours > 0 && mins > 0) return `${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h`;
  return `${mins} min`;
}
function formatLocalDateTime(value){
  if (!value) return "—";

  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return safeText(value);

  const dateText = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric"
  }).format(d);

  const timeText = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit"
  }).format(d);

  return `${dateText} · ${timeText}`;
}

function formatTimeDifference(minutes){
  const n = Number(minutes);
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "Same time";

  const abs = Math.abs(n);
  const hours = Math.floor(abs / 60);
  const mins = abs % 60;

  let duration = "";
  if (hours > 0 && mins > 0) {
    duration = `${hours}h ${mins}m`;
  } else if (hours > 0) {
    duration = `${hours}h`;
  } else {
    duration = `${mins} min`;
  }

  return n < 0 ? `${duration} earlier` : `${duration} later`;
}
function sortSimilarFlightRows(rows, sortKey){
  const items = Array.isArray(rows) ? [...rows] : [];

  const parseTime = (value) => {
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? Number.POSITIVE_INFINITY : d.getTime();
  };

  const parsePrice = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY;
  };

  const parseAirline = (row) =>
    String(row?.marketing_carrier || row?.airline_iata || "").toLowerCase();

  if (sortKey === "date_desc") {
    items.sort((a, b) => parseTime(b?.departure_local) - parseTime(a?.departure_local));
    return items;
  }

  if (sortKey === "price_asc") {
    items.sort((a, b) => parsePrice(a?.price) - parsePrice(b?.price));
    return items;
  }

  if (sortKey === "price_desc") {
    items.sort((a, b) => parsePrice(b?.price) - parsePrice(a?.price));
    return items;
  }

  if (sortKey === "airline_asc") {
    items.sort((a, b) => {
      const cmp = parseAirline(a).localeCompare(parseAirline(b));
      if (cmp !== 0) return cmp;
      return parseTime(a?.departure_local) - parseTime(b?.departure_local);
    });
    return items;
  }

  if (sortKey === "airline_desc") {
    items.sort((a, b) => {
      const cmp = parseAirline(b).localeCompare(parseAirline(a));
      if (cmp !== 0) return cmp;
      return parseTime(a?.departure_local) - parseTime(b?.departure_local);
    });
    return items;
  }

  // default: earliest departure first
  items.sort((a, b) => parseTime(a?.departure_local) - parseTime(b?.departure_local));
  return items;
}
// ---------- lookup state ----------
const lookupState = {
  loaded: false,
  airlines: [],
  airports: {},
  allowed_airports_by_airline: {},
  airlineAliasToCode: new Map(),
  airportAliasToCode: new Map(),
};

function registerAirlineAliases(airline){
  const candidates = [
    airline.code,
    airline.name,
    `${airline.name} (${airline.code})`,
    ...(Array.isArray(airline.aliases) ? airline.aliases : []),
  ];

  for (const c of candidates) {
    const key = normalizeForLookup(c);
    if (key) lookupState.airlineAliasToCode.set(key, airline.code);
  }
}

function registerAirportAliases(ap){
  const candidates = [
    ap.code,
    ap.name,
    ap.city,
    `${ap.city}, ${ap.state}`,
    `${ap.city} (${ap.code})`,
    `${ap.code} — ${ap.name}`,
    `${ap.city} (${ap.code}) — ${ap.name}`,
    ap.display,
  ];

  for (const c of candidates) {
    const key = normalizeForLookup(c);
    if (key) lookupState.airportAliasToCode.set(key, ap.code);
  }
}

function resolveAirlineCode(value){
  const raw = String(value ?? "").trim();
  if (!raw) return "";

  const maybeCode = extractCodeFromText(raw, 2);
  if (maybeCode && lookupState.airlines.some((a) => a.code === maybeCode)) {
    return maybeCode;
  }

  const key = normalizeForLookup(raw);
  return lookupState.airlineAliasToCode.get(key) || "";
}

function getAllowedAirportsForAirline(airlineCode){
  const code = String(airlineCode ?? "").toUpperCase();
  const allowed = lookupState.allowed_airports_by_airline?.[code];
  if (Array.isArray(allowed) && allowed.length) return allowed;
  return Object.keys(lookupState.airports || {});
}

function resolveAirportCode(value, airlineCode){
  const raw = String(value ?? "").trim();
  if (!raw) return "";

  const allowed = new Set(getAllowedAirportsForAirline(airlineCode));

  const maybeCode = extractCodeFromText(raw, 3);
  if (maybeCode && allowed.has(maybeCode)) {
    return maybeCode;
  }

  const key = normalizeForLookup(raw);
  const aliasCode = lookupState.airportAliasToCode.get(key);
  if (aliasCode && allowed.has(aliasCode)) {
    return aliasCode;
  }

  const plain = normalizeIata(raw);
  if (allowed.has(plain)) {
    return plain;
  }

  return "";
}

function getAirlineMatches(query){
  const q = normalizeForLookup(query);
  const rows = lookupState.airlines.map((a) => ({
    code: a.code,
    title: `${a.code} — ${a.name}`,
    subtitle: Array.isArray(a.aliases) ? a.aliases.join(" · ") : "",
    searchText: normalizeForLookup([a.code, a.name, ...(a.aliases || [])].join(" ")),
  }));

  if (!q) return rows.slice(0, 8);

  return rows
    .filter((r) => r.searchText.includes(q))
    .slice(0, 8);
}

function getAirportMatches(query, airlineCode){
  const q = normalizeForLookup(query);
  const allowed = getAllowedAirportsForAirline(airlineCode);

  const rows = allowed
    .map((code) => lookupState.airports?.[code])
    .filter(Boolean)
    .map((ap) => ({
      code: ap.code,
      title: `${ap.code} — ${ap.city || ap.name || ap.code}`,
      subtitle: ap.name && ap.city ? ap.name : (ap.state ? `${ap.city}, ${ap.state}` : ap.name),
      searchText: normalizeForLookup([
        ap.code,
        ap.name,
        ap.city,
        ap.state,
        ap.display
      ].join(" "))
    }));

  if (!q) return rows.slice(0, 10);

  return rows
    .filter((r) => r.searchText.includes(q))
    .slice(0, 10);
}

let activeMenu = null;

function closeAllSuggestionMenus(){
  document.querySelectorAll(".suggestions").forEach((el) => {
    el.classList.remove("open");
    el.innerHTML = "";
  });
  activeMenu = null;
}

function renderSuggestionMenu(menuEl, items, onSelect){
  menuEl.innerHTML = "";

  if (!items.length) {
    menuEl.classList.remove("open");
    if (activeMenu === menuEl) activeMenu = null;
    return;
  }

  items.forEach((item, idx) => {
    const row = document.createElement("div");
    row.className = "suggestionItem";
    if (idx === 0) row.classList.add("active");
    row.dataset.code = item.code || "";
    row.innerHTML = `
      <div class="suggestionTop">${safeText(item.title)}</div>
      <div class="suggestionSub">${safeText(item.subtitle || "")}</div>
    `;
    row.addEventListener("mousedown", (e) => {
      e.preventDefault();
      onSelect(item);
      closeAllSuggestionMenus();
    });
    menuEl.appendChild(row);
  });

  menuEl.classList.add("open");
  activeMenu = menuEl;
}

function moveActiveSuggestion(menuEl, delta){
  const items = Array.from(menuEl.querySelectorAll(".suggestionItem"));
  if (!items.length) return;

  let idx = items.findIndex((el) => el.classList.contains("active"));
  if (idx < 0) idx = 0;

  items[idx].classList.remove("active");
  idx = (idx + delta + items.length) % items.length;
  items[idx].classList.add("active");
  items[idx].scrollIntoView({ block: "nearest" });
}

function selectActiveSuggestion(menuEl){
  const active = menuEl.querySelector(".suggestionItem.active");
  if (active) {
    active.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
  }
}

async function loadLookups(){
  try {
    const r = await fetch("/api/lookups", { cache: "no-store" });
    if (!r.ok) return;
    const data = await r.json();
    if (!data?.ok) return;

    lookupState.loaded = true;
    lookupState.airlines = Array.isArray(data.airlines) ? data.airlines : [];
    lookupState.airports = data.airports || {};
    lookupState.allowed_airports_by_airline = data.allowed_airports_by_airline || {};

    lookupState.airlineAliasToCode.clear();
    lookupState.airportAliasToCode.clear();

    for (const airline of lookupState.airlines) registerAirlineAliases(airline);
    for (const ap of Object.values(lookupState.airports)) registerAirportAliases(ap);
  } catch (_e) {
    // Graceful fallback: user can still enter codes manually.
  }
}

// ---------- state ----------
const state = {
  activeTab: "prediction",
  lastInput: null,
  lastResponse: null,
  needsScheduleFallback: false,
  similarFlights: null,
  similarFlightsLoading: false,
  similarFlightsError: null,
  similarFlightsRequested: false,
    similarFlightsSort: "date_asc",
  similarFlightsVisibleCount: 5
};

// ---------- DOM ----------
const apiLabel = document.getElementById("apiLabel");
const apiDot = document.getElementById("apiDot");
const routeLabel = document.getElementById("routeLabel");
const content = document.getElementById("content");
const formError = document.getElementById("formError");

const airlineEl = document.getElementById("airline");
const flightnumEl = document.getElementById("flightnum");
const dateEl = document.getElementById("date");
const originEl = document.getElementById("origin");
const destEl = document.getElementById("dest");
const includeEl = document.getElementById("include");
const goBtn = document.getElementById("goBtn");

const airlineSuggestionsEl = document.getElementById("airlineSuggestions");
const originSuggestionsEl = document.getElementById("originSuggestions");
const destSuggestionsEl = document.getElementById("destSuggestions");

const scheduleFallbackBoxEl = document.getElementById("scheduleFallbackBox");
const scheduleFallbackWarningEl = document.getElementById("scheduleFallbackWarning");
const schedDepTimeEl = document.getElementById("schedDepTime");
const schedArrTimeEl = document.getElementById("schedArrTime");

function setScheduleFallbackVisible(on, warningText = ""){
  state.needsScheduleFallback = !!on;
  scheduleFallbackBoxEl.style.display = on ? "block" : "none";

  if (warningText) {
    scheduleFallbackWarningEl.textContent = warningText;
  } else {
    scheduleFallbackWarningEl.textContent =
      "Warning: this flight was not found automatically, but it may still exist. Enter the scheduled local departure and arrival times from your itinerary to continue.";
  }
}

function resetScheduleFallback(){
  state.needsScheduleFallback = false;
  scheduleFallbackBoxEl.style.display = "none";
  scheduleFallbackWarningEl.textContent =
    "Warning: this flight was not found automatically, but it may still exist. Enter the scheduled local departure and arrival times from your itinerary to continue.";
}

function resetSimilarFlightsState(){
  state.similarFlights = null;
  state.similarFlightsLoading = false;
  state.similarFlightsError = null;
  state.similarFlightsRequested = false;
  state.similarFlightsVisibleCount = 5;
}

function showAirlineSuggestions(){
  if (!lookupState.loaded) return;
  const matches = getAirlineMatches(airlineEl.value);
  renderSuggestionMenu(airlineSuggestionsEl, matches, (item) => {
    airlineEl.value = item.code;
    const code = item.code;
    const originCode = resolveAirportCode(originEl.value, code);
    const destCode = resolveAirportCode(destEl.value, code);
    if (originCode) originEl.value = originCode;
    if (destCode) destEl.value = destCode;
  });
}

function showOriginSuggestions(){
  if (!lookupState.loaded) return;
  const airlineCode = resolveAirlineCode(airlineEl.value);
  const matches = getAirportMatches(originEl.value, airlineCode);
  renderSuggestionMenu(originSuggestionsEl, matches, (item) => {
    originEl.value = item.code;
  });
}

function showDestSuggestions(){
  if (!lookupState.loaded) return;
  const airlineCode = resolveAirlineCode(airlineEl.value);
  const matches = getAirportMatches(destEl.value, airlineCode);
  renderSuggestionMenu(destSuggestionsEl, matches, (item) => {
    destEl.value = item.code;
  });
}

function wireAutocomplete(inputEl, menuEl, showFn, resolveFn){
  inputEl.addEventListener("focus", () => {
    showFn();
  });

  inputEl.addEventListener("input", () => {
    showFn();
  });

  inputEl.addEventListener("keydown", (e) => {
    if (!menuEl.classList.contains("open")) {
      if (e.key === "ArrowDown") {
        showFn();
        e.preventDefault();
      }
      return;
    }

    if (e.key === "ArrowDown") {
      moveActiveSuggestion(menuEl, 1);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      moveActiveSuggestion(menuEl, -1);
      e.preventDefault();
    } else if (e.key === "Enter") {
      selectActiveSuggestion(menuEl);
      e.preventDefault();
    } else if (e.key === "Escape") {
      closeAllSuggestionMenus();
      e.preventDefault();
    }
  });

  inputEl.addEventListener("blur", () => {
    setTimeout(() => {
      if (resolveFn) {
        const resolved = resolveFn(inputEl.value);
        if (resolved) inputEl.value = resolved;
      }
      if (activeMenu === menuEl) closeAllSuggestionMenus();
    }, 120);
  });
}

// default date = tomorrow
dateEl.value = addDaysISO(1);

wireAutocomplete(
  airlineEl,
  airlineSuggestionsEl,
  showAirlineSuggestions,
  (value) => resolveAirlineCode(value)
);

wireAutocomplete(
  originEl,
  originSuggestionsEl,
  showOriginSuggestions,
  (value) => resolveAirportCode(value, resolveAirlineCode(airlineEl.value))
);

wireAutocomplete(
  destEl,
  destSuggestionsEl,
  showDestSuggestions,
  (value) => resolveAirportCode(value, resolveAirlineCode(airlineEl.value))
);

document.addEventListener("mousedown", (e) => {
  if (!e.target.closest(".autocomplete")) {
    closeAllSuggestionMenus();
  }
});

// Tabs
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", async () => {
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    state.activeTab = tab.dataset.tab;

    if (state.activeTab === "similar" && state.lastResponse && !state.similarFlightsRequested) {
      await fetchSimilarFlights();
      return;
    }

    render();
  });
});

// ---------- API status ----------
async function refreshApiStatus(){
  apiLabel.textContent = "checking…";
  apiDot.className = "dot warn";
  try{
    const r = await fetch("/api/health", { cache: "no-store" });
    if (!r.ok) throw new Error("bad status " + r.status);
    apiLabel.textContent = "online";
    apiDot.className = "dot ok";
  }catch(e){
    apiLabel.textContent = "offline";
    apiDot.className = "dot bad";
  }
}
refreshApiStatus();
setInterval(refreshApiStatus, 15000);
loadLookups();

// ---------- request ----------
function buildPayload(){
  const airline = lookupState.loaded ? resolveAirlineCode(airlineEl.value) : normalizeIata(airlineEl.value);
  const flightnum = normalizeFlightnum(flightnumEl.value);
  const date = dateEl.value;
  const origin = lookupState.loaded
    ? resolveAirportCode(originEl.value, airline)
    : normalizeIata(originEl.value);
  const dest = lookupState.loaded
    ? resolveAirportCode(destEl.value, airline)
    : normalizeIata(destEl.value);

  if (!airline || airline.length !== 2) {
    return { error: "Please choose a valid airline. Current supported airlines are WN, UA, DL, and AA." };
  }
  if (!flightnum) return { error: "Flight number is required." };
  if (!date) return { error: "Date is required." };
  if (!origin || origin.length !== 3) {
    return { error: "Please choose a valid origin airport for the selected airline." };
  }
  if (!dest || dest.length !== 3) {
    return { error: "Please choose a valid destination airport for the selected airline." };
  }

  const sched_dep_time_24h = String(schedDepTimeEl.value || "").trim();
  const sched_arr_time_24h = String(schedArrTimeEl.value || "").trim();

  if (state.needsScheduleFallback) {
    if (!sched_dep_time_24h) {
      return { error: "Scheduled departure time is required because the flight was not found automatically." };
    }
    if (!sched_arr_time_24h) {
      return { error: "Scheduled arrival time is required because the flight was not found automatically." };
    }
  }

  const includeMode = includeEl.value;
  const include = (includeMode === "minimal")
    ? {}
    : {
        weather: true,
        flight_history: true,
        airport_stats: true,
        airline_stats: true
      };

  const payload = {
    airline,
    flightnum,
    date,
    origin,
    dest,
    include
  };

  if (sched_dep_time_24h) payload.sched_dep_time_24h = sched_dep_time_24h;
  if (sched_arr_time_24h) payload.sched_arr_time_24h = sched_arr_time_24h;

  airlineEl.value = airline;
  originEl.value = origin;
  destEl.value = dest;

  return payload;
}

async function runPredict(){
  formError.style.display = "none";
  const payload = buildPayload();
  if (payload.error){
    formError.textContent = payload.error;
    formError.style.display = "block";
    return;
  }

  state.lastInput = payload;
  state.lastResponse = null;
  resetSimilarFlightsState();
  routeLabel.textContent = `${payload.origin}→${payload.dest} · ${payload.date}`;

  goBtn.disabled = true;
  goBtn.textContent = "Checking…";

  try{
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify(payload)
    });
    const txt = await r.text();
    let data = null;
    try { data = JSON.parse(txt); } catch {}

    if (!r.ok){
      throw normalizePredictError(
        r.status,
        data,
        "Request failed (" + r.status + ")"
      );
    }

    state.lastResponse = data;
    resetScheduleFallback();

    state.activeTab = "prediction";
    document.querySelectorAll(".tab").forEach(t => {
      const on = (t.dataset.tab === "prediction");
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });

    render();
  }catch(e){
    state.lastResponse = null;

    if (e?.needsScheduleInputs) {
      setScheduleFallbackVisible(true, e.warning || e.message || "");
      formError.textContent = e.warning || "Flight was not found automatically, but it may still exist.";
      formError.style.display = "block";
    }

    content.innerHTML = renderPredictError(e);
  }finally{
    goBtn.disabled = false;
    goBtn.textContent = "Get delay risk";
  }
}

goBtn.addEventListener("click", runPredict);
function clearFallbackStateOnPrimaryInputChange() {
  resetScheduleFallback();
  formError.style.display = "none";
  formError.textContent = "";
}

[airlineEl, flightnumEl, dateEl, originEl, destEl].forEach((el) => {
  el.addEventListener("input", clearFallbackStateOnPrimaryInputChange);
  el.addEventListener("change", clearFallbackStateOnPrimaryInputChange);
});

async function fetchSimilarFlights(){
  if (!state.lastResponse || !state.lastInput) {
    state.similarFlightsError = "Run a prediction first.";
    state.similarFlightsRequested = true;
    render();
    return;
  }

  const requestedDepartureLocal = extractRequestedDepartureLocal(state.lastResponse, state.lastInput);
console.log("[similar-flights] requestedDepartureLocal =", requestedDepartureLocal);
console.log("[similar-flights] lastResponse raw =", state.lastResponse);
console.log(
  "[similar-flights] lastResponse json =",
  JSON.stringify(state.lastResponse, null, 2)
);
console.log("[similar-flights] lastInput =", state.lastInput);
  if (!requestedDepartureLocal) {
    state.similarFlightsError =
      "We could not determine the scheduled departure time for the selected flight, so we could not rank similar flights by time proximity.";
    state.similarFlightsRequested = true;
    render();
    return;
  }

  state.similarFlightsLoading = true;
  state.similarFlightsError = null;
  state.similarFlightsRequested = true;
  render();

  try {
    const body = {
      airline: state.lastInput.airline,
      origin_iata: state.lastInput.origin,
      destination_iata: state.lastInput.dest,
      departure_date: state.lastInput.date,
      requested_departure_local: requestedDepartureLocal,
      include: state.lastInput.include || {}
    };

    const r = await fetch("/api/similar-flights", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify(body)
    });

    const txt = await r.text();
    let data = null;
    try { data = JSON.parse(txt); } catch {}

    if (!r.ok || !data?.ok) {
      throw new Error(
        data?.user_message ||
        data?.error ||
        `Similar flight request failed (${r.status})`
      );
    }

    state.similarFlights = data;
  } catch (e) {
    state.similarFlightsError = String(e?.message || e || "Similar flight search failed.");
  } finally {
    state.similarFlightsLoading = false;
    render();
  }
}

// ---------- renderers ----------
function render(){
  if (!state.lastResponse){
    content.innerHTML = `<div class="empty">Run a prediction to see results.</div>`;
    return;
  }

  const tab = state.activeTab;
  if (tab === "prediction") renderPrediction(state.lastResponse);
  else if (tab === "similar") renderSimilarFlights();
  else if (tab === "weather") renderWeather(state.lastResponse);
  else if (tab === "flight_history") renderFlightHistory(state.lastResponse);
  else if (tab === "airport") renderAirport(state.lastResponse);
  else if (tab === "airline") renderAirline(state.lastResponse);
  else content.innerHTML = `<div class="empty">Unknown tab.</div>`;
}

function renderPrediction(resp){
  const p = resp.prediction || resp;

  const severityScore = p.severity_score ?? p.severity ?? p.risk_score ?? null;
  const expected = roundMinutes(p.expected_delay_minutes ?? p.expected_delay ?? p.expected_minutes);
  const predictedBin = p.predicted_bin ?? p.predicted_label ?? null;

  const labels = p.bin_labels ?? p.labels ?? null;
  const probs = p.bin_proba ?? p.bin_probs ?? p.probs ?? null;
  const thresholds = p.thresholds ?? p.bins ?? null;

  const bucketRiskScore = computeRiskScoreFromBuckets(labels, probs);
  const sevNorm = normalizeSeverity(severityScore);

  const scoreDisplay = (bucketRiskScore !== null)
    ? String(bucketRiskScore)
    : (sevNorm === null ? "—" : String(Math.round(sevNorm * 100)));

  const band = bandFromRisk100(
    bucketRiskScore !== null ? bucketRiskScore : (sevNorm === null ? NaN : sevNorm * 100)
  );

  const expectedDisplay = (expected === null) ? "—" : `${expected} min`;
  const binDisplay = predictedBin ? safeText(predictedBin) : "—";

  content.innerHTML = `
    <div class="riskCard">
      <div class="riskBig">
        <h3>Delay risk score</h3>
        <div class="riskScore" style="color: ${
          band.cls === "bandSafe" ? "rgba(56,211,159,.98)" :
          band.cls === "bandCaution" ? "rgba(247,201,72,.98)" :
          band.cls === "bandRisk" ? "rgba(255,138,76,.98)" :
          "rgba(255,77,109,.98)"
        }">${scoreDisplay}</div>
        <div class="riskBand ${band.cls}">
          ${safeText(band.name)}
        </div>
        <div class="footnote">Score is a normalized 0–100 indicator derived from the delay bucket probabilities.</div>
      </div>

      <div class="kpis">
        <div class="kpi">
          <div class="k">Expected delay</div>
          <div class="v">${expectedDisplay}</div>
        </div>
        <div class="kpi">
          <div class="k">Most likely bucket</div>
          <div class="v">${binDisplay}</div>
        </div>
        <div class="kpi">
          <div class="k">Bins</div>
          <div class="v">${Array.isArray(labels) ? labels.length : (Array.isArray(probs) ? probs.length : "—")}</div>
        </div>
        <div class="kpi">
          <div class="k">Thresholds (min)</div>
          <div class="v">${Array.isArray(thresholds) ? thresholds.join(", ") : "—"}</div>
        </div>
      </div>
    </div>

    <div class="sectionTitle">Probability by delay bucket</div>
    <div id="bars"></div>

    <div class="sectionTitle">What this means</div>
    <div class="twoCol">
      <div class="miniCard">
        <h4>Use it like a “risk meter”</h4>
        <div class="small">
          If the score is high, consider arriving a bit earlier, choosing a more flexible plan, or checking for earlier flights.
        </div>
      </div>
      <div class="miniCard">
        <h4>Why buckets?</h4>
        <div class="small">
          Buckets match what people care about: “on time-ish” vs. “meaningfully delayed”. The bars show how likely each outcome is.
        </div>
      </div>
    </div>
  `;

  const bars = document.getElementById("bars");
  const barList = document.createElement("div");
  barList.className = "barList";

  let L = Array.isArray(labels) ? labels : null;
  let P = Array.isArray(probs) ? probs : null;

  if (!L && P && Array.isArray(p.bin_labels_fallback)) L = p.bin_labels_fallback;

  if (!L && Array.isArray(thresholds) && thresholds.length){
    const t = thresholds.map(x => roundMinutes(x)).filter(x => Number.isFinite(x));
    const tmp = [];
    tmp.push(`< ${t[0]} min`);
    for (let i=0;i<t.length-1;i++){
      tmp.push(`${t[i]}–${t[i+1]} min`);
    }
    tmp.push(`≥ ${t[t.length-1]} min`);
    L = tmp;
  }

  if (!P || !P.length){
    bars.innerHTML = `<div class="empty">No bucket probabilities available for this response.</div>`;
    return;
  }

  const n = P.length;
  for (let i=0;i<n;i++){
    const label = (L && L[i]) ? String(L[i]) : `Bucket ${i+1}`;
    const pr = clamp01(P[i]);
    const prRound = roundProb(pr);

    const row = document.createElement("div");
    row.className = "barRow";
    row.innerHTML = `
      <div class="barLabel">${safeText(label)}</div>
      <div class="barTrack"><div class="barFill" style="width:${(pr*100).toFixed(2)}%"></div></div>
      <div class="barVal">${(prRound ?? 0).toFixed(3)}</div>
    `;
    barList.appendChild(row);
  }

  bars.appendChild(barList);
}

function renderSimilarFlights(){
  if (state.similarFlightsLoading) {
    content.innerHTML = `
      <div class="loadingBox">
        Searching similar flights on the selected date plus the day before and after, then scoring the closest departures…
      </div>
    `;
    return;
  }

  if (state.similarFlightsError) {
    content.innerHTML = `
      <div class="warningBox">${safeText(state.similarFlightsError)}</div>
      <button id="retrySimilarBtn" class="btn btnSecondary" type="button">Try again</button>
    `;
    const retryBtn = document.getElementById("retrySimilarBtn");
    if (retryBtn) retryBtn.addEventListener("click", fetchSimilarFlights);
    return;
  }

  if (!state.similarFlights) {
    content.innerHTML = `
      <div class="miniCard">
        <h4>Compare nearby departures</h4>
        <div class="small">
          We will search the same origin and destination on the selected date, plus the day before and after, and rank flights only by closeness to your selected departure time.
        </div>
      </div>
      <div style="margin-top:12px;">
        <button id="loadSimilarBtn" class="btn btnSecondary" type="button">Check similar flights</button>
      </div>
    `;
    const loadBtn = document.getElementById("loadSimilarBtn");
    if (loadBtn) loadBtn.addEventListener("click", fetchSimilarFlights);
    return;
  }

  const rows = Array.isArray(state.similarFlights.results) ? state.similarFlights.results : [];
  const sortedRows = sortSimilarFlightRows(rows, state.similarFlightsSort);
  const visibleRows = sortedRows.slice(0, state.similarFlightsVisibleCount);
  const hasMoreRows = sortedRows.length > visibleRows.length;

  const dates = Array.isArray(state.similarFlights.searched_dates)
    ? state.similarFlights.searched_dates.join(", ")
    : "selected date ± 1 day";

  if (!rows.length) {
    content.innerHTML = `
      <div class="miniCard">
        <h4>No similar flights found</h4>
        <div class="small">
          We searched ${safeText(dates)} but did not find any comparable departures for this route.
        </div>
      </div>
    `;
    return;
  }

  content.innerHTML = `
    <div class="similarTop">
      <div class="similarMeta">
        Searched: ${safeText(dates)} · showing ${safeText(visibleRows.length)} of ${safeText(rows.length)} nonstop candidates
      </div>

      <div class="similarControls">
        <label class="similarSortLabel" for="similarSortSelect">Sort by</label>
        <select id="similarSortSelect" class="similarSortSelect">
          <option value="date_asc" ${state.similarFlightsSort === "date_asc" ? "selected" : ""}>Date/time ↑</option>
          <option value="date_desc" ${state.similarFlightsSort === "date_desc" ? "selected" : ""}>Date/time ↓</option>
          <option value="price_asc" ${state.similarFlightsSort === "price_asc" ? "selected" : ""}>Price ↑</option>
          <option value="price_desc" ${state.similarFlightsSort === "price_desc" ? "selected" : ""}>Price ↓</option>
          <option value="airline_asc" ${state.similarFlightsSort === "airline_asc" ? "selected" : ""}>Airline A–Z</option>
          <option value="airline_desc" ${state.similarFlightsSort === "airline_desc" ? "selected" : ""}>Airline Z–A</option>
        </select>

        <button id="refreshSimilarBtn" class="btn btnSecondary" type="button">Refresh</button>
      </div>
    </div>

    <div class="similarList">
      ${visibleRows.map((row) => {
        const predictionOk = row?.prediction?.ok && row?.prediction?.data;
        const predictionPayload = predictionOk
          ? (row.prediction.data?.prediction || row.prediction.data)
          : null;

        const similarityBand = predictionPayload
          ? bandFromRisk100(
              computeRiskScoreFromBuckets(
                predictionPayload?.bin_labels ?? predictionPayload?.labels ?? null,
                predictionPayload?.bin_proba ?? predictionPayload?.bin_probs ?? predictionPayload?.probs ?? null
              ) ??
              (
                Number.isFinite(Number(predictionPayload?.severity_score))
                  ? (Math.max(0, Math.min(4, Number(predictionPayload.severity_score))) / 4) * 100
                  : NaN
              )
            )
          : null;

        const similarityScore = predictionPayload
          ? (
              computeRiskScoreFromBuckets(
                predictionPayload?.bin_labels ?? predictionPayload?.labels ?? null,
                predictionPayload?.bin_proba ?? predictionPayload?.bin_probs ?? predictionPayload?.probs ?? null
              ) ??
              (
                Number.isFinite(Number(predictionPayload?.severity_score))
                  ? Math.round((Math.max(0, Math.min(4, Number(predictionPayload.severity_score))) / 4) * 100)
                  : null
              )
            )
          : null;

        const pills = predictionOk ? extractPredictionPills(row.prediction.data) : [];

        return `
          <article class="similarFlight">
            <div class="similarFlightHead">
              <div>
                <div class="similarFlightTitle">
                  ${safeText(row.marketing_carrier || row.airline_iata || "Unknown carrier")}
                  ${safeText(row.flight_number || "")}
                </div>
                <div class="similarMeta">
                  ${safeText(row.origin_iata || "—")} → ${safeText(row.destination_iata || "—")}
                </div>
              </div>

              <div class="similarHeadBadges">
                ${
                  predictionOk && similarityBand && similarityScore !== null
                    ? `<div class="similarRiskBadge ${safeText(similarityBand.cls)}">${safeText(similarityBand.name)} · ${safeText(similarityScore)}</div>`
                    : ""
                }
                <div class="rankBadge">#${safeText(row.rank ?? "—")}</div>
              </div>
            </div>

            <div class="similarGrid">
              <div>
                <div class="metricLabel">Departure</div>
                <div class="metricValue">${safeText(formatLocalDateTime(row.departure_local))}</div>
              </div>
              <div>
                <div class="metricLabel">Arrival</div>
                <div class="metricValue">${safeText(formatLocalDateTime(row.arrival_local))}</div>
              </div>
              <div>
                <div class="metricLabel">Price</div>
                <div class="metricValue">${formatMoney(row.price, row.currency)}</div>
              </div>
              <div>
                <div class="metricLabel">Stops</div>
                <div class="metricValue">${safeText(formatStops(row.stop_count))}</div>
              </div>
              <div>
                <div class="metricLabel">Time difference</div>
                <div class="metricValue">${safeText(formatTimeDifference(row.similarity_minutes))}</div>
              </div>
              <div>
                <div class="metricLabel">Duration</div>
                <div class="metricValue">${safeText(formatDurationMinutes(row.duration_minutes))}</div>
              </div>
            </div>

            ${
              predictionOk
                ? `
                  <div class="predictionPills">
                    ${pills.length
                      ? pills.map((p) => `<div class="predictionPill">${safeText(p.label)}: ${safeText(p.value)}</div>`).join("")
                      : `<div class="predictionPill">Prediction returned</div>`}
                  </div>
                `
                : `
                  <div class="footnote">
                    Prediction unavailable${row?.prediction?.error ? `: ${safeText(row.prediction.error)}` : "."}
                  </div>
                `
            }
          </article>
        `;
      }).join("")}
    </div>

    ${
      hasMoreRows
        ? `
          <div class="similarLoadMoreWrap">
            <button id="loadMoreSimilarBtn" class="btn btnSecondary" type="button">
              Load more
            </button>
          </div>
        `
        : ""
    }
  `;

  const refreshBtn = document.getElementById("refreshSimilarBtn");
  if (refreshBtn) refreshBtn.addEventListener("click", fetchSimilarFlights);

  const sortSelect = document.getElementById("similarSortSelect");
  if (sortSelect) {
    sortSelect.addEventListener("change", (e) => {
      state.similarFlightsSort = e.target.value;
      state.similarFlightsVisibleCount = 5;
      renderSimilarFlights();
    });
  }

  const loadMoreBtn = document.getElementById("loadMoreSimilarBtn");
  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", () => {
      state.similarFlightsVisibleCount += 5;
      renderSimilarFlights();
    });
  }
}

function renderWeather(resp){
  const w = resp?.tabs?.weather ?? null;
  if (!w){
    content.innerHTML = `<div class="empty">No weather included in this response. Try “Include → All details”.</div>`;
    return;
  }

  const d = w.daily ?? {};
  const h = w.hourly ?? {};

  const KtoF = (K) => {
    const n = Number(K);
    if (!Number.isFinite(n)) return null;
    return Math.round((n - 273.15) * 9/5 + 32);
  };

  const tMaxF = KtoF(d.origin_temp_max_K);
  const tMinF = KtoF(d.origin_temp_min_K);

  const precipMm = Number(d.origin_daily_precip_sum_mm);
  const windKmhMax = Number(d.origin_daily_windspeed_max_kmh);

  const depTempF = KtoF(h.origin_dep_temp_K);
  const depPrecipMm = Number(h.origin_dep_precip_mm);
  const depWindKmh = Number(h.origin_dep_windspeed_kmh);

  const dailyCode = d.origin_daily_weathercode;
  const hourlyCode = h.origin_dep_hour_weathercode;

  const iconDaily = wxIconSvg(wxKindFromCode(dailyCode));
  const iconHourly = wxIconSvg(wxKindFromCode(hourlyCode));

  const tline = (tMinF != null && tMaxF != null) ? `${tMinF}°F – ${tMaxF}°F` : "—";
  const windMaxLine = Number.isFinite(windKmhMax) ? `${Math.round(windKmhMax)} km/h` : "—";
  const precipLine = Number.isFinite(precipMm) ? `${Math.round(precipMm * 10) / 10} mm` : "—";

  const depTempLine = (depTempF != null) ? `${depTempF}°F` : "—";
  const depWindLine = Number.isFinite(depWindKmh) ? `${Math.round(depWindKmh)} km/h` : "—";
  const depPrecipLine = Number.isFinite(depPrecipMm) ? `${Math.round(depPrecipMm * 10) / 10} mm` : "—";

  content.innerHTML = `
    <div class="twoCol">
      <div class="miniCard">
        <div class="wxRow">
          <div class="wxIcon">${iconDaily}</div>
          <div class="wxMeta">
            <p class="big">Daily outlook (origin)</p>
            <p class="muted">Temp: ${safeText(tline)} · Code: ${safeText(dailyCode ?? "—")}</p>
          </div>
        </div>
        <div class="footnote">Precip: ${safeText(precipLine)} · Max wind: ${safeText(windMaxLine)}</div>
      </div>

      <div class="miniCard">
        <div class="wxRow">
          <div class="wxIcon">${iconHourly}</div>
          <div class="wxMeta">
            <p class="big">At departure hour (origin)</p>
            <p class="muted">Temp: ${safeText(depTempLine)} · Code: ${safeText(hourlyCode ?? "—")}</p>
          </div>
        </div>
        <div class="footnote">Precip: ${safeText(depPrecipLine)} · Wind: ${safeText(depWindLine)}</div>
      </div>
    </div>

    <div class="sectionTitle">How to read this</div>
    <div class="miniCard">
      <div class="small">
        Icons summarize the weather codes. If you see rain/storm near departure, risk often increases.
      </div>
    </div>
  `;
}

function renderFlightHistory(resp){
  const fh = resp?.tabs?.flight_history ?? null;
  if (!fh){
    content.innerHTML = `<div class="empty">No flight-number history included. Try “Include → All details”.</div>`;
    return;
  }

  const support28 = fh.flightnum_od_support_count_last28d;
  const lowSupport28 = fh.flightnum_od_low_support_last28d;

  const m7  = roundMinutes(fh.flightnum_od_depdelay_mean_last7);
  const m14 = roundMinutes(fh.flightnum_od_depdelay_mean_last14);
  const m21 = roundMinutes(fh.flightnum_od_depdelay_mean_last21);
  const m28 = roundMinutes(fh.flightnum_od_depdelay_mean_last28);

  const vals = [m7, m14, m21, m28].map(v => (v == null ? 0 : v));
  const maxv = Math.max(1, ...vals);

  const rows = [
    { label: "Last 7 days",  v: m7 },
    { label: "Last 14 days", v: m14 },
    { label: "Last 21 days", v: m21 },
    { label: "Last 28 days", v: m28 }
  ];

  content.innerHTML = `
    <div class="twoCol">
      <div class="miniCard">
        <h4>Recent delay history (this flight #, this route)</h4>
        <div class="small">Average departure delay over rolling windows.</div>
        <div class="footnote">
          28-day sample size: <b>${safeText(support28 ?? "—")}</b>
          ${lowSupport28 ? `· Low-support flag: <b>${safeText(lowSupport28)}</b>` : ""}
        </div>
      </div>

      <div class="miniCard">
        <h4>What to look for</h4>
        <div class="small">
          If the recent 7-day average is much higher than 28-day, it may indicate a temporary issue (ops, weather patterns, etc.).
        </div>
      </div>
    </div>

    <div class="sectionTitle">Histogram-style summary</div>
    <div class="barList">
      ${rows.map(r => {
        const v = (r.v == null ? null : r.v);
        const pct = v == null ? 0 : (v / maxv) * 100;
        return `
          <div class="barRow">
            <div class="barLabel">${safeText(r.label)}</div>
            <div class="barTrack"><div class="barFill" style="width:${pct.toFixed(2)}%"></div></div>
            <div class="barVal">${v == null ? "—" : `${v}m`}</div>
          </div>
        `;
      }).join("")}
    </div>

    <div class="footnote">Minutes are rounded to the nearest minute.</div>
  `;
}

function renderAirport(resp){
  const a = resp?.tabs?.airport_stats ?? null;
  if (!a){
    content.innerHTML = `<div class="empty">No airport stats included. Try “Include → All details”.</div>`;
    return;
  }

  content.innerHTML = `
    <div class="miniCard">
      <h4>Airport stats</h4>
      <div class="small">
        Origin airport: <b>${safeText(a.origin ?? "—")}</b>
      </div>
      <div class="footnote">
        Your backend is currently returning only the airport code. Once you add metrics (delay rates, cancellations, etc.),
        this tab will show them as cards (no JSON).
      </div>
    </div>
  `;
}

function renderAirline(resp){
  const a = resp?.tabs?.airline_stats ?? null;
  if (!a){
    content.innerHTML = `<div class="empty">No airline stats included. Try “Include → All details”.</div>`;
    return;
  }

  content.innerHTML = `
    <div class="miniCard">
      <h4>Airline stats</h4>
      <div class="small">
        Airline: <b>${safeText(a.airline ?? "—")}</b>
      </div>
      <div class="footnote">
        Your backend is currently returning only the airline code. Once you include metrics (del15 rate, avg delay, etc.),
        we’ll render them as clean cards and charts.
      </div>
    </div>
  `;
}