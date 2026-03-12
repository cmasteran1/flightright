import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const app = express();

app.use(express.json({ limit: "1mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 8080;

// FlightRight prediction API.
const API_BASE =
  process.env.FLIGHTRIGHT_API_BASE || "https://api.flightrightus.com";
const API_KEY = process.env.FLIGHTRIGHT_API_KEY || "";

// GoflightLabs pricing API key.
const GOFLIGHTLABS_API_KEY = process.env.GOFLIGHTLABS_API_KEY || "";

// Serve frontend.
app.use(express.static(path.join(__dirname, "public")));

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "flightright-web" });
});

// For the UI status indicator.
app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${API_BASE}/health`, {
      headers: { Accept: "application/json" },
    });
    const txt = await r.text();
    res
      .status(r.status)
      .type(r.headers.get("content-type") || "text/plain")
      .send(txt);
  } catch (e) {
    res.status(502).json({
      ok: false,
      error: "Failed to reach API",
      details: String(e),
    });
  }
});

// Restore lookups proxy for autocomplete dropdowns.
app.get("/api/lookups", async (_req, res) => {
  try {
    const r = await fetch(`${API_BASE}/lookups`, {
      headers: { Accept: "application/json" },
    });
    const txt = await r.text();
    res
      .status(r.status)
      .type(r.headers.get("content-type") || "application/json")
      .send(txt);
  } catch (e) {
    res.status(502).json({
      ok: false,
      error: "Failed to reach lookups endpoint",
      details: String(e),
    });
  }
});

// Turn vague upstream errors into something a regular person can understand.
function buildUserFacingPredictError(status, rawMsg) {
  const raw = String(rawMsg || "").trim();
  const lower = raw.toLowerCase();

  if (
    lower.includes("same-day") ||
    lower.includes("same day") ||
    lower.includes("today")
  ) {
    return {
      ok: false,
      error: raw || "Prediction unavailable for today",
      user_title: "Prediction not available for today",
      user_message:
        "This tool works best for flights scheduled for tomorrow or later, and could not produce a reliable prediction for this date.",
      user_action: "Try checking the same flight for a later date.",
    };
  }

  if (lower.includes("flight not found") || lower.includes("not found")) {
    return {
      ok: false,
      error: raw || "Flight not found",
      user_title: "We couldn’t find that flight",
      user_message:
        "FlightRight could not confirm this flight automatically from the details provided.",
      user_action:
        "Double-check the airline, flight number, date, and route, then try again.",
    };
  }

  if (status >= 500) {
    return {
      ok: false,
      error: raw || "Upstream service error",
      user_title: "Temporary service problem",
      user_message:
        "FlightRight had trouble reaching the prediction service.",
      user_action: "Please try again in a minute.",
    };
  }

  return {
    ok: false,
    error: raw || "Request failed",
    user_title: "We couldn’t run this prediction",
    user_message:
      "We were not able to generate a delay estimate for this flight.",
    user_action: "Please double-check the flight details and try again.",
  };
}

// Existing proxy for prediction.
app.post("/api/predict", async (req, res) => {
  if (!API_KEY) {
    return res.status(500).json({
      ok: false,
      error:
        'Server missing FLIGHTRIGHT_API_KEY. Set it on flightright-web via fly secrets set FLIGHTRIGHT_API_KEY="..." -a flightright-web',
      user_title: "Service configuration problem",
      user_message:
        "The website is temporarily unable to contact the prediction service.",
      user_action: "Please try again later.",
    });
  }

  try {
    const r = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify(req.body ?? {}),
    });

    const contentType = r.headers.get("content-type") || "application/json";
    const text = await r.text();

    if (r.ok) {
      return res.status(r.status).type(contentType).send(text);
    }

    let data = null;
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }

    const nested =
      data?.detail && typeof data.detail === "object" ? data.detail : null;

    const alreadyHelpful =
      !!data?.user_title ||
      !!data?.user_message ||
      !!data?.user_action ||
      !!nested?.user_title ||
      !!nested?.user_message ||
      !!nested?.user_action ||
      !!nested?.needs_schedule_inputs;

    if (alreadyHelpful) {
      return res.status(r.status).json(nested || data);
    }

    const rawMsg = String(
      nested?.error ||
        data?.detail ||
        data?.error ||
        text ||
        ""
    ).trim();

    return res
      .status(r.status)
      .json(buildUserFacingPredictError(r.status, rawMsg));
  } catch (e) {
    res.status(502).json({
      ok: false,
      error: "Predict proxy failed",
      details: String(e),
      user_title: "Temporary service problem",
      user_message: "FlightRight had trouble reaching the prediction service.",
      user_action: "Please try again in a minute.",
    });
  }
});

/* =========================
   Similar flights helpers
   ========================= */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseLocalDateTime(input) {
  const d = new Date(input);
  return Number.isNaN(d.getTime()) ? null : d;
}

function addDaysYmd(ymd, deltaDays) {
  const [y, m, d] = String(ymd).split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  dt.setUTCDate(dt.getUTCDate() + deltaDays);
  return dt.toISOString().slice(0, 10);
}

function minutesBetween(a, b) {
  return Math.round((a.getTime() - b.getTime()) / 60000);
}

function normalizeCarrierIata(marketingCarrier, fallbackAirline) {
  const map = {
    "American Airlines": "AA",
    "Delta Air Lines": "DL",
    "United Airlines": "UA",
    "Southwest Airlines": "WN",
    "Alaska Airlines": "AS",
    JetBlue: "B6",
    "Spirit Airlines": "NK",
    "Frontier Airlines": "F9",
    "Hawaiian Airlines": "HA",
    "British Airways": "BA",
    Iberia: "IB",
    Finnair: "AY",
    "Turkish Airlines": "TK",
    "TAP AIR PORTUGAL": "TP",
    "Royal Air Maroc": "AT",
  };

  if (typeof fallbackAirline === "string" && fallbackAirline.trim()) {
    return fallbackAirline.trim().toUpperCase();
  }

  const name = String(marketingCarrier || "").trim();
  return map[name] || null;
}

function normalizePricedFlight(raw, query, requestedDeparture, fallbackAirline) {
  const dep = parseLocalDateTime(raw?.departure);
  const arr = parseLocalDateTime(raw?.arrival);
  if (!dep || !arr) return null;

  const origin = String(raw?.origin?.code || "").toUpperCase();
  const destination = String(raw?.destination?.code || "").toUpperCase();

  // Strict same-route filtering.
  if (origin !== query.origin_iata || destination !== query.destination_iata) {
    return null;
  }
  const stop_count = Number(raw?.stopCount ?? 0);
if (stop_count !== 0) return null;
  const airline_iata = normalizeCarrierIata(
    raw?.marketingCarrier,
    fallbackAirline
  );
  const flight_number = String(raw?.flightNumber || "").trim();

  if (!flight_number) return null;

  return {
    origin_iata: origin,
    destination_iata: destination,
    airline_iata,
    marketing_carrier: raw?.marketingCarrier || null,
    operating_carrier: raw?.operatingCarrier || null,
    flight_number,
    price: Number(raw?.price ?? NaN),
    currency: raw?.currency || "USD",
    departure: dep.toISOString(),
    arrival: arr.toISOString(),
    departure_local: raw?.departure,
    arrival_local: raw?.arrival,
    stop_count,
    duration_minutes: Number(raw?.durationInMinutes ?? NaN),
    similarity_minutes: minutesBetween(dep, requestedDeparture),
similarity_minutes_abs: Math.abs(minutesBetween(dep, requestedDeparture)),
    raw,
  };
}

async function fetchGoflightLabsOnce({ originIata, destinationIata, date }) {
  const url = new URL("https://www.goflightlabs.com/retrieveFlights");
  url.searchParams.set("access_key", GOFLIGHTLABS_API_KEY);
  url.searchParams.set("originIATACode", originIata);
  url.searchParams.set("destinationIATACode", destinationIata);
  url.searchParams.set("date", date);
  url.searchParams.set("sortBy", "best");
  url.searchParams.set("mode", "oneway");

  const r = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  const text = await r.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }

  if (!r.ok) {
    throw new Error(
      `GoflightLabs request failed (${r.status}): ${text || "no response body"}`
    );
  }
  console.log("[goflightlabs] date =", date, "response type =", Array.isArray(data) ? "array" : typeof data);
console.log("[goflightlabs] preview =", JSON.stringify(data)?.slice(0, 500));
  return data;
}

async function fetchPricedFlightsForDate({
  originIata,
  destinationIata,
  date,
}) {
  const maxAttempts = 10;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const data = await fetchGoflightLabsOnce({
      originIata,
      destinationIata,
      date,
    });

    if (Array.isArray(data)) {
      return data;
    }

    if (Array.isArray(data?.flights)) {
      return data.flights;
    }

    if (data?.status === "success" && Array.isArray(data?.data)) {
      return data.data;
    }

    if (data?.status === "processing") {
      if (attempt < maxAttempts) {
        await sleep(4000);
        continue;
      }
      throw new Error(
        "GoflightLabs search is still processing after multiple attempts."
      );
    }

    return [];
  }

  return [];
}

function dedupeCandidates(items) {
  const seen = new Set();
  const out = [];

  for (const item of items) {
    const key = [
      item.origin_iata,
      item.destination_iata,
      item.marketing_carrier || item.airline_iata || "",
      item.flight_number,
      item.departure_local,
    ].join("|");

    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }

  return out;
}

function buildPredictPayload(candidate, originalRequest) {
  const payload = {
    airline: candidate.airline_iata || originalRequest.airline || "",
    flightnum: candidate.flight_number,
    date: String(candidate.departure_local).slice(0, 10),
    include: originalRequest.include || {},
    origin: candidate.origin_iata,
    dest: candidate.destination_iata,
  };

  const depTime = String(candidate.departure_local || "").slice(11, 16);
  const arrTime = String(candidate.arrival_local || "").slice(11, 16);

  if (depTime) payload.sched_dep_time_24h = depTime;
  if (arrTime) payload.sched_arr_time_24h = arrTime;

  return payload;
}

async function predictCandidate(candidate, originalRequest) {
  if (!API_KEY) {
    return {
      ok: false,
      error: "Missing FLIGHTRIGHT_API_KEY",
    };
  }

  const body = buildPredictPayload(candidate, originalRequest);

  try {
    const r = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify(body),
    });

    const text = await r.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    if (!r.ok) {
      return {
        ok: false,
        status: r.status,
        error: data?.error || data?.detail || text || "Prediction failed",
        user_title: data?.user_title || null,
        user_message: data?.user_message || null,
        user_action: data?.user_action || null,
      };
    }

    return {
      ok: true,
      data,
    };
  } catch (e) {
    return {
      ok: false,
      error: String(e),
    };
  }
}
async function mapWithConcurrency(items, limit, asyncMapper) {
  const results = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const currentIndex = nextIndex;
      nextIndex += 1;

      if (currentIndex >= items.length) return;

      results[currentIndex] = await asyncMapper(items[currentIndex], currentIndex);
    }
  }

  const workerCount = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}
/* =========================
   Similar flights API
   ========================= */

app.post("/api/similar-flights", async (req, res) => {
  if (!GOFLIGHTLABS_API_KEY) {
    return res.status(500).json({
      ok: false,
      error: "Missing GOFLIGHTLABS_API_KEY",
      user_title: "Service configuration problem",
      user_message:
        "The website is temporarily unable to search similar flights.",
      user_action: "Please try again later.",
    });
  }

  const origin_iata = String(req.body?.origin_iata || req.body?.origin || "")
    .trim()
    .toUpperCase();
  const destination_iata = String(
    req.body?.destination_iata || req.body?.dest || req.body?.destination || ""
  )
    .trim()
    .toUpperCase();
  const departure_date = String(
    req.body?.departure_date || req.body?.date || ""
  ).trim();
  const requested_departure_local = String(
    req.body?.requested_departure_local ||
      req.body?.scheduled_departure_local ||
      ""
  ).trim();

  if (
    !origin_iata ||
    !destination_iata ||
    !departure_date ||
    !requested_departure_local
  ) {
    return res.status(400).json({
      ok: false,
      error:
        "origin_iata, destination_iata, departure_date, and requested_departure_local are required",
    });
  }

  const requestedDeparture = parseLocalDateTime(requested_departure_local);
  console.log("[/api/similar-flights] request body =", req.body);
console.log("[/api/similar-flights] requested_departure_local =", requested_departure_local);
  if (!requestedDeparture) {
    return res.status(400).json({
      ok: false,
      error: "requested_departure_local must be a valid local datetime string",
    });
  }

  try {
    const searchDates = [
      addDaysYmd(departure_date, -1),
      departure_date,
      addDaysYmd(departure_date, 1),
    ];

    const fetched = await Promise.all(
      searchDates.map(async (date) => {
        const flights = await fetchPricedFlightsForDate({
          originIata: origin_iata,
          destinationIata: destination_iata,
          date,
        });
        return { date, flights };
      })
    );

    const normalized = fetched
      .flatMap(({ date, flights }) =>
        flights.map((raw) =>
          normalizePricedFlight(
            raw,
            { origin_iata, destination_iata, departure_date: date },
            requestedDeparture,
            req.body?.airline
          )
        )
      )
      .filter(Boolean);

    const deduped = dedupeCandidates(normalized);

    deduped.sort((a, b) => {
      if (a.similarity_minutes_abs !== b.similarity_minutes_abs) {
  return a.similarity_minutes_abs - b.similarity_minutes_abs;
}
      return (a.price || Number.POSITIVE_INFINITY) -
        (b.price || Number.POSITIVE_INFINITY);
    });

    const topCandidates = deduped.slice(0, 20);

    const predictions = await mapWithConcurrency(
      topCandidates,
      4,
      (candidate) => predictCandidate(candidate, req.body)
    );

    const results = topCandidates.map((candidate, idx) => ({
      rank: idx + 1,
      similarity_minutes: candidate.similarity_minutes,
      similarity_minutes_abs: candidate.similarity_minutes_abs,
      airline_iata: candidate.airline_iata,
      marketing_carrier: candidate.marketing_carrier,
      operating_carrier: candidate.operating_carrier,
      flight_number: candidate.flight_number,
      origin_iata: candidate.origin_iata,
      destination_iata: candidate.destination_iata,
      departure_local: candidate.departure_local,
      arrival_local: candidate.arrival_local,
      price: Number.isFinite(candidate.price) ? candidate.price : null,
      currency: candidate.currency,
      stop_count: candidate.stop_count,
      duration_minutes: Number.isFinite(candidate.duration_minutes)
        ? candidate.duration_minutes
        : null,
      prediction: predictions[idx],
    }));

    return res.json({
      ok: true,
      query: {
        origin_iata,
        destination_iata,
        departure_date,
        requested_departure_local,
      },
      searched_dates: searchDates,
      total_candidates_found: deduped.length,
      results,
    });
  } catch (e) {
  console.error("[/api/similar-flights] error:", e);

  return res.status(502).json({
    ok: false,
    error: "Similar flights lookup failed",
    details: String(e),
    user_title: "We couldn’t search similar flights",
    user_message:
      "FlightRight had trouble retrieving comparable flight options.",
    user_action: String(e),
  });
}
});

// SPA fallback.
app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("web listening on " + PORT);
});