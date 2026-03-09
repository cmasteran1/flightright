import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
app.use(express.json({ limit: "1mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 8080;

// This is your API app (Fly) that actually runs the model.
const API_BASE = process.env.FLIGHTRIGHT_API_BASE || "https://api.flightrightus.com";

// Public key required for /predict (kept server-side, never sent to browser).
const API_KEY = process.env.FLIGHTRIGHT_API_KEY || "";

// Serve frontend
app.use(express.static(path.join(__dirname, "public")));

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "flightright-web" });
});

// For the UI status indicator
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
        "The flight details did not match a scheduled flight for the date you entered.",
      user_action:
        "Check the airline code, flight number, airports, and date, then try again.",
    };
  }

  if (
    lower.includes("insufficient") ||
    lower.includes("not enough data") ||
    lower.includes("unable to score") ||
    lower.includes("unavailable data")
  ) {
    return {
      ok: false,
      error: raw || "Insufficient data",
      user_title: "Not enough information to make a prediction",
      user_message:
        "We found the flight, but there was not enough valid data to generate a reliable result.",
      user_action: "Try a different date or check back later.",
    };
  }

  if (status === 400) {
    return {
      ok: false,
      error: raw || "Bad request",
      user_title: "Please check the flight details",
      user_message:
        "We couldn’t run the prediction because some of the flight information did not match what the system expected.",
      user_action:
        "Make sure the airline, flight number, airports, and date are correct, then try again.",
    };
  }

  if (status === 404) {
    return {
      ok: false,
      error: raw || "Not found",
      user_title: "We couldn’t find data for this flight",
      user_message:
        "The system could not find the flight or the supporting data needed to make a prediction.",
      user_action: "Verify the flight details or try a different date.",
    };
  }

  if (status === 422) {
    return {
      ok: false,
      error: raw || "Unprocessable request",
      user_title: "Not enough information to make a prediction",
      user_message:
        "We found the request, but there was not enough valid data to generate a reliable result.",
      user_action: "Try a different date or check back later.",
    };
  }

  if (status === 502 || status === 503 || status === 504) {
    return {
      ok: false,
      error: raw || "Temporary service problem",
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
    user_action:
      "Please double-check the flight details and try again.",
  };
}

// The UI calls THIS endpoint.
// Server adds X-API-Key and forwards to API_BASE/predict.
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

   const payload =
  data && typeof data?.detail === "object" && data.detail !== null
    ? data.detail
    : data;

const rawMsg = String(
  payload?.detail || payload?.error || data?.error || text || ""
).trim();

const alreadyHelpful =
  !!payload?.user_title ||
  !!payload?.user_message ||
  !!payload?.user_action ||
  !!payload?.error_code ||
  !!payload?.needs_schedule_inputs;

if (alreadyHelpful) {
  return res.status(r.status).json({
  ok: false,
  ...payload
});
}
    const generic =
      !rawMsg ||
      /^bad request\.?$/i.test(rawMsg) ||
      /^request failed/i.test(rawMsg) ||
      /^internal server error\.?$/i.test(rawMsg);

    if (generic) {
      return res
        .status(r.status)
        .json(buildUserFacingPredictError(r.status, rawMsg));
    }

    return res
      .status(r.status)
      .json(buildUserFacingPredictError(r.status, rawMsg));
  } catch (e) {
    res.status(502).json({
      ok: false,
      error: "Predict proxy failed",
      details: String(e),
      user_title: "Temporary service problem",
      user_message:
        "FlightRight had trouble reaching the prediction service.",
      user_action: "Please try again in a minute.",
    });
  }
});

// SPA fallback (safe even if you add routing later)
app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("web listening on " + PORT);
});