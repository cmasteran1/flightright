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

app.get("/health", (_req, res) => res.json({ ok: true, service: "flightright-web" }));

// For the UI status indicator
app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${API_BASE}/health`, { headers: { Accept: "application/json" } });
    const txt = await r.text();
    res.status(r.status).type(r.headers.get("content-type") || "text/plain").send(txt);
  } catch (e) {
    res.status(502).json({ ok: false, error: "Failed to reach API", details: String(e) });
  }
});

// The UI calls THIS endpoint. Server adds X-API-Key and forwards to API_BASE/predict.
app.post("/api/predict", async (req, res) => {
  if (!API_KEY) {
    return res.status(500).json({
      ok: false,
      error:
        "Server missing FLIGHTRIGHT_API_KEY. Set it on flightright-web via fly secrets set FLIGHTRIGHT_API_KEY=... -a flightright-web"
    });
  }

  try {
    const r = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
      },
      body: JSON.stringify(req.body ?? {})
    });

    const contentType = r.headers.get("content-type") || "application/json";
    const text = await r.text();
    res.status(r.status).type(contentType).send(text);
  } catch (e) {
    res.status(502).json({ ok: false, error: "Predict proxy failed", details: String(e) });
  }
});

// SPA fallback (safe even if you add routing later)
app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log("web listening on " + PORT);
});