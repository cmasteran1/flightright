// web/server.js
import express from "express";

const app = express();
const PORT = process.env.PORT || 8080;

// Backend API base (do NOT end with slash)
const API_BASE = (process.env.FLIGHTRIGHT_API_BASE || "https://api.flightrightus.com").replace(/\/+$/, "");

// Server-only secret (never exposed to browser)
const API_KEY = process.env.FLIGHTRIGHT_API_KEY;

if (!API_KEY) {
  console.warn(
    "[WARN] FLIGHTRIGHT_API_KEY is not set in the web app environment. /api/predict will fail until you set it."
  );
}

app.use(express.json({ limit: "256kb" }));

// --- UI (simple single-page HTML) ---
app.get("/", (_req, res) => {
  res.type("html").send(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>flightright</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b1020; color: #e7eaf3; }
      .wrap { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
      .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
      h1 { margin: 6px 0 10px; font-size: 28px; letter-spacing: 0.2px; }
      p { margin: 6px 0; color: rgba(231,234,243,0.85); }
      a { color: #9bd0ff; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
      label { display: block; font-size: 12px; color: rgba(231,234,243,0.75); margin-bottom: 6px; }
      input, select {
        width: 100%; box-sizing: border-box;
        padding: 10px 12px; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(0,0,0,0.35); color: #e7eaf3;
        outline: none;
      }
      input::placeholder { color: rgba(231,234,243,0.45); }
      .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      .btn {
        padding: 10px 14px; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(155,208,255,0.18);
        color: #e7eaf3; cursor: pointer;
        font-weight: 600;
      }
      .btn:disabled { opacity: 0.55; cursor: not-allowed; }
      .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06); }
      pre {
        margin-top: 12px;
        background: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 12px;
        overflow: auto;
        color: rgba(231,234,243,0.95);
      }
      .hint { font-size: 12px; color: rgba(231,234,243,0.65); margin-top: 10px; }
      .full { grid-column: 1 / -1; }
      @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <h1>flightright</h1>
            <p>Day-ahead delay risk for U.S. flights.</p>
            <p class="hint">API: <a href="${API_BASE}/health" target="_blank" rel="noreferrer">${API_BASE}/health</a></p>
          </div>
          <div class="pill" id="statusPill">idle</div>
        </div>

        <div class="grid" style="margin-top: 16px;">
          <div>
            <label>Airline (IATA)</label>
            <input id="airline" placeholder="WN" value="WN" />
          </div>
          <div>
            <label>Flight number</label>
            <input id="flightnum" placeholder="868" value="868" />
          </div>

          <div>
            <label>Date</label>
            <input id="date" type="date" />
          </div>
          <div>
            <label>Include tabs</label>
            <select id="includePreset">
              <option value="full" selected>Weather + history + stats</option>
              <option value="minimal">Minimal</option>
              <option value="no_weather">No weather</option>
            </select>
          </div>

          <div>
            <label>Origin (IATA)</label>
            <input id="origin" placeholder="BWI" value="BWI" />
          </div>
          <div>
            <label>Destination (IATA)</label>
            <input id="dest" placeholder="CVG" value="CVG" />
          </div>

          <div class="full">
            <div class="row">
              <button class="btn" id="predictBtn">Predict</button>
              <button class="btn" id="exampleBtn" type="button">Use example</button>
              <span class="hint" id="hint"></span>
            </div>
          </div>
        </div>

        <pre id="output" style="display:none;"></pre>
      </div>
    </div>

    <script>
      // default date = today in local tz
      const d = new Date();
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth()+1).padStart(2,'0');
      const dd = String(d.getDate()).padStart(2,'0');
      document.getElementById("date").value = \`\${yyyy}-\${mm}-\${dd}\`;

      const statusPill = document.getElementById("statusPill");
      const out = document.getElementById("output");
      const hint = document.getElementById("hint");
      const btn = document.getElementById("predictBtn");

      function setStatus(s) {
        statusPill.textContent = s;
      }

      function includeFromPreset(preset) {
        if (preset === "minimal") {
          return { weather: false, flight_history: false, airport_stats: false, airline_stats: false };
        }
        if (preset === "no_weather") {
          return { weather: false, flight_history: true, airport_stats: true, airline_stats: true };
        }
        return { weather: true, flight_history: true, airport_stats: true, airline_stats: true };
      }

      function prettySummary(resp) {
        const pred = resp?.prediction || {};
        const modelId = resp?.model?.id || "(unknown)";
        const band = pred?.severity_band || "(unknown)";
        const exp = pred?.expected_delay_minutes;
        const p15 = pred?.p_ge?.["15"];

        return [
          \`model: \${modelId}\`,
          \`severity: \${band}\`,
          (typeof exp === "number") ? \`expected delay: \${exp.toFixed(1)} min\` : null,
          (typeof p15 === "number") ? \`P(delay ≥ 15): \${(p15*100).toFixed(1)}%\` : null,
        ].filter(Boolean).join("\\n");
      }

      async function doPredict() {
        const airline = document.getElementById("airline").value.trim();
        const flightnum = document.getElementById("flightnum").value.trim();
        const date = document.getElementById("date").value;
        const origin = document.getElementById("origin").value.trim().toUpperCase();
        const dest = document.getElementById("dest").value.trim().toUpperCase();
        const preset = document.getElementById("includePreset").value;
        const include = includeFromPreset(preset);

        if (!airline || !flightnum || !date || !origin || !dest) {
          alert("Please fill airline, flight number, date, origin, and destination.");
          return;
        }

        setStatus("predicting…");
        hint.textContent = "";
        out.style.display = "none";
        btn.disabled = true;

        try {
          const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ airline, flightnum, date, origin, dest, include })
          });

          const text = await resp.text();
          let json;
          try { json = JSON.parse(text); } catch { json = { raw: text }; }

          if (!resp.ok) {
            setStatus(\`error (\${resp.status})\`);
            hint.textContent = json?.detail ? String(json.detail) : "Request failed.";
            out.style.display = "block";
            out.textContent = JSON.stringify(json, null, 2);
            return;
          }

          setStatus("ok");
          out.style.display = "block";
          out.textContent = prettySummary(json) + "\\n\\n" + JSON.stringify(json, null, 2);
        } catch (e) {
          setStatus("network error");
          hint.textContent = String(e);
        } finally {
          btn.disabled = false;
        }
      }

      document.getElementById("predictBtn").addEventListener("click", doPredict);
      document.getElementById("exampleBtn").addEventListener("click", () => {
        document.getElementById("airline").value = "WN";
        document.getElementById("flightnum").value = "868";
        document.getElementById("origin").value = "BWI";
        document.getElementById("dest").value = "CVG";
        document.getElementById("includePreset").value = "full";
      });
    </script>
  </body>
</html>`);
});

app.get("/health", (_req, res) => res.json({ ok: true }));

// --- Server-side proxy (keeps secrets off the client) ---
app.post("/api/predict", async (req, res) => {
  try {
    if (!API_KEY) {
      return res.status(500).json({ detail: "Server not configured (missing FLIGHTRIGHT_API_KEY in web app)." });
    }

    const upstream = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify(req.body ?? {}),
    });

    const text = await upstream.text();
    res.status(upstream.status);
    // best-effort json
    try {
      res.json(JSON.parse(text));
    } catch {
      res.type("text/plain").send(text);
    }
  } catch (err) {
    res.status(502).json({ detail: "Upstream error", error: String(err) });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`web listening on ${PORT}`);
  console.log(`proxying /api/predict -> ${API_BASE}/predict`);
});