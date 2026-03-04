async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  const txt = await res.text();
  let data;
  try { data = JSON.parse(txt); } catch { data = { raw: txt }; }
  if (!res.ok) {
    const msg = data?.detail || data?.error || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function val(id) {
  return document.getElementById(id).value.trim();
}

async function checkStatus() {
  const el = document.getElementById("status");
  try {
    const r = await getJSON("/health");
    el.textContent = r.ok ? "Service OK" : "Service not OK";
  } catch (e) {
    el.textContent = `Status check failed: ${e.message}`;
  }
}

async function predict() {
  const out = document.getElementById("out");
  out.textContent = "Running…";

  const apiKey = val("apikey");
  if (!apiKey) {
    out.textContent = "Missing API key (X-API-Key).";
    return;
  }

  const payload = {
    airline: val("airline"),
    flightnum: val("flightnum"),
    date: val("date"),
    origin: val("origin"),
    dest: val("dest"),
    include: {}
  };

  try {
    const data = await getJSON("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey
      },
      body: JSON.stringify(payload)
    });
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = `Error: ${e.message}`;
  }
}

window.addEventListener("load", () => {
  document.getElementById("btn").addEventListener("click", predict);
  checkStatus();
});