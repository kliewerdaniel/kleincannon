const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const STAGES = ["script", "tts", "align", "prompts", "images", "captions", "assemble"];
let logEl, stepsEl, resultEl, runBtn;

function setStage(name, state) {
  const li = $(`#steps li[data-stage="${name}"]`);
  if (li) {
    li.classList.remove("pending", "active", "done", "error");
    li.classList.add(state);
  }
}

function resetSteps() {
  STAGES.forEach((s) => setStage(s, "pending"));
}

function appendLog(line) {
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function connectSSE() {
  const es = new EventSource("/api/stream");
  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    handleEvent(ev);
  };
  es.onerror = () => { /* auto-reconnect handled by browser */ };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "connected":
      appendLog("• connected");
      break;
    case "run_start":
      appendLog(`▶ run start: ${ev.topic || ""}`);
      resetSteps();
      runBtn.disabled = true;
      break;
    case "stage":
      appendLog(`▷ ${ev.name}: ${ev.desc}`);
      setStage(ev.name, "active");
      break;
    case "log":
      appendLog(`  ${ev.line}`);
      break;
    case "done":
      appendLog("✔ done");
      STAGES.forEach((s) => setStage(s, "done"));
      resultEl.classList.remove("hidden");
      resultEl.innerHTML =
        `<strong>Final video:</strong> <a href="/${ev.result.mp4}" target="_blank">${ev.result.mp4}</a>` +
        ` &nbsp; <a href="/${ev.result.mp4}" download>⬇ download</a>` +
        ` <br><span class="meta">${ev.result.duration?.toFixed(1)}s · ${ev.result.id}</span>`;
      runBtn.disabled = false;
      loadEpisodes();
      break;
    case "error":
      appendLog("✖ " + ev.message);
      runBtn.disabled = false;
      break;
    default:
      appendLog(JSON.stringify(ev));
  }
}

async function loadServices() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    $$("#services .dot").forEach((d) => {
      const s = d.dataset.s;
      d.classList.toggle("on", !!j.services[s]);
      d.classList.toggle("off", !j.services[s]);
    });
  } catch {}
}

async function loadEpisodes() {
  try {
    const r = await fetch("/api/episodes");
    const j = await r.json();
    const grid = $("#gallery-grid");
    grid.innerHTML = j.episodes
      .map((id) => `<a class="ep" href="/episodes/${id}/${id}.mp4" target="_blank">${id}</a>`)
      .join("");
  } catch {}
}

function init() {
  logEl = $("#log");
  stepsEl = $("#steps");
  resultEl = $("#result");
  runBtn = $("#runbtn");

  const form = $("#runform");
  const manualWrap = $("#manual-wrap");
  const beatsWrap = $("#beats-wrap");
  const useManual = form.elements.use_manual;

  function syncManual() {
    const on = useManual.checked;
    manualWrap.classList.toggle("hidden", !on);
    beatsWrap.classList.toggle("hidden", on);
  }
  useManual.addEventListener("change", syncManual);
  syncManual();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());
    data.use_manual = useManual.checked;
    data.fast = form.elements.fast.checked;
    data.beats = parseInt(data.beats, 10);
    data.zoom_max = parseFloat(data.zoom_max);
    data.speed = parseFloat(data.speed);
    data.steps = data.steps ? parseInt(data.steps, 10) : null;
    data.cfg = data.cfg ? parseFloat(data.cfg) : null;
    data.seed = data.seed ? parseInt(data.seed, 10) : null;
    resultEl.classList.add("hidden");
    resultEl.innerHTML = "";
    appendLog("• submitting");
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const j = await r.json();
    if (!j.accepted) appendLog("✖ " + (j.error || "rejected"));
  });

  // populate the style dropdown from the API and live-update the speed label
  const speed = form.elements.speed;
  const speedVal = $("#speed-val");
  if (speed && speedVal) {
    const sync = () => { speedVal.textContent = (parseFloat(speed.value)).toFixed(2) + "×"; };
    speed.addEventListener("input", sync);
    sync();
  }
  (async () => {
    try {
      const sr = await fetch("/api/styles");
      const sj = await sr.json();
      const sel = $("#style-select");
      if (sel && sj.styles) {
        for (const s of sj.styles) {
          const o = document.createElement("option");
          o.value = s.name;
          o.textContent = `${s.name} — ${s.palette}`;
          sel.appendChild(o);
        }
      }
      if (speed && typeof sj.default_speed === "number") {
        speed.value = sj.default_speed;
        speedVal.textContent = sj.default_speed.toFixed(2) + "×";
      }
    } catch {}
  })();

  connectSSE();
  loadServices();
  loadEpisodes();
  setInterval(loadServices, 15000);
}

init();
