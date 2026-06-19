const FEATURES = window.__FEATURES__ || [];
const STATS = window.__STATS__ || {};

const controlsEl = document.getElementById("controls");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const statusBadge = document.getElementById("statusBadge");

const predLabel = document.getElementById("predLabel");
const predProb = document.getElementById("predProb");
const leafNode = document.getElementById("leafNode");
const leafSamples = document.getElementById("leafSamples");
const probFill = document.getElementById("probFill");
const probPct = document.getElementById("probPct");
const pathEl = document.getElementById("path");

const pathViz = document.getElementById("pathViz");

function setStatus(text){
  statusBadge.textContent = text;
}

function fmt(x, digits=3){
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(digits);
}

function buildControls(){
  controlsEl.innerHTML = "";

  FEATURES.forEach(feat => {
    const s = STATS[feat] || {min: 0, max: 10, mean: 0};
    const min = Number.isFinite(s.min) ? s.min : 0;
    const max = Number.isFinite(s.max) ? s.max : 10;
    const mean = Number.isFinite(s.mean) ? s.mean : (min + max) / 2;

    const id = `ctrl_${feat}`;
    const wrap = document.createElement("div");
    wrap.className = "control";
    wrap.dataset.feature = feat;

    wrap.innerHTML = `
      <div class="row">
        <label for="${id}"><b>${feat}</b></label>
        <div class="val"><span id="${id}_val">${fmt(mean, 3)}</span></div>
      </div>

      <input id="${id}" type="range" min="${min}" max="${max}" step="${(max-min)/400 || 0.01}" value="${mean}"/>

      <div class="row" style="margin-top:10px;">
        <div class="val">range: ${fmt(min, 2)} → ${fmt(max, 2)}</div>
        <input id="${id}_num" type="number" value="${mean}" step="0.01" />
      </div>
    `;

    const slider = wrap.querySelector(`#${id}`);
    const num = wrap.querySelector(`#${id}_num`);
    const val = wrap.querySelector(`#${id}_val`);

    const sync = (v) => {
      val.textContent = fmt(v, 3);
      slider.value = v;
      num.value = v;
    };

    slider.addEventListener("input", () => sync(slider.value));
    num.addEventListener("input", () => {
      const v = Number(num.value);
      if (Number.isFinite(v)) {
        // clamp
        const clamped = Math.min(max, Math.max(min, v));
        sync(clamped);
      }
    });

    controlsEl.appendChild(wrap);
  });
}

function collectFeatures(){
  const out = {};
  document.querySelectorAll(".control").forEach(ctrl => {
    const feat = ctrl.dataset.feature;
    const num = ctrl.querySelector(`input[type="number"]`);
    out[feat] = Number(num.value);
  });
  return out;
}

async function predict(){
  setStatus("Running…");

  const features = collectFeatures();
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({features})
  });

  const data = await res.json();

  if (!data.ok){
    setStatus("Error");
    predLabel.textContent = "—";
    predProb.textContent = "—";
    leafNode.textContent = "—";
    leafSamples.textContent = "—";
    probFill.style.width = "0%";
    probPct.textContent = "0%";
    pathEl.innerHTML = `<div class="step"><div class="decision"><b>Error:</b> ${escapeHtml(data.error || "Unknown")}</div></div>`;
    if (data.missing?.length){
      pathEl.innerHTML += `<div class="step"><div class="meta">Missing: ${data.missing.map(escapeHtml).join(", ")}</div></div>`;
    }
    return;
  }

  setStatus(data.warning ? ("OK — " + data.warning) : "OK");

  const p = Number(data.p_positive || 0);
  const label = data.label;

  predLabel.textContent = label === 1 ? "High risk (1)" : "Low risk (0)";
  predProb.textContent = fmt(p, 3);
  leafNode.textContent = `#${data.leaf?.node_id ?? "—"}`;
  leafSamples.textContent = `${data.leaf?.n_samples ?? "—"}`;

  const pct = Math.round(p * 100);
  probFill.style.width = `${pct}%`;
  probPct.textContent = `${pct}%`;

  renderPathDiagram(data.path || []);
  renderPath(data.path || []);
  
}

function renderPath(path){
  pathEl.innerHTML = "";
  path.forEach((step, idx) => {
    const div = document.createElement("div");
    div.className = "step";

    const decision = step.decision || "—";
    const node = step.node_id ?? "—";
    const imp = fmt(step.impurity, 3);
    const ns = step.n_samples ?? "—";
    const pp = fmt(step.p_positive, 3);

    div.innerHTML = `
      <div class="top">
        <div class="decision"><b>${idx === 0 ? "Root" : "Step " + idx}:</b> ${escapeHtml(decision)}</div>
        <div class="meta">node #${escapeHtml(node)} · impurity ${imp}</div>
      </div>
      <div class="mini">
        <div>p(risk=1): <b>${pp}</b></div>
        <div>samples: <b>${escapeHtml(ns)}</b></div>
        <div>feature: <b>${escapeHtml(step.feature || "—")}</b></div>
        ${step.threshold !== null && step.threshold !== undefined ? `<div>threshold: <b>${fmt(step.threshold,3)}</b></div>` : ""}
        ${step.value !== null && step.value !== undefined ? `<div>value: <b>${fmt(step.value,3)}</b></div>` : ""}
      </div>
    `;
    pathEl.appendChild(div);
  });
}

function reset(){
  buildControls();
  setStatus("Ready");
  predLabel.textContent = "—";
  predProb.textContent = "—";
  leafNode.textContent = "—";
  leafSamples.textContent = "—";
  probFill.style.width = "0%";
  probPct.textContent = "0%";
  pathEl.innerHTML = "";
}

function escapeHtml(x){
  return String(x)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}

buildControls();
predictBtn.addEventListener("click", () => predict().catch(err => {
  console.error(err);
  setStatus("Error");
}));
resetBtn.addEventListener("click", reset);

// Auto-run once on load for a baseline result
predict().catch(()=>{});

function renderPathDiagram(path){
  if (!pathViz) return;
  pathViz.innerHTML = "";

  if (!path || path.length === 0){
    pathViz.innerHTML = `<div class="small">No path steps available.</div>`;
    return;
  }

  // -------------------------
  // Layout config
  // -------------------------
  const nodeR = 12;
  const gapY = 120;
  const pad = 30;

  const containerW = pathViz.clientWidth;

  // how many nodes per row (cap so labels don’t collide)
  const maxCols = Math.min(
    5,
    Math.max(2, Math.floor(containerW / 180))
  );

  // 🔑 dynamic horizontal spacing
  const gapX = (containerW - pad * 2) / Math.max(1, maxCols - 1);

  const nodes = path.length + 1;   // steps + leaf
  const rows = Math.ceil(nodes / maxCols);

  const width = pathViz.clientWidth;
  const height = rows * gapY + pad;

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const mk = (n) => document.createElementNS(svgNS, n);

  function nodeXY(i){
    const row = Math.floor(i / maxCols);
    const col = i % maxCols;
    return {
      x: pad + col * gapX,
      y: pad + row * gapY + 50
    };
  }

  // -------------------------
  // Draw connections
  // -------------------------
  for (let i = 1; i < nodes; i++){
    const a = nodeXY(i - 1);
    const b = nodeXY(i);

    const line = mk("line");
    line.setAttribute("x1", a.x + nodeR);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x - nodeR);
    line.setAttribute("y2", b.y);

    // Vertical connector if wrapping to next row
    if (Math.floor(i / maxCols) !== Math.floor((i - 1) / maxCols)){
      line.setAttribute("x1", a.x);
      line.setAttribute("y1", a.y + nodeR);
      line.setAttribute("x2", b.x);
      line.setAttribute("y2", b.y - nodeR);
    }

    line.setAttribute("stroke", "rgba(255,255,255,0.35)");
    line.setAttribute("stroke-width", "2");
    svg.appendChild(line);
  }

  // -------------------------
  // Draw nodes
  // -------------------------
  for (let i = 0; i < nodes; i++){
    const {x, y} = nodeXY(i);

    const step = i < path.length ? path[i] : path[path.length - 1];
    const satisfied = step?.satisfied;

    let fill = "rgba(255,255,255,0.18)";
    let stroke = "rgba(255,255,255,0.55)";
    if (satisfied === true){
      fill = "rgba(34,197,94,0.35)";
      stroke = "rgba(34,197,94,0.9)";
    }
    if (satisfied === false){
      fill = "rgba(239,68,68,0.35)";
      stroke = "rgba(239,68,68,0.9)";
    }

    const c = mk("circle");
    c.setAttribute("cx", x);
    c.setAttribute("cy", y);
    c.setAttribute("r", nodeR);
    c.setAttribute("fill", fill);
    c.setAttribute("stroke", stroke);
    c.setAttribute("stroke-width", "2");
    svg.appendChild(c);

    // Label above
    const t = mk("text");
    t.setAttribute("x", x);
    t.setAttribute("y", y - 22);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "rgba(255,255,255,0.92)");
    t.setAttribute("font-size", "12");

    if (i < path.length){
      const s = path[i];
      t.textContent = `${s.feature} ${s.op} ${fmt(s.threshold, 2)}`;
    } else {
      t.textContent = "Leaf";
    }
    svg.appendChild(t);

    // Value below (if applicable)
    if (i < path.length){
      const t2 = mk("text");
      t2.setAttribute("x", x);
      t2.setAttribute("y", y + 28);
      t2.setAttribute("text-anchor", "middle");
      t2.setAttribute("fill", "rgba(255,255,255,0.7)");
      t2.setAttribute("font-size", "12");
      t2.textContent = `value: ${fmt(path[i].value, 2)}`;
      svg.appendChild(t2);
    }
  }

  pathViz.appendChild(svg);
}

// Allow "Enter / Return" key to trigger Run
document.addEventListener("keydown", (e) => {
  // Ignore if user is typing in a textarea (if you add one later)
  if (e.key === "Enter") {
    e.preventDefault();
    predict().catch(err => {
      console.error(err);
      setStatus("Error");
    });
  }
});