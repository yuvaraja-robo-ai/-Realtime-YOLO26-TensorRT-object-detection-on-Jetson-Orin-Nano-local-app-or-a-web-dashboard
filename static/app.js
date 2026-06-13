// Dashboard logic: WS stats, live controls, class filter, snapshot.
const $ = (id) => document.getElementById(id);

let selectedClasses = new Set();   // empty => all classes
let allNames = {};                 // id -> name

// ---- WebSocket stats ----------------------------------------------------
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    $("fps").textContent = d.fps ?? 0;
    $("gpu").textContent = d.gpu ?? "--";
    $("temp").textContent = d.temp ?? "--";
    $("ram").textContent =
      d.ram_used != null ? `${d.ram_used}/${d.ram_total}MB` : "--";
    $("model").textContent = "model: " + (d.model_name ?? "…");
    $("liveDot").classList.toggle("live", (d.fps ?? 0) > 0);
    renderCounts(d.counts || {}, d.detect_enabled);
    if (d.active_model) {
      $("modelSelect").value = d.active_model === "custom" ? "custom" : "base";
    }
    if (d.training) updateTraining(d.training);
  };
  ws.onclose = () => {
    $("liveDot").classList.remove("live");
    setTimeout(connectWS, 1500);   // auto-reconnect
  };
}

function renderCounts(counts, enabled) {
  const ul = $("counts");
  if (enabled === false) {
    ul.innerHTML = '<li class="muted">detection paused</li>';
    return;
  }
  const keys = Object.keys(counts).sort();
  if (!keys.length) {
    ul.innerHTML = '<li class="muted">no objects</li>';
    return;
  }
  ul.innerHTML = keys
    .map((k) => `<li><span>${k}</span><b>${counts[k]}</b></li>`)
    .join("");
}

// ---- Controls -----------------------------------------------------------
async function postConfig(body) {
  await fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

$("conf").addEventListener("input", (e) => {
  $("confVal").textContent = parseFloat(e.target.value).toFixed(2);
});
$("conf").addEventListener("change", (e) =>
  postConfig({ conf: parseFloat(e.target.value) })
);

$("detectToggle").addEventListener("change", (e) =>
  postConfig({ detect_enabled: e.target.checked })
);
$("trackToggle").addEventListener("change", (e) =>
  postConfig({ track: e.target.checked })
);
$("claheToggle").addEventListener("change", (e) =>
  postConfig({ preprocess_clahe: e.target.checked })
);

$("snapBtn").addEventListener("click", async () => {
  $("snapMsg").textContent = "saving…";
  const res = await fetch("/snapshot");
  if (!res.ok) { $("snapMsg").textContent = "no frame yet"; return; }
  const path = res.headers.get("X-Saved-Path") || "saved";
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "snapshot.jpg"; a.click();
  URL.revokeObjectURL(url);
  $("snapMsg").textContent = "saved on Orin: " + path;
});

// ---- Deep Scan (SAHI sliced inference for small objects) ----------------
$("deepScanBtn").addEventListener("click", async () => {
  const btn = $("deepScanBtn");
  btn.disabled = true;
  $("deepMsg").textContent = "scanning… (slicing frame, may take a few seconds)";
  try {
    const res = await fetch("/deepscan", { method: "POST" });
    if (!res.ok) {
      $("deepMsg").textContent = "scan failed: " + (await res.text());
      return;
    }
    const n = res.headers.get("X-Det-Count") || "?";
    const counts = JSON.parse(res.headers.get("X-Counts") || "{}");
    const blob = await res.blob();
    $("deepImg").src = URL.createObjectURL(blob);
    const keys = Object.keys(counts).sort();
    $("deepCounts").innerHTML = keys.length
      ? keys.map((k) => `<li><span>${k}</span><b>${counts[k]}</b></li>`).join("")
      : '<li class="muted">no objects</li>';
    $("deepCaption").textContent = `Deep Scan — ${n} objects`;
    $("deepModal").classList.remove("hidden");
    $("deepMsg").textContent = `found ${n} objects`;
  } finally {
    btn.disabled = false;
  }
});
$("deepClose").addEventListener("click", () =>
  $("deepModal").classList.add("hidden"));

// ---- Class filter -------------------------------------------------------
async function loadClasses() {
  const res = await fetch("/classes");
  allNames = (await res.json()).names;
  renderClassList("");
}

function pushClassFilter() {
  const ids = [...selectedClasses];
  postConfig({ classes: ids });  // [] => server treats as all
}

function renderClassList(filter) {
  const box = $("classList");
  const f = filter.toLowerCase();
  box.innerHTML = "";
  Object.entries(allNames).forEach(([id, name]) => {
    if (f && !name.toLowerCase().includes(f)) return;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedClasses.has(+id);
    cb.addEventListener("change", () => {
      cb.checked ? selectedClasses.add(+id) : selectedClasses.delete(+id);
      pushClassFilter();
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(name));
    box.appendChild(lab);
  });
}

$("classSearch").addEventListener("input", (e) => renderClassList(e.target.value));
$("clearClasses").addEventListener("click", () => {
  selectedClasses.clear();
  pushClassFilter();
  renderClassList($("classSearch").value);
});

// =========================================================================
//  TRAINING: capture -> label -> train -> hot-swap
// =========================================================================
let captures = [];
let capIdx = 0;
let boxes = [];           // current image: {cls_name, cx, cy, w, h}
let dsClasses = [];       // dataset class names
const img = new Image();
const canvas = $("labCanvas");
const ctx = canvas.getContext("2d");

async function refreshDsStats() {
  const s = await (await fetch("/dataset/stats")).json();
  $("dsStats").textContent =
    `dataset: ${s.num_images} imgs · ${s.num_labeled} labeled · ${s.classes.length} classes`;
  dsClasses = s.classes.map((c) => c.name);
  fillClassSelect();
}

function fillClassSelect() {
  const sel = $("labClass");
  sel.innerHTML = dsClasses.length
    ? dsClasses.map((c) => `<option>${c}</option>`).join("")
    : '<option value="">(add a class →)</option>';
}

// ---- capture ----
$("captureBtn").addEventListener("click", async () => {
  const r = await fetch("/capture", { method: "POST" });
  if (!r.ok) return;
  await refreshDsStats();
  $("dsStats").textContent += "  ✓ captured";
});

// ---- model selector ----
$("modelSelect").addEventListener("change", async (e) => {
  const r = await fetch("/model/select", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: e.target.value }),
  });
  if (!r.ok) { alert("No custom model trained yet."); e.target.value = "base"; }
});

// ---- training ----
$("epochs").addEventListener("input", (e) => ($("epVal").textContent = e.target.value));
$("trainBtn").addEventListener("click", async () => {
  const epochs = +$("epochs").value;
  const r = await fetch("/train", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs }),
  });
  const j = await r.json();
  if (!j.started) $("trainStatus").textContent = "not started: " + (j.reason || "?");
});

function updateTraining(t) {
  const el = $("trainStatus");
  el.classList.remove("err", "ok");
  let pct = 0;
  if (t.state === "training" && t.total) {
    pct = Math.round((t.epoch / t.total) * 100);
    el.textContent = `training… epoch ${t.epoch}/${t.total}`;
  } else if (t.state === "idle") {
    el.textContent = "idle";
  } else if (t.state === "done") {
    el.textContent = "✓ " + t.msg; el.classList.add("ok"); pct = 100;
  } else if (t.state === "error") {
    el.textContent = "error: " + t.msg; el.classList.add("err");
  } else {
    el.textContent = t.msg || t.state;
    pct = t.state === "exporting" || t.state === "swapping" ? 100 : 0;
  }
  $("trainBar").style.width = pct + "%";
}

// ---- labeling modal ----
async function openLabeler() {
  const data = await (await fetch("/captures")).json();
  captures = data.captures;
  dsClasses = data.classes;
  fillClassSelect();
  if (!captures.length) { alert("Capture some frames first."); return; }
  // jump to first unlabeled, else first
  capIdx = Math.max(0, captures.findIndex((c) => !c.labeled));
  $("labeler").classList.remove("hidden");
  loadCapture();
}

function loadCapture() {
  const cap = captures[capIdx];
  $("labCaption").textContent =
    `Label  ${capIdx + 1}/${captures.length}  ·  ${cap.id}`;
  boxes = [];
  $("labMsg").textContent = "";
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    redraw();
  };
  img.src = `/capture/${cap.id}?t=${Date.now()}`;
}

function redraw() {
  ctx.drawImage(img, 0, 0);
  boxes.forEach((b, i) => {
    const x = (b.cx - b.w / 2) * canvas.width;
    const y = (b.cy - b.h / 2) * canvas.height;
    const w = b.w * canvas.width, h = b.h * canvas.height;
    ctx.lineWidth = 3; ctx.strokeStyle = "#2ea043";
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#2ea043";
    ctx.font = "16px sans-serif";
    ctx.fillText(`${i}:${b.cls_name}`, x + 3, y + 16);
  });
  renderBoxList();
}

function renderBoxList() {
  $("boxList").innerHTML = boxes.length
    ? boxes.map((b, i) =>
        `<li><span>${i}: ${b.cls_name}</span><button data-i="${i}">del</button></li>`).join("")
    : '<li class="muted">no boxes</li>';
  $("boxList").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      boxes.splice(+btn.dataset.i, 1); redraw();
    }));
}

// draw new box by drag
let dragging = false, sx = 0, sy = 0;
function canvasXY(ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((ev.clientX - r.left) / r.width) * canvas.width,
    y: ((ev.clientY - r.top) / r.height) * canvas.height,
  };
}
canvas.addEventListener("mousedown", (e) => {
  dragging = true; const p = canvasXY(e); sx = p.x; sy = p.y;
});
canvas.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const p = canvasXY(e);
  redraw();
  ctx.lineWidth = 2; ctx.strokeStyle = "#58a6ff";
  ctx.strokeRect(sx, sy, p.x - sx, p.y - sy);
});
canvas.addEventListener("mouseup", (e) => {
  if (!dragging) return;
  dragging = false;
  const p = canvasXY(e);
  const x0 = Math.min(sx, p.x), y0 = Math.min(sy, p.y);
  const w = Math.abs(p.x - sx), h = Math.abs(p.y - sy);
  if (w < 5 || h < 5) return;
  const cls = $("labClass").value;
  if (!cls) { $("labMsg").textContent = "add a class first"; return; }
  boxes.push({
    cls_name: cls,
    cx: (x0 + w / 2) / canvas.width, cy: (y0 + h / 2) / canvas.height,
    w: w / canvas.width, h: h / canvas.height,
  });
  redraw();
});

$("suggestBtn").addEventListener("click", async () => {
  const cap = captures[capIdx];
  $("labMsg").textContent = "running model…";
  const r = await fetch("/label/suggest", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: cap.id, boxes: [] }),
  });
  const j = await r.json();
  (j.boxes || []).forEach((b) => boxes.push({
    cls_name: b.cls_name, cx: b.cx, cy: b.cy, w: b.w, h: b.h,
  }));
  $("labMsg").textContent = `added ${(j.boxes || []).length} candidates — fix classes & boxes`;
  redraw();
});

$("addClassBtn").addEventListener("click", async () => {
  const name = $("newClass").value.trim();
  if (!name) return;
  const r = await (await fetch("/class", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })).json();
  dsClasses = r.classes; fillClassSelect();
  $("labClass").value = name; $("newClass").value = "";
});

async function saveCapture() {
  const cap = captures[capIdx];
  await fetch("/label", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: cap.id, boxes }),
  });
  cap.labeled = true; cap.boxes = boxes.length;
  $("labMsg").textContent = "saved ✓";
  await refreshDsStats();
}
$("saveCap").addEventListener("click", saveCapture);
$("nextCap").addEventListener("click", () => {
  if (capIdx < captures.length - 1) { capIdx++; loadCapture(); }
});
$("prevCap").addEventListener("click", () => {
  if (capIdx > 0) { capIdx--; loadCapture(); }
});
$("openLabeler").addEventListener("click", openLabeler);
$("labClose").addEventListener("click", () => $("labeler").classList.add("hidden"));

// ---- boot ---------------------------------------------------------------
connectWS();
loadClasses();
refreshDsStats();
