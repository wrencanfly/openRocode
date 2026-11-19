/* openRocode dashboard: episodes table, a task x model comparison, and an episode view with Debug and Watch
   modes. No build step. */
"use strict";

const TOKEN = document.querySelector('meta[name="rocode-token"]').content;
const LIVE_ID = document.querySelector('meta[name="rocode-live"]').content;
const app = document.getElementById("app");
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/* ---- small helpers ------------------------------------------------------------------------- */

const icons = {
  logo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 20h14"/><path d="M8 20v-5l4-3"/><path d="M12 12l5-6"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><path d="M15.5 4.5l3 1.5M17 6l-1 3"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  prev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>',
  next: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  code: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 8l-4 4 4 4M15 8l4 4-4 4"/></svg>',
};

const MODELS = { "claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5",
                 "claude-fable-5-2": "Claude Fable 5.2", "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
                 "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-luna": "GPT-6 Luna", "scripted": "Scripted demo" };
const STATUS = { running: ["live", "Running"], succeeded: ["ok", "Succeeded"], failed: ["err", "Failed"], error: ["err", "Error"],
                 stopped: ["warn", "Stopped"], ended: ["", "Ended"] };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const modelName = (m) => MODELS[m] || m || "Unknown model";
const taskName = (id) => (id || "episode").replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const firstLine = (s) => String(s || "").split("\n")[0];
const statusOf = (s) => STATUS[s] || ["", s || ""];
const fmtTokens = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n || 0}`);
function fmtTime(s) {
  if (s == null) return "–";
  const m = Math.floor(s / 60), r = s - m * 60;
  return m ? `${m}:${r.toFixed(1).padStart(4, "0")}` : `${r.toFixed(1)}s`;
}
const when = (t) => (t ? new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "");
function argsText(name, args) {
  if (name === "run_code" || name === "done") return "";
  return Object.entries(args || {}).filter(([k, v]) => !(name === "nudge" && v === 0) && k !== "code")
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`).join(", ");
}
function highlight(code) {
  let out = esc(code);
  out = out.replace(/(#[^\n]*)/g, '<span class="com">$1</span>');
  out = out.replace(/(&quot;[^\n]*?&quot;|&#39;[^\n]*?&#39;)/g, '<span class="str">$1</span>');
  out = out.replace(/\b(print|if|elif|else|not|and|or|raise|for|in|range|while|True|False|None|import|def|return|break|continue)\b/g, '<span class="kw">$1</span>');
  out = out.replace(/(?<![\w#])(\d+(?:\.\d+)?)\b/g, '<span class="num">$1</span>');
  return out;
}
function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg; t.className = `toast show${isErr ? " err" : ""}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => (t.className = "toast"), 2200);
}

/* ---- theme --------------------------------------------------------------------------------- */

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function paintThemeButton() { $("#theme").innerHTML = currentTheme() === "dark" ? icons.sun : icons.moon; }
$("#theme").onclick = () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("rocode-theme", next); } catch (e) { /* storage unavailable */ }
  paintThemeButton();
};
paintThemeButton();
$(".wordmark").innerHTML = icons.logo + "openRocode";

/* ---- api ----------------------------------------------------------------------------------- */

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
async function control(action, text) {
  const r = await fetch("api/control", {
    method: "POST", headers: { "Content-Type": "application/json", "X-Rocode-Token": TOKEN },
    body: JSON.stringify({ episode: view.id, action, text }),
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

/* ---- routing ------------------------------------------------------------------------------- */

let poller = null;
function stopPolling() { clearInterval(poller); poller = null; }

function route() {
  stopPolling(); stopPlay();
  const h = location.hash.slice(1) || "/";
  const c = h.match(/^\/compare(?:\?(.*))?$/);
  if (c) return openCompare(new URLSearchParams(c[1] || "").get("batch") || "");
  const m = h.match(/^\/e\/([^?]+)(?:\?(.*))?$/);
  if (m) {
    const q = new URLSearchParams(m[2] || "");
    const mode = ["debug", "watch"].includes(q.get("mode")) ? q.get("mode") : null;
    const step = q.has("step") ? Math.max(1, parseInt(q.get("step"), 10) || 1) - 1 : null;  // 1-based in links
    return openEpisode(decodeURIComponent(m[1]), mode, step);
  }
  return openHome();
}
addEventListener("hashchange", route);

/* ---- home: episodes table ------------------------------------------------------------------ */

function sectionNav(current) {
  $("#bar-right").innerHTML = `<div class="segmented" role="group" aria-label="Section">
    <button data-go="#/" aria-pressed="${current === "episodes"}">Episodes</button>
    <button data-go="#/compare" aria-pressed="${current === "compare"}">Compare</button></div>`;
  $$("#bar-right [data-go]").forEach((b) => (b.onclick = () => { location.hash = b.dataset.go; }));
}

async function openHome() {
  $("#crumbs").innerHTML = `<span class="slash">/</span><span>Episodes</span>`;
  sectionNav("episodes");
  app.innerHTML = `<div class="home"><h1>Episodes</h1><p class="sub" id="home-sub">Loading…</p><div id="home-body"></div></div>`;
  const load = async () => {
    let data;
    try { data = await getJSON("api/episodes"); } catch (e) { $("#home-sub").textContent = `Could not load episodes: ${e.message}`; return; }
    $("#home-sub").innerHTML = `Every run recorded in <span class="mono">${esc(data.root)}</span>. Results come from the simulator, which the model never sees.`;
    if (!data.episodes.length) {
      $("#home-body").innerHTML = `<div class="empty-state"><p><b>No episodes yet.</b></p>
        <p>Start one and watch it here: <code>rocode run mug_to_coaster --view</code></p></div>`;
      return;
    }
    $("#home-body").innerHTML = `<div class="table" role="table" aria-label="Episodes">
      <div class="row head" role="row"><span>Task</span><span>Model</span><span>Result</span><span>Turns</span><span>Duration</span><span>Tokens in</span><span>Started</span></div>
      ${data.episodes.map((e) => {
        const [cls, label] = statusOf(e.status);
        return `<a class="row" role="row" href="#/e/${encodeURIComponent(e.id)}">
          <span class="task">${esc(taskName(e.task_id))}</span><span>${esc(modelName(e.model))}</span>
          <span class="st"><span class="dot ${cls}"></span>${label}</span><span class="num">${e.turns ?? "–"}</span>
          <span class="num">${fmtTime(e.seconds)}</span><span class="num">${fmtTokens((e.usage || {}).input_tokens)}</span>
          <span class="when">${esc(when(e.started))}</span></a>`;
      }).join("")}</div>`;
  };
  await load();
  poller = setInterval(load, 2500);
}

/* ---- compare: task x model matrix ---------------------------------------------------------- */

const DOT = { succeeded: "ok", failed: "err", error: "warn", stopped: "warn", ended: "", running: "live", pending: "pending" };
const pct = (r) => (r == null ? "–" : `${Math.round(r * 100)}%`);
const seedName = (s) => (s == null ? "default layout" : `seed ${s}`);

function runDots(cell) {
  return `<div class="runs">${cell.episodes.map((e) => {
    const label = `${seedName(e.seed)}: ${statusOf(e.status)[1] || e.status}${e.turns != null && e.status !== "pending" ? `, ${e.turns} turns` : ""}${e.why ? `. ${e.why}` : ""}`;
    return e.has_trace
      ? `<a class="run ${DOT[e.status] || ""}" href="#/e/${encodeURIComponent(e.id)}" title="${esc(label)}" aria-label="${esc(label)}"></a>`
      : `<span class="run ${DOT[e.status] || ""}" title="${esc(label)}" aria-label="${esc(label)}"></span>`;
  }).join("")}</div>`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function matrixCell(cell) {
  if (!cell) return `<td class="none">–</td>`;
  const waiting = cell.pending + cell.running;
  const aside = [cell.errors && plural(cell.errors, "error"), cell.stopped && `${cell.stopped} stopped`, waiting && `${waiting} to run`]
    .filter(Boolean).map((s) => `<span>${s}</span>`).join("");
  const meta = cell.scored
    ? `<span>${plural(cell.turns_median, "turn")}</span><span>${fmtTime(cell.seconds_median)}</span><span>${fmtTokens(cell.tokens_in_mean)} tokens in</span>`
    : "";
  const rate = cell.scored
    ? `<b>${cell.succeeded}</b><span class="of">/${cell.scored}</span><span class="pct">${pct(cell.rate)}</span>`
    : `<b class="none">–</b>`;
  return `<td><div class="rate">${rate}</div>
    <div class="meter"><i style="width:${(cell.rate || 0) * 100}%;background:var(--ok-dot)"></i></div>
    ${meta ? `<div class="cell-meta">${meta}</div>` : ""}${aside ? `<div class="cell-meta aside">${aside}</div>` : ""}${runDots(cell)}</td>`;
}

async function openCompare(batch) {
  $("#crumbs").innerHTML = `<span class="slash">/</span><span>Compare</span>`;
  sectionNav("compare");
  app.innerHTML = `<div class="home compare"><h1>Compare</h1>
    <p class="sub">Success comes from the simulator's ground truth, which the model never sees. Each dot is one episode; open it to see why.</p>
    <div class="cmp-bar"><label class="select"><span>Batch</span><select id="batch"></select></label><span class="mute" id="cmp-count"></span></div>
    <div id="cmp-body"><p class="mute">Loading…</p></div></div>`;
  const load = async () => {
    let d;
    try { d = await getJSON(`api/compare${batch ? `?batch=${encodeURIComponent(batch)}` : ""}`); } catch (e) {
      $("#cmp-body").innerHTML = `<div class="empty-state"><p><b>Could not load the comparison.</b></p><p>${esc(e.message)}</p></div>`; return false;
    }
    const sel = $("#batch");
    if (document.activeElement !== sel) {
      sel.innerHTML = `<option value="">All recorded episodes</option>` + d.batches.map((b) =>
        `<option value="${esc(b.id)}" ${b.id === batch ? "selected" : ""}>${esc(when(b.created))} · ${b.models.length} model${b.models.length === 1 ? "" : "s"}, ${b.tasks.length} task${b.tasks.length === 1 ? "" : "s"} · ${b.done}/${b.planned}${b.status === "running" ? " running" : b.status === "stopped" ? " stopped" : ""}</option>`).join("");
      sel.onchange = () => { location.hash = sel.value ? `#/compare?batch=${encodeURIComponent(sel.value)}` : "#/compare"; };
    }
    $("#cmp-count").textContent = `${d.episodes} episode${d.episodes === 1 ? "" : "s"}`;
    if (!d.episodes) {
      $("#cmp-body").innerHTML = `<div class="empty-state"><p><b>Nothing to compare yet.</b></p>
        <p>Run the same tasks with several models and seeds:</p>
        <p><code>rocode bench mug_to_coaster --model anthropic:claude-opus-5-5 --model openai:gpt-6.1-sol --repeat 3 --view</code></p></div>`;
      return false;
    }
    const cols = d.columns;
    const head = cols.map((c) => `<th scope="col"><span class="m">${esc(modelName(c.model))}</span>${c.policy ? `<span class="p">${esc(c.policy)}</span>` : ""}</th>`).join("");
    const rows = d.tasks.map((t) => `<tr><th scope="row">${esc(taskName(t))}</th>${cols.map((c) => matrixCell(d.cells[t][c.key])).join("")}</tr>`).join("");
    const total = d.tasks.length > 1 ? `<tr class="total"><th scope="row">All tasks</th>${cols.map((c) => matrixCell(d.totals[c.key])).join("")}</tr>` : "";
    const failures = d.tasks.flatMap((t) => cols.flatMap((c) => ((d.cells[t][c.key] || {}).episodes || [])
      .filter((e) => ["failed", "error", "stopped"].includes(e.status)).map((e) => ({ ...e, task: t, model: c.model }))));
    $("#cmp-body").innerHTML = `<div class="matrix-wrap"><table class="matrix"><thead><tr><th scope="col">Task</th>${head}</tr></thead>
        <tbody>${rows}${total}</tbody></table></div>
      <p class="foot mute">Rates and medians count episodes the model played to a verdict. Errors (API or harness) and emergency stops are listed beside them, not counted against the model. With a few episodes per cell, a difference of one episode can be chance.</p>
      ${failures.length ? `<h2>Why episodes failed</h2><div class="table fails">${failures.map((e) => `
        <${e.has_trace ? `a href="#/e/${encodeURIComponent(e.id)}"` : "div"} class="frow">
          <span class="st"><span class="dot ${DOT[e.status]}"></span>${esc(statusOf(e.status)[1])}</span>
          <span>${esc(taskName(e.task))}</span><span>${esc(modelName(e.model))}</span><span class="mono mute">${esc(seedName(e.seed))}</span>
          <span class="why">${esc(e.why)}</span></${e.has_trace ? "a" : "div"}>`).join("")}</div>` : ""}`;
    return d.batches.some((b) => b.status === "running" && (!batch || b.id === batch)) || cols.some((c) => d.totals[c.key].running);
  };
  if (await load()) poller = setInterval(async () => { if (!(await load())) stopPolling(); }, 2500);
}

/* ---- episode view -------------------------------------------------------------------------- */

const view = { id: null, ep: null, steps: [], idx: 0, follow: true, mode: "debug", sig: "" };

function flattenSteps(ep) {
  const out = [];
  const visit = (call, turn, parent) => {
    if (call.children && call.children.length) { call.children.forEach((c) => visit(c, turn, call)); return; }
    const r = call.result || {}, data = r.data || {};
    const obs = data.observation || (r.obs || []).slice(-1)[0];
    if (obs && ep.observations.some((o) => o.id === obs)) out.push({ call, parent, turn, obs });
  };
  (ep.setup || []).forEach((c) => visit(c, 0, null));
  ep.turns.forEach((t) => t.calls.forEach((c) => visit(c, t.turn, null)));
  return out;
}
const obsById = (id) => view.ep.observations.find((o) => o.id === id);

async function openEpisode(id, mode, step) {
  Object.assign(view, { id, ep: null, steps: [], idx: 0, follow: true, sig: "" });
  app.innerHTML = `<div class="home"><p class="mute">Loading episode…</p></div>`;
  try { view.ep = await getJSON(`api/episodes/${encodeURIComponent(id)}`); } catch (e) {
    app.innerHTML = `<div class="home"><div class="empty-state"><p><b>Could not open this episode.</b></p><p>${esc(e.message)}</p><p><a href="#/">Back to all episodes</a></p></div></div>`;
    return;
  }
  view.mode = mode || (view.ep.live ? "watch" : "debug");
  ingest(view.ep);
  view.idx = step != null ? Math.min(step, view.steps.length - 1) : Math.max(0, view.steps.length - 1);
  view.follow = view.idx === view.steps.length - 1;
  mount();
  if (view.ep.live) poller = setInterval(refresh, 800);
}

function ingest(ep) {
  view.ep = ep;
  view.steps = flattenSteps(ep);
  view.sig = `${ep.status}|${ep.observations.length}|${ep.turns.length}|${ep.turns.map((t) => t.calls.length + ":" + t.notes.length + ":" + (t.text || "").length).join(",")}|${ep.paused}`;
}

async function refresh() {
  let ep;
  try { ep = await getJSON(`api/episodes/${encodeURIComponent(view.id)}`); } catch (e) { return; }
  const before = view.sig;
  const atEnd = view.idx >= view.steps.length - 1;
  ingest(ep);
  if (view.sig === before) return;
  if (view.follow && atEnd) view.idx = Math.max(0, view.steps.length - 1);
  update();
  if (!ep.live) { stopPolling(); toast(`Episode ${statusOf(ep.status)[1].toLowerCase()}`); }
}

function mount() {
  const ep = view.ep;
  $("#crumbs").innerHTML = `<span class="slash">/</span><a href="#/">Episodes</a><span class="slash">/</span>
    <span>${esc(taskName(ep.task_id))} · ${esc(modelName(ep.model))}</span>`;
  $("#bar-right").innerHTML = `<span class="badge" id="status"></span>
    <div class="segmented" role="group" aria-label="Layout">
      <button data-mode="debug" aria-pressed="${view.mode === "debug"}">Debug</button>
      <button data-mode="watch" aria-pressed="${view.mode === "watch"}">Watch</button></div>`;
  $$("#bar-right [data-mode]").forEach((b) => b.onclick = () => { location.hash = `#/e/${encodeURIComponent(view.id)}?mode=${b.dataset.mode}`; });
  app.innerHTML = view.mode === "watch" ? watchSkeleton() : debugSkeleton();
  wireControls();
  wireReplay();
  update();
}

function update() {
  const ep = view.ep;
  const [cls, label] = statusOf(ep.status);
  $("#status").innerHTML = `<span class="dot ${cls}"></span>${ep.paused ? "Paused" : label}`;
  if (view.mode === "watch") updateWatch(); else updateDebug();
  updateControls();
}

/* ---- shared pieces ------------------------------------------------------------------------- */

function frames(o, tagWithId) {
  if (!o) return `<div class="frame empty" style="grid-column:1/-1">${view.ep.live ? "Waiting for the first pictures…" : esc(view.ep.error || "No pictures were taken in this episode.")}</div>`;
  return Object.entries(o.images).map(([cam, src]) =>
    `<div class="frame"><img src="${esc(src)}" alt="${esc(cam)} camera, ${esc(o.id)}"><span class="tag">${tagWithId ? esc(o.id) + " · " : ""}${esc(cam)}</span></div>`).join("");
}

function gripText(r) {
  if (!r) return ["–", ""];
  if (r.holding) return ["Holding", `${r.opening_cm} cm`];
  if (r.opening_cm <= 0.4) return ["Closed", "empty"];
  return ["Open", `${r.opening_cm} cm`];
}

function telemetry(selected, width, height) {
  const pts = view.steps.map((s) => ({ s, r: (obsById(s.obs) || {}).readings || null }));
  if (pts.length < 2) return `<p class="mute" style="margin:8px 0">The chart appears after a few steps.</p>`;
  const n = pts.length - 1, padL = 34, padR = 8, padT = 10, padB = 18;
  const x = (i) => padL + (i / n) * (width - padL - padR);
  const maxH = Math.max(20, ...pts.map((p) => (p.r && p.r.height_cm) || 0));
  const y = (v, max) => padT + (1 - v / max) * (height - padT - padB);
  const path = (vals, max, step) => {
    let d = "";
    vals.forEach((v, i) => { if (v == null) return; const px = x(i).toFixed(1), py = y(v, max).toFixed(1); d += !d ? `M${px},${py}` : step ? `H${px}V${py}` : `L${px},${py}`; });
    return d;
  };
  const hs = pts.map((p) => p.r && p.r.height_cm), os = pts.map((p) => p.r && p.r.opening_cm), fs = pts.map((p) => (p.r && p.r.contact_n) || 0);
  const grid = [0, maxH / 2, maxH].map((v) => `<line x1="${padL}" x2="${width - padR}" y1="${y(v, maxH)}" y2="${y(v, maxH)}" stroke="var(--hairline)"/>
    <text x="${padL - 6}" y="${y(v, maxH) + 4}" text-anchor="end" fill="var(--mute)" font-size="10" font-family="var(--mono)">${Math.round(v)}</text>`).join("");
  const half = (x(1) - x(0)) / 2;
  const turns = pts.map((p, i) => (i > 0 && p.s.turn !== pts[i - 1].s.turn ? `<line x1="${x(i) - half}" x2="${x(i) - half}" y1="${padT}" y2="${height - padB}" stroke="var(--hairline)" stroke-dasharray="2 3"/>` : "")).join("");
  const forces = fs.map((f, i) => (f > 0 ? `<line x1="${x(i)}" x2="${x(i)}" y1="${height - padB}" y2="${height - padB - Math.min(1, f / 10) * 26}" stroke="var(--warning-dot)" stroke-width="3" stroke-linecap="round"/>` : "")).join("");
  const dots = hs.map((v, i) => (v == null ? "" : `<circle cx="${x(i)}" cy="${y(v, maxH)}" r="${i === selected ? 4.5 : 2.5}" fill="${i === selected ? "var(--ink)" : "var(--elevated)"}" stroke="var(--ink)" stroke-width="1.5"/>`)).join("");
  const every = Math.ceil(pts.length / 14);
  const labels = pts.map((p, i) => (i % every === 0 || i === selected ? `<text x="${x(i)}" y="${height - 4}" text-anchor="middle" fill="${i === selected ? "var(--ink)" : "var(--mute)"}" font-size="10" font-family="var(--mono)">${esc(p.s.obs)}</text>` : "")).join("");
  const hits = pts.map((p, i) => `<rect data-i="${i}" x="${x(i) - half}" y="0" width="${half * 2}" height="${height}" fill="transparent" style="cursor:pointer"><title>${esc(p.s.obs)}</title></rect>`).join("");
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" role="img" aria-label="Fingertip height, gripper opening and contact force over the episode">
    ${grid}${turns}<line x1="${x(selected)}" x2="${x(selected)}" y1="${padT - 4}" y2="${height - padB + 4}" stroke="var(--link)" stroke-width="1.5"/>
    <path d="${path(os, 9, true)}" fill="none" stroke="var(--link)" stroke-width="1.5" stroke-dasharray="4 3"/>
    <path d="${path(hs, maxH)}" fill="none" stroke="var(--ink)" stroke-width="1.75" stroke-linejoin="round"/>
    ${forces}${dots}${labels}${hits}</svg>`;
}
const legend = `<span><span class="legend"><i style="background:var(--ink)"></i>Fingertip height, cm</span><span class="legend"><i style="background:var(--link)"></i>Gripper opening, cm</span><span class="legend"><i style="background:var(--warning-dot)"></i>Contact force</span></span>`;

function wireChart(root) { $$("rect[data-i]", root).forEach((r) => r.onclick = () => go(+r.dataset.i)); }

function controlsHTML(dock) {
  return `<label class="composer"><span class="sr">Message the model</span>
      <input id="say" maxlength="2000" autocomplete="off">
      ${dock ? "" : `<button class="btn primary icon-btn" id="send" aria-label="Send to the model">${icons.send}</button>`}</label>
    <button class="btn" id="pause"></button>
    <button class="btn danger" id="stop">${icons.stop}Stop</button>`;
}

function wireControls() {
  const say = $("#say");
  const send = async () => {
    const text = say.value.trim();
    if (!text) return;
    try { await control("say", text); say.value = ""; toast("Sent. The model reads it before its next move."); } catch (e) { toast(e.message, true); }
  };
  say.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  if ($("#send")) $("#send").onclick = send;
  $("#pause").onclick = async () => {
    try { const r = await control(view.ep.paused ? "resume" : "pause"); view.ep.paused = r.paused; updateControls(); toast(r.paused ? "Pausing at the next safe point" : "Resumed"); } catch (e) { toast(e.message, true); }
  };
  $("#stop").onclick = async () => {
    try { await control("stop"); toast("Emergency stop sent"); } catch (e) { toast(e.message, true); }
  };
}

function updateControls() {
  const ep = view.ep, can = ep.controllable && ep.live;
  const say = $("#say");
  say.disabled = !can; $("#pause").disabled = !can; $("#stop").disabled = !can;
  if ($("#send")) $("#send").disabled = !can;
  say.placeholder = can ? "Tell the model something before its next move…"
    : ep.live ? "Controls work for the run started with rocode run --view" : "This run has ended. Controls are for live runs.";
  $("#pause").innerHTML = ep.paused ? `${icons.play}Resume` : `${icons.pause}Pause`;
}

/* ---- replay -------------------------------------------------------------------------------- */

let player = null;
function stopPlay() { clearInterval(player); player = null; paintPlay(); }
function paintPlay() { const b = $("#play"); if (b) b.innerHTML = player ? `${icons.pause}Pause replay` : `${icons.play}Replay`; }
function togglePlay() {
  if (player) return stopPlay();
  if (view.idx >= view.steps.length - 1) view.idx = 0;
  player = setInterval(() => { if (view.idx >= view.steps.length - 1) stopPlay(); else go(view.idx + 1, true); }, 900);
  paintPlay(); update();
}
function go(i, fromPlayer) {
  if (!view.steps.length) return;
  view.idx = Math.max(0, Math.min(view.steps.length - 1, i));
  view.follow = view.idx === view.steps.length - 1;
  if (!fromPlayer && player) stopPlay();
  update();
}
function wireReplay() {
  const p = $("#prev"), n = $("#next"), pl = $("#play");
  if (p) p.onclick = () => go(view.idx - 1);
  if (n) n.onclick = () => go(view.idx + 1);
  if (pl) pl.onclick = togglePlay;
  paintPlay();
}
document.addEventListener("keydown", (e) => {
  if (!view.ep || ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) || !location.hash.startsWith("#/e/")) return;
  if (e.key === "ArrowRight" || e.key === "j") { e.preventDefault(); go(view.idx + 1); }
  if (e.key === "ArrowLeft" || e.key === "k") { e.preventDefault(); go(view.idx - 1); }
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
});

/* ---- debug mode ---------------------------------------------------------------------------- */

function debugSkeleton() {
  return `<div class="debug">
    <section class="timeline" id="tl" aria-label="Timeline"></section>
    <section class="stage" aria-label="Camera views">
      <div class="stage-bar"><span class="where" id="where"></span><span class="mute mono" id="when"></span><span class="spacer"></span>
        <button class="btn icon-btn" id="prev" aria-label="Previous step">${icons.prev}</button>
        <button class="btn" id="play"></button>
        <button class="btn icon-btn" id="next" aria-label="Next step">${icons.next}</button></div>
      <div class="views" id="views"></div>
      <div class="scrub"><div class="ticks" id="ticks"></div><div class="ops">${controlsHTML(false)}</div></div>
    </section></div>`;
}

function callRow(c, stepIndex) {
  const r = c.result;
  const mark = !r ? `<span class="mark run">●</span>` : r.error ? `<span class="mark err">✕</span>` : `<span class="mark ok">✓</span>`;
  const res = c.denied ? `Refused: ${c.denied}` : c.name === "done" ? (c.args || {}).summary : r ? firstLine(r.text) : "Running…";
  return `<div class="call" ${stepIndex != null ? `data-step="${stepIndex}" role="button" tabindex="0"` : ""} aria-current="${stepIndex === view.idx}">
    ${mark}<span class="sig">${esc(c.name)}<span>(${esc(argsText(c.name, c.args))})</span></span>
    <span class="obs">${esc((r && r.data && r.data.observation) || "")}</span><span class="res">${esc(res)}</span></div>`;
}

function noteRow(n) {
  const label = { OperatorInput: "You said", Paused: "Paused", Resumed: "Resumed", EStop: "Emergency stop", Notice: "Note" }[n.kind] || n.kind;
  return `<div class="note"><b>${label}</b><span>${esc(n.text)}</span><span class="spacer"></span><span class="mono mute">${fmtTime(n.t)}</span></div>`;
}

function updateDebug() {
  const ep = view.ep, tl = $("#tl"), scroll = tl.scrollTop;
  const stepOf = new Map(view.steps.map((s, i) => [s.call, i]));
  const u = ep.usage || {};
  let html = `<div class="ep-head"><h1>${esc(taskName(ep.task_id))}</h1><p>${esc(ep.task)}</p>
    <dl class="stats"><div><dt>Model</dt><dd>${esc(modelName(ep.model))}</dd></div><div><dt>Turns</dt><dd>${ep.turns.length}</dd></div>
    <div><dt>Duration</dt><dd>${fmtTime(ep.seconds)}</dd></div><div><dt>Tokens</dt><dd>${fmtTokens(u.input_tokens)} in, ${fmtTokens(u.output_tokens)} out</dd></div></dl></div>`;
  ep.notes.forEach((n) => (html += `<div class="turn">${noteRow(n)}</div>`));
  for (const t of ep.turns) {
    html += `<div class="turn"><div class="turn-label"><b>Turn ${t.turn}</b><span class="mono">${fmtTime(t.t)}</span></div>`;
    if (t.reasoning) html += `<details class="reasoning"><summary><b>Thinking</b><span class="pv">${esc(firstLine(t.reasoning).replace(/\*\*/g, "").slice(0, 90))}</span></summary><p>${esc(t.reasoning.replace(/\*\*/g, ""))}</p></details>`;
    if (t.text) html += `<p class="thought">${esc(t.text)}</p>`;
    for (const c of t.calls) {
      if (c.name === "run_code") {
        const r = c.result;
        const out = r ? r.text.split("\n").filter((l) => !/^\s+\d+\. /.test(l) && !/^(Robot calls:|Program |Output:)/.test(l)).join("\n").trim() : "";
        const state = !r ? `<span class="mark run">running</span>` : r.error ? `<span style="color:var(--error)">failed</span>` : `<span style="color:var(--ok)">finished</span>`;
        html += `<div class="program"><div class="head">${icons.code} run_code <span class="mute">· ${c.children.length} robot call${c.children.length === 1 ? "" : "s"}</span><span class="spacer"></span>${state}</div>
          <pre class="code">${highlight((c.args.code || "").trim())}</pre>
          <div class="calls">${c.children.map((ch) => callRow(ch, stepOf.get(ch))).join("")}</div>
          ${out ? `<details class="out"${r && r.error ? " open" : ""}><summary>Program output · ${out.split("\n").length} lines</summary><pre>${esc(out)}</pre></details>` : ""}</div>`;
      } else {
        html += callRow(c, stepOf.get(c));
      }
    }
    t.notes.forEach((n) => (html += noteRow(n)));
    if (ep.live && t === ep.turns[ep.turns.length - 1] && !t.calls.length) html += `<div class="thinking"><span class="dot live"></span>The model is deciding…</div>`;
    html += `</div>`;
  }
  const [cls, label] = statusOf(ep.status);
  if (ep.status === "error") {
    html += `<div class="finish"><div class="card err"><b style="color:var(--error)">The episode ended with an error</b><p class="body" style="margin:6px 0 0">${esc(ep.error)}</p></div></div>`;
  } else if (!ep.live) {
    const claim = ep.finish ? `The model said: ${esc(ep.finish.summary)}` : `Ended by ${esc(ep.reason || "unknown")}.`;
    html += `<div class="finish"><div class="card"><div style="display:flex;gap:8px;align-items:center;margin-bottom:6px"><span class="dot ${cls}"></span><b>${label}</b>
      <span class="mute">according to the simulator</span></div><p class="body" style="margin:0">${claim}</p></div></div>`;
  }
  tl.innerHTML = html;
  tl.scrollTop = scroll;
  $$(".call[data-step]", tl).forEach((el) => {
    el.onclick = () => go(+el.dataset.step);
    el.onkeydown = (e) => { if (e.key === "Enter") go(+el.dataset.step); };
  });
  const cur = $(`.call[data-step="${view.idx}"]`, tl);
  if (cur && (view.follow || player)) cur.scrollIntoView({ block: "nearest" });

  const st = view.steps[view.idx], o = st && obsById(st.obs);
  $("#where").textContent = st ? (st.turn === 0 ? "First look" : `${st.call.name}${st.parent ? " in run_code" : ""}`) : "";
  $("#when").textContent = o ? `${o.id} · ${fmtTime(o.t)}` : "";
  const r = o && o.readings;
  const [gripLabel, gripDetail] = gripText(r);
  $("#views").innerHTML = frames(o, false) + (o ? `<div class="readout">
      <div class="cell"><div class="k">Fingertip height</div><div class="v">${r ? r.height_cm : "–"}<small> cm</small></div><div class="meter"><i style="width:${r ? Math.min(100, r.height_cm / 25 * 100) : 0}%"></i></div></div>
      <div class="cell"><div class="k">Gripper</div><div class="v">${gripLabel} <small>${gripDetail}</small></div><div class="meter"><i style="width:${r ? r.opening_cm / 9 * 100 : 0}%"></i></div></div>
      <div class="cell"><div class="k">Contact force</div><div class="v">${r ? r.contact_n : "–"}<small> N</small></div><div class="meter"><i style="width:${r ? Math.min(100, r.contact_n / 10 * 100) : 0}%;background:var(--warning-dot)"></i></div></div>
    </div><div class="chart-card"><div class="chart-head"><b>Over the episode</b>${legend}</div>${telemetry(view.idx, 720, 150)}</div>` : "");
  wireChart($("#views"));
  $("#ticks").innerHTML = view.steps.map((s, i) => `<button data-i="${i}" aria-current="${i === view.idx}" aria-label="Step ${i + 1}, ${esc(s.obs)}">${esc(s.obs)}</button>`).join("");
  $$("#ticks button").forEach((b) => (b.onclick = () => go(+b.dataset.i)));
  const ticks = $("#ticks"), tick = $(`#ticks button[data-i="${view.idx}"]`);
  if (tick) ticks.scrollLeft = Math.max(0, tick.offsetLeft - ticks.offsetLeft - (ticks.clientWidth - tick.offsetWidth) / 2);
}

/* ---- watch mode ---------------------------------------------------------------------------- */

function watchSkeleton() {
  return `<div class="watch">
      <div class="title-row" id="titlerow"></div>
      <section class="big-stage" id="bigstage" aria-label="Camera views"></section>
      <section class="caption" id="caption" aria-live="polite"></section>
      <section class="chart"><div class="chart-head"><b>Telemetry</b>${legend}</div><div id="chart"></div></section>
      <div class="strip-head"><b>Every picture the model saw</b><span class="mute" id="count"></span><span class="spacer"></span>
        <button class="btn icon-btn" id="prev" aria-label="Previous picture">${icons.prev}</button>
        <button class="btn" id="play"></button>
        <button class="btn icon-btn" id="next" aria-label="Next picture">${icons.next}</button></div>
      <section class="strip" id="strip" aria-label="Pictures"></section>
    </div>
    <div class="dock" role="group" aria-label="Operator controls"><span class="badge" id="dockstate"></span><span class="sep"></span>${controlsHTML(true)}</div>`;
}

function updateWatch() {
  const ep = view.ep, u = ep.usage || {};
  const [cls, label] = statusOf(ep.status);
  $("#titlerow").innerHTML = `<div><h1>${esc(taskName(ep.task_id))}</h1><p class="task">${esc(ep.task)}</p></div>
    <dl class="facts"><div><dt>Model</dt><dd>${esc(modelName(ep.model))}</dd></div><div><dt>Result</dt><dd>${label}</dd></div>
    <div><dt>Turns</dt><dd>${ep.turns.length}</dd></div><div><dt>Duration</dt><dd>${fmtTime(ep.seconds)}</dd></div>
    <div><dt>Tokens</dt><dd>${fmtTokens(u.input_tokens)} / ${fmtTokens(u.output_tokens)}</dd></div></dl>`;
  $("#dockstate").innerHTML = `<span class="dot ${cls}"></span>${ep.live ? (ep.paused ? "Paused" : "Live") : `Replay · ${label}`}`;
  const st = view.steps[view.idx], o = st && obsById(st.obs);
  $("#bigstage").innerHTML = frames(o, true);
  const turn = st && ep.turns.find((t) => t.turn === st.turn);
  const r = o && o.readings;
  if (st) {
    const sig = st.turn === 0 ? `<span class="sig">First look</span>` :
      `<span class="sig">${esc(st.call.name)}<span>(${esc(argsText(st.call.name, st.call.args))})</span></span>${st.parent ? `<span class="in-program">run_code</span>` : ""}`;
    const res = st.turn === 0 ? ep.task : firstLine(st.call.result && st.call.result.text);
    const [gl, gd] = gripText(r);
    $("#caption").innerHTML = `<div class="step-no">${String(view.idx + 1).padStart(2, "0")}</div>
      <div>${sig}<div class="res">${esc(res)}</div>${turn && turn.text ? `<div class="said"><b>Model:</b> ${esc(turn.text)}</div>` : ""}</div>
      <dl class="readings"><div><dt>Height</dt><dd>${r ? r.height_cm + " cm" : "–"}</dd></div><div><dt>Gripper</dt><dd>${gl} ${gd}</dd></div>
        <div><dt>Force</dt><dd>${r ? r.contact_n + " N" : "–"}</dd></div><div><dt>Time</dt><dd>${o ? fmtTime(o.t) : "–"}</dd></div></dl>`;
  } else {
    $("#caption").innerHTML = ep.live ? `<div></div><div class="thinking"><span class="dot live"></span>Waiting for the first step…</div><div></div>` : "";
  }
  $("#chart").innerHTML = telemetry(view.idx, 1272, 140);
  wireChart($("#chart"));
  $("#count").textContent = `${view.steps.length} pictures, ${ep.turns.length} turns`;
  $("#strip").innerHTML = view.steps.map((s, i) => {
    const ob = obsById(s.obs);
    const newTurn = i === 0 || view.steps[i - 1].turn !== s.turn;
    return `<button class="shot" data-i="${i}" ${newTurn ? `data-turn="${s.turn ? "Turn " + s.turn : "Start"}"` : ""} aria-current="${i === view.idx}">
      <div class="frame"><img src="${esc(Object.values(ob.images)[0])}" alt="${esc(ob.id)}" loading="lazy"></div>
      <div class="lbl"><span>${esc(ob.id)}</span><span>${s.turn === 0 ? "first look" : esc(s.call.name)}</span></div></button>`;
  }).join("");
  $$(".shot").forEach((b) => (b.onclick = () => go(+b.dataset.i)));
  // Scroll only the strip sideways; scrollIntoView would also scroll the page on every step.
  const strip = $("#strip"), cur = $(`.shot[data-i="${view.idx}"]`);
  if (cur) strip.scrollLeft = Math.max(0, cur.offsetLeft - strip.offsetLeft - (strip.clientWidth - cur.offsetWidth) / 2);
}

/* ---- start --------------------------------------------------------------------------------- */

if (!location.hash && LIVE_ID) location.hash = `#/e/${encodeURIComponent(LIVE_ID)}?mode=watch`;
else route();
