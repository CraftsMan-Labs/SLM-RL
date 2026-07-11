"""The playground page: one self-contained HTML document (inline CSS +
vanilla JS, no CDN, no external requests) -- same pattern as
`slm_rl/webui/page.py`. Read-WRITE: the "run experiment" and "evolve"
buttons POST to this server (see server.py's docstring for the trust
model).
"""

from __future__ import annotations

PAGE: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SLM-RL — workshop playground</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    color-scheme: dark;
    --bg: #14161a;
    --card: #1d2026;
    --border: #2c313a;
    --text: #e6e8eb;
    --muted: #8b93a1;
    --ok: #3ddc84;
    --warn: #f2b632;
    --bad: #f2494c;
    --accent: #5b8def;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 1rem;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: baseline;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 1.1rem; margin: 0; }
  header .stat { color: var(--muted); font-size: 0.9rem; }
  .layout { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-start; }
  .col { flex: 1 1 380px; min-width: 320px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
  }
  .card h2 { font-size: 0.95rem; margin: 0 0 0.7rem 0; }
  label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 0.15rem; }
  .field { margin-bottom: 0.6rem; }
  input[type=text], input[type=number], select, textarea {
    width: 100%;
    background: #0f1115;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 4px;
    padding: 0.35rem 0.5rem;
    font-size: 0.85rem;
    font-family: inherit;
  }
  textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    min-height: 260px;
    white-space: pre;
  }
  .knob-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 0.8rem; }
  button {
    background: var(--accent);
    color: #0f1115;
    border: none;
    border-radius: 4px;
    padding: 0.45rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.small {
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    background: #23262d;
    color: var(--text);
    border: 1px solid var(--border);
    font-weight: 400;
  }
  #msg { font-size: 0.8rem; margin-top: 0.5rem; min-height: 1.1em; }
  #msg.err { color: var(--bad); }
  #msg.ok { color: var(--ok); }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; }
  tr.baseline { color: var(--muted); }
  .status { font-size: 0.72rem; padding: 0.1rem 0.4rem; border-radius: 4px; background: #23262d; }
  .status.complete { color: var(--ok); }
  .status.running { color: var(--warn); }
  .mix { font-size: 0.72rem; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>SLM-RL — workshop playground</h1>
  <span class="stat" id="stat-game">game: —</span>
</header>
<div class="layout">
  <div class="col">
    <div class="card">
      <h2>New experiment</h2>
      <div class="field">
        <label for="f-name">name</label>
        <input type="text" id="f-name" placeholder="e.g. tighter-loop">
      </div>
      <div class="knob-grid" id="knob-grid"></div>
      <div class="field" style="margin-top:0.6rem;">
        <label for="f-agent">agent</label>
        <select id="f-agent"><option value="solver">solver (teacher)</option><option value="random">random</option></select>
      </div>
      <div class="field">
        <label for="f-episodes">episodes</label>
        <input type="number" id="f-episodes" value="30" min="1" max="200">
      </div>
      <div class="field">
        <label for="f-seed">seed</label>
        <input type="number" id="f-seed" value="20000">
      </div>
      <button id="btn-run" type="button">run experiment</button>
      <div id="msg"></div>
    </div>
    <div class="card">
      <h2>reward code</h2>
      <textarea id="f-reward" spellcheck="false"></textarea>
    </div>
  </div>
  <div class="col" style="flex-basis: 500px;">
    <div class="card">
      <h2>scoreboard</h2>
      <table>
        <thead>
          <tr>
            <th>name</th><th>episodes</th><th>mean</th><th>median</th><th>max</th>
            <th>top actions</th><th>interventions</th><th>status</th><th></th>
          </tr>
        </thead>
        <tbody id="scoreboard"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
(function () {
  "use strict";

  const knobGrid = document.getElementById("knob-grid");
  const scoreboardEl = document.getElementById("scoreboard");
  const msgEl = document.getElementById("msg");
  const statGame = document.getElementById("stat-game");
  let knobSchema = [];

  function knobField(knob) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const label = document.createElement("label");
    label.textContent = knob.label;
    label.setAttribute("for", "knob-" + knob.key);
    wrap.appendChild(label);
    let input;
    if (knob.type === "enum") {
      input = document.createElement("select");
      (knob.choices || []).forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        if (c === knob.default) opt.selected = true;
        input.appendChild(opt);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (knob.type === "float") input.step = "any";
      if (knob.min !== undefined) input.min = knob.min;
      if (knob.max !== undefined) input.max = knob.max;
      input.value = knob.default;
    }
    input.id = "knob-" + knob.key;
    input.dataset.key = knob.key;
    input.dataset.type = knob.type;
    wrap.appendChild(input);
    return wrap;
  }

  function collectKnobValues() {
    const values = {};
    knobSchema.forEach((knob) => {
      const el = document.getElementById("knob-" + knob.key);
      if (!el) return;
      if (knob.type === "int") values[knob.key] = parseInt(el.value, 10);
      else if (knob.type === "float") values[knob.key] = parseFloat(el.value);
      else values[knob.key] = el.value;
    });
    return values;
  }

  function escapeHtml(s) {
    return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  }

  function renderScoreboard(rows) {
    scoreboardEl.innerHTML = "";
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.name === "baseline") tr.className = "baseline";
      const mix = Object.entries(row.action_mix || {})
        .slice(0, 3)
        .map(([action, pct]) => `${escapeHtml(action)} ${pct}%`)
        .join(", ");
      tr.innerHTML = `
        <td>${escapeHtml(row.name)}</td>
        <td>${row.episodes ?? 0}</td>
        <td>${row.mean_score ?? "—"}</td>
        <td>${row.median_score ?? "—"}</td>
        <td>${row.max_score ?? "—"}</td>
        <td class="mix">${mix}</td>
        <td>${row.intervention_episodes ?? 0}</td>
        <td><span class="status ${row.status}">${row.status}</span></td>
        <td>${row.name === "baseline" ? "" : '<button class="small" data-evolve="' + escapeHtml(row.name) + '">▶ evolve</button>'}</td>
      `;
      scoreboardEl.appendChild(tr);
    });
    scoreboardEl.querySelectorAll("[data-evolve]").forEach((btn) => {
      btn.addEventListener("click", () => launchEvolve(btn.dataset.evolve));
    });
  }

  async function refreshScoreboard() {
    try {
      const res = await fetch("/api/experiments");
      if (!res.ok) return;
      renderScoreboard(await res.json());
    } catch (e) {
      // transient network hiccup; next poll retries
    }
  }

  async function launchEvolve(name) {
    try {
      const res = await fetch(`/api/experiments/${encodeURIComponent(name)}/evolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generations: 3 }),
      });
      const body = await res.json();
      if (!res.ok) {
        showMsg(body.error || "evolve failed", true);
      } else {
        showMsg(`evolve launched for ${name}`, false);
      }
    } catch (e) {
      showMsg("network error launching evolve", true);
    }
  }

  function showMsg(text, isError) {
    msgEl.textContent = text;
    msgEl.className = isError ? "err" : "ok";
  }

  async function runExperiment() {
    const name = document.getElementById("f-name").value.trim();
    if (!name) {
      showMsg("name is required", true);
      return;
    }
    const payload = {
      name,
      knob_values: collectKnobValues(),
      reward_code: document.getElementById("f-reward").value,
      agent: document.getElementById("f-agent").value,
      episodes: parseInt(document.getElementById("f-episodes").value, 10),
      seed: parseInt(document.getElementById("f-seed").value, 10),
    };
    try {
      const res = await fetch("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        showMsg(body.error || "failed to launch", true);
      } else {
        showMsg(`launched ${name}`, false);
        refreshScoreboard();
      }
    } catch (e) {
      showMsg("network error", true);
    }
  }

  // tab-key inserts spaces in the reward-code textarea, instead of moving
  // focus away (the default browser behavior for <textarea>).
  const rewardEl = document.getElementById("f-reward");
  rewardEl.addEventListener("keydown", (ev) => {
    if (ev.key !== "Tab") return;
    ev.preventDefault();
    const start = rewardEl.selectionStart;
    const end = rewardEl.selectionEnd;
    rewardEl.value = rewardEl.value.slice(0, start) + "    " + rewardEl.value.slice(end);
    rewardEl.selectionStart = rewardEl.selectionEnd = start + 4;
  });

  async function init() {
    const [knobsRes, templateRes] = await Promise.all([
      fetch("/api/knobs"),
      fetch("/api/reward-template"),
    ]);
    knobSchema = await knobsRes.json();
    knobSchema.forEach((knob) => knobGrid.appendChild(knobField(knob)));
    const template = await templateRes.json();
    rewardEl.value = template.template || "";
    statGame.textContent = "game: " + (new URLSearchParams(window.location.search).get("game") || "space-invaders");
    document.getElementById("btn-run").addEventListener("click", runExperiment);
    refreshScoreboard();
    setInterval(refreshScoreboard, 3000);
  }

  init();
})();
</script>
</body>
</html>
"""
