"""The live-play viewer page: one self-contained HTML document (inline CSS
+ vanilla JS, no CDN, no external requests). Embedded as a Python string so
serving it needs zero package-data plumbing.
"""

from __future__ import annotations

PAGE: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SLM-RL — live play</title>
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
  #conn { font-size: 0.85rem; padding: 0.15rem 0.5rem; border-radius: 999px; }
  #conn.connecting { background: #3a3520; color: var(--warn); }
  #conn.open { background: #163a26; color: var(--ok); }
  #conn.closed { background: #3a1f20; color: var(--bad); }
  #episodes {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
  }
  .card.win { border-left-color: var(--ok); }
  .card.loss { border-left-color: var(--bad); }
  .card-head {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }
  .step {
    padding: 0.4rem 0;
    border-top: 1px solid var(--border);
  }
  .step:first-of-type { border-top: none; }
  .row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .pegs { display: flex; gap: 3px; }
  .peg {
    width: 16px; height: 16px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.25);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 9px; color: #000;
  }
  .peg.R { background: #e5484d; }
  .peg.G { background: #30a46c; }
  .peg.B { background: #3b82f6; }
  .peg.Y { background: #ffd60a; }
  .peg.O { background: #f5820a; }
  .peg.P { background: #a855f7; }
  .peg.unk { background: #8b93a1; color: #14161a; }
  .badge {
    font-size: 0.75rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: #23262d;
    color: var(--muted);
  }
  .badge.bad { background: #3a1f20; color: var(--bad); }
  .chip {
    font-size: 0.7rem;
    padding: 0.05rem 0.35rem;
    border-radius: 4px;
    background: #3a3520;
    color: var(--warn);
  }
  .reward { font-size: 0.8rem; color: var(--muted); }
  details { margin-top: 0.3rem; }
  summary { cursor: pointer; font-size: 0.78rem; color: var(--muted); }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.78rem;
    background: #0f1115;
    padding: 0.5rem;
    border-radius: 6px;
    margin: 0.3rem 0 0 0;
    max-height: 240px;
    overflow-y: auto;
  }
  .outcome { font-weight: 600; }
  .outcome.win { color: var(--ok); }
  .outcome.loss { color: var(--bad); }
  .watch-btn {
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: #23262d;
    color: var(--text);
    cursor: pointer;
  }
  .watch-btn:hover { background: #2c313a; }
  #screen-panel {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem;
    display: none;
    z-index: 10;
  }
  #screen-panel.open { display: block; }
  #screen-panel .screen-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    color: var(--muted);
  }
  #screen-panel img {
    display: block;
    width: 320px;
    height: 240px;
    image-rendering: pixelated;
    background: #000;
    border-radius: 4px;
  }
  #screen-close {
    cursor: pointer;
    background: none;
    border: none;
    color: var(--muted);
    font-size: 0.9rem;
  }
  #screen-close:hover { color: var(--text); }
  #screen-msg {
    font-size: 0.75rem;
    color: var(--warn);
    margin-top: 0.3rem;
    max-width: 320px;
  }
</style>
</head>
<body>
<header>
  <h1>SLM-RL — live play</h1>
  <span class="stat" id="stat-run">run: —</span>
  <span class="stat" id="stat-gen">gen: —</span>
  <span class="stat" id="stat-episodes">episodes: 0</span>
  <span class="stat" id="stat-wins">wins: 0</span>
  <span id="conn" class="connecting">connecting…</span>
</header>
<div id="episodes"></div>
<div id="screen-panel">
  <div class="screen-head">
    <span id="screen-title">episode</span>
    <button id="screen-close" title="stop watching">close ✕</button>
  </div>
  <img id="screen-img" alt="game screen">
  <div id="screen-msg"></div>
</div>
<script>
(function () {
  "use strict";
  const MAX_CARDS = 30;

  const episodesEl = document.getElementById("episodes");
  const connEl = document.getElementById("conn");
  const statRun = document.getElementById("stat-run");
  const statGen = document.getElementById("stat-gen");
  const statEpisodes = document.getElementById("stat-episodes");
  const statWins = document.getElementById("stat-wins");

  const cards = new Map(); // episode_id -> {el, stepsEl}
  const seenEpisodes = new Set();
  let wins = 0;
  let currentGen = null;

  function pegHtml(ch) {
    const known = "RGBYOP".includes(ch);
    return `<span class="peg ${known ? ch : "unk"}" title="${ch}">${known ? "" : ch}</span>`;
  }

  function guessToPegs(guess) {
    if (!guess) return "";
    return [...String(guess)].map(pegHtml).join("");
  }

  function ensureCard(ev) {
    let entry = cards.get(ev.episode_id);
    if (entry) return entry;

    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `
      <div class="card-head">
        <span>episode ${ev.episode_id ?? "?"} · gen ${ev.generation ?? "?"} · model ${ev.model_id ?? "?"} · seed ${ev.seed ?? "?"}</span>
        <span class="row">
          <button class="watch-btn" type="button">▶ watch</button>
          <span class="card-outcome"></span>
        </span>
      </div>
      <div class="steps"></div>
    `;
    el.querySelector(".watch-btn").addEventListener("click", () => {
      openScreen(ev.episode_id);
    });
    episodesEl.insertBefore(el, episodesEl.firstChild);
    entry = { el, stepsEl: el.querySelector(".steps"), outcomeEl: el.querySelector(".card-outcome") };
    cards.set(ev.episode_id, entry);

    if (!seenEpisodes.has(ev.episode_id)) {
      seenEpisodes.add(ev.episode_id);
      statEpisodes.textContent = `episodes: ${seenEpisodes.size}`;
    }

    // Bound memory: drop oldest cards past MAX_CARDS.
    while (episodesEl.children.length > MAX_CARDS) {
      const last = episodesEl.lastChild;
      const epId = [...cards.entries()].find(([, v]) => v.el === last)?.[0];
      if (epId !== undefined) cards.delete(epId);
      episodesEl.removeChild(last);
    }
    return entry;
  }

  function renderStep(entry, ev) {
    const stepEl = document.createElement("div");
    stepEl.className = "step";
    const statusBadge = ev.parse_status && ev.parse_status !== "ok"
      ? `<span class="badge bad">${ev.parse_status}</span>`
      : `<span class="badge">${ev.parse_status ?? ""}</span>`;
    const flags = ev.monitor_flags && typeof ev.monitor_flags === "object"
      ? Object.keys(ev.monitor_flags).map((k) => `<span class="chip">${k}</span>`).join("")
      : "";
    stepEl.innerHTML = `
      <div class="row">
        <span class="pegs">${guessToPegs(ev.parsed_action)}</span>
        <span class="reward">reward ${ev.reward ?? "—"} · cum ${ev.cum_reward ?? "—"}</span>
        ${statusBadge}
        ${flags}
      </div>
      <details>
        <summary>completion &amp; observed</summary>
        <pre>${escapeHtml(ev.observed ?? "")}</pre>
        <pre>${escapeHtml(ev.completion ?? "")}</pre>
      </details>
    `;
    entry.stepsEl.insertBefore(stepEl, entry.stepsEl.firstChild);

    if (ev.terminated || ev.truncated) {
      const outcome = ev.outcome || (ev.truncated ? "truncated" : "");
      entry.el.classList.remove("win", "loss");
      if (outcome === "win") {
        entry.el.classList.add("win");
        wins += 1;
        statWins.textContent = `wins: ${wins}`;
      } else if (outcome === "loss" || ev.truncated) {
        entry.el.classList.add("loss");
      }
      entry.outcomeEl.innerHTML =
        `<span class="outcome ${outcome === "win" ? "win" : outcome === "loss" ? "loss" : ""}">${outcome}</span> · cum ${ev.cum_reward ?? "—"}`;
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function handleEvent(ev) {
    if (ev.generation !== undefined && ev.generation !== null) {
      currentGen = ev.generation;
      statGen.textContent = `gen: ${currentGen}`;
    }
    const entry = ensureCard(ev);
    renderStep(entry, ev);
  }

  const params = new URLSearchParams(window.location.search);
  statRun.textContent = `run: ${params.get("run") || "—"}`;

  // All-gens grid (plan 020): a `?gen=N` on THIS page's URL is forwarded to
  // both the events and frames endpoints below. Appended (never replacing
  // the whole query string) so any other params a future caller adds keep
  // working; relative URLs ("events", "frames?...") are unchanged, so
  // mounting under /watch/<name>/ or /gens/<name>/... (plan 014's trick)
  // still resolves correctly.
  const gen = params.get("gen");
  const genQS = gen !== null ? `gen=${encodeURIComponent(gen)}` : "";

  // Live game screen (plan 010): the panel's <img> points at /frames?episode=,
  // a multipart/x-mixed-replace stream of re-simulated PNG frames — no JS
  // image decoding needed, the browser paints each part as it arrives.
  // Clearing `src` is what ends the stream server-side (client disconnect
  // -> the server's stop event fires -> the replay env is closed).
  const screenPanel = document.getElementById("screen-panel");
  const screenImg = document.getElementById("screen-img");
  const screenTitle = document.getElementById("screen-title");
  const screenMsg = document.getElementById("screen-msg");
  const screenClose = document.getElementById("screen-close");

  function openScreen(episodeId) {
    screenTitle.textContent = `episode ${episodeId}`;
    screenMsg.textContent = "";
    screenPanel.classList.add("open");
    screenImg.onerror = () => {
      screenMsg.textContent =
        "no live screen for this episode (non-Atari game, or the atari extra isn't installed)";
    };
    const epQS = `episode=${encodeURIComponent(episodeId)}`;
    screenImg.src = `frames?${genQS ? epQS + "&" + genQS : epQS}`;
  }

  function closeScreen() {
    screenPanel.classList.remove("open");
    screenImg.onerror = null;
    screenImg.src = "";
  }

  screenClose.addEventListener("click", closeScreen);

  function connect() {
    connEl.className = "connecting";
    connEl.textContent = "connecting…";
    // Plain `new EventSource("events")` when no ?gen= is present -- kept as
    // a literal call (not string-built) so it stays byte-for-byte what it
    // was before plan 020, per test_playground_watch.py's regression guard.
    const es = genQS ? new EventSource(`events?${genQS}`) : new EventSource("events");
    es.onopen = () => {
      connEl.className = "open";
      connEl.textContent = "connected";
    };
    es.onerror = () => {
      connEl.className = "closed";
      connEl.textContent = "disconnected — retrying…";
    };
    es.onmessage = (msg) => {
      try {
        handleEvent(JSON.parse(msg.data));
      } catch (e) {
        // Ignore malformed events; the stream keeps going.
      }
    };
  }
  connect();
})();
</script>
</body>
</html>
"""
