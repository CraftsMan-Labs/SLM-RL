"""The playground page: one self-contained HTML document (inline CSS +
vanilla JS, no CDN, no external requests) -- same pattern as
`slm_rl/webui/page.py`. Read-WRITE: the "run experiment" and "evolve"
buttons POST to this server (see server.py's docstring for the trust
model).

Tutorial mode (plan 023): every knob/field/tab/button/scoreboard column
gets an (i) info icon -- hover (desktop) shows a tooltip, click pins it
(dismissed on outside-click). Copy lives in `tutorial_content.py`
(CARDS dict) and is injected here as a JSON blob (`<script id=
"tutorial-data" type="application/json">`) the page's own JS reads at
init time -- this module never re-templates PAGE per-request (PAGE is
served byte-identical on every GET /, see server.py::_serve_page), so
copy changes only require restarting the playground process, not any
new templating machinery. PAGE is built with plain string concatenation,
not `str.format`, because the existing template's CSS/JS is full of
literal `{`/`}` that `.format` would choke on.
"""

from __future__ import annotations

import html
import json

from slm_rl.playground.experiments import BACKEND_CHOICES
from slm_rl.playground.tutorial_content import (
    CARDS,
    INTRO_ANTI_DOOM_LOOP,
    INTRO_DIAGRAM,
    INTRO_STAGES,
)

_TUTORIAL_DATA_JSON = json.dumps(CARDS)

_INTRO_STAGE_ROWS = "\n".join(
    f'          <li><strong>{html.escape(stage)}</strong> — {html.escape(desc)}</li>'
    for stage, desc in INTRO_STAGES
)

_PAGE_TEMPLATE: str = """<!doctype html>
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
  .field p.hint { color: var(--muted); font-size: 0.72rem; margin: 0.2rem 0 0 0; }
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
  #watch-panel, #compare-panel, #log-panel { display: none; }
  #watch-panel.open, #compare-panel.open, #log-panel.open { display: block; }
  #watch-panel .watch-head, #compare-panel .watch-head, #log-panel .watch-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
  }
  #watch-panel iframe {
    width: 100%;
    height: 620px;
    border: 0;
    border-radius: 6px;
    background: #0f1115;
  }
  #compare-scores { margin-bottom: 0.6rem; }
  .compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
  }
  .compare-grid .panel-head {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
  }
  .compare-grid iframe {
    width: 100%;
    height: 480px;
    border: 0;
    border-radius: 6px;
    background: #0f1115;
  }
  /* Signup gate (plan 021): a full-viewport overlay shown until /api/profile
     returns something other than 404. Not a lockout -- "skip for now" saves
     name-only and dismisses it just like a real save does. */
  #signup-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(10, 11, 14, 0.75);
    z-index: 50;
    align-items: center;
    justify-content: center;
  }
  #signup-overlay.open { display: flex; }
  #signup-card {
    width: 100%;
    max-width: 380px;
    margin: 1rem;
  }
  #signup-card p.hint { color: var(--muted); font-size: 0.78rem; margin: 0 0 0.6rem 0; }
  #signup-card .actions { display: flex; gap: 0.5rem; margin-top: 0.6rem; }
  #signup-card button.secondary {
    background: #23262d;
    color: var(--text);
    border: 1px solid var(--border);
  }
  #signed-in-strip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--muted);
  }
  #signed-in-strip button.small { margin-left: 0.2rem; }
  button.small[disabled] { opacity: 0.4; cursor: not-allowed; }

  /* Tutorial mode (plan 023): (i) info icons + hover/click cards, an
     intro panel, and a header toggle that hides every icon at once. */
  #tutorial-toggle {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
  }
  #tutorial-toggle input { cursor: pointer; }
  body.tutorial-off .info-icon { display: none !important; }
  .info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.1em;
    height: 1.1em;
    margin-left: 0.3em;
    border-radius: 50%;
    background: #23262d;
    color: var(--accent);
    border: 1px solid var(--border);
    font-size: 0.72rem;
    font-style: normal;
    line-height: 1;
    cursor: help;
    position: relative;
    vertical-align: middle;
  }
  .info-icon .info-card {
    display: none;
    position: absolute;
    z-index: 40;
    top: 1.4em;
    left: 0;
    width: 260px;
    max-width: 70vw;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.7rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    cursor: auto;
    white-space: normal;
    font-weight: 400;
  }
  .info-icon .info-card h3 {
    margin: 0 0 0.3rem 0;
    font-size: 0.82rem;
    color: var(--text);
  }
  .info-icon .info-card p {
    margin: 0;
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.4;
  }
  .info-icon:hover .info-card { display: block; }
  .info-icon.pinned .info-card { display: block; }
  #intro-panel {
    cursor: default;
  }
  #intro-panel .intro-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
  }
  #intro-panel .intro-head h2 { margin: 0; }
  #intro-panel .intro-body { margin-top: 0.7rem; }
  #intro-panel.collapsed .intro-body { display: none; }
  #intro-panel pre {
    background: #0f1115;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    overflow-x: auto;
    font-size: 0.72rem;
    line-height: 1.35;
    color: var(--text);
  }
  #intro-panel ul { margin: 0.6rem 0; padding-left: 1.2rem; font-size: 0.8rem; color: var(--muted); }
  #intro-panel ul li { margin-bottom: 0.25rem; }
  #intro-panel .anti-doom-loop { font-size: 0.8rem; color: var(--muted); margin: 0.4rem 0 0 0; }
  #intro-toggle-caret { color: var(--muted); font-size: 0.8rem; }
  .filter-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.7rem;
  }
  .filter-chips button.chip {
    padding: 0.15rem 0.55rem;
    font-size: 0.75rem;
    background: #23262d;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-weight: 400;
    cursor: pointer;
  }
  .filter-chips button.chip.active {
    border-color: var(--accent);
    color: var(--accent);
  }
  /* Hardware tier banner (plan 026 Phase E) */
  #tier-banner {
    font-size: 0.82rem;
    color: var(--muted);
    padding: 0.45rem 0.75rem;
    margin-bottom: 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  #tier-banner strong { color: var(--text); font-weight: 600; }
  /* HF token side panel (plan 026 Phase D) — collapsible aside */
  #hf-panel {
    flex: 0 1 240px;
    min-width: 200px;
  }
  #hf-panel.collapsed .hf-body { display: none; }
  #hf-panel .hf-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    margin-bottom: 0;
  }
  #hf-panel .hf-head h2 { margin: 0; }
  #hf-panel .hf-body { margin-top: 0.7rem; }
  #hf-panel ol {
    margin: 0;
    padding-left: 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.45;
  }
  #hf-panel ol li { margin-bottom: 0.4rem; }
  #hf-panel a { color: var(--accent); }
  #hf-panel .hf-note { font-size: 0.72rem; color: var(--muted); margin: 0.5rem 0 0 0; }
  #hf-toggle-caret { color: var(--muted); font-size: 0.8rem; }
  #signup-card ol.hf-steps {
    margin: 0 0 0.7rem 0;
    padding-left: 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.45;
  }
  #signup-card ol.hf-steps li { margin-bottom: 0.3rem; }
  #signup-card a { color: var(--accent); }
  @media (max-width: 900px) {
    #hf-panel { flex: 1 1 100%; min-width: 0; }
  }
  #log-body {
    margin: 0;
    max-height: 320px;
    overflow: auto;
    background: #0f1115;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    line-height: 1.35;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
  }
</style>
</head>
<body>
<div id="signup-overlay">
  <div class="card" id="signup-card">
    <h2>sign up<i class="info-icon" data-card="signup_card">i<span class="info-card"></span></i></h2>
    <p class="hint">
      Name + (optional) Hugging Face token, stored locally on THIS machine
      only (<code>~/.../playground/profile.json</code>, file mode 0600) --
      not sent anywhere but the Hub, and only when you click publish. Skip
      the token and add it later from the profile card; publish buttons
      stay disabled until then.
    </p>
    <ol class="hf-steps">
      <li>Create a free account at <a href="https://huggingface.co/join" target="_blank" rel="noopener">huggingface.co</a></li>
      <li>Open <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Settings → Access Tokens</a> and create a write token</li>
      <li>Paste the token below (optional — skip anytime)</li>
    </ol>
    <div class="field">
      <label for="su-name">name</label>
      <input type="text" id="su-name" placeholder="e.g. Ada">
    </div>
    <div class="field">
      <label for="su-token">Hugging Face token (optional)</label>
      <input type="text" id="su-token" placeholder="hf_...">
    </div>
    <div id="signup-msg"></div>
    <div class="actions">
      <button id="signup-save" type="button">save</button>
      <button class="secondary" id="signup-skip" type="button">skip for now</button>
    </div>
  </div>
</div>
<header>
  <h1>SLM-RL — workshop playground</h1>
  <span class="stat" id="stat-game">game: —</span>
  <label id="tutorial-toggle"><input type="checkbox" id="tutorial-checkbox" checked> tutorial</label>
  <span id="signed-in-strip"></span>
</header>
<div id="tier-banner">detecting hardware…</div>
<div class="card" id="intro-panel">
  <div class="intro-head" id="intro-head">
    <h2>How it works</h2>
    <span id="intro-toggle-caret">▲ hide</span>
  </div>
  <div class="intro-body">
    <pre>__INTRO_DIAGRAM__</pre>
    <ul>
__INTRO_STAGE_ROWS__
    </ul>
    <p class="anti-doom-loop">__INTRO_ANTI_DOOM_LOOP__</p>
  </div>
</div>
<div class="layout">
  <aside class="card" id="hf-panel">
    <div class="hf-head" id="hf-head">
      <h2>Hugging Face token</h2>
      <span id="hf-toggle-caret">▲ hide</span>
    </div>
    <div class="hf-body">
      <ol>
        <li>Create a free account at <a href="https://huggingface.co/join" target="_blank" rel="noopener">huggingface.co</a></li>
        <li>Open <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">Settings → Access Tokens</a> and create a write token</li>
        <li>Paste it in the sign-up / profile card (optional)</li>
      </ol>
      <p class="hf-note">Token is optional. Skip anytime — publish stays disabled until you add one; rollouts, evolve, and theater work without it.</p>
    </div>
  </aside>
  <div class="col">
    <div class="card">
      <h2>New experiment</h2>
      <div class="field">
        <label for="f-game">game</label>
        <select id="f-game"></select>
      </div>
      <div class="field">
        <label for="f-name">name</label>
        <input type="text" id="f-name" placeholder="e.g. tighter-loop">
      </div>
      <div class="field">
        <label for="f-preset">model preset</label>
        <select id="f-preset">
          <option value="">— tier default / custom —</option>
        </select>
      </div>
      <div class="field">
        <label for="f-model">model (optional)<i class="info-icon" data-card="model_field">i<span class="info-card"></span></i></label>
        <input type="text" id="f-model" placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct — blank = tier default">
        <p class="hint">Presets are official HF org IDs only; free-text still allowed. 8GB: &le;1B.</p>
      </div>
      <div class="field">
        <label for="f-backend">backend<i class="info-icon" data-card="backend_field">i<span class="info-card"></span></i></label>
        <select id="f-backend">
          __BACKEND_OPTIONS__
        </select>
      </div>
      <div class="knob-grid" id="knob-grid"></div>
      <div class="field" style="margin-top:0.6rem;">
        <label for="f-agent">agent<i class="info-icon" data-card="teacher_select">i<span class="info-card"></span></i></label>
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
      <div class="field" style="flex-basis:100%;">
        <label for="f-dataset-url">dataset URL (workshop pack)</label>
        <input type="text" id="f-dataset-url" placeholder="org/slm-rl-space-invaders" autocomplete="off">
      </div>
      <div class="field" style="flex-basis:100%;">
        <label for="f-adapter-url">SFT model URL (published LoRA)</label>
        <input type="text" id="f-adapter-url" placeholder="org/slm-rl-boxing" autocomplete="off">
      </div>
      <div class="field">
        <label for="f-dqn-url">dqn URL (optional)</label>
        <input type="text" id="f-dqn-url" placeholder="same repo or omit" autocomplete="off">
      </div>
      <div class="field">
        <label for="f-generations">evolve gens</label>
        <input type="number" id="f-generations" value="1" min="1" max="50">
      </div>
      <button id="btn-run" type="button">run experiment</button>
      <div id="msg"></div>
    </div>
    <div class="card">
      <h2>reward code<i class="info-icon" data-card="reward_tab">i<span class="info-card"></span></i></h2>
      <textarea id="f-reward" spellcheck="false"></textarea>
    </div>
  </div>
  <div class="col" style="flex-basis: 500px;">
    <div class="card">
      <h2>scoreboard</h2>
      <div class="filter-chips" id="game-filters"></div>
      <table>
        <thead>
          <tr>
            <th>name<i class="info-icon" data-card="scoreboard_name">i<span class="info-card"></span></i></th>
            <th>game</th>
            <th>model<i class="info-icon" data-card="scoreboard_model">i<span class="info-card"></span></i></th>
            <th>episodes<i class="info-icon" data-card="scoreboard_episodes">i<span class="info-card"></span></i></th>
            <th>mean<i class="info-icon" data-card="scoreboard_mean">i<span class="info-card"></span></i></th>
            <th>median<i class="info-icon" data-card="scoreboard_median">i<span class="info-card"></span></i></th>
            <th>max<i class="info-icon" data-card="scoreboard_max">i<span class="info-card"></span></i></th>
            <th>top actions<i class="info-icon" data-card="scoreboard_actions">i<span class="info-card"></span></i></th>
            <th>interventions<i class="info-icon" data-card="scoreboard_interventions">i<span class="info-card"></span></i></th>
            <th>status<i class="info-icon" data-card="scoreboard_status">i<span class="info-card"></span></i></th>
            <th title="evolve">▶<i class="info-icon" data-card="evolve_button">i<span class="info-card"></span></i></th>
            <th title="watch">👁<i class="info-icon" data-card="watch_link">i<span class="info-card"></span></i></th>
            <th title="A/B">A/B<i class="info-icon" data-card="ab_button">i<span class="info-card"></span></i></th>
            <th title="gens">gens<i class="info-icon" data-card="gens_link">i<span class="info-card"></span></i></th>
            <th title="play again">play<i class="info-icon" data-card="play_again_button">i<span class="info-card"></span></i></th>
            <th title="publish">pub<i class="info-icon" data-card="publish_button">i<span class="info-card"></span></i></th>
          </tr>
        </thead>
        <tbody id="scoreboard"></tbody>
      </table>
    </div>
    <div class="card" id="watch-panel">
      <div class="watch-head">
        <h2 id="watch-title" style="margin:0;">live view — —</h2>
        <button class="small" id="watch-close" type="button">close ✕</button>
      </div>
      <iframe id="watch-frame" title="live view"></iframe>
    </div>
    <div class="card" id="log-panel">
      <div class="watch-head">
        <h2 id="log-title" style="margin:0;">log — —</h2>
        <button class="small" id="log-close" type="button">close ✕</button>
      </div>
      <pre id="log-body"></pre>
    </div>
    <div class="card" id="compare-panel">
      <div class="watch-head">
        <h2 id="compare-title" style="margin:0;">A/B — —</h2>
        <button class="small" id="compare-close" type="button">close ✕</button>
      </div>
      <div id="compare-scores" class="mix"></div>
      <div class="compare-grid">
        <div>
          <div class="panel-head">base (gen 0)</div>
          <iframe id="compare-frame-base" title="base"></iframe>
        </div>
        <div>
          <div class="panel-head">champion</div>
          <iframe id="compare-frame-champion" title="champion"></iframe>
        </div>
      </div>
    </div>
    <div class="card" id="play-again-panel" style="display:none;">
      <div class="watch-head">
        <h2 id="play-again-title" style="margin:0;">play again — —</h2>
        <button class="small" id="play-again-close" type="button">close ✕</button>
      </div>
      <div class="knob-grid" style="margin-top:0.5rem;">
        <div>
          <label>gen (or check champion)<i class="info-icon" data-card="play_again_button">i<span class="info-card"></span></i></label>
          <input id="pa-gen" type="number" min="0" step="1" value="0">
        </div>
        <div>
          <label>&nbsp;</label>
          <label style="color:var(--text);"><input id="pa-champion" type="checkbox"> use champion</label>
        </div>
        <div>
          <label>episodes</label>
          <input id="pa-episodes" type="number" min="1" max="200" value="10">
        </div>
        <div>
          <label>seed</label>
          <input id="pa-seed" type="number" value="20000">
        </div>
        <div>
          <label>temperature</label>
          <input id="pa-temperature" type="number" min="0" max="2" step="0.1" value="0.2">
        </div>
      </div>
      <div style="margin-top:0.6rem;">
        <button id="pa-go" type="button">play</button>
      </div>
    </div>
    <div class="card" id="publish-panel" style="display:none;">
      <h2>publish result</h2>
      <div id="publish-result" class="mix"></div>
    </div>
  </div>
</div>
<script id="tutorial-data" type="application/json">__TUTORIAL_DATA_JSON__</script>
<script>
(function () {
  "use strict";

  // --- Tutorial mode (plan 023): cards are pure data (tutorial_content.py's
  // CARDS dict, embedded above as JSON) -- this block wires the (i) icons
  // already present in the static HTML, plus the header toggle and the
  // intro-panel collapse. knobField() (below) calls wireInfoIcons() itself
  // for the (i) icon it builds client-side for each knob.
  const TUTORIAL_CARDS = JSON.parse(document.getElementById("tutorial-data").textContent);

  function cardHtml(key) {
    const card = TUTORIAL_CARDS[key];
    if (!card) return "";
    return `<h3>${escapeHtml(card.title)}</h3><p>${escapeHtml(card.body)}</p>`;
  }

  // Fills every <i class="info-icon" data-card="..."> already in the DOM
  // (static markup) or newly appended (knob grid, scoreboard rows) with its
  // card body, and wires click-to-pin / outside-click-to-dismiss. Safe to
  // call repeatedly (e.g. after every scoreboard re-render) -- re-filling an
  // already-filled icon is a harmless no-op.
  function wireInfoIcons(root) {
    (root || document).querySelectorAll(".info-icon[data-card]").forEach((icon) => {
      const span = icon.querySelector(".info-card");
      if (span && !span.dataset.filled) {
        span.innerHTML = cardHtml(icon.dataset.card);
        span.dataset.filled = "1";
      }
      if (icon.dataset.wired) return;
      icon.dataset.wired = "1";
      icon.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const wasPinned = icon.classList.contains("pinned");
        document.querySelectorAll(".info-icon.pinned").forEach((el) => el.classList.remove("pinned"));
        if (!wasPinned) icon.classList.add("pinned");
      });
    });
  }
  document.addEventListener("click", () => {
    document.querySelectorAll(".info-icon.pinned").forEach((el) => el.classList.remove("pinned"));
  });

  // Tutorial toggle: default ON, "?"-style switch in the header hides every
  // (i) icon at once via the body.tutorial-off CSS rule above -- presenters
  // declutter without the icons' markup being removed (no re-render needed
  // when toggled back on).
  const tutorialCheckbox = document.getElementById("tutorial-checkbox");
  tutorialCheckbox.addEventListener("change", () => {
    document.body.classList.toggle("tutorial-off", !tutorialCheckbox.checked);
  });

  // Intro panel: expanded by default on first visit, collapse state
  // remembered per-browser in localStorage so it doesn't reappear every
  // reload once a presenter has dismissed it.
  const introPanel = document.getElementById("intro-panel");
  const introHead = document.getElementById("intro-head");
  const introCaret = document.getElementById("intro-toggle-caret");
  const INTRO_COLLAPSE_KEY = "slm-rl-playground-intro-collapsed";

  function setIntroCollapsed(collapsed) {
    introPanel.classList.toggle("collapsed", collapsed);
    introCaret.textContent = collapsed ? "▼ show" : "▲ hide";
    try {
      window.localStorage.setItem(INTRO_COLLAPSE_KEY, collapsed ? "1" : "0");
    } catch (e) {
      // localStorage unavailable (private browsing, etc) -- collapse state
      // just doesn't persist across reloads; not fatal.
    }
  }

  introHead.addEventListener("click", () => {
    setIntroCollapsed(!introPanel.classList.contains("collapsed"));
  });

  (function initIntroCollapseState() {
    let stored = null;
    try {
      stored = window.localStorage.getItem(INTRO_COLLAPSE_KEY);
    } catch (e) {
      // ignore; defaults to expanded below
    }
    setIntroCollapsed(stored === "1");
  })();

  // HF side panel (plan 026 Phase D): same collapse pattern as intro.
  const hfPanel = document.getElementById("hf-panel");
  const hfHead = document.getElementById("hf-head");
  const hfCaret = document.getElementById("hf-toggle-caret");
  const HF_COLLAPSE_KEY = "slm-rl-playground-hf-collapsed";

  function setHfCollapsed(collapsed) {
    hfPanel.classList.toggle("collapsed", collapsed);
    hfCaret.textContent = collapsed ? "▼ show" : "▲ hide";
    try {
      window.localStorage.setItem(HF_COLLAPSE_KEY, collapsed ? "1" : "0");
    } catch (e) {
      // ignore
    }
  }

  hfHead.addEventListener("click", () => {
    setHfCollapsed(!hfPanel.classList.contains("collapsed"));
  });

  (function initHfCollapseState() {
    let stored = null;
    try {
      stored = window.localStorage.getItem(HF_COLLAPSE_KEY);
    } catch (e) {
      // ignore
    }
    setHfCollapsed(stored === "1");
  })();

  const knobGrid = document.getElementById("knob-grid");
  const scoreboardEl = document.getElementById("scoreboard");
  const gameFiltersEl = document.getElementById("game-filters");
  const msgEl = document.getElementById("msg");
  const statGame = document.getElementById("stat-game");
  const gameSelect = document.getElementById("f-game");
  let knobSchema = [];
  let allScoreboardRows = [];
  let gameFilter = "all";
  const LAST_GAME_KEY = "slm-rl-playground-last-game";
  // Signup gate state (plan 021): `hasToken` gates the scoreboard's publish
  // buttons -- re-rendered by renderScoreboard on every poll, so saving a
  // token later (no reload needed) flips existing rows from disabled to
  // enabled on the next 3s tick.
  let hasToken = false;
  const tierBanner = document.getElementById("tier-banner");
  const presetSelect = document.getElementById("f-preset");
  const modelInput = document.getElementById("f-model");
  const backendSelect = document.getElementById("f-backend");

  function applyHardware(hw) {
    // Banner: "tier: any-8gb · default LiquidAI/LFM2.5-350M"
    tierBanner.innerHTML =
      "tier: <strong>" + escapeHtml(hw.tier) + "</strong> · default " +
      "<strong>" + escapeHtml(hw.model) + "</strong>" +
      " <span class=\"mix\">(" + escapeHtml(hw.backend) + ")</span>";
    presetSelect.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— tier default / custom —";
    presetSelect.appendChild(blank);
    (hw.presets || []).forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.model;
      opt.textContent = p.label || p.model;
      if (p.backend) opt.dataset.backend = p.backend;
      presetSelect.appendChild(opt);
    });
  }

  async function loadHardware() {
    try {
      const res = await fetch("/api/hardware");
      if (!res.ok) {
        tierBanner.textContent = "tier: unknown (hardware probe failed)";
        return;
      }
      applyHardware(await res.json());
    } catch (e) {
      tierBanner.textContent = "tier: unknown (hardware probe failed)";
    }
  }

  presetSelect.addEventListener("change", () => {
    const opt = presetSelect.options[presetSelect.selectedIndex];
    if (!opt || !opt.value) return;
    modelInput.value = opt.value;
    const suggested = opt.dataset.backend;
    if (suggested) {
      // Prefer the preset's suggested backend when that option exists.
      for (let i = 0; i < backendSelect.options.length; i++) {
        if (backendSelect.options[i].value === suggested) {
          backendSelect.selectedIndex = i;
          break;
        }
      }
    }
  });

  function selectedGame() {
    return gameSelect.value || "";
  }

  function setStatGame(game) {
    statGame.textContent = "game: " + (game || "—");
  }

  async function loadKnobs(game) {
    const url = game ? `/api/knobs?game=${encodeURIComponent(game)}` : "/api/knobs";
    const res = await fetch(url);
    if (!res.ok) return;
    knobSchema = await res.json();
    knobGrid.innerHTML = "";
    knobSchema.forEach((knob) => knobGrid.appendChild(knobField(knob)));
  }

  function populateGameSelect(games, defaultGame) {
    let preferred = defaultGame;
    try {
      const stored = window.localStorage.getItem(LAST_GAME_KEY);
      if (stored && games.includes(stored)) preferred = stored;
    } catch (e) {
      // localStorage unavailable — use server default
    }
    if (!preferred || !games.includes(preferred)) preferred = games[0] || "";
    gameSelect.innerHTML = "";
    games.forEach((g) => {
      const opt = document.createElement("option");
      opt.value = g;
      opt.textContent = g;
      if (g === preferred) opt.selected = true;
      gameSelect.appendChild(opt);
    });
    setStatGame(preferred);
  }

  function renderGameFilters(rows) {
    const games = [...new Set(rows.map((r) => r.game).filter(Boolean))].sort();
    if (gameFilter !== "all" && !games.includes(gameFilter)) {
      gameFilter = "all";
    }
    gameFiltersEl.innerHTML = "";
    const chips = ["all", ...games];
    chips.forEach((g) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip" + (gameFilter === g ? " active" : "");
      btn.textContent = g === "all" ? "all" : g;
      btn.addEventListener("click", () => {
        gameFilter = g;
        renderScoreboard(allScoreboardRows);
      });
      gameFiltersEl.appendChild(btn);
    });
  }

  function knobField(knob) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const label = document.createElement("label");
    label.textContent = knob.label;
    label.setAttribute("for", "knob-" + knob.key);
    wrap.appendChild(label);
    // Every KNOBS entry gets an (i) card (plan 023 required coverage) --
    // data-card is the knob's own key, so tutorial_content.CARDS needs one
    // entry per Knob.key with no per-knob wiring here beyond the lookup.
    const icon = document.createElement("i");
    icon.className = "info-icon";
    icon.dataset.card = knob.key;
    icon.textContent = "i";
    const infoSpan = document.createElement("span");
    infoSpan.className = "info-card";
    icon.appendChild(infoSpan);
    label.appendChild(icon);
    wireInfoIcons(wrap);
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
    allScoreboardRows = rows;
    renderGameFilters(rows);
    const filtered = gameFilter === "all"
      ? rows
      : rows.filter((r) => r.game === gameFilter);
    scoreboardEl.innerHTML = "";
    filtered.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.name === "baseline") tr.className = "baseline";
      const mix = Object.entries(row.action_mix || {})
        .slice(0, 3)
        .map(([action, pct]) => `${escapeHtml(action)} ${pct}%`)
        .join(", ");
      const canPublish = row.name !== "baseline" && hasToken;
      const publishTitle = hasToken ? "" : 'title="add a Hugging Face token in your profile to publish"';
      // Provenance (plan 022 design decision 4): "tier default" when the
      // experiment didn't override model/backend, so A/B comparisons across
      // model choices are legible at a glance without opening experiment.json.
      const modelLabel = row.model || "tier default";
      const modelCell = row.backend
        ? escapeHtml(modelLabel) + ' <span class="mix">(' + escapeHtml(row.backend) + ')</span>'
        : escapeHtml(modelLabel);
      const modelTitleAttr = escapeHtml(modelLabel + (row.backend ? ` (backend: ${row.backend})` : ""));
      const gameLabel = row.game || "—";
      tr.innerHTML = `
        <td>${escapeHtml(row.name)}</td>
        <td class="mix">${escapeHtml(gameLabel)}</td>
        <td class="mix" title="${modelTitleAttr}">${modelCell}</td>
        <td>${row.episodes ?? 0}</td>
        <td>${row.mean_score ?? "—"}</td>
        <td>${row.median_score ?? "—"}</td>
        <td>${row.max_score ?? "—"}</td>
        <td class="mix">${mix}</td>
        <td>${row.intervention_episodes ?? 0}</td>
        <td><span class="status ${row.status}">${row.status}</span></td>
        <td>${row.name === "baseline" ? "" : '<button class="small" data-evolve="' + escapeHtml(row.name) + '">▶ evolve</button>'}</td>
        <td><a class="small" href="/watch/${encodeURIComponent(row.name)}/" data-watch="${escapeHtml(row.name)}" target="_blank" style="text-decoration:none;">watch</a></td>
        <td>${row.name === "baseline" ? "" : '<button class="small" data-compare="' + escapeHtml(row.name) + '">A/B</button>'}</td>
        <td><a class="small" href="/gens/${encodeURIComponent(row.name)}/" target="_blank" style="text-decoration:none;">gens</a></td>
        <td>${row.name === "baseline" ? "" : '<button class="small" data-play="' + escapeHtml(row.name) + '">play</button>'}</td>
        <td>${row.name === "baseline" ? "" : '<button class="small" data-publish="' + escapeHtml(row.name) + '" ' + publishTitle + (canPublish ? "" : " disabled") + '>publish</button>'}</td>
      `;
      scoreboardEl.appendChild(tr);
    });
    scoreboardEl.querySelectorAll("[data-evolve]").forEach((btn) => {
      btn.addEventListener("click", () => launchEvolve(btn.dataset.evolve));
    });
    // Plain left-click embeds the viewer in the panel below (preventDefault
    // stops the navigation); middle-click / ctrl-click / cmd-click still
    // opens the real <a href> in a new tab (no listener runs for those).
    scoreboardEl.querySelectorAll("[data-watch]").forEach((link) => {
      link.addEventListener("click", (ev) => {
        if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        ev.preventDefault();
        openWatch(link.dataset.watch);
      });
    });
    scoreboardEl.querySelectorAll("[data-compare]").forEach((btn) => {
      btn.addEventListener("click", () => launchTheaterAndCompare(btn.dataset.compare));
    });
    scoreboardEl.querySelectorAll("[data-play]").forEach((btn) => {
      btn.addEventListener("click", () => openPlayAgainForm(btn.dataset.play));
    });
    scoreboardEl.querySelectorAll("[data-publish]").forEach((btn) => {
      btn.addEventListener("click", () => publishExperiment(btn.dataset.publish));
    });
  }

  function showHttpError(res, body, fallback) {
    // 409 Busy (plan 013 / 026): surface the server message clearly so a
    // second quick/evolve click isn't mistaken for a silent no-op.
    const err = (body && body.error) || fallback;
    if (res.status === 409 && err && !String(err).startsWith("Busy")) {
      showMsg("Busy: " + err, true);
    } else {
      showMsg(err, true);
    }
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
    const datasetUrl = (document.getElementById("f-dataset-url").value || "").trim();
    const adapterUrl = (document.getElementById("f-adapter-url").value || "").trim();
    const dqnUrl = (document.getElementById("f-dqn-url").value || "").trim();
    let generations = parseInt(document.getElementById("f-generations").value, 10);
    if (!Number.isFinite(generations) || generations < 1) {
      generations = (datasetUrl || adapterUrl) ? 1 : 3;
    }
    try {
      const res = await fetch(`/api/experiments/${encodeURIComponent(name)}/evolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          generations,
          dataset_url: datasetUrl || null,
          adapter_url: adapterUrl || null,
          dqn_url: dqnUrl || null,
        }),
      });
      const body = await res.json();
      if (!res.ok) {
        showHttpError(res, body, "evolve failed");
      } else {
        showMsg(`evolve launched for ${name}`, false);
        openLog(name, "evolve");
      }
    } catch (e) {
      showMsg("network error launching evolve", true);
    }
  }

  function showMsg(text, isError) {
    msgEl.textContent = text;
    msgEl.className = isError ? "err" : "ok";
  }

  // Live view (plan 014): one <iframe>, swapped src -- not one per row.
  const watchPanel = document.getElementById("watch-panel");
  const watchFrame = document.getElementById("watch-frame");
  const watchTitle = document.getElementById("watch-title");

  function openWatch(name) {
    watchTitle.textContent = `live view — ${name}`;
    watchPanel.classList.add("open");
    watchFrame.src = `/watch/${encodeURIComponent(name)}/`;
  }

  function closeWatch() {
    watchPanel.classList.remove("open");
    // Clearing src (not just hiding the panel) stops the embedded page's
    // SSE/frame streams -- the browser closes an iframe's in-flight
    // connections when its src changes away from what loaded them.
    watchFrame.src = "about:blank";
  }

  // Live log tail (plan 026 Phase F): poll GET .../log?kind= every ~1s while
  // the panel is open (opened on run / evolve / theater launch).
  const logPanel = document.getElementById("log-panel");
  const logTitle = document.getElementById("log-title");
  const logBody = document.getElementById("log-body");
  let logPoll = null;
  let logTarget = null; // { name, kind }

  function openLog(name, kind) {
    logTarget = { name, kind };
    logTitle.textContent = `log — ${name} (${kind})`;
    logBody.textContent = "";
    logPanel.classList.add("open");
    if (logPoll) clearInterval(logPoll);
    const poll = async () => {
      if (!logTarget) return;
      try {
        const url = `/api/experiments/${encodeURIComponent(logTarget.name)}/log?kind=${encodeURIComponent(logTarget.kind)}`;
        const res = await fetch(url);
        if (!res.ok) return;
        const text = await res.text();
        logBody.textContent = text;
        logBody.scrollTop = logBody.scrollHeight;
      } catch (e) {
        // transient; next tick retries
      }
    };
    poll();
    logPoll = setInterval(poll, 1000);
  }

  function closeLog() {
    logPanel.classList.remove("open");
    if (logPoll) {
      clearInterval(logPoll);
      logPoll = null;
    }
    logTarget = null;
    logBody.textContent = "";
  }

  // A/B compare (plan 020): launch the theater subprocess for `name`, then
  // poll /api/experiments/<name>/theater-scores until both sides have
  // written at least one record, showing the two viewer pages side by side
  // meanwhile (the tailer polls, so "watch" works even mid-exhibition).
  const comparePanel = document.getElementById("compare-panel");
  const compareTitle = document.getElementById("compare-title");
  const compareScoresEl = document.getElementById("compare-scores");
  const compareFrameBase = document.getElementById("compare-frame-base");
  const compareFrameChampion = document.getElementById("compare-frame-champion");
  let comparePoll = null;

  function openCompare(name) {
    compareTitle.textContent = `A/B — ${name}`;
    compareScoresEl.textContent = "launching exhibition…";
    comparePanel.classList.add("open");
    compareFrameBase.src = `/theater/${encodeURIComponent(name)}/base/`;
    compareFrameChampion.src = `/theater/${encodeURIComponent(name)}/champion/`;
    if (comparePoll) clearInterval(comparePoll);
    const poll = async () => {
      try {
        const res = await fetch(`/api/experiments/${encodeURIComponent(name)}/theater-scores`);
        if (!res.ok) return;
        const scores = await res.json();
        renderCompareScores(scores);
      } catch (e) {
        // transient; next tick retries
      }
    };
    poll();
    comparePoll = setInterval(poll, 3000);
  }

  function renderCompareScores(scores) {
    const side = (label, s) => {
      if (!s) return `${label}: no data yet`;
      const parts = [`episodes ${s.episodes ?? 0}`];
      if (s.win_rate !== null && s.win_rate !== undefined) parts.push(`win rate ${s.win_rate}`);
      if (s.mean_score !== null && s.mean_score !== undefined) parts.push(`mean score ${s.mean_score}`);
      return `${label}: ${parts.join(" · ")} (${s.status})`;
    };
    compareScoresEl.textContent = `${side("base", scores.base)}  |  ${side("champion", scores.champion)}`;
  }

  function closeCompare() {
    comparePanel.classList.remove("open");
    if (comparePoll) {
      clearInterval(comparePoll);
      comparePoll = null;
    }
    compareFrameBase.src = "about:blank";
    compareFrameChampion.src = "about:blank";
  }

  async function launchTheaterAndCompare(name) {
    try {
      const res = await fetch(`/api/experiments/${encodeURIComponent(name)}/theater`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const body = await res.json();
      if (!res.ok) {
        showHttpError(res, body, "theater launch failed");
        return;
      }
      showMsg(`exhibition launched for ${name}`, false);
    } catch (e) {
      showMsg("network error launching theater", true);
      return;
    }
    openLog(name, "theater");
    openCompare(name);
  }

  // --- Play again (plan 026 Phase G): optional form → POST play-again.
  // Default journey never opens this; evolve → A/B is enough.
  const playAgainPanel = document.getElementById("play-again-panel");
  const playAgainTitle = document.getElementById("play-again-title");
  const paGen = document.getElementById("pa-gen");
  const paChampion = document.getElementById("pa-champion");
  let playAgainName = null;

  function openPlayAgainForm(name) {
    playAgainName = name;
    playAgainTitle.textContent = `play again — ${name}`;
    playAgainPanel.style.display = "block";
    fillInfoCards(playAgainPanel);
  }

  function closePlayAgainForm() {
    playAgainPanel.style.display = "none";
    playAgainName = null;
  }

  paChampion.addEventListener("change", () => {
    paGen.disabled = paChampion.checked;
  });

  document.getElementById("play-again-close").addEventListener("click", closePlayAgainForm);

  document.getElementById("pa-go").addEventListener("click", async () => {
    if (!playAgainName) return;
    const payload = {
      champion: paChampion.checked,
      gen: paChampion.checked ? null : parseInt(paGen.value, 10),
      episodes: parseInt(document.getElementById("pa-episodes").value, 10),
      seed: parseInt(document.getElementById("pa-seed").value, 10),
      temperature: parseFloat(document.getElementById("pa-temperature").value),
    };
    try {
      const res = await fetch(
        `/api/experiments/${encodeURIComponent(playAgainName)}/play-again`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      const body = await res.json();
      if (!res.ok) {
        showHttpError(res, body, "play-again failed");
        return;
      }
      showMsg(
        `play-again launched for ${playAgainName} (gen ${body.generation})`,
        false,
      );
      openLog(playAgainName, "theater");
      openWatchPlay(playAgainName);
    } catch (e) {
      showMsg("network error launching play-again", true);
    }
  });

  function openWatchPlay(name) {
    // Reuse the watch panel pointed at theater/play/ (same layout as A/B sides).
    const watchPanel = document.getElementById("watch-panel");
    const watchTitle = document.getElementById("watch-title");
    const watchFrame = document.getElementById("watch-frame");
    watchTitle.textContent = `play again — ${name}`;
    watchFrame.src = `/theater/${encodeURIComponent(name)}/play/`;
    watchPanel.classList.add("open");
  }

  // --- Publish (plan 021): POST /api/experiments/<name>/publish, result
  // (repo URLs or per-side error strings -- partial success is normal and
  // shown as such) rendered in its own panel below A/B compare.
  const publishPanel = document.getElementById("publish-panel");
  const publishResultEl = document.getElementById("publish-result");

  function renderPublishResult(name, result) {
    publishPanel.style.display = "block";
    const parts = [];
    if (result.model_repo) {
      parts.push(result.model_error
        ? `model: FAILED (${result.model_error})`
        : `model: <a href="https://huggingface.co/${result.model_repo}" target="_blank">${result.model_repo}</a>`);
    } else if (result.message) {
      parts.push(`model: ${escapeHtml(result.message)}`);
    }
    if (result.dataset_repo) {
      parts.push(result.dataset_error
        ? `dataset: FAILED (${result.dataset_error})`
        : `dataset: <a href="https://huggingface.co/datasets/${result.dataset_repo}" target="_blank">${result.dataset_repo}</a>`);
    }
    publishResultEl.innerHTML = `<strong>${escapeHtml(name)}</strong> — ${parts.join(" &nbsp;|&nbsp; ")}`;
  }

  async function publishExperiment(name) {
    showMsg(`publishing ${name}…`, false);
    try {
      const res = await fetch(`/api/experiments/${encodeURIComponent(name)}/publish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const body = await res.json();
      if (!res.ok) {
        showMsg(body.error || "publish failed", true);
        return;
      }
      showMsg(`publish finished for ${name}`, false);
      renderPublishResult(name, body);
    } catch (e) {
      showMsg("network error publishing", true);
    }
  }

  // --- Signup gate (plan 021) ------------------------------------------
  const signupOverlay = document.getElementById("signup-overlay");
  const signupMsgEl = document.getElementById("signup-msg");
  const signedInStrip = document.getElementById("signed-in-strip");

  function renderSignedInStrip(profile) {
    hasToken = !!profile.has_token;
    const tokenNote = hasToken ? `token ${escapeHtml(profile.token_masked)}` : "no token (publish disabled)";
    signedInStrip.innerHTML =
      `signed in as ${escapeHtml(profile.name)} — ${tokenNote} ` +
      `<button class="small" id="btn-edit-profile" type="button">profile</button>`;
    document.getElementById("btn-edit-profile").addEventListener("click", openSignup);
  }

  function openSignup() {
    signupOverlay.classList.add("open");
  }

  function closeSignup() {
    signupOverlay.classList.remove("open");
  }

  async function saveProfile(name, token) {
    const res = await fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, hf_token: token || null }),
    });
    const body = await res.json();
    if (!res.ok) {
      signupMsgEl.textContent = body.error || "could not save profile";
      signupMsgEl.className = "err";
      return false;
    }
    renderSignedInStrip(body);
    closeSignup();
    refreshScoreboard();
    return true;
  }

  async function checkProfile() {
    try {
      const res = await fetch("/api/profile");
      if (res.status === 404) {
        openSignup();
        return;
      }
      if (!res.ok) return;
      renderSignedInStrip(await res.json());
    } catch (e) {
      // transient network hiccup; the overlay stays closed either way --
      // it only opens on a definite 404, never on a fetch error.
    }
  }

  async function runExperiment() {
    const name = document.getElementById("f-name").value.trim();
    if (!name) {
      showMsg("name is required", true);
      return;
    }
    const payload = {
      name,
      game: selectedGame(),
      knob_values: collectKnobValues(),
      reward_code: document.getElementById("f-reward").value,
      agent: document.getElementById("f-agent").value,
      episodes: parseInt(document.getElementById("f-episodes").value, 10),
      seed: parseInt(document.getElementById("f-seed").value, 10),
      model: document.getElementById("f-model").value.trim() || null,
      backend: document.getElementById("f-backend").value || null,
    };
    try {
      const res = await fetch("/api/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        showHttpError(res, body, "failed to launch");
      } else if (body.warnings && body.warnings.length) {
        showMsg(`launched ${name} — ${body.warnings.join("; ")}`, false);
        openLog(name, "rollout");
        refreshScoreboard();
      } else {
        showMsg(`launched ${name}`, false);
        openLog(name, "rollout");
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
    // Wire every (i) icon already present in the static HTML (header
    // column headers, signup card, model/backend/agent labels, reward-code
    // tab) -- knobField() wires its own icon as each knob field is
    // appended below.
    wireInfoIcons(document);
    const [gamesRes, templateRes] = await Promise.all([
      fetch("/api/games"),
      fetch("/api/reward-template"),
    ]);
    loadHardware();
    const gamesPayload = await gamesRes.json();
    const games = gamesPayload.games || [];
    populateGameSelect(games, gamesPayload.default);
    await loadKnobs(selectedGame());
    const template = await templateRes.json();
    rewardEl.value = template.template || "";
    gameSelect.addEventListener("change", () => {
      const g = selectedGame();
      setStatGame(g);
      try {
        window.localStorage.setItem(LAST_GAME_KEY, g);
      } catch (e) {
        // ignore
      }
      loadKnobs(g);
    });
    document.getElementById("btn-run").addEventListener("click", runExperiment);
    document.getElementById("watch-close").addEventListener("click", closeWatch);
    document.getElementById("log-close").addEventListener("click", closeLog);
    document.getElementById("compare-close").addEventListener("click", closeCompare);

    document.getElementById("signup-save").addEventListener("click", () => {
      const name = document.getElementById("su-name").value.trim();
      const token = document.getElementById("su-token").value.trim();
      if (!name) {
        signupMsgEl.textContent = "name is required";
        signupMsgEl.className = "err";
        return;
      }
      saveProfile(name, token);
    });
    document.getElementById("signup-skip").addEventListener("click", () => {
      const name = document.getElementById("su-name").value.trim();
      if (!name) {
        signupMsgEl.textContent = "name is required, even to skip the token";
        signupMsgEl.className = "err";
        return;
      }
      saveProfile(name, "");
    });

    checkProfile();
    refreshScoreboard();
    setInterval(refreshScoreboard, 3000);
  }

  init();
})();
</script>
</body>
</html>
"""


def _backend_options_html() -> str:
    opts = []
    for choice in BACKEND_CHOICES:
        # "tier default" is the pseudo-choice meaning no override — empty value.
        value = "" if choice == "tier default" else choice
        opts.append(f'<option value="{html.escape(value)}">{html.escape(choice)}</option>')
    return "\n          ".join(opts)


def _render_page() -> str:
    """Substitutes the `__TOKEN__`-style placeholders left in
    `_PAGE_TEMPLATE` with tutorial content. Plain `str.replace`, not
    `str.format` -- the template's CSS/JS is full of literal `{`/`}` that
    `.format` would try (and fail) to parse as fields."""
    diagram_html = html.escape(INTRO_DIAGRAM)
    anti_doom_loop_html = html.escape(INTRO_ANTI_DOOM_LOOP)
    page = _PAGE_TEMPLATE
    page = page.replace("__INTRO_DIAGRAM__", diagram_html)
    page = page.replace("__INTRO_STAGE_ROWS__", _INTRO_STAGE_ROWS)
    page = page.replace("__INTRO_ANTI_DOOM_LOOP__", anti_doom_loop_html)
    page = page.replace("__TUTORIAL_DATA_JSON__", _TUTORIAL_DATA_JSON)
    page = page.replace("__BACKEND_OPTIONS__", _backend_options_html())
    return page


# Rendered once at import time (not per-request -- server.py serves this
# byte-identical on every GET /, same as before plan 023).
PAGE: str = _render_page()

__all__ = ["PAGE"]
