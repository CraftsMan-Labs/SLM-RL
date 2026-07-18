# SLM-RL Frontend Design

> **Product:** Workshop playground for a self-improving game gymnasium  
> **Stack target:** Dedicated Vue frontend in `web/` talking to the existing Python playground/webui backend  
> **Visual system:** [Framer](../open-design/design-systems/framer/) — void black, Framer Blue, motion-first, product-as-hero  
> **Component recipes:** [open-design/design-templates](../open-design/design-templates/) — dashboard shell, live-dashboard primitives, web-prototype sections  
> **UI nouns:** The product says **project** in the UI. The backend still stores **experiments** (`POST /api/experiments`). Map 1:1 — never invent a second persistence layer.

This document is the contract for the Vue app in `web/`. It replaces the inline HTML in `slm_rl/playground/page.py` — do not reuse that layout, CSS, or form dump. Keep the **stdlib Python HTTP API** as the backend.

---

## 0. Goals & non-goals

### Goals
- Ship a clean Framer-aligned Vue SPA with a deliberate first-run story (welcome → HF → projects).
- Journey: welcome scroll → Hugging Face credentials → project list / create → project workspace (game + knobs + run).
- Connect exclusively through the existing JSON / SSE / frame endpoints (no rewrite of training/rollout logic).
- Ship Framer tokens as CSS variables so screens stay one-accent and dark-first.

### Non-goals
- Multi-user auth / shared server accounts (profile stays local `~/.../playground/profile.json`).
- Redesigning games, trainers, or eval-gate semantics.
- Reusing the old playground HTML/CSS from `page.py`.
- Introducing a second accent color palette or light theme for v1.
- Chart libraries, icon kits, or component libraries that fight the Framer recipe.

---

## 0.1 First-run journey (locked)

```
┌─────────────────┐     scroll      ┌─────────────────┐     token saved     ┌─────────────────┐
│  1. Welcome     │ ──────────────▶ │  2. Hugging Face│ ──────────────────▶ │  3. Projects    │
│  What we'll do  │                 │  Account + key  │                     │  List / create  │
└─────────────────┘                 └─────────────────┘                     └────────┬────────┘
                                                                                    │ open
                                                                                    ▼
                                                                          ┌─────────────────┐
                                                                          │  4. Workspace   │
                                                                          │  Game + knobs   │
                                                                          └─────────────────┘
```

1. **Welcome** (`/`) — Hero: “Hey, welcome to the platform.” Explain the loop (play → train → compare → publish) as they scroll. No forms yet.
2. **Hugging Face** (same page, next section) — Ask them to create an HF account and paste an API key (plus display name). Saving credentials unlocks the rest of the app. **Token required** to leave welcome (stricter than the old optional-skip gate).
3. **Projects** (`/projects`) — List existing projects; create a new one with **name only**, then pick the **game** the project trains on. Opening a project navigates to its workspace URL.
4. **Workspace** (`/projects/:name`) — Tune parameters, run rollout/evolve, watch progress. From here they can always navigate back to the project list.

**Route guard:** If `GET /api/profile` is 404 or `has_token === false`, redirect to `/`. If profile has a token and they hit `/`, offer a short path to `/projects` (don’t force re-onboarding).

---

## 1. Visual Theme & Atmosphere

SLM-RL’s UI should feel like a **night workshop for models that play games**: pure black void (`#000000`), cold electric blue (`#0099ff`), and the live product surface (scoreboard, theater, frame replay) as the hero art — not illustrations or decorative icons.

Typography is compressed and kinetic: display headings use tight negative tracking; body stays Inter-like with refined OpenType where available. Primary CTAs are **solid white pills** on black; Framer Blue is reserved for links, focus rings, live indicators, and card containment rings.

**Key characteristics**
- Void black canvas — absolute dark, never warm charcoal
- One accent: Framer Blue (`#0099ff`)
- Pill buttons (40px–100px radius) — no squared CTAs
- Product UI as marketing: theater split, live episode cards, Atari frame replay
- Frosted surfaces via `rgba(255,255,255,0.1)` — not heavy backdrop-blur glass
- Dense within panels, spacious between sections (void = dramatic pause)

---

## 2. Design System Source of Truth

Import / mirror these from the Framer package (do not invent parallel token names):

| Asset | Role |
|-------|------|
| `open-design/design-systems/framer/DESIGN.md` | Visual intent, do/don’t |
| `open-design/design-systems/framer/tokens.css` | `:root` CSS variables |
| `open-design/design-systems/framer/components.html` | Reference selectors / states |
| `open-design/design-systems/framer/components.manifest.json` | Allowed component groups |
| `open-design/design-systems/framer/tailwind-v4.css` | Optional Tailwind v4 bridge |

### Token map (bind in Vue as CSS variables)

```css
:root {
  --bg: #000000;
  --surface: #090909;
  --fg: #ffffff;
  --fg-2: #a6a6a6;
  --muted: rgba(255, 255, 255, 0.6);
  --meta: rgba(255, 255, 255, 0.4);
  --border: rgba(0, 153, 255, 0.15);
  --border-soft: rgba(255, 255, 255, 0.06);
  --accent: #0099ff;
  --accent-on: #ffffff;
  --success: #16a34a;
  --warn: #eab308;
  --danger: #dc2626;
  --font-display: "GT Walsheim Medium", "Inter", -apple-system, sans-serif;
  --font-body: "Inter Variable", "Inter", -apple-system, sans-serif;
  --font-mono: "Azeret Mono", ui-monospace, Menlo, monospace;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --radius-pill: 9999px;
  --elev-ring: 0 0 0 1px var(--border);
  --elev-raised:
    0 0.5px 0 0.5px rgba(255, 255, 255, 0.1),
    0 10px 30px rgba(0, 0, 0, 0.25);
  --focus-ring: 0 0 0 3px color-mix(in oklab, var(--accent), transparent 70%);
  --motion-fast: 150ms;
  --motion-base: 200ms;
  --ease-standard: cubic-bezier(0.2, 0, 0, 1);
  --container-max: 1200px;
}
```

**Font note:** GT Walsheim may be unavailable; fall back to a geometric sans with the same weight-500 + negative tracking rules. Prefer loading Inter Variable + a mono face; never Inter Display as the hero face.

---

## 3. Template → Screen Mapping

Use Open Design templates as **layout/component recipes**, always re-skinned with Framer tokens. Do not copy the old playground.

| Screen | Template recipe | What to take |
|--------|-----------------|--------------|
| Welcome hero + scroll | `web-prototype` hero + feature triplet | Compressed display headline, one accent CTA, void spacing |
| HF credentials panel | Framer `.card` + `.field` | Pill primary CTA; blue focus rings |
| Projects list | `dashboard` main + Framer cards | List of projects; create card; empty state |
| Create project | Framer form | Name + game select only |
| Project workspace | `dashboard` shell + `live-dashboard` + watch feed | Knobs, run actions, episode stream, frame replay, log |
| Theater A/B (later) | `TheaterSplit` product panels | Base vs champion side-by-side |

### Template hard rules
- **Single accent, ≤2 uses per viewport.**
- **No purple gradients, emoji strips, glassmorphic KPI cards, left-border accent cards.**
- **Charts:** hand-rolled SVG only.
- Product surfaces (workspace, theater) are the hero — not decorative art.

---

## 4. Information Architecture (Vue routes)

| Route | Guard | Purpose | Backend |
|-------|-------|---------|---------|
| `/` | — | Welcome scroll + HF credentials | `GET\|POST /api/profile` |
| `/projects` | requires `has_token` | Project list + create | `GET /api/experiments`, `GET /api/games` |
| `/projects/:name` | requires `has_token` | Project workspace | knobs, create/evolve/log, … |

Client noun **project** ↔ API noun **experiment** (`name` is the id). Creating a project calls `POST /api/experiments` with `{ name, game, … }`.

---

## 5. Screen Specs

### 5.1 Welcome (`/`) — first viewport
- Full-bleed void. Brand **SLM-RL** as hero-level signal.
- One headline: welcome to the platform.
- One short supporting sentence: what we’re going to do (train a small model on a game, improve it, publish).
- One CTA group: **Scroll to continue** (or anchor to HF section).
- No forms, no KPI strips, no project list in the first viewport.

### 5.2 Welcome scroll — “what we’ll do”
- Three tight steps (play → train → compare/publish) as a feature row or vertical stack.
- Dense within, spacious between (Framer void rhythm).

### 5.3 Hugging Face credentials (same page)
- Card on `--surface` + `--elev-ring`.
- Steps: create account → Settings → Access Tokens → paste.
- Fields: **display name** (required), **HF API key** (required to continue).
- Primary pill: **Save and continue** → `POST /api/profile` → navigate to `/projects`.
- No “skip for now” on this journey (token unlocks projects).

### 5.4 Projects (`/projects`)
- Top bar: brand + signed-in name + link back only if needed.
- **Existing projects** as a clean list/cards (name, game, status, mean score).
- **Create project**: ask for **name** only first; then **game** select (workshop slate).
- Open → `/projects/:name`.
- Empty state: one sentence + create CTA.

### 5.5 Project workspace (`/projects/:name`)
- Back to projects list always available.
- Show selected game; allow confirming/changing before first run if backend allows (create-time game is fixed today — show game as read-only after create, knobs editable).
- Knob grid, model/backend optional, **Run** / **Evolve** / **Theater**, publish when token present.
- **Live episode stream** in-page (not the Python `/watch/:name/` HTML): SSE → `EpisodeCard` list + optional frame panel (`Watch screen`). Consume `/watch/:name/events` and `/watch/:name/frames` directly.
- Live log panel (rollout / evolve / theater tails).
- Busy: honor `409` with a clear toast.
- Do **not** deep-link attendees to the stdlib HTML viewer; keep those routes as API/media endpoints only.

### 5.6 Later (not blocking workspace shell)
- Theater A/B split UI (`TheaterSplit`), gens grid — same APIs as today, Framer-skinned.
---

## 6. Component Inventory (Vue)

Map Framer / template primitives to Vue SFCs. Reuse these names so agents and humans share vocabulary.

| Component | Based on | Notes |
|-----------|----------|-------|
| `AppShell` | `dashboard` sidebar + topbar | Sticky aside + header |
| `LivePill` | `live-dashboard` live-pill | 3 states; pulse respects `prefers-reduced-motion` |
| `KpiCard` | live-dashboard kpi | Tabular nums; tween on update; **no** progress bar under value |
| `Sparkline` | live-dashboard spark SVG | Accent stroke 2px, fill 10% alpha |
| `UiButton` | Framer `.btn-primary` / `.btn-secondary` | Primary = white pill + black text; secondary = frosted |
| `UiField` | Framer `.field` | Focus = `#0099ff` ring |
| `UiCard` | Framer `.card` | `--elev-ring`; hover may intensify ring only |
| `Badge` / `StatusPill` | Framer badge + live-dashboard pills | Map experiment states: idle / running / promoted / failed |
| `ScoreboardTable` | live-dashboard db-row / web-prototype `ds-table` | Dense; mono for numerics |
| `KnobGrid` | form-row grid | 2-col desktop → 1-col &lt;809px |
| `EpisodeCard` | Framer card + activity row | Guess / reward / parse / monitor flags |
| `TheaterSplit` | hero-split | Base vs champion |
| `SignupOverlay` | modal card | Gate only |
| `TierBanner` | callout | Hardware tier + preset hint |
| `Toast` | subtle bottom toast | Prefer muted meta text over red banners for soft errors |

### Button rules (Framer)
- Primary: `background: #fff; color: #000; border-radius: var(--radius-pill)`
- Secondary: `background: rgba(255,255,255,0.1); color: #fff; border-radius: var(--radius-pill)`
- Ghost: text only; hover frosted
- Never square CTAs; never blue-filled primary for default actions (blue = chromatic accent only)

---

## 7. Backend Connection Contract

The Vue app talks to the playground process (default `http://127.0.0.1:8780`). During migration, either:

1. **Dev proxy** — Vite `server.proxy` → playground host, or  
2. **CORS** — allow localhost origin on the Python server (prefer proxy first to avoid changing trust model).

### 7.1 JSON API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/games` | `{ games, default }` |
| `GET` | `/api/knobs?game=` | Knob schema for form |
| `GET` | `/api/experiments` | Scoreboard rows |
| `POST` | `/api/experiments` | Create project. Body may include `launch: false` (Vue default — run from workspace). |
| `POST` | `/api/experiments/:name/rollout` | Quick Run game (solver/random episodes) |
| `POST` | `/api/experiments/:name/evolve` | Launch evolve. Body: `{ generations, dataset_url?, dqn_url? }`. `dataset_url` = public HF pack (skips live warm-start); empty = DIY live `--warm-start`. |
| `POST` | `/api/experiments/:name/theater` | Launch exhibition |
| `POST` | `/api/experiments/:name/play-again` | Re-run exhibition seeds |
| `GET` | `/api/experiments/:name/theater-scores` | Live A/B scores |
| `GET` | `/api/experiments/:name/log?kind=` | Text tail (`rollout` \| `evolve` \| `theater`) |
| `POST` | `/api/experiments/:name/publish` | HF publish |
| `GET` | `/api/reward-template` | Default reward hook source |
| `GET` | `/api/profile` | Masked profile or **404** |
| `POST` | `/api/profile` | Save `{ name, hf_token? }` |
| `GET` | `/api/hardware` | Detected tier + presets |
| `GET` | `/api/packs` | Local bake packs + `{ baking }` |
| `GET` | `/api/packs/log` | Bake subprocess log tail |
| `POST` | `/api/packs/bake` | Bake pack(s) via UI. Body: `{ game?, all?, episodes?, dqn_decisions?, push?, push_prefix? }` |

**Status conventions to surface in UI**
- `400` — validation; show field-level or toast message from `{ error }`
- `404` — missing experiment / profile
- `409` — busy lock or missing HF token for publish — **never crash**; clear busy copy

### 7.2 Streaming / media (viewer)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/watch/:name/` | Legacy HTML viewer (do not link from Vue; media endpoints below are the contract) |
| `GET` | `/watch/:name/events?gen=` | SSE episode stream |
| `GET` | `/watch/:name/frames?episode=` | PNG frame stream / replay |
| `GET` | `/theater/:name/:side/` | Legacy theater HTML (`base` \| `champion`) |
| `GET` | `/theater/:name/:side/events` | SSE |
| `GET` | `/theater/:name/:side/frames` | Frames |
| `GET` | `/gens/:name/` | Legacy multi-gen grid HTML |

Vue consumes **events + frames** directly (EventSource + `<img>` multipart) inside the project workspace — never scrape or embed the Python HTML pages.

### 7.3 Frontend data layer (recommended)
- `src/api/client.ts` — fetch wrapper, JSON errors
- `src/api/experiments.ts`, `profile.ts`, `hardware.ts`, `games.ts`
- `src/composables/useEventSource.ts` — SSE with reconnect
- `src/composables/useExperimentPoll.ts` — scoreboard + log polling
- Pinia (or lightweight stores): `profile`, `hardware`, `experiments`, `ui` (tutorial on/off)

Do not duplicate business logic (knob defaults, gate metrics). Trust API payloads; render only.

---

## 8. Workshop UX Constraints (must not regress)

From plans 021 / 026 and `docs/ARCHITECTURE.md`:

1. **UI-first** for attendees — CLI is instructor bootstrap only.
2. **Local profile only** — token masked on GET; never log tokens.
3. **Publish optional** — skip never locks the playground.
4. **Concurrency** — at most one quick + one evolve + one theater; show 409 clearly.
5. **Scoreboard** — all experiments; game column + filter chips.
6. **Tutorial mode** — info affordances on knobs/actions; toggleable.
7. **8GB ethos** — frontend must stay light (no huge analytics SDKs); theater still loads one model side at a time on the backend.
8. **Trust model** — reward-code execution is local-attendee-trust; bind UI assumptions to localhost.

---

## 9. Typography & Layout Rules

### Type roles (app-scaled; marketing hero sizes optional on landing only)

| Role | Token / size | Tracking |
|------|--------------|----------|
| App title / section | `--text-2xl`–`--text-3xl` display, weight 500 | `--tracking-display` |
| Card title | 22–24px body/display | slight negative |
| Body / UI | 15px Inter | −0.15px nav |
| Caption / hint | 12–14px `--muted` / `--fg-2` | normal |
| Code / logs / seeds | `--font-mono` 12–13px | normal |
| KPI value | 32px, weight 600, `tabular-nums` | −0.01em |

### Layout
- Max content width ~1200px; dashboard main may go full remaining width beside sidebar.
- Spacing: 8px base; card padding 16–24px; section gaps generous on marketing blocks, tighter in ops panels.
- Breakpoints (Framer-aligned): mobile &lt;809px, tablet 809–1199, desktop &gt;1199.
- Mobile: collapse sidebar to drawer; stack theater columns; full-width pills.

---

## 10. Motion

- Prefer CSS transitions (`--motion-fast` / `--motion-base`, `--ease-standard`).
- High-impact moments only: page enter stagger for KPI row; theater score tick; live-pill pulse.
- KPI number tween ~600ms on refresh (`live-dashboard`).
- Honor `prefers-reduced-motion: reduce` (disable pulse / stagger).
- No long decorative animation loops.

---

## 11. Do’s and Don’ts

### Do
- Use pure `#000000` backgrounds and Framer tokens exclusively for color.
- Keep primary CTAs white pills; blue for focus/links/live/rings only.
- Let theater, scoreboard, and frame replay be the visual centerpiece.
- Handle 404 profile and 409 busy as first-class UI states.
- Proxy the existing API — don’t invent parallel REST shapes without updating Python.

### Don’t
- Warm grays (`#1a1a1a`, `#2d2d2d`) as page background.
- Extra accent colors, purple gradients, or light theme in v1.
- Squared buttons or heavy drop shadows.
- Glassmorphism / blur on KPI cards.
- Emoji icon strips or decorative illustrations.
- Chart libraries or dashboard kits that override the token system.
- Send HF tokens anywhere except `POST /api/profile` / publish path the backend already owns.

---

## 12. Vue Project Skeleton (suggested)

```
slm-rl-web/                 # sibling or packages/web
  package.json              # vue 3 + vite + vue-router + pinia
  index.html
  src/
    main.ts
    App.vue
    styles/
      tokens.css            # copied/adapted from framer/tokens.css
      base.css
    api/
    components/
      shell/
      ui/                   # UiButton, UiField, UiCard, LivePill, …
      experiments/
      theater/
      watch/                # LiveWatchPanel, EpisodeCard, ScreenPanel
    composables/            # useEventSource, useWatchStream, …
    stores/
    views/
      WelcomeView.vue
      ProjectsView.vue
      ProjectView.vue       # includes live watch panel
      TheaterView.vue       # later
    router/
  vite.config.ts            # proxy /api, /watch, /theater, /gens → :8780
```

Backend remains `uv run slm-rl playground` / Docker `8780`. Frontend `npm run dev` proxies to it.

---

## 13. Implementation Phases

1. **Scaffold** — Vue + Vite + tokens.css + AppShell + proxy.
2. **Read path** — games, hardware, knobs, experiments, profile gate.
3. **Write path** — create experiment, evolve, theater, play-again, publish.
4. **Live surfaces** — log tail, SSE episode cards, frames.
5. **Replace embeds** — remove dependence on Python-generated HTML pages.
6. **Polish** — tutorial cards, motion, empty/busy/error states, a11y (focus rings, 44px targets).

---

## 14. Acceptance Checklist

- [ ] Void black + single Framer Blue accent; white primary pills
- [ ] Signup 404 → overlay; skip leaves app usable
- [ ] New experiment + scoreboard match API fields (incl. game filter)
- [ ] 409 busy and publish-without-token are explicit UI states
- [ ] Project workspace live episode SSE + frames work without scraping HTML / opening `/watch/...`
- [ ] No token leakage in UI logs or network panel beyond intended POSTs
- [ ] Mobile: sidebar drawer, stacked theater, usable touch targets
- [ ] `prefers-reduced-motion` respected on live-pill / KPI tween

---

## 15. Agent / builder prompt stubs

Use when generating Vue screens:

- “Build `AppShell` on void black with 240px sidebar, Inter 15px nav, white pill CTA, Framer Blue ring on active nav item.”
- “Scoreboard: live-dashboard KPI row + dense table; tabular nums; game filter pills with blue ring active state.”
- “Signup card: `--surface` + `--elev-ring`, white Save pill, ghost Skip; overlay until profile exists.”
- “TheaterSplit: two product panels with `--elev-raised`, live score strip, SSE-driven EpisodeCards.”

When refining: change one component at a time; verify tracking on display headings; ensure blue appears only on interactive/live accents.
