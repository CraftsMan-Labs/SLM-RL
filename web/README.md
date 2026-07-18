# SLM-RL Web

Vue 3 frontend for the workshop playground. Visual system: Framer void / blue (`../DESIGN.md`).

## Docker (recommended — hot reload)

From the repo root:

```bash
docker compose up --build
# UI  → http://127.0.0.1:5173/
# API → http://127.0.0.1:8780/
```

- **Vue:** Vite HMR via bind mount `./web` + polling on Docker Desktop.
- **API:** `watchfiles` restarts the playground when `slm_rl/` or `configs/` change.

## Local (without Docker)

```bash
# Terminal 1 — backend
uv run slm-rl playground   # http://127.0.0.1:8780

# Terminal 2 — frontend
cd web
npm install
npm run dev                # http://127.0.0.1:5173  (proxies /api → :8780)
```

## Journey

1. `/` — Welcome + HF credentials
2. `/projects` — Bake packs (instructor) + list/create projects
3. `/projects/:name` — **Run game**, Evolve, Theater, Live view, Publish

One command: `docker compose up --build` → browser only.
