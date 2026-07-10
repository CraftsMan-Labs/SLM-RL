# Roadmap

Each phase is a shippable milestone. Status: **skeleton complete** (interfaces, configs, tier detection, docs, tests) — Phase 1 is next.

## Phase 1 — Full loop on Mastermind, 8GB-first

The milestone that proves the whole thesis end-to-end.

- Mastermind engine + renderer + engine unit tests
- `LlamaCppBackend` / `MLXBackend` (8GB path) + `TransformersBackend` (CUDA/MPS)
- `LLMAgent` with the D3 parsing ladder + golden tests
- `EpisodeRunner` + basic `DoomLoopMonitor` (action-repeat + invalid-streak signals)
- Streaming JSONL writer + chunked parquet consolidation
- **`reject_sft` trainer first** (mlx-lm / transformers+PEFT), then `grpo` (TRL + LoRA, KL-to-champion, entropy logging)
- Fixed-seed eval suite + `EvalGate` + `ModelRegistry` promote/rollback
- 8GB memory-budget CI test

**Exit criteria**: `slm-rl evolve --game mastermind --generations 5` shows measurable win-rate improvement gen0→gen3, (a) via `reject_sft` on an 8GB machine with LFM2.5-350M, and (b) via GRPO on a CUDA card.

## Phase 2 — Competitive + stochastic

- Hybrid-RL seam 1+2 on Mastermind (HYBRID_RL.md): exact consistency solver as teacher-Agent → `reject_sft` warm-start, and as top-k menu pruner — validates the teacher pipeline with zero DQN training
- Connect-4 engine + heuristic bot (shallow minimax)
- `OpponentPool` with frozen-generation league; ELO module with bot anchors
- Blackjack engine + basic-strategy eval baseline; variance handling (larger GRPO groups, paired-seed eval)
- `VLLMBackend` with multi-LoRA (fast CUDA rollouts, cheap frozen opponents)

## Phase 3 — Atari + anti-doom hardening

- `gym_adapter.py` + ALE Freeway RAM→text renderer, decision-point downsampling
- Further ALE games as self-contained plugins, easiest-first ramp: Pong → Breakout → Space Invaders (→ Ms. Pac-Man as a flagship-tier stretch) — see PLUGIN_GUIDE.md §6
- Full monitor suite: state-hash revisits, reward stagnation, reflect/backtrack/truncate ladder + degenerate-agent tests
- Entropy-floor alarm + auto-remediation; optional antidoom hygiene stage
- SFT warm-start pipeline from heuristic/teacher traces (tiny models need it for Atari)
- `teachers/dqn.py` — CleanRL-pattern single-file DQN over `Game.vector_obs()` (Atari RAM bytes) as the learned teacher: warm-start traces, top-k menus, V(s) reward shaping for GRPO (HYBRID_RL.md, D11)

## Phase 4 — Dominion + product polish

- Dominion v1 (see DECISIONS.md D8) with heavy engine unit tests; reward shaping tuning
- Streamlit dashboard: win-rate/ELO/entropy/intervention-rate across generations
- `slm-rl new-game` scaffold + plugin docs
- `slm-rl export --merge --gguf` for Mac play of trained models

## Phase 5 — Ecosystem (stretch)

- `openenv_bridge` server packaging + TRL `environment_factory` training path
- Offline RL / distillation from the accumulated dataset product
- Cross-game curriculum (one adapter trained across games)
- Catan as the first external-plugin proof
