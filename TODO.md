# TODO — Four-game CUDA-optional RL

Keep only: **boxing**, **space-invaders**, **freeway**, **demon-attack**.  
Delete everything else. Every keeper needs DQN pack + SFT dataset + GRPO (CUDA or MPS).

---

## 1. Cull non-keeper games

- [x] Delete game modules/configs for: `mastermind`, `mastermind-easy`, `pong`, `breakout`, `bowling`, `connect4`, `blackjack`, `wordle`, `2048`, `minesweeper`
- [x] Remove their teachers
- [x] Update `slm_rl/teachers/__init__.py` registrations
- [x] Update `pyproject.toml` game entry points (exactly 4)
- [x] Update `slm_rl/packs.py` `ATARI_GAMES` / `DQN_GAMES`
- [x] Retarget defaults (`configs/default.yaml`, CLI `--game`) → `boxing`
- [x] Delete or rewrite game-specific tests; fix shared fixtures
- [x] README four-game catalog
- [ ] Docs pass: LIFECYCLE / ARCHITECTURE / DECISIONS / PLUGIN_GUIDE (still mention deleted games in places)
- [x] Confirm registry lists exactly these 4 games

---

## 2. Generic GRPO (CUDA optional)

- [x] Replace mastermind-only gate in `slm_rl/datagen/grpo_export.py` with generic export
- [x] `game_ctx` includes: legal actions, step reward, discounted return, target action
- [x] Replace Mastermind rewards in `slm_rl/training/grpo.py` with:
  - [x] `format_reward` (parseable / legal ACTION)
  - [x] `return_reward` (from recorded discounted return / step reward)
- [x] Device policy: CUDA bf16 · MPS fp32 · CPU fallback (no MPS fp16)
- [x] MPS-safe batch sizes + grad accumulation
- [x] `configs/hardware.yaml`: all tiers → `transformers` + `train: grpo` (incl. any-8gb / 350M)
- [x] Unit tests for GRPO export/rewards
- [x] Replace old mastermind `test_grpo_*` tests

---

## 3. SFT hygiene

- [x] Strip DQN `Q-values rank…` rationales in `sft_export` (ACTION-only)
- [x] Boxing renderer: relative geometry + edge / punch-range hints
- [x] `boxing-sft-002` ACTION-only train complete (loss ~0.26, ~90% token acc)
- [x] Re-publish boxing LoRA from `boxing-sft-002`
- [ ] Re-publish boxing pack SFT jsonl (ACTION-only) on HF dataset repo

---

## 4. Bake + publish packs (per game)

For each game: `dqn.pt` + top-quantile rollouts + `dataset/sft.jsonl` + `MANIFEST.json` → HF `…/slm-rl-<game>`.

### boxing
- [x] DQN trained
- [x] Pack / dataset published (`BLANK/slm-rl-boxing`)
- [x] ACTION-only SFT LoRA (`boxing-sft-002`) trained + published to `BLANK/slm-rl-boxing`
- [ ] Refresh dataset SFT file on HF to ACTION-only
- [ ] GRPO generation after warm-start (MPS smoke)

### space-invaders
- [ ] Train high-scoring DQN
- [ ] Bake pack (demos + SFT)
- [ ] Publish HF pack
- [ ] Warm-start reject_sft
- [ ] GRPO generation

### freeway
- [ ] Train high-scoring DQN
- [ ] Bake pack (demos + SFT)
- [ ] Publish HF pack
- [ ] Warm-start reject_sft
- [ ] GRPO generation

### demon-attack
- [ ] Train high-scoring DQN
- [ ] Bake pack (demos + SFT)
- [ ] Publish HF pack
- [ ] Warm-start reject_sft
- [ ] GRPO generation

---

## 5. Validation

- [x] `boxing-sft-002` finishes with finite loss / non-nan grads
- [x] Unit suite green
- [ ] Smoke: `evolve` boxing — reject_sft warm-start → 1× GRPO on MPS
- [ ] Theater A/B: base vs champion shows varied actions (not always UP/RIGHT)
- [ ] Quick smoke GRPO path for one other Atari keeper

---

## Out of scope (do not reopen without a new plan)

- Keeping deleted games behind feature flags
- MLX-native GRPO
- Vision / VL games
- Mastermind pruner / elimination rewards
