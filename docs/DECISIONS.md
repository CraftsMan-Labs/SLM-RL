# Design Decisions

Each decision records the choice, the rationale, and what would make us revisit it.

## D1. Env abstraction: our own Gymnasium-style core + thin OpenEnv bridge

All games implement our text-native `Game` ABC. Thin adapters in `slm_rl/bridges/`: `gym_adapter.py` wraps external Gymnasium envs (ALE) in; `openenv_bridge.py` wraps our games out as OpenEnv FastAPI servers.

**Why**: OpenEnv v0.4.x is experimental with breaking releases — coupling game logic to it puts churn at the center of the codebase. In-process rollouts are 10–100× cheaper than HTTP round-trips per step, which dominates on a single GPU (rollout generation is ~70% of GRPO wall-clock). OpenEnv is an optional pinned extra (`slm-rl[openenv]`).
**Revisit if**: OpenEnv stabilizes at 1.0 and TRL's multi-turn `environment_factory` path clearly outperforms our per-decision flattening (see Risks).

## D2. Multi-agent as single-agent view + opponent pool

The env always presents a single-agent view; `step()` internally advances opponent turns via an injected `OpponentPolicy`. `OpponentPool` sources: `RandomBot`, per-game heuristic bots, frozen earlier generations (LoRA adapter swaps). Default league mix: 20% heuristic / 30% latest champion / 50% uniform over past K champions.

**Why**: keeps the training loop identical for solitaire and competitive games; prioritized fictitious self-play avoids overfitting to the latest self.

## D3. Action interface: enumerated legal-move menu, not JSON

Observation + numbered legal-move menu; free-text reasoning allowed but the completion must end with `ACTION: <index or canonical string>`. Parsing ladder: exact → fuzzy → one retry with error feedback (−0.05) → random legal move substituted with `invalid_action=True` (−0.2). Forfeit only on a configurable invalid streak.

**Why**: 350M–2B models are unreliable at JSON tool-calls; menus turn a generation problem into a selection problem — exactly what RL can improve. Random substitution keeps episodes informative and blocks reward-hacking via early termination. Constrained decoding (grammar on the final line) is a backend capability flag: on for datagen quality, off during training so format-following remains learnable.

## D4. Anti-doom-loop: two-level design

See ARCHITECTURE.md ("Anti-doom-loop design"). Rollout-level `DoomLoopMonitor` with the reflect → backtrack → truncate ladder; training-level entropy floor, KL-to-champion, reward hygiene, and the `EvalGate` with auto-remediation and rollback.

**Why**: the two failure modes are different — an *agent* stuck in a loop needs an in-episode intervention (including true backtracking via state snapshots); a *policy* collapsing onto one branch needs training-time pressure (entropy, KL) and a promotion gate so a degenerate generation never becomes the champion.

## D5. Hardware tiers: config-driven table, first match wins

`configs/hardware.yaml` ordered most → least capable, ending with the unconditional `any-8gb` floor. Detection (`platform/hardware.py`) resolves against the table only — models are never hardcoded. `--tier/--model/--backend` override anytime.

**Why**: the user's requirement — "assume a person arrives with a Mac that has 8GB of RAM" — plus graceful scaling to 16/24GB. Config-driven so new models (or new tiers) are YAML edits, not code changes.

## D6. Config & CLI: plain YAML + pydantic v2, Typer — not Hydra

Precedence: `configs/default.yaml` → `configs/games/<game>.yaml` → user YAML → CLI flags.

**Why**: Hydra's composition power isn't needed, and its sys.argv/cwd hijacking fights a Typer CLI. pydantic gives validation and typed access.

## D7. Game plugin contract

Implement the `Game` ABC (`reset`, `step` with reward+terminal inside, `state_hash`, `system_prompt`, optional `snapshot/restore`, `heuristic_opponents()`, `eval_suite()`); register with `@register_game("name")` in-repo or the `slm_rl.games` entry-point group from any pip package; ship a `configs/games/<name>.yaml`. See PLUGIN_GUIDE.md.

**Why**: entry points are the standard Python plugin mechanism — Catan or a driving sim onboard without touching this repo. Built-in games use the same mechanism so it's continuously exercised.

## D8. Dominion v1 scope

2 players; fixed kingdom of 8 non-attack base cards (Village, Smithy, Laboratory, Market, Festival, Council Room, Workshop, Mine) + treasures/victories; simplified turn (enumerated action plays, treasures auto-played, auto cleanup) → ≤~20 legal moves per decision. Opponent: Big Money (+ BM-Smithy). Reward: ±1 terminal, +0.01·ΔVP shaping clipped; end on Provinces empty / any 3 piles / 40-turn cap (score by VP at cap). Deferred: attacks/reactions, 3–4 players, kingdom randomization, nested sub-decisions (the card list was chosen to avoid them).

**Why**: full Dominion is a multi-month engine project; this subset preserves the strategic core (engine vs money trade-off) at a tractable action-space size.

## D9. Atari via ALE, RAM → text

In-process `ale-py` with `obs_type="ram"`; Freeway first (simplest decodable RAM map). RAM decoding is a per-game plugin file (`games/atari/ram_maps/`). The OpenEnv Docker client mode (per the user's reference example) is available through the bridge for machines with headroom.

**Why**: text-native observations keep small language models in their element (no vision tower needed for gameplay); Freeway's RAM map is well understood. Per-game RAM reverse-engineering is real effort — acknowledged, not generalized.

## D10. Tier-adaptive training: GRPO where possible, rejection-sampling SFT everywhere

Two `TrainingStrategy` implementations behind one interface (`training/base.py`):

- **`grpo`** (CUDA tiers): TRL `GRPOTrainer` + PEFT LoRA on a 4-bit base, per-decision flattening (each decision = one prompt, G resamples; terminal reward broadcast with discounting), entropy/KL safeguards.
- **`reject_sft`** (universal — the 8GB path): expert iteration / rejection-sampling (STaR/ReST-style). Play N episodes with temperature sampling → keep winning / top-quantile, monitor-clean trajectories with a diversity quota → LoRA-SFT on the (prompt → winning action) pairs → same EvalGate. Runs via `mlx-lm` on Apple Silicon and transformers+PEFT on CPU.

**Why**: GRPO on Apple Silicon/MLX is not yet mature (mid-2026), and the user requires the *full* loop on 8GB. `reject_sft` is also the gen-0 warm-start GRPO needs, so it's built first. Known trade-off: it improves more slowly and can plateau (it only reinforces behaviors the model already samples) — mitigated by sampling temperature/diversity and documented honestly; GRPO tiers exist for faster progress.
**Revisit if**: MLX GRPO (`mlx-tune`, `MLX-GRPO`) reaches TRL-level stability.

## D11. Classical-RL teachers around the LLM loop (CleanRL pattern)

Cheap classical learners (exact solvers where they exist; otherwise a
CleanRL-style single-file DQN over a `Game.vector_obs()` hook) scaffold the
LLM loop at three seams: teacher trajectories → `reject_sft` warm-start;
teacher top-k → pruned action menus; teacher V(s) → potential-based GRPO
reward shaping. Teachers implement the same `Agent` ABC, so the entire
datagen stack works on them unchanged. **Eval stays LLM-only** — teachers
assist training, never the gate. See HYBRID_RL.md.

**Why**: measured (grpo-1p2b, Jul 2026) — pure-LLM GRPO spent ~2 GPU-hours to
move win rate 0.4%→0.6%, blocked on exploration-from-zero and repetition.
Classical steps are ~10⁴× cheaper; letting them do exploration/dense-signal
discovery reserves LLM compute for language reasoning and evaluation.
CleanRL (MIT) is a pattern reference (single-file, torch+Gymnasium), not a
dependency.
**Revisit if**: teacher-distilled policies cap the LLM's ceiling (imitation
lock-in) — then reduce teacher mixing per generation rather than dropping it.

Implementation notes (Mastermind, Jul 2026):

- **Pruning happens in the EpisodeRunner**, not in an Agent wrapper — records
  persist the runner's observation, so runner-level pruning gives SFT/GRPO
  exports the pruned menu for free. Pruned episodes alternate with
  format-mode episodes (`teacher.pruner_fraction`) because the gate eval is
  format-mode: training only on menus would create a train/eval mismatch.
- **Warm-start is just generation 1** (`evolve --warm-start`): the solver
  plays rollout (no inference backend), strategy is forced to `reject_sft`,
  and the normal dataset→train→eval→gate path runs. Idempotent on resume.
  Teachers stamp `model_id="teacher:..."` and build LLM-identical prompts
  via `build_messages` — without those the SFT export would yield 0 pairs.
- **The GRPO consistency reward was replaced by an elimination reward**
  (seam 3 realized exactly: Φ(s) = −log|consistent(s)|). Two measured defects
  forced this: a repeated wrong guess scored (k−1)/k under the consistency
  fraction (near-max — the reward *fed* the repeat doom loop, now −1.0), and
  under a pruned menu every option is consistent, so group reward std hit
  zero (dead gradient). Elimination varies per option even on all-consistent
  menus and on turn 0.
- **Auto-remediation no longer escalates entropy_bonus** — one doubling
  (0.01→0.02) measurably sent train entropy to 7.82 (random play). LR still
  halves (floored at 1e-6) and both knobs reset on the next promotion.
- **Dual eval**: the gate eval never sees the pruner; `metrics.eval_pruned`
  (small with-pruner suite) is the product metric, recorded only. On
  Mastermind the exact pruner saturates it (even a random policy wins ~100%
  under consistent-candidate menus) — the honest signal stays the gate.
