# Hybrid RL: Classical Teachers Around the SLM Loop

Combine old-school deep RL (CleanRL-style DQN/PPO) with the LLM generation
loop to raise quality and cut training time. Classical RL is sample-hungry
but ~10⁴× cheaper per step; the LLM loop is the opposite — so let cheap
learners do the exploration and dense-signal discovery, and spend LLM compute
only where language reasoning matters.

**Motivating evidence** (grpo-1p2b run, July 2026): ~2 GPU-hours of pure-LLM
GRPO moved Mastermind win rate 0.4% → 0.6% (rejected by the gate); every eval
episode averaged ~2.0 doom-loop interventions (repeated guesses). The
bottlenecks are exploration-from-zero and repetition — both things a cheap
teacher solves structurally.

## Reference implementation: CleanRL

[CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT) is the pattern, not a
dependency: **single-file, standalone algorithm implementations** (~300–400
lines each) on torch + Gymnasium — `dqn.py`, `dqn_atari.py`, `ppo.py`, `c51`,
`sac`, etc. We adopt:

- The single-file philosophy: one self-contained `teachers/dqn.py`, readable
  top to bottom, no framework. Matches our lazy-import/optional-extra rules.
- Their `dqn.py` (vector observations, MLP Q-network, replay buffer, target
  network, epsilon-greedy) — **not** `dqn_atari.py`'s pixel conv stack: our
  games are text/RAM-native, so the teacher consumes a `Game.vector_obs()`
  hook (Atari: the 128 RAM bytes, already our D9 observation source).
- torch is already installed on training tiers; the teacher adds no deps.

## The interoperability layer: RolloutRecord

The two worlds never call each other's internals — they meet at the
trajectory schema. A `DQNAgent` implements the existing `Agent` ABC, so
`EpisodeRunner`, `RolloutWriter`, `consolidate`, `sft_export`, and the HF
dataset push all work on teacher games unchanged (`model_id="teacher:dqn"`).

```
game engine (pure Python, + vector_obs() hook)
        │
        ├──► teacher (CleanRL-pattern DQN; minutes, thousands of episodes)
        │        │
        │        ├─ trajectories ─► RolloutRecord parquet ─► reject_sft
        │        │                  (gen-0 warm-start: LLM imitates teacher wins)
        │        ├─ Q(s,·) top-k ─► pruned action menus        (ROLLOUT phase)
        │        └─ V(s) ────────► potential-based shaping     (TRAIN phase)
        │
        └──► LLM loop unchanged: rollout → dataset → train → EVAL → gate
                                                     (teacher NEVER in eval)
```

## Four integration seams

1. **Teacher-as-Agent → warm-start (highest leverage).** Train the teacher in
   minutes, play N episodes at engine speed, distill wins through the
   existing `reject_sft` path into the gen-0 adapter. Deletes the R7
   zero-win bootstrap: GRPO refines a competent policy instead of hoping to
   sample wins from a base model.
2. **Q-head as menu pruner (kills repetition structurally).** The teacher's
   top-k actions (minus anything already played) become the LLM's numbered
   menu — every game becomes a small-menu game (D3's sweet spot: selection,
   not generation). The LLM's job shifts to choosing among strong candidates,
   which is where language reasoning actually adds value.
3. **V(s) as reward shaping (learned replacement for hand-coded rewards).**
   `grpo_export` stamps Φ(s)=max_a Q(s,a) into `game_ctx` at dataset-build
   time; GRPO gains a `γΦ(s′) − Φ(s)` term (potential-based ⇒ optimal policy
   unchanged). This generalizes the hand-built Mastermind consistency reward
   to any game with a trainable teacher — the plan for Atari, where
   "consistency" has no hand-codable analogue.
4. **Gate purity (non-negotiable).** Eval stays LLM-only on the frozen suite.
   Teachers assist training but never inflate measured skill — promotion
   still means *the language model* got better. Side dividend: teacher
   win-rate per game config is an automatic difficulty/curriculum signal.

## Build order

1. **Mastermind exact solver as teacher** (~30 lines, no DQN training):
   consistency filtering is exact for this game. Validates the whole
   teacher-as-Agent → warm-start → GRPO pipeline immediately, and doubles as
   the menu pruner (propose only feedback-consistent, unplayed codes).
2. **`teachers/dqn.py`** (CleanRL `dqn.py` pattern, MLP over `vector_obs()`)
   when Freeway lands (Phase 3) — Atari is where a learned teacher earns its
   keep. Blackjack's basic-strategy table is another exact-teacher case.
3. **Shaping term** in GRPO rewards once teacher Q-values exist on disk.

Honest framing: a DQN will flatly beat our SLMs at Atari scores. That's fine —
the product is a language-native reasoning agent and its dataset; classical
RL's role is to make the SLM's education faster and denser, not to compete.
