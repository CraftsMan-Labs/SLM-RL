# Adding a Game

Games are plugins. A game is pure Python, seed-deterministic, text-native, and has **no ML dependencies** — it knows nothing about models, training, or rewards shaping beyond its own rules.

## 1. Implement the `Game` ABC

```python
from slm_rl.games.base import ActionSpec, Game, Observation, StepResult
from slm_rl.games.registry import register_game

@register_game("mygame")
class MyGame(Game):
    def reset(self, seed=None) -> Observation:
        """Initialize deterministic-given-seed state; return the first
        Observation (rendered text + the full legal-action menu)."""

    def step(self, action: ActionSpec) -> StepResult:
        """Apply the action, advance any opponent turns internally
        (self.opponent), compute reward and terminal INSIDE the game
        (rule-based, verifiable). Invalid actions never reach you —
        the agent layer guarantees `action` is from legal_actions."""

    def state_hash(self) -> str:
        """Stable hash of the current state — powers loop detection."""

    def system_prompt(self) -> str:
        """Rules preamble for the model. Keep it tight: the 8GB tier has a
        2048-token context budget for the whole conversation."""

    @classmethod
    def eval_suite(cls) -> "EvalSuite":
        """Fixed seeds + primary metric; the EvalGate compares generations
        on this frozen suite."""
```

Optional overrides:
- `snapshot() / restore()` — state checkpoints for the backtrack intervention (default pickles `__dict__`; override for efficiency or to exclude caches).
- `heuristic_opponents()` — named scripted opponents (e.g. Big Money for Dominion). Strongly recommended for competitive games: they anchor the ELO league and serve as eval baselines.
- `vector_obs()` — the state as a flat float vector (Atari: the 128 RAM bytes). Enables classical-RL teachers (CleanRL-pattern DQN) for warm-start traces, menu pruning, and reward shaping — see HYBRID_RL.md.

## 2. Design the observation for a small model

- Render compact text; every token counts on the 8GB tier.
- Enumerate **all** legal actions as `ActionSpec`s — the model picks from a numbered menu, it never free-forms moves.
- Keep per-decision menus small (≲20 entries). If your game has huge action spaces, decompose the turn into sub-decisions.

## 3. Ship a config

`configs/games/mygame.yaml`:

```yaml
max_turns: 50
eval_episodes: 300
monitor:
  interventions: [reflect, truncate]   # add `backtrack` if snapshots make sense
extra:                                 # your game-specific knobs
  board_size: 9
```

## 4. Register

- **In this repo**: the `@register_game("mygame")` decorator plus an entry in `pyproject.toml` under `[project.entry-points."slm_rl.games"]`.
- **Your own package** (the Catan / driving-sim path): just declare the entry point —

```toml
[project.entry-points."slm_rl.games"]
catan = "catan_pkg.env:CatanGame"
```

`pip install your-package` next to `slm-rl` and the game appears in `slm-rl info` / `--game catan`. No changes to SLM-RL required.

## 5. Test

Engine tests are pure Python and must not need a model: legal-move generation, terminal detection, scoring, determinism given a seed, and `state_hash` stability. See `tests/` for the pattern.

## 6. Atari games as plugins

Every ALE game follows the same self-contained recipe — a `Game` subclass
holding an `ale-py` env with `obs_type="ram"`, plus three per-game pieces and
**no changes to the training/orchestration side**:

1. **RAM decoder** (`games/atari/ram_maps/<game>.py`) — byte offsets → named
   state variables. The AtariARI annotations (Anand et al.) document these
   offsets for ~22 games; being on that list is the practical feasibility test.
2. **Text renderer** — the decoded variables as a compact observation
   (≲15 lines; the 8GB tier budget applies).
3. **Action map + frame-skip** — the minimal discrete action set, with one
   LLM decision covering ~4–8 frames. ALE steps synchronously, so slow
   decisions are fine — the game waits.

Candidate ramp, easiest first (all AtariARI-annotated):

| Game | State to render | Why this position |
|---|---|---|
| Freeway | chicken y, car x's | 3 actions, one goal — the Phase 3 opener |
| Pong | ball x/y, paddle y | whole state is 3 numbers; good doom-loop test (paddle jitter) |
| Breakout | + brick count | Pong plus dense score reward — suits GRPO |
| Space Invaders | player x, alien columns, bombs, lives | dense reward; dodge-vs-shoot timing gets coarse under frame-skip |
| Ms. Pac-Man | pellet map, 4 ghosts, maze | flagship-tier: long horizon, big observation, maze reasoning — attempt after the ramp, likely with VL models reading frames instead of RAM text |

Dense-score games (Pong, Breakout, Space Invaders) fit the GRPO strategy
especially well: the environment reward itself provides group variance that
sparse-win games like Mastermind need engineered rewards for.

Space Invaders landed first (plan 008, out-of-ramp-order by user request) via
exactly this recipe — `GymnasiumGameAdapter` + `SpaceInvadersRenderer` +
`games/atari/ram_maps/space_invaders.py` — trained via `reject_sft` (GRPO
stays Mastermind-only for now).

Pong landed next (plan 016, ALE game pack) via the identical recipe —
`games/atari/pong.py` + `games/atari/ram_maps/pong.py` — also `reject_sft`.

Freeway landed third — `games/atari/freeway.py` +
`games/atari/ram_maps/freeway.py` — replacing the dead `atari_freeway`
NotImplementedError stub this table used to describe as "the Phase 3
opener." Its car-avoidance variant (car x's, per the table below) did not
verify (see `ram_maps/freeway.py`); it ships as a pure hold-UP teacher.

Breakout landed fourth (plan 016), completing the ALE game pack —
`games/atari/breakout.py` + `games/atari/ram_maps/breakout.py`. Its ball_x
and player_x bytes live in different pixel frames of reference; the
calibration constant (`ball_x - player_x == -5` at paddle contact) was
probe-verified across 5 seeds with zero variance. The table above still
reflects the original candidate ramp, not landing order.

Blackjack landed (plan 018) — text-native, stochastic rewards: the first
game in the pack where the right move can still lose. `games/blackjack/env.py`
+ `teachers/blackjack_solver.py` (basic strategy encoded as data tables keyed
by (player_total, dealer_upcard), not nested ifs). HIT/STAND only (no
double/split/insurance) against an infinite deck (`random.Random(seed)`
drives every card, so probabilities stay clean and analyzable). A natural
(21 on the opening two cards) is detected in `step`, not `reset` -- `reset`
must always leave `legal_actions` non-empty, since every agent type
(`RandomBot` included) indexes into it unconditionally.

Wordle landed next (plan 018) — text-native, Mastermind's consistency-
filtering lesson in a familiar skin: `games/wordle/env.py` +
`games/wordle/words.py` (curated ~500-word embedded answer/guess list, no
network fetch) + `teachers/wordle_solver.py` (consistency filter over
`score_guess` feedback strings, structurally identical to
`mastermind_solver.consistent_candidates`). The action space (hundreds of
words) is far above `MENU_LIMIT`, so the LLM plays in format mode exactly
like standard Mastermind. Duplicate-letter feedback (greens consume first,
then yellows left-to-right against the REMAINING letter counts) is the
classic correctness trap here -- `tests/test_wordle.py` has hand-computed
goldens for it, including a guess with MORE copies of a letter than the
answer has (`"sassy"` vs `"vases"` -> `"YGG--"`, not every 's' scoring).

2048 landed last in the pack (plan 018) — greedy vs planning: every move
looks locally good (any merge raises the score), but the winning long game
needs some merges sacrificed to keep the board maneuverable.
`games/game2048/env.py` (package `game2048` -- a Python package can't start
with a digit -- but the registry/config/CLI-facing name is the natural
`2048`, verified end-to-end: `load_game_config("2048")`, `--game 2048`, and
the playground experiment API all work with no digit-related friction) +
`teachers/game2048_priority.py`. The plan's baseline teacher was pure fixed
priority (LEFT>DOWN>RIGHT>UP, first legal wins); a one-ply greedy variant
(highest immediate merge sum, same order as tiebreak) measured meaningfully
better over 30 seeds (mean score 3166 vs 2012, ~57% higher) and ships
instead, per the plan's own "ship the greedy variant if it measures
better" clause. The classic slide/merge trap (`[2,2,4,0]` -> `LEFT` ->
`[4,4,0,0]`, never `[8,0,0,0]` -- a merged tile never merges again in the
same move) has hand-computed goldens in `tests/test_2048.py`. Monitor
thresholds (`action_repeat_threshold`, `ngram_loop_threshold`,
`reward_stagnation_window`) were measured against the teacher's own
streak/scoreless behavior rather than reused from Mastermind's structural
defaults -- 2048 legitimately produces long same-direction streaks that
aren't doom loops.

Connect-4 landed (plan 019) — the repo's FIRST two-player game.
`games/connect4/env.py` + `teachers/connect4_lookahead.py`. The opponent is
config-driven and entirely internal to the game (`extra.opponent`: random /
greedy / lookahead, resolved in `__init__` via `heuristic_opponents()`) --
`run_suite`/`EpisodeRunner`/`BatchedEpisodeRunner` are untouched, still
constructing `game_cls(game_cfg)` with no opponent argument; after the
agent's move, `step()` advances the opponent internally so the agent's next
observation already shows both moves. First-mover alternates by seed parity
(`seed % 2 == 0` -> agent first; otherwise the opponent's opening move is
played inside `reset()` itself, deterministic per seed) so eval is balanced.
The executor's initial fully-deterministic build exposed a second instance
of the ALE seed-invariance failure: with both sides deterministic, 40
seeded episodes collapsed to exactly TWO distinct games and the outcome was
100% determined by first-mover parity (teacher won every agent-first seed
via a 2-ply bottom-row double-threat fork neither 1-ply bot can see, and
lost every opponent-first seed to the mirror-image fork — win rate exactly
0.5, zero draws, zero episode diversity). Review fix (same doctrine as
Atari's `noop_start_max`): `extra.opponent_epsilon` (default 0.25) makes
the opponent's NEUTRAL moves seeded-random while win-now/block-now always
fire. Measured after the fix: 189 distinct games over 300 seeds, teacher
win 0.587 / draw 0.187 vs `greedy` — a real skill edge instead of a parity
coin-flip, still below the 1-ply mirror ceiling that a deeper opponent
would expose. Ladder spread: ~0.98 vs `random`, 0.587 vs `greedy` (see
`tests/test_connect4_teacher.py` and `configs/games/connect4.yaml` for the
sweep numbers). Two-player lesson for future games: deterministic-vs-
deterministic play NEEDS an explicit episode-diversity mechanism.

## Vision: reading frames instead of RAM text (plan 011)

Every shipped Atari game renders decoded RAM as text — no vision needed.
The VL stack from plan 011 is still in-tree for a future pixel-observation
game: `VLAgent` (`agents/vl_agent.py`) + `VLTransformersBackend`
(`inference/transformers_vl_be.py`, factory name `"transformers-vl"`),
wired via `slm-rl rollout --agent vl`. Observations carry a PNG frame in
`Observation.metadata["frame_png"]`; recorded `prompt_messages` replace
the image with an `image_ref` placeholder (note + frame sha1), never raw
pixels or a PIL object.

Vision records must not poison text SFT: any pixel game sets
`Game.export_exempt() -> True` (default False on the ABC).
`datagen/sft_export.py` skips export-exempt episodes with a loud, logged
count. Recipe for a new vision game: pixel-observation `Game` subclass +
`export_exempt() -> True` + an agent that builds `{"type": "image", ...}`
content parts (redacted before recording). Prefer reusing `VLAgent`, which
imports `llm_agent.action_instruction`/`parse_action` rather than forking
them. Scope today is rollout + live viewer only — no training, no
eval-gate, no teacher, no 8GB claim (CUDA + `cuda` extra;
llama.cpp/MLX multimodal is out of scope).
