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
