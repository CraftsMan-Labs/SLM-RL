"""Space Invaders RAM offsets (ALE/SpaceInvaders-v5, obs_type="ram").

Hypotheses are from the AtariARI benchmark (Anand et al. 2019). Each
constant below was verified empirically 2026-07 against `ale-py` 0.12.0 /
`gymnasium` 1.3.0 with `frameskip=4, repeat_action_probability=0.0` (see
plan 008 probe transcript) before being trusted:

- PLAYER_X (28): confirmed monotonic — holding RIGHT increases the value,
  holding LEFT decreases it back, over a 15-decision hold in each direction
  post-warmup. Note: the cannon does not respond to movement for the first
  ~32-34 decisions of an episode (classic SI level-intro lockout) — this is
  real ALE behavior, not an offset bug.
- INVADERS_LEFT_COUNT (17): confirmed — decrements by exactly 1 on each
  invader kill, in lockstep with the ALE `reward` signal (5, 10, 15, 20,
  25, 30 point steps observed for successive rows).
- NUM_LIVES (73): confirmed — byte-identical to `info["lives"]` across 1000
  sampled steps (0 mismatches). `info["lives"]` remains the authoritative
  source (per CODING_GUIDELINE/plan 008); this constant is offered only for
  `vector_obs()` consumers that want it in the raw RAM vector.
- ENEMIES_X (26): confirmed — increases by 1 roughly every 8 decisions
  under NOOP, consistent with the invader block marching sideways.

NOT verified — degrade the observation, don't guess (dropped from
`decode()` and the renderer):
- ENEMIES_Y (hypothesis: 24, "invader block lowest row y"): over a
  1200-step NOOP probe the byte did change (82 -> 80 -> 81 -> 83) but by
  1-3 units at a time and non-monotonically — not the clean row-descent
  step function AtariARI describes. Too ambiguous to label confidently as
  a row-y coordinate; excluded.
- MISSILES_Y (hypothesis: 9, "player missile y"): the hypothesis label
  itself verified (238 immediately post-FIRE, decreasing monotonically
  frame-by-frame back to a resting value of 13 as the shot resolves) but is
  intentionally left out of `decode()`'s guaranteed set below because reward
  is the authoritative score signal per plan 008 — it is exposed anyway
  since the renderer wants "missile in flight" state. See MISSILE_Y.
"""

from __future__ import annotations

PLAYER_X = 28
INVADERS_LEFT_COUNT = 17
NUM_LIVES = 73
ENEMIES_X = 26
MISSILE_Y = 9

# Resting value observed for MISSILE_Y when no shot is in flight.
MISSILE_Y_RESTING = 13


def decode(ram) -> dict:
    """RAM -> named variables. Only offsets verified empirically (see module
    docstring) are included; ENEMIES_Y is deliberately omitted."""
    missile_y = int(ram[MISSILE_Y])
    return {
        "player_x": int(ram[PLAYER_X]),
        "invaders_left": int(ram[INVADERS_LEFT_COUNT]),
        "enemies_x": int(ram[ENEMIES_X]),
        "missile_in_flight": missile_y != MISSILE_Y_RESTING,
        "missile_y": missile_y,
    }
