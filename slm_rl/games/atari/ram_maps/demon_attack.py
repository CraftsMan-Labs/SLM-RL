"""Demon Attack RAM offsets (ALE/DemonAttack-v5, obs_type="ram").

Hypotheses are from the AtariARI benchmark (Anand et al. 2019). Each
constant below was verified empirically 2026-07 against `ale-py` 0.12.0 /
`gymnasium` 1.3.0 with `frameskip=4, repeat_action_probability=0.0` (plan
026 Phase B probe) before being trusted:

- PLAYER_X (22): confirmed responsive to LEFT/RIGHT. Observed range ~1-247
  with wraparound (the ship wraps the screen). Holding LEFT from spawn
  (~200) settles at the left-edge floor ~17; holding RIGHT from that edge
  walks the byte downward then wraps to the high side (~241) and continues
  (frameskip=1 probe: 241, 225, 209, ... stepping by ~16). A raw
  player_x vs enemy_x comparison with wrap handling (if |dx|>128 flip
  direction) is enough for a working tracker (see teachers/demon_attack.py).
  Do NOT assume the byte is a simple non-wrapping 0-160 screen x.
- ENEMY_X1/X2/X3 (17/18/19): confirmed -- each sweeps 0-249 over a
  LEFTFIRE episode (78/62/71 unique values in 200 decisions).
- ENEMY_Y1/Y2/Y3 (69/70/71): confirmed -- each varies in-flight (e.g.
  enemy_y1 99-149, enemy_y2 74-134, enemy_y3 48-90 over the same probe).
- MISSILE_Y (21): confirmed -- under pure NOOP the byte stays at exactly 3
  (idle sentinel); after FIRE it sweeps 12, 24, 36, ... up to ~156 then
  returns to 3. decode() reports missile_in_flight as missile_y != 3.
- LEVEL (62): confirmed incrementing across a longer episode (0 -> 1 when
  the wave advanced around decision ~600 under LEFTFIRE). Flavor only.
- NUM_LIVES (114): NOT verified for this ALE build -- over a LEFTFIRE
  episode ram[114] stayed stuck at 3 while info["lives"] moved 4 -> 5 ->
  ... (ALE life counter). Dropped from decode(); use info["lives"].

Core variables (STOP-if-unverifiable): player_x, enemy positions,
missile_y -- verified above. num_lives AtariARI hypothesis dropped.
"""

from __future__ import annotations

PLAYER_X = 22
ENEMY_X1 = 17
ENEMY_X2 = 18
ENEMY_X3 = 19
ENEMY_Y1 = 69
ENEMY_Y2 = 70
ENEMY_Y3 = 71
MISSILE_Y = 21
LEVEL = 62

MISSILE_IDLE = 3


def decode(ram) -> dict:
    """RAM -> named variables. Only offsets verified empirically (see module
    docstring) are included. num_lives (AtariARI 114) did not verify and is
    omitted."""
    missile_y = int(ram[MISSILE_Y])
    return {
        "player_x": int(ram[PLAYER_X]),
        "enemy_x": [
            int(ram[ENEMY_X1]),
            int(ram[ENEMY_X2]),
            int(ram[ENEMY_X3]),
        ],
        "enemy_y": [
            int(ram[ENEMY_Y1]),
            int(ram[ENEMY_Y2]),
            int(ram[ENEMY_Y3]),
        ],
        "missile_y": missile_y,
        "missile_in_flight": missile_y != MISSILE_IDLE,
        "level": int(ram[LEVEL]),
    }
