"""Boxing RAM offsets (ALE/Boxing-v5, obs_type="ram").

Hypotheses are from the AtariARI benchmark (Anand et al. 2019). Each
constant below was verified empirically 2026-07 against `ale-py` 0.12.0 /
`gymnasium` 1.3.0 with `frameskip=4, repeat_action_probability=0.0` (plan
026 Phase B probe) before being trusted:

- PLAYER_X (32): confirmed -- holding "RIGHT" for 20 decisions increases
  the byte (30 -> 70, saturating near observed max 77 under longer holds);
  holding "LEFT" from mid-ring decreases it (70 -> 30, saturating at the
  observed left-edge floor of 30). Matches the intuitive reading.
- PLAYER_Y (34): confirmed -- holding "DOWN" increases the byte (4 -> 39
  over 20 decisions, up to ~87 under longer holds); holding "UP" decreases
  it (39 -> 3, saturating at the observed top-edge floor of 3). y=0 is top
  of the ring, larger y is toward the bottom.
- ENEMY_X (33) / ENEMY_Y (35): confirmed -- both vary under NOOP while the
  CPU opponent moves (enemy_x observed 50-109, enemy_y 3-87 across a
  mixed-action episode). Not player-controlled; exposed for the chase
  teacher / renderer flavor.
- PLAYER_SCORE (18) / ENEMY_SCORE (19): confirmed against reward sign --
  over an 800-decision mixed episode, every reward=+1.0 step coincided with
  player_score incrementing by 1, and every reward<0 step coincided with
  enemy_score incrementing (by 1 or 2; ALE awards -2 on some punches).
  Authoritative score remains reward-derived (plan 008 discipline); these
  bytes are renderer flavor only.
- CLOCK (17): confirmed varying over the episode (observed 6-89, 54 unique
  values in one run) -- flavor only, not used by decode().

Core variables (STOP-if-unverifiable): player_x, player_y, enemy_x,
enemy_y -- all four verified above.
"""

from __future__ import annotations

PLAYER_X = 32
PLAYER_Y = 34
ENEMY_X = 33
ENEMY_Y = 35
PLAYER_SCORE = 18
ENEMY_SCORE = 19


def decode(ram) -> dict:
    """RAM -> named variables. Only offsets verified empirically (see module
    docstring) are included."""
    return {
        "player_x": int(ram[PLAYER_X]),
        "player_y": int(ram[PLAYER_Y]),
        "enemy_x": int(ram[ENEMY_X]),
        "enemy_y": int(ram[ENEMY_Y]),
        "player_score_ram": int(ram[PLAYER_SCORE]),
        "enemy_score_ram": int(ram[ENEMY_SCORE]),
    }
