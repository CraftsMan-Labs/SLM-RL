"""Freeway RAM offsets (ALE/Freeway-v5, obs_type="ram").

Hypotheses are from the AtariARI benchmark (Anand et al. 2019). Each
constant below was verified empirically 2026-07 against `ale-py` 0.12.0 /
`gymnasium` 1.3.0 with `frameskip=4, repeat_action_probability=0.0` (plan
016 probe transcript) before being trusted:

- PLAYER_Y (14): confirmed monotonic and its direction determined -- ALE's
  own "UP" action meaning increases the byte (6 -> 126 over a 30-decision
  hold, chicken advancing toward the goal), "DOWN" decreases it back (122 ->
  6). Score/reset behavior: the byte resets to exactly 6 immediately after a
  successful crossing (reward=+1 observed at score-byte increments). Range
  observed under a full hold-UP episode: 6 (start) to 126 (goal, just before
  the crossing-complete reset).
- SCORE (103): confirmed -- increments by exactly 1 on every reward=+1 step
  (crossings), in lockstep with PLAYER_Y resetting to 6. Authoritative score
  for gameplay purposes is still `info`/reward-derived (plan 008
  discipline); this byte is exposed only as renderer flavor.
- CAR_X (108-117, one byte per lane): confirmed -- over a 600-decision
  hold-UP probe, ALL 10 bytes changed value (verified: every lane index 0-9
  appeared in the changed-lanes set), and at any single decision the 10
  values are pairwise distinct (verified: 10/10 distinct values sampled at
  decision 5) -- consistent with 10 independently-moving cars, one per lane.

NOT verified / out of scope for `decode()`'s "current lane" claim:
- Which CAR_X byte is "the chicken's current lane" (plan 016 asks for
  "car in the chicken's lane and adjacent lanes"): a naive
  `lane = (player_y - 6) // 12` estimate was tried and correlated against
  ~290 observed hit-like events (player_y decreasing while holding UP,
  i.e. knocked back by a car) over a 2048-decision episode -- the supposed
  "current lane" car_x value at the moment of each hit did NOT cluster
  around a consistent collision-zone x (values were scattered across the
  full 0-159 range for every lane tested). Either the linear lane-index
  formula is wrong, or collision depends on sprite geometry not capturable
  as "car_x is near threshold T" -- too ambiguous to ship a per-lane
  mapping confidently. All 10 raw car_x values ARE exposed by decode() (the
  bytes themselves are verified), but decode() does NOT claim to identify
  which one is "the current lane" -- that judgment is left to the renderer
  (which shows all 10) rather than baked into a wrong index. The teacher
  (plan 016) therefore ships as pure hold-UP, not a car-avoiding variant
  (see teachers/freeway.py for the measured comparison this decision is
  based on).

Core variable per plan 016 (STOP-if-unverifiable list): player_y --
verified above.
"""

from __future__ import annotations

PLAYER_Y = 14
SCORE = 103
CAR_X_START = 108
NUM_LANES = 10

# Score resets PLAYER_Y to exactly this value on a successful crossing.
PLAYER_Y_START = 6


def decode(ram) -> dict:
    """RAM -> named variables. Only offsets verified empirically (see module
    docstring) are included. car_x is a 10-element list (lane 0..9); no
    claim is made about which index is the chicken's "current" lane (see
    docstring NOT-verified section)."""
    return {
        "player_y": int(ram[PLAYER_Y]),
        "car_x": [int(ram[CAR_X_START + i]) for i in range(NUM_LANES)],
    }
