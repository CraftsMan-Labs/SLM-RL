"""The reward-code tab's starting template, served verbatim by
`GET /api/reward-template`. Pure data (a string) -- no imports beyond
`__future__`, so this module is trivially stdlib-only.
"""

from __future__ import annotations

TEMPLATE: str = '''\
def shape_reward(ctx: dict) -> float:
    """Reshape this decision's reward. Called once per decision, after the
    built-in reward has already been computed -- you are wrapping it, not
    replacing the environment.

    ctx fields:
      raw_points        ALE points scored this decision (before scaling)
      default_reward     what the built-in formula produced:
                          raw_points / score_scale, plus life_loss_penalty
                          if a life was lost this decision
      score              cumulative raw score so far this episode
      lives_lost         bool: True if a life was lost this decision
      lives               lives remaining (None if the game has no concept
                          of lives)
      turn                decision index this episode (0-based)
      terminated          bool: episode ended (loss of all lives, etc.)
      truncated           bool: episode cut short (max_turns, monitor)
      vector_obs          list[float], 128 RAM bytes / 255.0

    Two things this hook CANNOT affect (by design, see docs/ARCHITECTURE.md
    "Workshop playground"):
      1. Monitor penalties (retry/fallback/truncate) -- those are applied
         later, in the rollout runner, over the value this function returns.
      2. The eval gate's primary metric -- that's mean RAW score parsed from
         `outcome` ("score:<n>"), which this hook never touches. Reshaping
         training-time reward changes what reject_sft selects as "good"
         demonstrations; it cannot make the gate itself easier to pass.
    """
    # --- default: reproduces the built-in formula exactly -----------------
    return ctx["default_reward"]

    # --- example 1: small survival bonus per turn --------------------------
    # Encourages staying alive longer, independent of scoring. Try this if
    # episodes are ending too early relative to their score.
    #
    # return ctx["default_reward"] + 0.01

    # --- example 2: harsher life-loss penalty -------------------------------
    # The default life_loss_penalty is a game knob (see the knob panel); this
    # shows how to go further, e.g. doubling it only late in the episode when
    # a life is especially costly to lose.
    #
    # reward = ctx["default_reward"]
    # if ctx["lives_lost"] and ctx["turn"] > 100:
    #     reward -= 0.5
    # return reward
'''
