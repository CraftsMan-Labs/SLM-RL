"""The reward-code tab's starting template, served verbatim by
`GET /api/reward-template`. Pure data (a string) -- no imports beyond
`__future__`, so this module is trivially stdlib-only.
"""

from __future__ import annotations

TEMPLATE: str = '''\
def shape_reward(ctx: dict) -> float:
    """Reshape this decision's reward. Called once per decision after the
    built-in reward is computed -- wrapping it, not replacing the env.

    ctx always has: default_reward, score, turn, terminated, truncated.
    Per-game ALE fields are documented in docs/ARCHITECTURE.md
    ("Workshop playground").

    This hook cannot affect monitor penalties or the eval gate's primary
    metric (by design -- see that same docs section).
    """
    return ctx["default_reward"]
'''
