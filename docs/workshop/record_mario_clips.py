#!/usr/bin/env python3
"""Record World 1-1 Mario DQN clips for the deck and Colab.

    python docs/workshop/record_mario_clips.py

Downloads the pinned public checkpoints, plays untrained / mid / pretrained
policies, and writes MP4s plus ``clip_manifest.json`` under
``docs/workshop/assets/mario``. Falls back without raising if the emulator
is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mario_lab import FALLBACK_DIR, record_mario_clips  # noqa: E402


def main() -> int:
    result = record_mario_clips(FALLBACK_DIR, workdir=FALLBACK_DIR / "checkpoints")
    print(result.get("mode"), result.get("reason"))
    for row in result.get("metrics") or []:
        print(
            f"{row['stage']}: x_pos={row['x_pos']} "
            f"reward={row['mean_ep_reward']} clip={row['clip']}"
        )
    if result.get("clips"):
        for stage, path in result["clips"].items():
            print(f"  {stage} → {path}")
    return 0 if result.get("mode") == "live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
