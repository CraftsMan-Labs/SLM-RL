#!/usr/bin/env python3
"""Copy Vue-deck stills and clips into this repo for Colab.

    python docs/workshop/import_deck_assets.py

Looks for ``../SLM-RL-Presentation/public/assets``. No-ops with a message if
that tree is missing. Workshop SVGs are *not* imported — they are authored
here and synced the other way.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "assets" / "deck"
SRC_CANDIDATES = [
    HERE.parents[2] / "SLM-RL-Presentation" / "public" / "assets",
    Path("/home/rishub/Desktop/projects/enterprises/craftsmanlabs/SLM-RL-Presentation/public/assets"),
]

DECK_FILES = (
    "HeroVisual.png",
    "SLM.png",
    "Why-Games.png",
    "SpaceInv.png",
    "Observer-Learn-Play-React.png",
    "GRPO.png",
    "dqn-game-master.png",
    "dqn-today-plus-tomorrow.png",
    "dqn-flashcards-target.png",
    "dqn-tourist-then-regular.png",
    "enemy_destroyed.png",
    "better_score.png",
    "life_lost.png",
    "doom_loop.png",
    "meme_never_move.png",
    "doom.png",
    "meet-the-teacher.mp4",
    "base-0.mp4",
    "RL-trained.mp4",
    "FreeWay_trained.mp4",
)


def main() -> None:
    src = next((p for p in SRC_CANDIDATES if p.is_dir()), None)
    if src is None:
        print("presentation repo not found; skip deck import")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: list[str] = []
    for name in DECK_FILES:
        path = src / name
        if not path.is_file():
            missing.append(name)
            continue
        shutil.copy2(path, DEST / name)
        copied += 1
    print(f"copied {copied} files → {DEST}")
    if missing:
        print("missing in presentation public/assets:", ", ".join(missing))


if __name__ == "__main__":
    main()
