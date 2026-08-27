#!/usr/bin/env python3
"""Copy workshop SVGs into the sibling presentation repo.

    python docs/workshop/sync_presentation_assets.py

Looks for ``../SLM-RL-Presentation/public/assets/workshop``. No-ops with a
message if that tree is missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "assets" / "diagrams"
DEST_CANDIDATES = [
    HERE.parents[2] / "SLM-RL-Presentation" / "public" / "assets" / "workshop",
    Path("/home/rishub/Desktop/projects/enterprises/craftsmanlabs/SLM-RL-Presentation/public/assets/workshop"),
]


def main() -> None:
    dest = next((p for p in DEST_CANDIDATES if p.parent.is_dir()), None)
    if dest is None:
        print("presentation repo not found; skip asset sync")
        return
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(SRC.glob("*.svg")):
        shutil.copy2(src, dest / src.name)
        copied += 1
    extra = HERE / "assets" / "mario"
    if extra.is_dir():
        mario_dest = dest / "mario"
        mario_dest.mkdir(parents=True, exist_ok=True)
        for src in extra.iterdir():
            if src.is_file():
                shutil.copy2(src, mario_dest / src.name)
                copied += 1
    print(f"copied {copied} files → {dest}")


if __name__ == "__main__":
    main()
