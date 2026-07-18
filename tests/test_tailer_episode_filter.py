"""Frame-replay path must not json.loads every line of a multi-GB JSONL."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from slm_rl.webui.tailer import iter_run_records


def test_iter_run_records_scopes_generation_and_skips_other_episodes(tmp_path: Path):
    g0 = tmp_path / "generations" / "gen_000" / "rollouts" / "a.jsonl"
    g1 = tmp_path / "generations" / "gen_001" / "rollouts" / "b.jsonl"
    g0.parent.mkdir(parents=True)
    g1.parent.mkdir(parents=True)

    junk = {"episode_id": "other", "generation": 0, "parsed_action": "NOOP", "seed": 1}
    hit = {
        "episode_id": "want",
        "generation": 1,
        "parsed_action": "FIRE",
        "seed": 7,
        "game": "demon-attack",
    }
    g0.write_text(json.dumps(junk) + "\n" * 1, encoding="utf-8")
    # Many non-matching lines in gen_001 before the hit
    with g1.open("w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({**junk, "generation": 1, "step_idx": i}) + "\n")
        f.write(json.dumps({**hit, "step_idx": 0}) + "\n")
        f.write(json.dumps({**hit, "step_idx": 1, "terminated": True}) + "\n")

    stop = threading.Event()
    stop.set()
    rows = list(
        iter_run_records(
            tmp_path, stop=stop, generation=1, episode_id="want",
        )
    )
    assert [r["step_idx"] for r in rows] == [0, 1]
    assert all(r["episode_id"] == "want" for r in rows)
    # gen_000 file was never needed; scoped walk only opens gen_001
    assert not any(r.get("generation") == 0 for r in rows)
