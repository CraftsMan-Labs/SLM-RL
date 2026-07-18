"""RolloutWriter truncates so restarted generations do not inflate GRPO data."""

from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter


def _rec(episode_id: str, step_idx: int = 0) -> RolloutRecord:
    return RolloutRecord(
        run_id="r",
        generation=1,
        game="boxing",
        episode_id=episode_id,
        step_idx=step_idx,
        seed=0,
        model_id="m",
        adapter_ref=None,
        opponent_id=None,
        prompt_messages=[{"role": "user", "content": "x"}],
        completion="ACTION: UP",
        parsed_action="UP",
        legal_actions=["UP"],
        parse_status="ok",
        reward=0.0,
        shaped_reward=0.0,
        cum_reward=0.0,
        terminated=False,
        truncated=False,
        outcome=None,
        state_hash="h",
    )


def test_rollout_writer_truncates_existing_file(tmp_path):
    path = tmp_path / "boxing.jsonl"
    with RolloutWriter(path) as w:
        w.write(_rec("old-ep"))
    assert path.read_text().count("\n") == 1

    with RolloutWriter(path) as w:
        w.write(_rec("new-ep"))
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "new-ep" in lines[0]
    assert "old-ep" not in lines[0]
