"""select_episodes selection/quota logic + export skipping — the non-trivial
reject_sft data path, tested without any model."""

import json

from slm_rl.config.schema import TrainConfig
from slm_rl.datagen.sft_export import export_sft_dataset, select_episodes


def rec(ep, step, action, cum_reward, outcome=None, parse_status="ok", dirty=False):
    return {
        "episode_id": ep, "step_idx": step, "parsed_action": action,
        "cum_reward": cum_reward, "outcome": outcome, "parse_status": parse_status,
        "monitor_flags": {"intervention": {"kind": "reflect"}} if dirty else {},
        "prompt_messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "legal_actions": [action],
    }


def write_jsonl(tmp_path, records):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def test_wins_are_selected(tmp_path):
    records = [
        rec("win", 0, "A", 1.0, outcome="win"),
        rec("loss", 0, "B", 0.0, outcome="loss"),
    ]
    cfg = TrainConfig(selection_quantile=0.0)  # only wins qualify by quantile
    selected = select_episodes(write_jsonl(tmp_path, records), cfg)
    ids = {s[0]["episode_id"] for s in selected}
    assert "win" in ids and "loss" not in ids


def test_top_quantile_selected_when_no_wins(tmp_path):
    records = [rec(f"e{i}", 0, "A", float(i), outcome="loss") for i in range(10)]
    cfg = TrainConfig(selection_quantile=0.25)
    selected = select_episodes(write_jsonl(tmp_path, records), cfg)
    returns = sorted(s[-1]["cum_reward"] for s in selected)
    assert min(returns) >= 7  # top 25% of 0..9


def test_monitor_flagged_dropped(tmp_path):
    records = [
        rec("clean", 0, "A", 1.0, outcome="win"),
        rec("dirty", 0, "A", 1.0, outcome="win", dirty=True),
    ]
    cfg = TrainConfig(exclude_monitor_flagged=True)
    ids = {s[0]["episode_id"] for s in select_episodes(write_jsonl(tmp_path, records), cfg)}
    assert ids == {"clean"}


def test_diversity_quota_caps_duplicate_sequences(tmp_path):
    # 5 winning episodes, identical action sequence ("A",) -> quota keeps 2
    records = [rec(f"e{i}", 0, "A", 1.0, outcome="win") for i in range(5)]
    cfg = TrainConfig(max_duplicate_action_sequences=2)
    selected = select_episodes(write_jsonl(tmp_path, records), cfg)
    assert len(selected) == 2


def test_export_skips_fallback_random_and_writes_canonical(tmp_path):
    records = [
        rec("win", 0, "GOOD", 1.0, outcome="win"),
        rec("win", 1, "RANDO", 1.0, outcome="win", parse_status="fallback_random"),
    ]
    cfg = TrainConfig()
    out = tmp_path / "sft.jsonl"
    n = export_sft_dataset(write_jsonl(tmp_path, records), out, cfg)
    assert n == 1  # fallback_random step skipped
    row = json.loads(out.read_text().strip())
    assert row["completion"][0]["content"] == "ACTION: GOOD"
    assert len(row["prompt"]) == 2  # clean system+user only
