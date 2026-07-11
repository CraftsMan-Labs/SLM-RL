"""select_episodes selection/quota logic + export skipping — the non-trivial
reject_sft data path, tested without any model."""

import json

from slm_rl.config.schema import TrainConfig
from slm_rl.datagen.sft_export import export_sft_dataset, select_episodes


def rec(ep, step, action, cum_reward, outcome=None, parse_status="ok", dirty=False, model_id="", completion=""):
    return {
        "episode_id": ep, "step_idx": step, "parsed_action": action,
        "cum_reward": cum_reward, "outcome": outcome, "parse_status": parse_status,
        "monitor_flags": {"intervention": {"kind": "reflect"}} if dirty else {},
        "prompt_messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "legal_actions": [action],
        "model_id": model_id,
        "completion": completion,
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


def test_win_turn_cap_drops_long_wins_keeps_short(tmp_path):
    # 8-step win exceeds cap=5 and is dropped; 4-step win survives. The long
    # win deliberately carries the HIGHEST return in the batch (per-turn
    # format reward means long wins outscore short ones in real runs), so it
    # sits in the top-quantile `top` set — the cap must exclude it from the
    # final union anyway, not merely from the wins list.
    records = [
        rec("long", 0, "A", 0.1, outcome=None),
        rec("long", 1, "A", 0.3, outcome=None),
        rec("long", 2, "A", 0.5, outcome=None),
        rec("long", 3, "A", 0.7, outcome=None),
        rec("long", 4, "A", 0.9, outcome=None),
        rec("long", 5, "A", 1.1, outcome=None),
        rec("long", 6, "A", 1.3, outcome=None),
        rec("long", 7, "A", 1.5, outcome="win"),
        rec("short", 0, "B", 0.2, outcome=None),
        rec("short", 1, "B", 0.4, outcome=None),
        rec("short", 2, "B", 0.6, outcome=None),
        rec("short", 3, "B", 0.8, outcome="win"),
    ]
    cfg = TrainConfig(win_turn_cap=5)
    ids = {s[0]["episode_id"] for s in select_episodes(write_jsonl(tmp_path, records), cfg)}
    assert "short" in ids
    assert "long" not in ids


def test_win_turn_cap_keeps_shortest_when_all_wins_too_long(tmp_path):
    # both wins exceed cap=2; the shortest (3 steps) is kept rather than
    # losing all winning demonstrations. The longer win has the higher
    # return (so it sits in the top-quantile set) and must still be dropped.
    records = [
        rec("longer", 0, "A", 0.3, outcome=None),
        rec("longer", 1, "A", 0.6, outcome=None),
        rec("longer", 2, "A", 0.9, outcome=None),
        rec("longer", 3, "A", 1.2, outcome="win"),
        rec("shortest", 0, "B", 0.3, outcome=None),
        rec("shortest", 1, "B", 0.6, outcome=None),
        rec("shortest", 2, "B", 0.9, outcome="win"),
    ]
    cfg = TrainConfig(win_turn_cap=2)
    ids = {s[0]["episode_id"] for s in select_episodes(write_jsonl(tmp_path, records), cfg)}
    assert ids == {"shortest"}


def test_sft_win_final_dup_repeats_only_final_pair(tmp_path):
    # distinct actions per step so duplication can be attributed to the
    # correct (final) pair rather than merely counting total rows.
    records = [
        rec("win", 0, "AAAA", 0.5, outcome=None),
        rec("win", 1, "BBBB", 0.6, outcome=None),
        rec("win", 2, "CCCC", 1.0, outcome="win"),
    ]
    cfg = TrainConfig(sft_win_final_dup=3)
    out = tmp_path / "sft.jsonl"
    n = export_sft_dataset(write_jsonl(tmp_path, records), out, cfg)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert n == 5  # step0 x1 + step1 x1 + step2(final) x3
    contents = [r["completion"][0]["content"] for r in rows]
    assert contents.count("ACTION: AAAA") == 1
    assert contents.count("ACTION: BBBB") == 1
    assert contents.count("ACTION: CCCC") == 3


def test_sft_win_final_dup_duplicates_teacher_rationale_verbatim(tmp_path):
    # the duplicated final pair must apply to the row's FINAL content: for a
    # teacher episode that means the verbatim rationale completion, not a
    # rebuilt "ACTION: X" string.
    records = [
        rec(
            "teacher-win", 0, "RRRR", 1.0, outcome="win",
            model_id="teacher:x", completion="Because the feedback narrows it down.\nACTION: RRRR",
        ),
    ]
    cfg = TrainConfig(sft_win_final_dup=2)
    out = tmp_path / "sft.jsonl"
    n = export_sft_dataset(write_jsonl(tmp_path, records), out, cfg)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert n == 2
    for row in rows:
        assert row["completion"][0]["content"] == "Because the feedback narrows it down.\nACTION: RRRR"


def test_export_keeps_teacher_rationale_but_rebuilds_llm_completion(tmp_path):
    # teacher records (model_id starts with "teacher:") keep their raw
    # completion verbatim (process supervision, plan 002); model-generated
    # records are still rebuilt from parsed_action since they can contain
    # retry junk.
    records = [
        rec(
            "teacher-ep", 0, "RRRR", 1.0, outcome="win",
            model_id="teacher:x", completion="Because...\nACTION: RRRR",
        ),
        rec(
            "llm-ep", 0, "GGBB", 1.0, outcome="win",
            model_id="llm", completion="uh let me think... ACTION: WRONG\nACTION: GGBB",
        ),
    ]
    cfg = TrainConfig()
    out = tmp_path / "sft.jsonl"
    export_sft_dataset(write_jsonl(tmp_path, records), out, cfg)
    rows = [json.loads(line) for line in out.read_text().splitlines()]

    teacher_row = next(r for r in rows if r["completion"][0]["content"].startswith("Because"))
    assert teacher_row["completion"][0]["content"] == "Because...\nACTION: RRRR"

    llm_row = next(r for r in rows if not r["completion"][0]["content"].startswith("Because"))
    assert llm_row["completion"][0]["content"] == "ACTION: GGBB"
