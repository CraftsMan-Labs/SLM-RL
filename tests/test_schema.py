from slm_rl.datagen.schema import SCHEMA_VERSION, RolloutRecord


def make_record(**kw) -> RolloutRecord:
    base = dict(
        run_id="r1",
        generation=3,
        game="boxing",
        episode_id="ep-001",
        step_idx=4,
        seed=42,
        model_id="LiquidAI/LFM2.5-350M",
        adapter_ref="gen_003/adapter",
        opponent_id=None,
        prompt_messages=[{"role": "user", "content": "board...\n1) r0c0"}],
        completion="I think... ACTION: 1",
        parsed_action="r0c0",
        legal_actions=["r0c0", "r0c1"],
        parse_status="ok",
        reward=0.0,
        shaped_reward=0.05,
        cum_reward=0.05,
        terminated=False,
        truncated=False,
        outcome=None,
        state_hash="abc123",
        monitor_flags={"action_repeat": 1},
        timestamp="2026-07-09T00:00:00Z",
    )
    base.update(kw)
    return RolloutRecord(**base)


def test_json_round_trip():
    rec = make_record()
    restored = RolloutRecord.from_json(rec.to_json())
    assert restored == rec


def test_schema_version_stamped():
    assert make_record().schema_version == SCHEMA_VERSION


def test_round_trip_preserves_unicode_and_nesting():
    rec = make_record(
        completion="ACTION: 1 ✓",
        monitor_flags={"interventions": [{"kind": "reflect", "penalty": -0.05}]},
    )
    restored = RolloutRecord.from_json(rec.to_json())
    assert restored.completion == "ACTION: 1 ✓"
    assert restored.monitor_flags["interventions"][0]["kind"] == "reflect"
