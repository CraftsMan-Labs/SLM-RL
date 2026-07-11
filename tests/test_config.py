from pathlib import Path

from slm_rl.config.loader import (
    CONFIG_DIR,
    deep_merge,
    load_game_config,
    load_run_config,
    load_tiers,
)
from slm_rl.config.schema import RunConfig


def test_deep_merge_nested_override():
    base = {"a": 1, "train": {"lr": 1e-4, "rank": 16}}
    out = deep_merge(base, {"train": {"lr": 5e-5}})
    assert out["train"] == {"lr": 5e-5, "rank": 16}
    assert out["a"] == 1
    assert base["train"]["lr"] == 1e-4  # no mutation


def test_default_yaml_loads_into_run_config():
    cfg = load_run_config()
    assert isinstance(cfg, RunConfig)
    assert cfg.game == "mastermind"
    assert cfg.max_context_tokens == 2048  # 8GB budget rule
    assert cfg.train.strategy == "reject_sft"


def test_cli_style_overrides_win():
    cfg = load_run_config(game="dominion", overrides={"generations": 9, "model": None})
    assert cfg.game == "dominion"
    assert cfg.generations == 9
    assert cfg.model is None  # None overrides are dropped, default kept


def test_rollout_batch_size_override_lands_in_train_config():
    cfg = load_run_config(game="mastermind", overrides={"train": {"rollout_batch_size": 4}})
    assert cfg.train.rollout_batch_size == 4


def test_rollout_batch_size_omitted_keeps_default():
    cfg = load_run_config(game="mastermind")
    assert cfg.train.rollout_batch_size == 1  # current behavior unchanged


def test_selection_quantile_override_lands_in_train_config():
    cfg = load_run_config(game="mastermind", overrides={"train": {"selection_quantile": 0.5}})
    assert cfg.train.selection_quantile == 0.5


def test_selection_quantile_omitted_keeps_default():
    cfg = load_run_config(game="mastermind")
    assert cfg.train.selection_quantile == 0.25  # current behavior unchanged


def test_every_shipped_game_config_parses():
    for path in (CONFIG_DIR / "games").glob("*.yaml"):
        cfg = load_game_config(path.stem)
        assert cfg.name == path.stem
        assert cfg.max_turns > 0


def test_game_config_missing_file_uses_defaults(tmp_path: Path):
    cfg = load_game_config("mastermind", config_dir=tmp_path)
    assert cfg.name == "mastermind"
    assert cfg.monitor.action_repeat_threshold == 3


def test_tier_table_parses_and_ends_with_universal_floor():
    tiers = load_tiers()
    assert len(tiers) >= 2
    floor = tiers[-1]
    assert floor.name == "any-8gb"
    # the floor must match ANY host: no conditions
    assert floor.os is None and floor.min_ram_gb is None and floor.min_cuda_vram_gb is None
    assert floor.train == "reject_sft"  # full loop works on 8GB
