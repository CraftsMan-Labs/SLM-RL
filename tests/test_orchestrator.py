from pathlib import Path

from slm_rl.orchestrator.paths import RunPaths
from slm_rl.orchestrator.registry import ModelRegistry


def test_run_paths_layout(tmp_path: Path):
    paths = RunPaths(tmp_path, "run1")
    assert paths.adapter(7) == tmp_path / "run1" / "generations" / "gen_007" / "adapter"
    assert paths.registry.name == "registry.json"


def test_registry_promote_and_rollback(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    assert reg.champion == 0  # gen 0 = base model

    reg.promote(1, "win_rate +0.05")
    assert reg.champion == 1

    reg.reject(2, "entropy below floor")
    reg.reject(3, "no improvement")
    assert reg.champion == 1  # rollback = pointer never moved
    assert reg.consecutive_failures == 2  # triggers auto-remediation

    reg.promote(4, "win_rate +0.03")
    assert reg.consecutive_failures == 0

    # persistence
    reloaded = ModelRegistry(tmp_path / "registry.json")
    assert reloaded.champion == 4


def test_imports_work_without_ml_extras():
    """The 8GB rule: core package must import with no torch/trl/vllm installed.
    (If torch happens to be installed this still passes — the real guarantee
    is that these imports never require it.)"""
    import slm_rl
    import slm_rl.agents
    import slm_rl.config
    import slm_rl.datagen
    import slm_rl.eval
    import slm_rl.games
    import slm_rl.inference
    import slm_rl.orchestrator
    import slm_rl.platform
    import slm_rl.rollout
    import slm_rl.training

    assert slm_rl.__version__
