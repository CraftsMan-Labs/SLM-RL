"""EvalGate: checkpoint gating — a new generation is promoted only if it
beats the champion on the frozen suite without regressing on doom-loop,
invalid-action, or entropy criteria (D4, training level). Rollback is simply
never moving the champion pointer."""

from __future__ import annotations

from typing import Any

from slm_rl.config.schema import GateConfig

# Keys run_suite always emits; workshop stubs must match or gate.decide dies.
_GATE_RATE_KEYS = ("invalid_rate", "intervention_rate")


def stub_eval_metrics(primary: float, *, note: str = "") -> dict[str, Any]:
    """Skipped baseline / imported-SFT champion metrics — gate-safe shape."""
    out: dict[str, Any] = {
        "primary": float(primary),
        "invalid_rate": 0.0,
        "intervention_rate": 0.0,
        "skipped": True,
    }
    if note:
        out["note"] = note
    return out


def coerce_gate_metrics(metrics: dict) -> dict[str, Any]:
    """Guard: fill rate keys so thin stubs cannot KeyError decide()."""
    out = dict(metrics)
    for key in _GATE_RATE_KEYS:
        out.setdefault(key, 0.0)
    return out


class EvalGate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def decide(self, champion_metrics: dict, candidate_metrics: dict) -> tuple[bool, str]:
        """Both dicts come from eval.suites.run_suite (must contain `primary`,
        `invalid_rate`, `intervention_rate`, optional `mean_entropy`).
        Returns (promote, reason)."""
        # Guard + belt: coerce missing rates; .get keeps decide crash-proof.
        champion_metrics = coerce_gate_metrics(champion_metrics)
        candidate_metrics = coerce_gate_metrics(candidate_metrics)

        champ_primary = champion_metrics["primary"]
        cand_primary = candidate_metrics["primary"]
        if cand_primary < champ_primary + self.cfg.min_improvement:
            return False, (
                f"primary {cand_primary:.4f} < champion {champ_primary:.4f} "
                f"+ margin {self.cfg.min_improvement}"
            )

        cand_invalid = float(candidate_metrics.get("invalid_rate") or 0.0)
        if cand_invalid > self.cfg.max_invalid_rate:
            return False, (
                f"invalid_rate {cand_invalid:.4f} > "
                f"max {self.cfg.max_invalid_rate}"
            )

        champ_iv = float(champion_metrics.get("intervention_rate") or 0.0)
        cand_iv = float(candidate_metrics.get("intervention_rate") or 0.0)
        # allow a small absolute floor so a champion at exactly 0 doesn't
        # make any nonzero candidate rate an automatic failure
        ceiling = max(champ_iv * self.cfg.max_intervention_rate_ratio, 0.01)
        if cand_iv > ceiling:
            return False, (
                f"intervention_rate {cand_iv:.4f} > ceiling {ceiling:.4f} "
                f"(champion {champ_iv:.4f} x {self.cfg.max_intervention_rate_ratio})"
            )

        entropy = candidate_metrics.get("mean_entropy")
        if entropy is not None and entropy < self.cfg.min_mean_entropy:
            return False, (
                f"mean_entropy {entropy:.4f} < floor {self.cfg.min_mean_entropy}"
            )

        return True, (
            f"primary {champ_primary:.4f} -> {cand_primary:.4f}, "
            f"invalid_rate {cand_invalid:.4f}, "
            f"intervention_rate {cand_iv:.4f}"
        )
