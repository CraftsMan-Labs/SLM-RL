"""EvalGate: checkpoint gating — a new generation is promoted only if it
beats the champion on the frozen suite without regressing on doom-loop,
invalid-action, or entropy criteria (D4, training level). Rollback is simply
never moving the champion pointer."""

from __future__ import annotations

from slm_rl.config.schema import GateConfig


class EvalGate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def decide(self, champion_metrics: dict, candidate_metrics: dict) -> tuple[bool, str]:
        """Both dicts come from eval.suites.run_suite (must contain `primary`,
        `invalid_rate`, `intervention_rate`, optional `mean_entropy`).
        Returns (promote, reason)."""
        champ_primary = champion_metrics["primary"]
        cand_primary = candidate_metrics["primary"]
        if cand_primary < champ_primary + self.cfg.min_improvement:
            return False, (
                f"primary {cand_primary:.4f} < champion {champ_primary:.4f} "
                f"+ margin {self.cfg.min_improvement}"
            )

        if candidate_metrics["invalid_rate"] > self.cfg.max_invalid_rate:
            return False, (
                f"invalid_rate {candidate_metrics['invalid_rate']:.4f} > "
                f"max {self.cfg.max_invalid_rate}"
            )

        champ_iv = champion_metrics["intervention_rate"]
        cand_iv = candidate_metrics["intervention_rate"]
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
            f"invalid_rate {candidate_metrics['invalid_rate']:.4f}, "
            f"intervention_rate {cand_iv:.4f}"
        )
