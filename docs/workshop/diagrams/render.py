#!/usr/bin/env python3
"""Render workshop flow diagrams to deterministic dark-theme SVGs.

Re-run from the repo:

    python docs/workshop/diagrams/render.py

Writes ``docs/workshop/assets/diagrams/*.svg``. Source graphs live beside
this file as ``*.mmd`` for review; the SVG layout is authored here so Colab
never has to execute Mermaid.
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams"

BG = "#171512"
DEEP = "#0D0C0B"
TEXT = "#F2E7D5"
MUTED = "#CFC2AF"
ACCENT = "#D89B55"
LINE = "#60584E"
SOFT = "#A98D6B"

DIAGRAMS: dict[str, dict] = {}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _node(x: float, y: float, w: float, h: float, label: str, *, kind: str = "box") -> str:
    cx, cy = x + w / 2, y + h / 2
    lines = label.split("\n")
    text_y = cy - (len(lines) - 1) * 8
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else 16
        tspans.append(
            f'<tspan x="{cx:.1f}" dy="{dy}">{_esc(line)}</tspan>'
        )
    text = (
        f'<text x="{cx:.1f}" y="{text_y:.1f}" text-anchor="middle" '
        f'fill="{TEXT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="13" font-weight="600">{"".join(tspans)}</text>'
    )
    if kind == "decision":
        pts = f"{cx:.1f},{y:.1f} {x + w:.1f},{cy:.1f} {cx:.1f},{y + h:.1f} {x:.1f},{cy:.1f}"
        shape = (
            f'<polygon points="{pts}" fill="{DEEP}" stroke="{ACCENT}" '
            f'stroke-width="1.6"/>'
        )
    else:
        shape = (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="8" fill="{DEEP}" stroke="{LINE}" stroke-width="1.4"/>'
        )
    return shape + text


def _arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> str:
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    parts = [
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#arrow)"/>'
    ]
    if label:
        parts.append(
            f'<text x="{mid_x:.1f}" y="{mid_y - 8:.1f}" text-anchor="middle" '
            f'fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
            f'font-size="11">{_esc(label)}</text>'
        )
    return "".join(parts)


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">\n'
        f"<title>{_esc(title)}</title>\n"
        f'<rect width="100%" height="100%" fill="{BG}"/>\n'
        "<defs>\n"
        '  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">\n'
        f'    <path d="M 0 0 L 10 5 L 0 10 z" fill="{ACCENT}"/>\n'
        "  </marker>\n"
        "</defs>\n"
        f"{body}\n"
        "</svg>\n"
    )


def _row(labels: list[tuple[str, str]], *, y: float = 70, w: float = 150, h: float = 56) -> str:
    n = len(labels)
    width = 64 + n * w + (n - 1) * 48
    parts = []
    xs = []
    for i, (kind, label) in enumerate(labels):
        x = 32 + i * (w + 48)
        xs.append(x)
        parts.append(_node(x, y, w, h, label, kind=kind))
        if i:
            parts.append(_arrow(xs[i - 1] + w, y + h / 2, x, y + h / 2))
    return width, "".join(parts)


def evolve_loop() -> str:
    labels = [
        ("box", "ROLLOUT"),
        ("box", "DATASET"),
        ("box", "TRAIN"),
        ("box", "EVAL"),
        ("box", "GATE"),
        ("box", "champion"),
    ]
    width, row = _row(labels, y=64, w=128, h=58)
    extra = _arrow(32 + 5 * 176 + 64, 150, 32 + 64, 150, "")
    # wrap-around under the row
    x0 = 32 + 128 / 2
    x1 = 32 + 5 * (128 + 48) + 128 / 2
    wrap = (
        f'<path d="M {x1:.1f} 122 C {x1:.1f} 168, {x0:.1f} 168, {x0:.1f} 122" '
        f'fill="none" stroke="{ACCENT}" stroke-width="1.6" '
        f'marker-end="url(#arrow)"/>'
        f'<text x="{(x0 + x1) / 2:.1f}" y="186" text-anchor="middle" fill="{SOFT}" '
        f'font-family="Inter, Helvetica, sans-serif" font-size="12">'
        f"promote or keep the old champion</text>"
    )
    return _svg(max(int(width), 980), 210, row + wrap + extra, "Evolve loop")


def hardware_tier() -> str:
    parts = [
        _node(300, 24, 200, 48, "detect_host"),
        _node(300, 112, 200, 64, "VRAM ≥ 20 GB?", kind="decision"),
        _node(36, 220, 200, 52, "cuda-24gb\nGemma E2B"),
        _node(300, 220, 200, 64, "VRAM ≥ 6 GB?", kind="decision"),
        _node(300, 328, 200, 52, "cuda-8-16gb\nLFM 1.2B"),
        _node(564, 328, 200, 52, "any-8gb\nLFM 350M"),
        _arrow(400, 72, 400, 112),
        _arrow(300, 144, 136, 220, "yes"),
        _arrow(400, 176, 400, 220, "no"),
        _arrow(400, 284, 400, 328, "yes"),
        _arrow(500, 252, 664, 328, "no"),
    ]
    return _svg(800, 410, "".join(parts), "Hardware tier selection")


def games_pipeline() -> str:
    _, row = _row(
        [
            ("box", "ALE RAM"),
            ("box", "ram_map"),
            ("box", "Observation.text"),
            ("box", "legal_actions"),
            ("box", "game.step"),
        ],
        w=150,
    )
    return _svg(980, 200, row, "Atari RAM becomes a text menu")


def config_merge() -> str:
    _, row = _row(
        [
            ("box", "default.yaml"),
            ("box", "games/GAME.yaml"),
            ("box", "overrides / knobs"),
            ("box", "RunConfig"),
        ],
        w=170,
    )
    return _svg(900, 200, row, "Config merge: last writer wins")


def parse_action() -> str:
    parts = [
        _node(24, 70, 150, 56, "obs.text + menu"),
        _node(214, 70, 140, 56, "backend.generate"),
        _node(394, 70, 140, 56, "raw_completion"),
        _node(574, 62, 150, 72, "parse_action", kind="decision"),
        _node(764, 24, 140, 48, "ok"),
        _node(764, 90, 140, 48, "one retry"),
        _node(764, 156, 140, 48, "fallback_random"),
        _arrow(174, 98, 214, 98),
        _arrow(354, 98, 394, 98),
        _arrow(534, 98, 574, 98),
        _arrow(724, 80, 764, 48, "ACTION / index"),
        _arrow(724, 98, 764, 114, "fail"),
        _arrow(834, 138, 834, 156),
    ]
    return _svg(930, 230, "".join(parts), "Generate, parse, or fall back")


def rollout_dataset() -> str:
    parts = [
        _node(24, 36, 140, 52, "LLMAgent"),
        _node(24, 120, 140, 52, "Game"),
        _node(220, 78, 160, 52, "EpisodeRunner"),
        _node(440, 78, 170, 52, "DoomLoopMonitor"),
        _node(670, 78, 150, 52, "JSONL"),
        _node(870, 78, 150, 52, "parquet"),
        _arrow(164, 62, 220, 94),
        _arrow(164, 146, 220, 114),
        _arrow(380, 104, 440, 104),
        _arrow(610, 104, 670, 104),
        _arrow(820, 104, 870, 104),
    ]
    return _svg(1048, 210, "".join(parts), "Rollout to dataset")


def dqn_hybrid() -> str:
    parts = [
        _node(40, 28, 160, 48, "raw SLM"),
        _node(40, 110, 160, 48, "DQN teacher"),
        _node(40, 192, 160, 48, "homework demos"),
        _node(40, 274, 160, 48, "SFT copies habits"),
        _node(280, 274, 160, 48, "SLM plays alone"),
        _node(520, 274, 180, 48, "frozen exam"),
        _node(520, 160, 180, 64, "beats champion?", kind="decision"),
        _node(520, 48, 180, 48, "promote"),
        _arrow(120, 76, 120, 110),
        _arrow(120, 158, 120, 192),
        _arrow(120, 240, 120, 274),
        _arrow(200, 298, 280, 298),
        _arrow(440, 298, 520, 298),
        _arrow(610, 274, 610, 224),
        _arrow(610, 160, 610, 96, "yes"),
        f'<path d="M 520 192 C 280 192, 200 80, 200 76" fill="none" '
        f'stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#arrow)"/>',
        f'<text x="300" y="150" fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="12">no — keep the old line</text>',
    ]
    return _svg(740, 360, "".join(parts), "DQN teacher hybrid loop")


def packs() -> str:
    _, row = _row(
        [
            ("box", "bake_pack"),
            ("box", "MANIFEST + dqn.pt"),
            ("box", "push_pack"),
            ("box", "HF dataset"),
            ("box", "resolve_pack"),
        ],
        w=155,
    )
    return _svg(1000, 200, row, "Bake, push, and resolve a pack")


def train_strategies() -> str:
    parts = [
        _node(300, 24, 160, 48, "parquet"),
        f'<rect x="24" y="100" width="300" height="210" rx="12" fill="none" '
        f'stroke="{LINE}" stroke-dasharray="5 4"/>',
        f'<text x="40" y="122" fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="12">reject_sft — copy the best</text>',
        _node(48, 140, 250, 44, "keep top quantile"),
        _node(48, 200, 250, 44, "prompt / completion"),
        _node(48, 260, 250, 44, "SFTTrainer"),
        f'<rect x="436" y="100" width="300" height="210" rx="12" fill="none" '
        f'stroke="{LINE}" stroke-dasharray="5 4"/>',
        f'<text x="452" y="122" fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="12">GRPO — play and score</text>',
        _node(460, 140, 250, 44, "one situation"),
        _node(460, 200, 250, 44, "sample K answers"),
        _node(460, 260, 250, 44, "score vs the game"),
        _node(300, 348, 160, 48, "adapter/"),
        _arrow(380, 72, 173, 140),
        _arrow(380, 72, 585, 140),
        _arrow(173, 184, 173, 200),
        _arrow(173, 244, 173, 260),
        _arrow(585, 184, 585, 200),
        _arrow(585, 244, 585, 260),
        _arrow(173, 304, 360, 348),
        _arrow(585, 304, 400, 348),
    ]
    return _svg(760, 420, "".join(parts), "reject_sft vs GRPO")


def eval_gate() -> str:
    parts = [
        _node(260, 20, 180, 48, "eval_suite seeds"),
        _node(80, 120, 160, 48, "base model"),
        _node(460, 120, 160, 48, "adapter"),
        _node(260, 220, 180, 64, "EvalGate.decide", kind="decision"),
        _node(80, 330, 160, 48, "promote"),
        _node(460, 330, 160, 48, "reject"),
        _arrow(350, 68, 160, 120),
        _arrow(350, 68, 540, 120),
        _arrow(160, 168, 320, 220),
        _arrow(540, 168, 380, 220),
        _arrow(300, 284, 160, 330, "margin + hygiene"),
        _arrow(400, 284, 540, 330, "else"),
    ]
    return _svg(700, 410, "".join(parts), "Eval gate promote or reject")


def theater() -> str:
    parts = [
        _node(40, 80, 170, 56, "seeds ≥ 20 000"),
        _node(270, 24, 160, 48, "base plays"),
        _node(270, 144, 160, 48, "champion plays"),
        _node(500, 80, 160, 56, "theater JSONL"),
        _node(730, 80, 190, 56, "hstack replay"),
        _arrow(210, 94, 270, 48),
        _arrow(210, 122, 270, 168),
        _arrow(430, 48, 500, 94),
        _arrow(430, 168, 500, 122),
        _arrow(660, 108, 730, 108),
    ]
    return _svg(950, 220, "".join(parts), "Base vs champion theater")


def publish() -> str:
    parts = [
        _node(40, 80, 140, 52, "run_dir"),
        _node(240, 80, 180, 52, "publish_experiment"),
        _node(500, 24, 220, 52, "username/slm-rl-colab"),
        _node(500, 136, 250, 52, "username/slm-rl-colab-data"),
        _arrow(180, 106, 240, 106),
        _arrow(420, 96, 500, 50),
        _arrow(420, 116, 500, 162),
    ]
    return _svg(780, 220, "".join(parts), "Publish an experiment")


def game_abc() -> str:
    _, row = _row(
        [
            ("box", "Game ABC"),
            ("box", "@register_game"),
            ("box", "EpisodeRunner"),
            ("box", "export / train / eval"),
        ],
        w=175,
    )
    return _svg(920, 200, row, "Custom game plugin path")


def dqn_loop() -> str:
    """Beginner play loop. The choice is a current guess, never an optimum."""
    parts = [
        f'<rect x="28" y="28" width="748" height="228" rx="12" fill="none" '
        f'stroke="{LINE}" stroke-dasharray="5 4"/>',
        f'<text x="44" y="50" fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="12">agent looks and guesses</text>',
        f'<rect x="800" y="56" width="196" height="200" rx="12" fill="none" '
        f'stroke="{LINE}" stroke-dasharray="5 4"/>',
        f'<text x="816" y="78" fill="{SOFT}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="12">world replies</text>',
        _node(48, 68, 140, 52, "state"),
        _node(220, 68, 176, 52, "online network"),
        _node(428, 68, 140, 52, "Q-values"),
        _node(600, 68, 168, 52, "ε-greedy choice"),
        _node(598, 168, 172, 64, "chosen action\ncurrent best guess"),
        _node(816, 160, 164, 72, "environment"),
        _node(48, 284, 220, 56, "reward + next state"),
        _arrow(188, 94, 220, 94),
        _arrow(396, 94, 428, 94),
        _arrow(568, 94, 600, 94),
        _arrow(684, 120, 684, 168),
        _arrow(770, 200, 816, 196),
        (
            f'<path d="M 898 232 L 898 312 L 268 312" fill="none" '
            f'stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#arrow)"/>'
        ),
        (
            f'<path d="M 48 312 L 16 312 L 16 94 L 48 94" fill="none" '
            f'stroke="{ACCENT}" stroke-width="1.6" marker-end="url(#arrow)"/>'
        ),
        f'<text x="510" y="356" text-anchor="middle" fill="{MUTED}" '
        f'font-family="Inter, Helvetica, sans-serif" font-size="13">'
        f"Look, score every move, then pick. Sometimes try something new.</text>",
    ]
    return _svg(
        1020,
        376,
        "".join(parts),
        "DQN play loop: state, online network, Q-values, "
        "epsilon-greedy choice, chosen action, environment, reward and next state",
    )


def dqn_q_values() -> str:
    parts = [
        _node(40, 80, 150, 56, "Mario frame"),
        _node(250, 52, 200, 40, "RIGHT → 0.4"),
        _node(250, 116, 200, 40, "RIGHT+A → 0.8"),
        _node(520, 80, 170, 56, "current guess:\nRIGHT+A"),
        _arrow(190, 108, 250, 72),
        _arrow(190, 108, 250, 136),
        _arrow(450, 136, 520, 108),
    ]
    return _svg(730, 210, "".join(parts), "Score Mario actions and choose the current best guess")


def dqn_bellman() -> str:
    parts = [
        _node(40, 80, 180, 56, "reward now"),
        _node(280, 80, 40, 56, "+"),
        _node(360, 80, 200, 56, "γ × best next Q"),
        _node(620, 80, 160, 56, "target"),
        _arrow(220, 108, 280, 108),
        _arrow(320, 108, 360, 108),
        _arrow(560, 108, 620, 108),
        f'<text x="360" y="176" fill="{MUTED}" font-family="Inter, Helvetica, sans-serif" '
        f'font-size="13">γ = 0.99 in slm_rl.teachers.dqn — almost all of tomorrow counts.</text>',
    ]
    return _svg(820, 210, "".join(parts), "Bellman target without the math wall")


def dqn_replay() -> str:
    parts = [
        _node(40, 70, 180, 70, "play one step\n(s, a, r, s')"),
        _node(300, 70, 180, 70, "store in replay\nbuffer"),
        _node(560, 70, 180, 70, "sample a batch\nand learn"),
        _arrow(220, 105, 300, 105),
        _arrow(480, 105, 560, 105),
    ]
    return _svg(780, 200, "".join(parts), "Experience replay")


def dqn_target() -> str:
    parts = [
        _node(80, 40, 220, 64, "online net\nlearns every few steps"),
        _node(80, 160, 220, 64, "target net\ncopied every 1000 updates"),
        _node(420, 100, 220, 64, "stable target\nfor the Bellman backup"),
        _arrow(300, 72, 420, 120),
        _arrow(300, 192, 420, 148),
    ]
    return _svg(700, 270, "".join(parts), "Online network vs target network")


def dqn_encoders() -> str:
    parts = [
        _node(40, 40, 280, 80, "Mario demo\npixels → CNN"),
        _node(40, 160, 280, 80, "SLM-RL teacher\nRAM vector → MLP"),
        _node(400, 100, 260, 80, "same DQN loop\nreplay · target · ε-greedy"),
        _arrow(320, 80, 400, 126),
        _arrow(320, 200, 400, 154),
    ]
    return _svg(700, 280, "".join(parts), "Same DQN, different eyes")


def dqn_math() -> str:
    parts = [
        _node(24, 28, 150, 56, "reward r"),
        _node(194, 28, 40, 56, "+"),
        _node(254, 28, 200, 56, "γ × best next Q"),
        _node(474, 28, 40, 56, "="),
        _node(534, 28, 150, 56, "target"),
        _node(24, 140, 200, 56, "predicted Q(s, a)"),
        _node(254, 140, 200, 56, "vs"),
        _node(484, 140, 200, 56, "target value"),
        _arrow(224, 168, 254, 168),
        _arrow(454, 168, 484, 168),
        f'<text x="352" y="230" text-anchor="middle" fill="{MUTED}" '
        f'font-family="Inter, Helvetica, sans-serif" font-size="13">'
        f"Mario: a coin now plus almost all of the best jump after it. γ = 0.99.</text>",
    ]
    return _svg(720, 260, "".join(parts), "Bellman target in one line")


def trace_to_pair() -> str:
    parts = [
        _node(24, 70, 150, 64, "DQN plays\nstate → action"),
        _node(214, 70, 150, 64, "JSONL\ntrace row"),
        _node(404, 70, 160, 64, "select\nclean + top"),
        _node(604, 70, 170, 64, "SFT pair\nACTION: RIGHT"),
        _arrow(174, 102, 214, 102),
        _arrow(364, 102, 404, 102),
        _arrow(564, 102, 604, 102),
        f'<text x="400" y="176" text-anchor="middle" fill="{MUTED}" '
        f'font-family="Inter, Helvetica, sans-serif" font-size="13">'
        f"Machine-made homework. Not an eval label.</text>",
    ]
    return _svg(800, 210, "".join(parts), "Teacher trace to SFT pair")


RENDERERS = {
    "evolve-loop": evolve_loop,
    "hardware-tier": hardware_tier,
    "games-pipeline": games_pipeline,
    "config-merge": config_merge,
    "parse-action": parse_action,
    "rollout-dataset": rollout_dataset,
    "dqn-hybrid": dqn_hybrid,
    "packs": packs,
    "train-strategies": train_strategies,
    "eval-gate": eval_gate,
    "theater": theater,
    "publish": publish,
    "game-abc": game_abc,
    "dqn-loop": dqn_loop,
    "dqn-q-values": dqn_q_values,
    "dqn-bellman": dqn_bellman,
    "dqn-replay": dqn_replay,
    "dqn-target": dqn_target,
    "dqn-encoders": dqn_encoders,
    "dqn-math": dqn_math,
    "trace-to-pair": trace_to_pair,
}


def render_all(dest: Path | None = None) -> list[Path]:
    dest = dest or OUT
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fn in RENDERERS.items():
        path = dest / f"{name}.svg"
        path.write_text(fn(), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    written = render_all()
    print(f"wrote {len(written)} diagrams under {OUT}")


if __name__ == "__main__":
    main()
