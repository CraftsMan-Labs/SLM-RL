#!/usr/bin/env python3
"""Bake N teacher episodes, split train/holdout, SFT on train, live-eval champion.

Example:
  .venv/bin/python scripts/bake_split_sft_eval.py \\
    --game boxing --episodes 30 --train-n 20 --holdout-n 10 \\
    --run-id boxing-sft-30split --device mps --eval-limit 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _episode_score(steps: list[dict]) -> float:
    last = steps[-1]
    outcome = last.get("outcome")
    if isinstance(outcome, str) and outcome.startswith("score:"):
        try:
            return float(outcome.split(":", 1)[1])
        except ValueError:
            pass
    return float(last.get("cum_reward", 0.0))


def _write_episode_jsonl(episodes: list[list[dict]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for steps in episodes:
            for rec in steps:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def _build_pack(
    pack_dir: Path,
    *,
    game: str,
    episodes: list[list[dict]],
    dqn_src: Path | None,
    role: str,
) -> dict:
    from slm_rl.config.schema import TrainConfig
    from slm_rl.datagen.sft_export import export_sft_dataset
    from slm_rl.packs import update_manifest, write_manifest

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    rollouts = pack_dir / "rollouts"
    dataset = pack_dir / "dataset"
    rollouts.mkdir(parents=True)
    dataset.mkdir(parents=True)

    jsonl = rollouts / f"{game}.jsonl"
    n_steps = _write_episode_jsonl(episodes, jsonl)
    scores = {
        steps[0]["episode_id"]: {
            "outcome": steps[-1].get("outcome"),
            "cum_reward": steps[-1].get("cum_reward"),
            "steps": len(steps),
            "score": _episode_score(steps),
        }
        for steps in episodes
    }
    cfg = TrainConfig(selection_quantile=1.0, exclude_monitor_flagged=True)
    n_pairs = export_sft_dataset(jsonl, dataset / "sft.jsonl", cfg)
    has_dqn = False
    if dqn_src is not None and dqn_src.is_file():
        shutil.copy2(dqn_src, pack_dir / "dqn.pt")
        has_dqn = True
    write_manifest(
        pack_dir,
        game=game,
        n_episodes=len(episodes),
        has_dqn=has_dqn,
        n_episodes_raw=len(episodes),
        selection_quantile=1.0,
    )
    update_manifest(
        pack_dir,
        n_sft_pairs=n_pairs,
        n_steps=n_steps,
        role=role,
        episode_ids=[steps[0]["episode_id"] for steps in episodes],
        episode_scores=scores,
    )
    return {
        "pack": str(pack_dir),
        "n_episodes": len(episodes),
        "n_steps": n_steps,
        "n_sft_pairs": n_pairs,
        "scores": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="./runs")
    ap.add_argument("--game", default="boxing")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument("--holdout-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--run-id", default="boxing-sft-30split")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--backend", default="transformers")
    ap.add_argument(
        "--eval-limit",
        type=int,
        default=20,
        help="Live eval episodes (frozen suite is 100; default 20 for workshop smoke)",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--push", default=None, help="Optional HF dataset repo to push train pack")
    args = ap.parse_args()

    if args.train_n + args.holdout_n != args.episodes:
        raise SystemExit(
            f"train-n ({args.train_n}) + holdout-n ({args.holdout_n}) "
            f"must equal episodes ({args.episodes})"
        )

    from slm_rl.datagen.sft_export import group_episodes
    from slm_rl.packs import bake_pack, cache_slug, packs_root, push_pack
    from slm_rl.teachers.dqn_checkpoint import find_dqn_checkpoint

    home = Path(args.home)
    from slm_rl.hf_auth import hf_token

    token = hf_token()
    out_root = packs_root(home)
    print(f"[pipeline] bake {args.episodes} {args.game} demos (keep all)", flush=True)
    pack_all = bake_pack(
        args.game,
        out_root,
        episodes=args.episodes,
        dqn_decisions=0,
        device=args.device,
        seed=args.seed,
        push=None,
        token=token,
        selection_quantile=1.0,
    )
    # bake_pack writes to out_root/<game>/ — move aside so we can build split packs
    raw_all = out_root / f"{args.game}-raw-{args.episodes}"
    if raw_all.exists():
        shutil.rmtree(raw_all)
    shutil.move(str(pack_all), str(raw_all))
    jsonl = raw_all / "rollouts" / f"{args.game}.jsonl"
    grouped = group_episodes(jsonl)
    # Stable order by first-seen episode_id (seed order from bake)
    ordered = sorted(grouped.values(), key=lambda steps: steps[0].get("seed", 0))
    if len(ordered) < args.episodes:
        print(
            f"[pipeline] warning: only {len(ordered)} episodes baked "
            f"(requested {args.episodes})",
            flush=True,
        )
    train_eps = ordered[: args.train_n]
    holdout_eps = ordered[args.train_n : args.train_n + args.holdout_n]
    dqn_src = find_dqn_checkpoint(args.game, home)
    train_pack = out_root / f"{args.game}-train{args.train_n}"
    holdout_pack = out_root / f"{args.game}-holdout{args.holdout_n}"
    train_info = _build_pack(
        train_pack, game=args.game, episodes=train_eps, dqn_src=dqn_src, role="train",
    )
    holdout_info = _build_pack(
        holdout_pack, game=args.game, episodes=holdout_eps, dqn_src=dqn_src, role="holdout",
    )
    print("[pipeline] train pack:", json.dumps(train_info, indent=2), flush=True)
    print("[pipeline] holdout pack:", json.dumps(holdout_info, indent=2), flush=True)

    # Cache slugs so evolve --dataset-url resolves locally without Hub download.
    train_repo = f"BLANK/slm-rl-{args.game}-train{args.train_n}"
    holdout_repo = f"BLANK/slm-rl-{args.game}-holdout{args.holdout_n}"
    for repo, pack in ((train_repo, train_pack), (holdout_repo, holdout_pack)):
        link = out_root / cache_slug(repo)
        if link.exists() or link.is_symlink():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(pack.resolve())
        print(f"[pipeline] linked {repo} -> {pack}", flush=True)

    summary = {
        "raw_pack": str(raw_all),
        "train": train_info,
        "holdout": holdout_info,
        "train_repo": train_repo,
        "holdout_repo": holdout_repo,
        "run_id": args.run_id,
    }
    (home / f"{args.run_id}-split-summary.json").parent.mkdir(parents=True, exist_ok=True)
    summary_path = home / f"{args.run_id}-split-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.push:
        url = push_pack(train_pack, args.push, token=token)
        print(f"[pipeline] pushed train pack → {args.push} ({url})", flush=True)

    if args.skip_train:
        print("[pipeline] skip-train set; done after bake/split", flush=True)
        print(json.dumps(summary, indent=2))
        return

    # Cap live eval via a full configs/ copy with eval_episodes overridden
    # (frozen suite prefix; pairing across gens preserved).
    import yaml
    from slm_rl.config.loader import CONFIG_DIR, load_yaml

    run_cfg_dir = home / args.run_id / "_pipeline_config"
    if run_cfg_dir.exists():
        shutil.rmtree(run_cfg_dir)
    shutil.copytree(CONFIG_DIR, run_cfg_dir)
    game_yaml = run_cfg_dir / "games" / f"{args.game}.yaml"
    base_game = load_yaml(game_yaml)
    base_game["eval_episodes"] = int(args.eval_limit)
    game_yaml.write_text(yaml.safe_dump(base_game, sort_keys=False), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "slm_rl.cli", "evolve",
        "--game", args.game,
        "--generations", "1",
        "--run-id", args.run_id,
        "--dataset-url", train_repo,
        "--dqn-url", train_repo,
        "--train-strategy", "reject_sft",
        "--model", args.model,
        "--backend", args.backend,
        "--selection-quantile", "1.0",
        "--skip-baseline",
        "--warm-start",
        "--config-dir", str(run_cfg_dir),
    ]
    if args.skip_eval:
        cmd.append("--skip-eval")
    print("[pipeline] evolve:", " ".join(cmd), flush=True)
    print(f"[pipeline] live eval capped at {args.eval_limit} frozen seeds", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

    # Holdout imitation note (offline): count ACTION labels only.
    holdout_pairs = holdout_info["n_sft_pairs"]
    print(
        f"[pipeline] holdout kept offline for imitation checks "
        f"({holdout_info['n_episodes']} eps, {holdout_pairs} pairs) — "
        "live gate uses frozen eval seeds, not these demos.",
        flush=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
