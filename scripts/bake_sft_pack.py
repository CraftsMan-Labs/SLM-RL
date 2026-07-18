#!/usr/bin/env python3
"""Bake a top-quantile teacher pack, export ~N SFT pairs, optional HF push.

Uses an existing DQN when --dqn-decisions 0 (finds runs/teachers/dqn-<game>.pt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_rl.config.schema import TrainConfig
from slm_rl.datagen.sft_export import export_sft_dataset, group_episodes
from slm_rl.packs import bake_pack, packs_root, read_manifest, update_manifest


def _trim_to_steps(jsonl: Path, max_steps: int) -> dict:
    """Keep highest-return episodes until ~max_steps records remain."""
    episodes = list(group_episodes(jsonl).values())

    def final_return(steps: list[dict]) -> float:
        return float(steps[-1].get("cum_reward", 0.0))

    episodes.sort(key=final_return, reverse=True)
    kept: list[list[dict]] = []
    n = 0
    for steps in episodes:
        if n >= max_steps:
            break
        kept.append(steps)
        n += len(steps)
    tmp = jsonl.with_suffix(".jsonl.trim")
    with tmp.open("w", encoding="utf-8") as fh:
        for steps in kept:
            for rec in steps:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(jsonl)
    return {"n_episodes": len(kept), "n_steps": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="./runs")
    ap.add_argument("--game", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--dqn-decisions", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--selection-quantile", type=float, default=0.25)
    ap.add_argument("--target-pairs", type=int, default=5000)
    ap.add_argument("--push", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from slm_rl.hf_auth import hf_token

    token = hf_token()
    home = Path(args.home)
    pack = bake_pack(
        args.game,
        packs_root(home),
        episodes=args.episodes,
        dqn_decisions=args.dqn_decisions,
        device=args.device,
        seed=args.seed,
        push=None,  # push after SFT export
        token=token,
        selection_quantile=args.selection_quantile,
    )
    jsonl = pack / "rollouts" / f"{args.game}.jsonl"
    # Cap rollout volume so SFT export lands near target_pairs.
    trim = _trim_to_steps(jsonl, max_steps=args.target_pairs * 2)
    print(f"[bake-sft] trimmed rollouts → {trim}", flush=True)

    sft_path = pack / "dataset" / "sft.jsonl"
    cfg = TrainConfig(selection_quantile=1.0)  # already filtered
    n_pairs = export_sft_dataset(jsonl, sft_path, cfg)
    if n_pairs > args.target_pairs:
        lines = sft_path.read_text(encoding="utf-8").splitlines()
        sft_path.write_text("\n".join(lines[: args.target_pairs]) + "\n", encoding="utf-8")
        n_pairs = args.target_pairs
        print(f"[bake-sft] capped sft.jsonl at {n_pairs} pairs", flush=True)
    else:
        print(f"[bake-sft] wrote {n_pairs} SFT pairs → {sft_path}", flush=True)

    man = read_manifest(pack)
    update_manifest(
        pack,
        n_sft_pairs=n_pairs,
        n_episodes=trim["n_episodes"],
        n_steps=trim["n_steps"],
        target_pairs=args.target_pairs,
    )
    print(f"[bake-sft] manifest keys={sorted(read_manifest(pack))}", flush=True)

    if args.push:
        from slm_rl.packs import push_pack

        url = push_pack(pack, args.push, token=token)
        update_manifest(pack, hf_repo=args.push, hf_commit_url=url)
        print(f"[bake-sft] pushed https://huggingface.co/datasets/{args.push}", flush=True)
        print(f"[bake-sft] commit {url}", flush=True)
    print(json.dumps({"pack": str(pack), "n_sft_pairs": n_pairs, "manifest": man}, indent=2))


if __name__ == "__main__":
    main()
