"""Workshop bake packs: local cache under runs/packs + public HF resolve.

Layout:
  <pack>/MANIFEST.json
  <pack>/dqn.pt            # Atari optional
  <pack>/rollouts/*.jsonl  # teacher demos (same shape as evolve warm-start)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ATARI_GAMES = frozenset({
    "space-invaders", "freeway", "boxing", "demon-attack",
})
DQN_GAMES = ATARI_GAMES


def is_atari(game: str) -> bool:
    return game in ATARI_GAMES or any(game.startswith(g) for g in ATARI_GAMES)


def normalize_repo_id(url: str) -> str:
    """Accept `org/name` or https://huggingface.co/[datasets/]org/name[...]."""
    s = url.strip().rstrip("/")
    if not s:
        raise ValueError("empty Hugging Face repo URL")
    m = re.search(r"huggingface\.co/(?:datasets/|models/)?([^/\s]+/[^/\s#?]+)", s)
    if m:
        return m.group(1)
    if s.startswith("http://") or s.startswith("https://"):
        raise ValueError(f"not a Hugging Face repo URL: {url!r}")
    if "/" not in s:
        raise ValueError(f"expected org/name repo id, got {url!r}")
    return s


def resolve_adapter(url: str, home: Path | str, game: str) -> Path:
    """Download a published PEFT LoRA (`adapter/`) from an HF *model* repo.

    Layout expected (publish_experiment): `adapter/adapter_config.json` +
    `adapter/adapter_model.safetensors`. Cached under
    `<home>/packs/<slug>__adapter/adapter/`.
    """
    from huggingface_hub import snapshot_download

    from slm_rl.hf_auth import hf_token

    root = packs_root(home)
    repo_id = normalize_repo_id(url)
    local = root / (cache_slug(repo_id) + "__adapter")
    adapter_dir = local / "adapter"
    if (adapter_dir / "adapter_config.json").is_file() and (
        (adapter_dir / "adapter_model.safetensors").is_file()
        or (adapter_dir / "adapter_model.bin").is_file()
    ):
        return adapter_dir

    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)
    print(f"[packs] downloading adapter from model {repo_id} (game={game}) → {local}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local),
        allow_patterns=["adapter/**", "adapter_config.json", "adapter_model.*"],
        token=hf_token(),
    )
    # Some uploads nest under adapter/; others put PEFT files at repo root.
    if not (adapter_dir / "adapter_config.json").is_file():
        root_cfg = local / "adapter_config.json"
        if root_cfg.is_file():
            adapter_dir.mkdir(parents=True, exist_ok=True)
            for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md"):
                src = local / name
                if src.is_file():
                    shutil.copy2(src, adapter_dir / name)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(
            f"no PEFT adapter/ found in model repo {repo_id!r} "
            "(expected adapter/adapter_config.json)"
        )
    return adapter_dir


def import_adapter_as_champion(
    run_dir: Path,
    adapter_src: Path,
    *,
    model_id: str,
    game: str,
    reason: str | None = None,
) -> Path:
    """Copy a LoRA into gen_001/adapter and promote it as champion.

    Used when a workshop attendee pastes a published SFT model URL so evolve
    can start RL (gen 2+) without re-running reject_sft.
    """
    from slm_rl.orchestrator.registry import ModelRegistry

    run_dir = Path(run_dir)
    dest = run_dir / "generations" / "gen_001" / "adapter"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_src, dest)

    from slm_rl.config.schema import DEFAULT_STUB_PRIMARY
    from slm_rl.eval.gate import stub_eval_metrics

    reason = reason or "imported published SFT adapter as RL initialization"
    stub_eval = stub_eval_metrics(
        DEFAULT_STUB_PRIMARY, note="imported HF SFT adapter",
    )
    manifest = {
        "base_model": model_id,
        "backend": "transformers",
        "strategy": "imported_adapter",
        "game": game,
        "adapter": "adapter/",
        "source": str(adapter_src),
    }
    (dest.parent / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    metrics = {
        "train": {"imported": True},
        "eval": stub_eval,
        "gate": {"promoted": True, "reason": reason},
    }
    (dest.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    # Gate comparisons for gen 2+ read champion eval/results.json.
    eval_dir = dest.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "results.json").write_text(
        json.dumps(stub_eval, indent=2), encoding="utf-8",
    )

    registry = ModelRegistry(run_dir / "registry.json")
    if registry.champion < 1:
        registry.promote(1, reason)
    return dest


def cache_slug(url: str) -> str:
    repo = normalize_repo_id(url)
    return repo.replace("/", "__")


def packs_root(home: Path | str) -> Path:
    return Path(home) / "packs"


def write_manifest(
    out: Path,
    *,
    game: str,
    n_episodes: int,
    has_dqn: bool,
    hf_repo: str | None = None,
    n_episodes_raw: int | None = None,
    selection_quantile: float | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "game": game,
        "n_episodes": n_episodes,
        "has_dqn": has_dqn,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if hf_repo:
        manifest["hf_repo"] = hf_repo
    if n_episodes_raw is not None:
        manifest["n_episodes_raw"] = n_episodes_raw
    if selection_quantile is not None:
        manifest["selection_quantile"] = selection_quantile
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / "MANIFEST.json"
    if not path.is_file():
        raise ValueError(f"pack missing MANIFEST.json: {pack_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(pack_dir: Path, **fields: Any) -> dict[str, Any]:
    """Merge fields into an existing MANIFEST.json (e.g. hf_repo after push)."""
    man = read_manifest(pack_dir)
    man.update(fields)
    (pack_dir / "MANIFEST.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8",
    )
    return man


# Orgs people type as placeholders expecting the UI to fill their HF username.
_PLACEHOLDER_ORGS = frozenset({
    "blank", "your-org", "your_org", "your-username", "your_username",
    "username", "org", "<org>", "{org}", "{username}", "<username>",
})


def resolve_push_repo(repo: str, username: str | None) -> str:
    """Normalize `org/name`; rewrite placeholder orgs to `username/name`."""
    repo = normalize_repo_id(repo)
    org, sep, name = repo.partition("/")
    if not sep or not name:
        raise ValueError(f"expected org/name repo id, got {repo!r}")
    if org.lower() not in _PLACEHOLDER_ORGS:
        return repo
    if not username:
        raise ValueError(
            f"push repo {repo!r} uses a placeholder org ({org!r}). "
            "Add an HF token on Welcome (so we can resolve your username), "
            "or paste your real org/name (e.g. alice/slm-rl-boxing)."
        )
    return f"{username}/{name}"


def resolve_push_prefix(prefix: str, username: str | None) -> str:
    """Like resolve_push_repo but keeps a prefix used as `{prefix}-{game}`."""
    prefix = prefix.strip().rstrip("-")
    if "/" not in prefix:
        # bare "slm-rl" → "{username}/slm-rl"
        if not username:
            raise ValueError(
                f"push prefix {prefix!r} needs an HF username — add a token on Welcome "
                "or use org/slm-rl"
            )
        return f"{username}/{prefix}"
    return resolve_push_repo(prefix, username)


def validate_manifest(manifest: dict[str, Any], game: str) -> None:
    ver = manifest.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported pack schema_version={ver!r} (need {SCHEMA_VERSION}); "
            "re-download pack / ask instructor"
        )
    pack_game = manifest.get("game")
    if pack_game != game:
        raise ValueError(f"pack is for {pack_game!r}, project is {game!r}")


def materialize_rollouts(pack_dir: Path, dest_rollouts: Path) -> int:
    src = pack_dir / "rollouts"
    if not src.is_dir():
        raise ValueError(f"pack missing rollouts/: {pack_dir}")
    files = sorted(src.glob("*.jsonl"))
    if not files:
        raise ValueError(f"pack has no *.jsonl under rollouts/: {pack_dir}")
    dest_rollouts.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_rollouts / f.name)
    return len(files)


def resolve_pack(url: str, home: Path | str, game: str) -> Path:
    """Cache-first under `<home>/packs/<slug>/`. Uses HF_TOKEN when available."""
    root = packs_root(home)
    local = root / cache_slug(url)
    manifest_path = local / "MANIFEST.json"
    if not manifest_path.is_file():
        _download_dataset(normalize_repo_id(url), local)
    validate_manifest(read_manifest(local), game)
    return local


def resolve_dqn(url: str, home: Path | str, game: str) -> Path:
    """Download/cache `dqn.pt`. URL may be same dataset repo or a model repo."""
    root = packs_root(home)
    # ponytail: share slug with dataset when same org/name; else own folder
    slug = cache_slug(url) + "__dqn"
    local = root / slug
    pt = local / "dqn.pt"
    if pt.is_file():
        return pt
    # Prefer sibling pack dir if instructor dropped dqn.pt next to demos
    sibling = root / cache_slug(url) / "dqn.pt"
    if sibling.is_file():
        return sibling
    _download_file(normalize_repo_id(url), "dqn.pt", local)
    if not pt.is_file():
        raise ValueError(f"dqn.pt not found in {url!r}")
    _ = game  # reserved for future game-id check inside checkpoint
    return pt


def _download_dataset(repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    from slm_rl.hf_auth import hf_token

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".tmp-{dest.name}-{hashlib.sha1(repo_id.encode()).hexdigest()[:8]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"[packs] downloading dataset {repo_id} → {dest}")
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(tmp), token=hf_token(),
    )
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def _download_file(repo_id: str, filename: str, dest_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    from slm_rl.hf_auth import hf_token

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[packs] downloading {filename} from {repo_id}")
    # Try dataset first, then model
    for repo_type in ("dataset", "model"):
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type=repo_type,
                local_dir=str(dest_dir), token=hf_token(),
            )
            # hf may nest; ensure dest_dir/filename exists
            p = Path(path)
            target = dest_dir / filename
            if p.resolve() != target.resolve() and p.is_file():
                shutil.copy2(p, target)
            return
        except Exception as exc:  # noqa: BLE001 — try next repo_type
            last = exc
    raise ValueError(f"could not download {filename} from {repo_id!r}: {last}")


def push_pack(pack_dir: Path, repo_id: str, *, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    from slm_rl.datagen.hf_push import create_and_upload_folder

    api = HfApi(token=token)
    info = create_and_upload_folder(
        api,
        repo_id,
        repo_type="dataset",
        folder_path=pack_dir,
        path_in_repo="",
        commit_message=f"bake-pack {pack_dir.name}",
        private=False,
    )
    return info.commit_url


def list_local_packs(home: Path | str) -> list[dict[str, Any]]:
    root = packs_root(home)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        man = child / "MANIFEST.json"
        if not man.is_file():
            continue
        try:
            data = read_manifest(child)
        except (ValueError, json.JSONDecodeError):
            continue
        hf_repo = data.get("hf_repo")
        out.append({
            "slug": child.name,
            "path": str(child),
            "game": data.get("game"),
            "n_episodes": data.get("n_episodes"),
            "n_episodes_raw": data.get("n_episodes_raw"),
            "selection_quantile": data.get("selection_quantile"),
            "has_dqn": data.get("has_dqn"),
            "created_at": data.get("created_at"),
            "hf_repo": hf_repo,
            # Prefer the Hub id attendees should paste; fall back to local slug.
            "repo_hint": hf_repo or child.name.replace("__", "/"),
        })
    return out


def bake_pack(
    game: str,
    out_root: Path | str,
    *,
    episodes: int = 1000,
    dqn_decisions: int = 50_000,
    device: str = "cpu",
    seed: int = 0,
    push: str | None = None,
    token: str | None = None,
    selection_quantile: float = 0.25,
) -> Path:
    """Bake one game pack under out_root/<game>/. Prints progress (log-friendly).

    After demos, keeps only the top `selection_quantile` fraction of episodes
    by return (plus wins) — same filter as reject_sft. Use 1.0 to keep all.
    """
    from slm_rl.config.loader import load_game_config
    from slm_rl.datagen.sft_export import filter_rollouts_top_quantile
    from slm_rl.datagen.writer import RolloutWriter
    from slm_rl.games.registry import get_game
    from slm_rl.rollout.runner import EpisodeRunner
    from slm_rl.teachers import make_teacher

    pack_dir = Path(out_root) / game
    pack_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir = pack_dir / "rollouts"
    if rollouts_dir.exists():
        shutil.rmtree(rollouts_dir)
    rollouts_dir.mkdir(parents=True)

    game_cfg = load_game_config(game)
    dqn_path = None
    if game in DQN_GAMES and dqn_decisions > 0:
        from slm_rl.teachers.dqn import train_dqn

        dqn_path = pack_dir / "dqn.pt"
        print(f"[bake] {game}: train-dqn decisions={dqn_decisions}", flush=True)
        train_dqn(game_cfg, decisions=dqn_decisions, out_path=dqn_path, device=device, seed=seed)
    elif game in DQN_GAMES:
        # Reuse an already-trained teacher (train-dqn or prior pack) — do not
        # silently fall back to the heuristic when dqn_decisions=0.
        from slm_rl.teachers.dqn_checkpoint import find_dqn_checkpoint

        home = Path(out_root).parent  # .../runs/packs → .../runs
        found = find_dqn_checkpoint(game, home=home) or find_dqn_checkpoint(game)
        if found is not None:
            dqn_path = pack_dir / "dqn.pt"
            if found.resolve() != dqn_path.resolve():
                shutil.copy2(found, dqn_path)
            print(f"[bake] {game}: using existing DQN {found}", flush=True)
        else:
            print(
                f"[bake] {game}: no DQN checkpoint found; teacher will be heuristic/solver",
                flush=True,
            )

    agent, model_id = make_teacher(
        game_cfg, seed=seed, dqn_checkpoint=str(dqn_path) if dqn_path else None,
    )
    game_cls = get_game(game)
    out_jsonl = rollouts_dir / f"{game}.jsonl"
    wins = 0
    print(f"[bake] {game}: teacher demos episodes={episodes} ({model_id})", flush=True)
    with RolloutWriter(out_jsonl) as writer:
        for i in range(episodes):
            runner = EpisodeRunner(
                game_cls(game_cfg), agent, game_cfg, writer=writer,
                run_id=f"bake-{game}", generation=0, model_id=model_id,
            )
            summary = runner.run_episode(seed + i, episode_id=f"bake-{seed + i}")
            wins += summary["outcome"] == "win"
            if (i + 1) % 50 == 0 or i + 1 == episodes:
                print(f"[bake] {game}: {i + 1}/{episodes} episodes", flush=True)
    print(
        f"[bake] {game}: filter top-quantile={selection_quantile} "
        f"(1.0 keeps all)",
        flush=True,
    )
    filt = filter_rollouts_top_quantile(
        out_jsonl, selection_quantile=selection_quantile,
    )
    n_kept = int(filt["n_kept"])
    n_steps = int(filt["n_steps"])
    write_manifest(
        pack_dir,
        game=game,
        n_episodes=n_kept,
        has_dqn=dqn_path is not None,
        n_episodes_raw=int(filt["n_raw"]),
        selection_quantile=float(filt["selection_quantile"]),
    )
    print(
        f"[bake] {game}: win_rate={wins}/{episodes} "
        f"kept={n_kept}/{filt['n_raw']} steps={n_steps} → {pack_dir}",
        flush=True,
    )
    if push:
        url = push_pack(pack_dir, push, token=token)
        update_manifest(
            pack_dir,
            hf_repo=push,
            hf_commit_url=url,
        )
        hub = f"https://huggingface.co/datasets/{push}"
        print(f"[bake] pushed {push}: {url}", flush=True)
        print(
            f"[bake] dataset (not model) page: {hub}/tree/main",
            flush=True,
        )
        print(
            f"[bake] paste for attendees / Teacher=dqn: {push}",
            flush=True,
        )
    return pack_dir


def bake_packs(
    *,
    home: Path | str,
    games: list[str],
    episodes: int = 1000,
    dqn_decisions: int = 50_000,
    device: str = "cpu",
    seed: int = 0,
    push: str | None = None,
    push_prefix: str | None = None,
    token: str | None = None,
    selection_quantile: float = 0.25,
) -> list[Path]:
    out_root = packs_root(home)
    out_root.mkdir(parents=True, exist_ok=True)
    done: list[Path] = []
    for gname in games:
        repo = push
        if push_prefix:
            repo = f"{push_prefix.rstrip('-')}-{gname}"
        done.append(
            bake_pack(
                gname, out_root,
                episodes=episodes, dqn_decisions=dqn_decisions,
                device=device, seed=seed, push=repo, token=token,
                selection_quantile=selection_quantile,
            )
        )
    return done


def _main(argv: list[str] | None = None) -> None:
    """Subprocess entry for playground bake (not a user-facing CLI)."""
    import argparse

    from slm_rl.games.registry import available_games
    from slm_rl.hf_auth import hf_token

    ap = argparse.ArgumentParser(prog="python -m slm_rl.packs")
    ap.add_argument("--home", required=True)
    ap.add_argument("--game")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--dqn-decisions", type=int, default=50_000)
    ap.add_argument(
        "--selection-quantile", type=float, default=0.25,
        help="Keep top fraction of demos by return (1.0 = keep all)",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push")
    ap.add_argument("--push-prefix")
    args = ap.parse_args(argv)
    if args.all:
        games = available_games()
    elif args.game:
        games = [args.game]
    else:
        raise SystemExit("need --game NAME or --all")
    bake_packs(
        home=args.home,
        games=games,
        episodes=args.episodes,
        dqn_decisions=args.dqn_decisions,
        device=args.device,
        seed=args.seed,
        push=args.push,
        push_prefix=args.push_prefix,
        token=hf_token(),
        selection_quantile=args.selection_quantile,
    )


if __name__ == "__main__":
    _main()
