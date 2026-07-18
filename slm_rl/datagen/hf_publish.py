"""Publish a playground experiment's datasets + champion adapter to the
attendee's own Hugging Face account (plan 021 design decision 4).

Two independent sides, each best-effort and independently reported so a
partial failure (e.g. model upload works, dataset upload's network hiccups)
is visible rather than swallowed:

  dataset side: reuses `slm_rl.datagen.hf_push.push_generation` per existing
    behavior, one call per generation directory present, repo
    `{username}/slm-rl-{experiment}-data` (dataset repo type).
  model side (NEW): resolves the run's champion generation from
    `registry.json` (same resolution `slm_rl.theater.exhibition` already
    does: `ModelRegistry(...).champion` + `RunPaths(...).adapter(gen)`),
    uploads that generation's `adapter/` dir plus a generated README.md
    model card to repo `{username}/slm-rl-{experiment}` (model repo type).
    Champion 0 (no promotion has ever happened) means no adapter directory
    exists yet -- datasets-only publish, with a message explaining why.

Token hygiene (CODING_GUIDELINE + plan 021 hard rule 3): every huggingface_hub
call here takes `token=` explicitly on `HfApi(token=...)` (never ambient
`HfApi()` -- this module publishes attendee data to the attendee's account).
Per-call `token=` on create_repo/upload_* is redundant once the client is
constructed with the token and is omitted (plan 025 G6). Errors surface
`str(e)` only -- never a repr that could echo the token.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# HF repo id slug (no owner/). Allow letters/digits/._- ; strip owner if pasted.
_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")


def normalize_publish_slug(repo_name: str | None, *, experiment: str) -> str:
    """Return the model repo slug (no `owner/`). Dataset is `{slug}-data`."""
    raw = (repo_name or "").strip()
    if not raw:
        return f"slm-rl-{experiment}"
    if "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1]
    raw = raw.strip()
    if not _REPO_SLUG_RE.match(raw):
        raise ValueError(
            f"invalid repo name {raw!r}: use 1–96 chars of letters, digits, "
            "., _, or - (no spaces); owner/ is filled from your HF token"
        )
    if raw.endswith("-data"):
        raise ValueError(
            "repo name should be the model id only — we publish the dataset as "
            f"{raw}-data automatically"
        )
    return raw


def publish_repo_ids(username: str, experiment: str, repo_name: str | None = None) -> tuple[str, str]:
    """(model_repo_id, dataset_repo_id) under the attendee's username."""
    slug = normalize_publish_slug(repo_name, experiment=experiment)
    return f"{username}/{slug}", f"{username}/{slug}-data"


@dataclass
class PublishResult:
    dataset_repo: str | None
    dataset_error: str | None
    model_repo: str | None
    model_error: str | None
    message: str | None  # e.g. "champion 0: datasets only, no adapter yet"

    def to_json(self) -> dict:
        # UI rebuilds Hub links from repo ids (playground/page.py) — no
        # model_url/dataset_url fields (plan 025 G7).
        return {
            "dataset_repo": self.dataset_repo,
            "dataset_error": self.dataset_error,
            "model_repo": self.model_repo,
            "model_error": self.model_error,
            "message": self.message,
        }


def _game_title(game: str) -> str:
    return game.replace("-", " ").title()


def _render_model_card(
    *,
    game: str,
    experiment: str,
    base_model: str | None,
    champion_generation: int,
    gate_metrics: dict,
    repo_id: str | None = None,
    dataset_repo: str | None = None,
) -> str:
    """HF model card with YAML metadata + transformers/PEFT load snippet."""
    base = base_model or "LiquidAI/LFM2.5-350M"
    title = _game_title(game)
    hub_id = repo_id or f"BLANK/slm-rl-{game}"
    owner = hub_id.split("/", 1)[0] if "/" in hub_id else "BLANK"
    # Workshop packs use game id; attendee publish pairs model + `{slug}-data`.
    if owner == "BLANK":
        dataset_id = dataset_repo or f"BLANK/slm-rl-{game}"
        dqn_id = f"BLANK/slm-rl-{game}-dqn"
    else:
        dataset_id = dataset_repo or f"{owner}/slm-rl-{experiment}-data"
        dqn_id = None
    eval_metrics = gate_metrics.get("eval", {})
    gate = gate_metrics.get("gate", {})
    train = gate_metrics.get("train", {})

    related_rows = [
        f"| **Dataset pack** | [{dataset_id}](https://huggingface.co/datasets/{dataset_id}) |"
    ]
    if dqn_id:
        related_rows.append(
            f"| **DQN teacher** | [{dqn_id}](https://huggingface.co/{dqn_id}) |"
        )
    related = "\n".join(related_rows)

    cli_dqn = f" \\\n  --dqn-url {dqn_id}" if dqn_id else ""

    yaml = "\n".join(
        [
            "---",
            "library_name: peft",
            f"base_model: {base}",
            "pipeline_tag: text-generation",
            "tags:",
            "  - lora",
            "  - peft",
            "  - transformers",
            "  - reinforcement-learning",
            "  - atari",
            "  - slm-rl",
            f"  - {game}",
            "license: apache-2.0",
            "---",
            "",
        ]
    )

    body = f"""# {hub_id}

PEFT LoRA adapter that warm-starts **{title}** play for
[{base}](https://huggingface.co/{base})
in the [SLM-RL](https://github.com/CraftsMan-Labs/SLM-RL) workshop.

| | |
|---|---|
| **Game** | `{game}` |
| **Base model** | `{base}` |
| **Adapter layout** | `adapter/` (PEFT `adapter_config.json` + weights) |
| **Training** | `reject_sft` on DQN teacher demos |
| **Champion generation** | {champion_generation} |
| **Promoted** | {gate.get("promoted", "unknown")} ({gate.get("reason", "n/a")}) |
{related}

Paste `{hub_id}` as the playground **adapter URL** (and usually the same id
as the **dataset URL**).

## Install

```bash
pip install "transformers>=4.46" peft accelerate torch
```

## Load with transformers + PEFT

Weights live under the `adapter/` subfolder — pass `subfolder="adapter"`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "{base}"
ADAPTER = "{hub_id}"  # this repo

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
dtype = torch.bfloat16 if device != "cpu" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
model = PeftModel.from_pretrained(model, ADAPTER, subfolder="adapter")
model.to(device).eval()

messages = [
    {{"role": "system", "content": "You play {title}. Reply with ACTION: <id>."}},
    {{"role": "user", "content": "Legal actions: 1) NOOP 2) UP\\nChoose."}},
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Download only the adapter files

```python
from huggingface_hub import snapshot_download

path = snapshot_download("{hub_id}", allow_patterns="adapter/*")
# then: PeftModel.from_pretrained(base_model, f"{{path}}/adapter")
```

## Workshop / SLM-RL CLI

```bash
slm-rl evolve --game {game} \\
  --dataset-url {dataset_id} \\
  --adapter-url {hub_id}{cli_dqn} \\
  --generations 2
```

## Train metrics (if recorded)

```json
{json.dumps({"train": train, "eval": eval_metrics, "gate": gate}, indent=2, sort_keys=True)}
```

Trained with [SLM-RL](https://github.com/CraftsMan-Labs/SLM-RL).
"""
    return yaml + body


def _dataset_side(
    *, token: str, username: str, experiment: str, run_dir: Path, dataset_repo: str,
) -> tuple[str | None, str | None]:
    """Returns (repo_id, error)."""
    from slm_rl.datagen.hf_push import push_generation

    gen_dirs = sorted(run_dir.glob("generations/gen_*"))
    if not gen_dirs:
        return dataset_repo, "no generations found to publish"

    try:
        for gen_dir in gen_dirs:
            gen_num = int(gen_dir.name.split("_")[1])
            push_generation(
                dataset_repo, experiment, gen_num, gen_dir, private=False, token=token,
            )
    except Exception as e:  # noqa: BLE001 - any hub/network error is reported, not fatal to the model side
        return dataset_repo, str(e)
    return dataset_repo, None


def _model_side(
    *,
    token: str,
    username: str,
    experiment: str,
    game: str,
    run_dir: Path,
    model_repo: str,
    dataset_repo: str,
) -> tuple[str | None, str | None, str | None]:
    """Returns (repo_id, error, message)."""
    from slm_rl.datagen.hf_push import create_and_upload_folder
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.orchestrator.registry import ModelRegistry

    registry = ModelRegistry(run_dir / "registry.json")
    champ_gen = registry.champion
    if champ_gen <= 0:
        return None, None, (
            f"experiment {experiment!r} has no promoted champion yet "
            f"(registry champion={champ_gen}); publishing datasets only."
        )

    paths = RunPaths(run_dir.parent, run_dir.name)
    adapter_dir = paths.adapter(champ_gen)
    if not adapter_dir.exists():
        return None, None, (
            f"champion generation {champ_gen} has no adapter/ directory on disk; "
            "publishing datasets only."
        )

    manifest_path = paths.manifest(champ_gen)
    base_model = None
    if manifest_path.exists():
        base_model = json.loads(manifest_path.read_text(encoding="utf-8")).get("base_model")

    metrics_path = paths.metrics(champ_gen)
    gate_metrics = {}
    if metrics_path.exists():
        gate_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    from huggingface_hub import HfApi

    repo_id = model_repo
    card = _render_model_card(
        game=game, experiment=experiment, base_model=base_model,
        champion_generation=champ_gen, gate_metrics=gate_metrics,
        repo_id=repo_id,
        dataset_repo=dataset_repo,
    )
    try:
        api = HfApi(token=token)
        create_and_upload_folder(
            api, repo_id, repo_type="model", folder_path=adapter_dir,
            path_in_repo="adapter", commit_message=f"{experiment}: champion gen {champ_gen}",
            private=False,
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
            repo_id=repo_id, repo_type="model", commit_message="model card",
        )
        return repo_id, None, None
    except Exception as e:  # noqa: BLE001 - reported to the caller, never fatal
        return repo_id, str(e), None


def publish_experiment(
    *,
    token: str,
    username: str,
    experiment: str,
    game: str,
    run_dir: Path,
    repo_name: str | None = None,
) -> PublishResult:
    """Publish both sides for one playground experiment. `run_dir` is the
    experiment's own run dir (`ExperimentDir.run_dir`); `token`/`username`
    come from the caller's loaded profile -- this function never reads a
    profile itself and never falls back to an ambient token.

    `repo_name` is the model repo slug (no owner/). Dataset is published as
    `{slug}-data`. Default slug: `slm-rl-{experiment}`.
    """
    run_dir = Path(run_dir)
    model_repo_id, dataset_repo_id = publish_repo_ids(username, experiment, repo_name)

    model_repo, model_error, model_message = _model_side(
        token=token, username=username, experiment=experiment, game=game,
        run_dir=run_dir, model_repo=model_repo_id, dataset_repo=dataset_repo_id,
    )
    dataset_repo, dataset_error = _dataset_side(
        token=token, username=username, experiment=experiment, run_dir=run_dir,
        dataset_repo=dataset_repo_id,
    )

    return PublishResult(
        dataset_repo=dataset_repo, dataset_error=dataset_error,
        model_repo=model_repo, model_error=model_error,
        message=model_message,
    )
