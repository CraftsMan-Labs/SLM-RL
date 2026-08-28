# Mario DQN assets

Used to teach DQN intuition on Super Mario Bros. World 1-1. Chapter 5 can train a pixel CNN live and evaluate it for a user-set
step budget. Recorded clips remain the fallback. The Atari RAM-vector
teacher is still the workshop critical path.

## Recorded clips

Real emulator recordings from checksummed public Nature-style CNN
checkpoints (`alanfrancis442/mario-ai`, 4×84×84, `RIGHT` / `RIGHT+A`):

- `mario-untrained.mp4` — random weights; dies at the first Goomba
- `mario-mid.mp4` — early checkpoint; starts jumping, still dies early
- `mario-pretrained.mp4` — later checkpoint; clears the first pit and
  reaches the first pipe
- `clip_manifest.json` — distances, rewards, clip checksums

Refresh (needs `gym-super-mario-bros`, `nes-py`, `torch`, `ffmpeg`):

```bash
python docs/workshop/record_mario_clips.py
```

Checkpoints download into `checkpoints/` (gitignored). Prefer the public
Hugging Face repo `CraftsMan-Labs/mario-dqn-workshop` (`hf-repo/` is the
upload template). GitHub staged files stay the checksummed fallback.

Weights are educational; Nintendo owns Super Mario Bros.

A T4 smoke checklist lives in `SMOKE_TEST.md`.

## Fallback

If the emulator, network, or codec is missing:

- `fallback_storyboard.svg` — schematic untrained / mid / pretrained frames
- `fallback_metrics.jsonl` — recorded World 1-1 distance curve for narration

## Compatibility

- CPython 3.14: `gym-super-mario-bros==9.1.0` and `nes-py>=9.0.1`
- Colab 3.11/3.12: `gym==0.25.2`, `gym-super-mario-bros==7.4.0`, `nes-py<9`
- Action space is the 2-way `[["right"], ["right", "A"]]`, not gym's 5-way
  `RIGHT_ONLY` (that list starts with `NOOP` and breaks the public nets)
