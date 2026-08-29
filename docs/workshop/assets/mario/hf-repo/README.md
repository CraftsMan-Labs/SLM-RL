---
license: other
tags:
  - reinforcement-learning
  - dqn
  - workshop
library_name: pytorch
---

# mario-dqn-workshop

Educational Super Mario Bros. World 1-1 Deep Q-Network (DQN) checkpoints for the SLM-RL Colab workshop.

These weights exist so a 15–20 minute live-training cell can start from a warm-start policy and so a final evaluation cell can show a stronger public policy. They are not a product, a Nintendo-approved model, or a replacement for the workshop’s Atari RAM-vector teacher.

## Files

| File | Role |
|---|---|
| `warm-start.chkpt` | Early World 1-1 play. Default starting point for live training. |
| `final.chkpt` | Later World 1-1 play. Default public evaluation policy. |
| `config.json` | Action map, frame stack, and preprocessing. |
| `checksums.json` | SHA-256 pins for the checkpoint files. |
| `metrics.json` | Recorded evaluation distances and rewards. |

A random / untrained baseline is generated locally. It is not stored here.

## Environment

- Title: Super Mario Bros. World 1-1 (`SuperMarioBros-1-1-v0`)
- Observation: 4 stacked 84×84 grayscale frames
- Actions: `RIGHT`, `RIGHT+A`
- Encoder: Nature-style CNN
- Frame skip: 4

## How the notebook uses this repo

The Colab cell `MARIO_MODEL_REPO` defaults to `BLANK/mario-dqn-workshop` and pins `MARIO_MODEL_REVISION`. Downloads are anonymous. A checksum mismatch or network failure falls back to the committed workshop clips.

Live training writes `local-trained.chkpt` on the Colab disk. Participant uploads stay off unless the participant supplies their own token.

## Limitations

- A short workshop run does not equal full training.
- Super Mario Bros is owned by Nintendo. These weights are educational evidence only.
- The workshop’s production teacher remains the Atari RAM-vector MLP in `slm_rl.teachers.dqn`.
