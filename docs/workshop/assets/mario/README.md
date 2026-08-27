# Mario DQN fallback

Used when `gym-super-mario-bros` / `nes-py` will not install on the Colab
runtime, the checkpoint is missing, or the emulator crashes.

- `fallback_storyboard.svg` — schematic untrained / mid / pretrained frames
- `fallback_metrics.jsonl` — typical World 1-1 distance curve for narration

The live path is optional. Chapter 5's Atari RAM-vector teacher is the
workshop critical path.

## Compatibility spike (2026-08-27)

- CPython 3.14: `gym-super-mario-bros==9.1.0` and `nes-py>=9.0.1` install and
  `SuperMarioBros-1-1-v0` + `RIGHT_ONLY` reset successfully (240×256 RGB).
- Colab is usually 3.11/3.12. Use the legacy pin in `mario_lab.pinned_packages()`
  (`gym==0.25.2`, `gym-super-mario-bros==7.4.0`, `nes-py<9`).
- No checksummed public DQN checkpoint is vendored. Live mode is inference plus
  a bounded Bellman update on whatever weights are present (random if none).
  Label it that way; do not claim a full trainer resume.
- Missing emulator, torch, or checksum → `mode="fallback"` and these assets.
