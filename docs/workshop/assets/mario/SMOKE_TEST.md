# Mario Colab GPU smoke test

Run this once on a T4 after the notebook is regenerated.

1. Runtime → Change runtime type → T4 GPU.
2. Run Chapter 0 and the knobs cell.
3. In **Play before you train**, pick `boxing`, click an action, Reset, and Auto-repeat.
4. Switch the dropdown to `mario`. Confirm packages install only then. Play `RIGHT` and `RIGHT+A`.
5. If Mario install fails, confirm Atari controls still work and the error is printed.
6. In Chapter 5, leave `TRAINING_MODE=warm-start`, set `TRAIN_MINUTES=2`, and run **Train Mario live**. Confirm a progress row, a checkpoint path, and a chart.
7. Re-run the same cell and confirm it resumes from `local-trained.chkpt`.
8. Run **Evaluate the trained DQN** with `EVAL_STEPS=400`, then again with `EVAL_SOURCE=public-final`.
9. Confirm a missing Hugging Face download falls back to the committed clips without breaking later chapters.

Pass the smoke test only if each numbered step completes or fails with the documented fallback.
