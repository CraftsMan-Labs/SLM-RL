# Training dataset pipeline (instructor)

How we produce workshop warm-start packs and publish them to Hugging Face.
This is the **day-before / bake** path for teachers and LoRAs. The attendee
evolve loop after that is [PIPELINE.md](PIPELINE.md).

## Step by step

1. **Train DQN (long run)**  
   Run DQN for **2 million decisions** on the target game (e.g. Boxing) to get
   a strong teacher checkpoint (`dqn.pt`). This is the top-performing teacher
   we will demo from.

2. **Ask for approval**  
   Pause and get an explicit go-ahead before baking packs or publishing
   anything public.

3. **Review how it plays**  
   Watch the DQN teacher (live / screen replay). Confirm it looks competent
   and that episodes finish to natural game-over (not monitor-truncated junk).

4. **Bake demo datasets from the teacher**  
   With the approved DQN running (or from its checkpoint), generate rollouts
   and keep the **top ~30%** of episodes by score. From those, promote the
   strongest slice (**top ~10%**) into the workshop dataset pack
   (rollouts + SFT jsonl + `dqn.pt` in the pack).  
   Practical workshop default we ship: a small fixed pack of the best
   full-to-death demos (e.g. top 30 episodes) rather than an unbounded dump.

5. **Publish the dataset on Hugging Face**  
   Push the pack to the canonical dataset repo for the game, e.g.
   `BLANK/slm-rl-<game>` (dataset). Use the same naming for every game;
   do **not** invent one-off names like `*-dqn-top30` or `*-train30`.

6. **Publish the DQN teacher and place Hub collections**  
   Publish the DQN checkpoint as its own model repo,
   `BLANK/slm-rl-<game>-dqn` (`dqn.pt` only).  
   Put artifacts into the three collections with clear roles:
   - **SLM RL workshop datasets** — `BLANK/slm-rl-<game>` (dataset)
   - **SLM RL models** — LoRA adapters (added after SFT, step 8)
   - **SLM RL dqn** — `BLANK/slm-rl-<game>-dqn`

7. **SFT (warm-start LoRA)**  
   After the dataset (and DQN) are on the Hub, pull/use them and run
   **reject_sft** (supervised fine-tune) on the base SLM
   (`LiquidAI/LFM2.5-350M`) against the pack’s SFT pairs. This produces the
   warm-start LoRA under `adapter/`.

8. **Publish the LoRA with the same naming**  
   Upload the champion adapter to `BLANK/slm-rl-<game>` (**model** repo,
   same id as the dataset). Refresh the **SLM RL models** collection.
   Attendees paste that id as both dataset URL and adapter URL in the
   playground.

## Naming cheat sheet

| Artifact | Repo id | Hub type |
| -------- | ------- | -------- |
| Demo pack + SFT jsonl | `BLANK/slm-rl-<game>` | dataset |
| SFT LoRA (`adapter/`) | `BLANK/slm-rl-<game>` | model |
| DQN teacher (`dqn.pt`) | `BLANK/slm-rl-<game>-dqn` | model |

`<game>` is the playground id (`boxing`, `space-invaders`, …).

## Related

- Attendee evolve / gate loop: [PIPELINE.md](PIPELINE.md)
- Day-of workshop lifecycle: [LIFECYCLE.md](LIFECYCLE.md)
