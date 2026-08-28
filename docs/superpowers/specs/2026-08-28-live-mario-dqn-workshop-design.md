# Live Mario DQN workshop design

## Goal

Let participants play the workshop games before training, watch a Mario Deep Q-Network (DQN) improve during a real Colab training run, and evaluate a trained policy for a user-controlled number of steps. Keep the presentation and notebook synchronized, and preserve recorded evidence when the live emulator or GPU path fails.

## Audience and teaching sequence

The workshop serves a mixed audience. The interaction should establish intuition before introducing DQN mechanics.

1. **Play before training.** Near the start of the notebook, participants select Mario or a supported Atari game and use clickable controls to play a short attempt.
2. **Predict behavior.** Chapter 5 retains the action-value, Bellman target, and exploration checkpoints.
3. **Watch a baseline attempt.** The notebook runs an untrained Mario policy and asks participants to identify what failed.
4. **Train live.** A 15–20 minute cell performs genuine DQN updates and periodically evaluates the current policy.
5. **Evaluate the result.** A final cell runs a trained policy for `EVAL_STEPS`, defaulting to 10,000, and reports gameplay and metrics.
6. **Generalize the pattern.** The deck explains that the same control, training, and evaluation loop applies to the workshop's Atari environments even though their observation encoders differ.

## Notebook experience

### Reusable game controls

Add a small environment adapter with this interface:

- `reset()`
- `step(action_id)`
- `render_rgb()`
- `action_labels`
- `metrics()`
- `close()`

Provide Mario and Atari implementations. An `ipywidgets` panel uses the adapter to render:

- the current game frame;
- one button per legal action;
- **Reset** and bounded **Auto-repeat** controls;
- cumulative reward, score or distance, and step count;
- a clear stopped, terminal, or dependency-error state.

The widget belongs near the top of the notebook under a “Play before you train” heading. Mario dependencies install lazily when Mario is selected, so the normal Atari setup is not disturbed when participants skip Mario. Button-based controls are required because a normal Colab cell cannot reliably capture real-time keyboard input.

### Live Mario training

Extend `docs/workshop/mario_lab.py` with a chunked training API. It must:

- train for a time or decision budget;
- resume from a checkpoint;
- emit progress after each chunk;
- evaluate and capture a short attempt at configurable intervals;
- report reward, distance, loss, epsilon, decisions, and deaths;
- save resumable local checkpoints;
- avoid raising through the notebook boundary when an optional dependency fails.

The notebook exposes:

- `TRAINING_MODE`: `warm-start` by default or `from-scratch`;
- `TRAIN_MINUTES`: 15 by default, bounded to a workshop-safe range;
- `EVAL_INTERVAL`: controls how often a fresh attempt is shown;
- `SAVE_TO_DRIVE`: off by default.

The default run resumes from the public warm-start checkpoint. This makes improvement visible within the selected workshop budget. The from-scratch option remains available with copy explaining that a short run may not produce competent play.

### Final evaluation

Add a final Chapter 5 cell with:

- `EVAL_SOURCE`: `local-trained` or `public-final`;
- `EVAL_STEPS`: 10,000 by default;
- a documented safe upper bound;
- a rendered video or streamed frame sequence;
- summary metrics for total reward, farthest distance, deaths, completed episodes, and best attempt.

The cell compares the local workshop checkpoint with the public final checkpoint when both are available. It labels each result by source so a short live run is not presented as a fully trained model.

## Hugging Face checkpoint repository

Create a public model repository named `mario-dqn-workshop` under the user's Hugging Face account. The full repository ID remains configurable as `MARIO_MODEL_REPO` because the account name has not been supplied.

Publish:

- a warm-start checkpoint;
- a final checkpoint;
- preprocessing and action-space configuration;
- training metadata and evaluation metrics;
- checksums;
- a model card describing the educational purpose, environment, limitations, and Nintendo ownership of Super Mario Bros.

Generate the random baseline locally instead of storing random weights.

Notebook downloads must pin a repository revision and verify the expected SHA-256 checksum. Files should use the Hugging Face cache. A download failure must route to the existing recorded clips and explain the fallback. Participant uploads stay disabled unless the participant supplies a token and explicitly enables publishing.

## Presentation changes

Add two Chapter 5 slides and update both talk tracks:

1. **Watch learning happen** — aligns with the chunked live-training cell and explains attempts, epsilon, reward, distance, and loss without implying that loss alone measures gameplay quality.
2. **Let the trained DQN play** — aligns with the configurable evaluation cell, names the 10,000-step default, and distinguishes the locally improved checkpoint from the public final checkpoint.

The early games section should cue the “Play before you train” widget. The existing three recorded Mario clips remain the fallback evidence and should not duplicate the new live-training slide.

## Failure handling

- If Mario package installation fails, show the package error and retain Atari controls.
- If the Mario environment cannot start, show the committed fallback storyboard and clips.
- If the public checkpoint cannot download or fails checksum validation, stop using that checkpoint and explain why.
- If CUDA is unavailable, reduce the default live-training budget or allow evaluation-only mode.
- If video encoding fails, stream sampled frames and keep the metric summary.
- If a Colab session disconnects, allow resume from a local or Drive checkpoint when one exists.
- Always close environments when a widget is reset, replaced, or disposed.

## Verification

Add automated coverage for:

- Mario and Atari adapter reset, step, render, terminal, and close behavior;
- widget action dispatch and bounded auto-repeat;
- checkpoint download, revision pinning, and checksum rejection;
- chunked training callbacks, checkpoint save, and resume;
- final evaluation step limits and metric aggregation;
- dependency and encoder fallback paths;
- notebook cell order and required parameters;
- new presentation slide IDs, talk-track parity, and asset resolution.

Regenerate `colab_workshop.ipynb`, run workshop and Mario tests, and build the Vue presentation. A manual Colab GPU smoke test should cover package installation, one human-controlled attempt, a short live-training run, checkpoint resume, and final evaluation.

## Implementation plan

1. Introduce the shared playable-environment adapter and unit tests.
2. Build the reusable `ipywidgets` game panel and add it to the early notebook flow.
3. Add Hugging Face checkpoint configuration, verified downloads, and metadata parsing.
4. Refactor Mario DQN training into resumable chunks with progress callbacks and periodic attempts.
5. Add configurable final evaluation and fallback rendering.
6. Add the two presentation slides and synchronize Python and JavaScript talk tracks.
7. Regenerate notebook and assets, run automated verification, and complete a Colab GPU smoke test.
8. Create and populate the public `mario-dqn-workshop` repository after the Hugging Face account name and credentials are available.

## Scope boundaries

- The workshop will not claim real-time keyboard control inside a standard Colab cell.
- The 15–20 minute warm-start run demonstrates real learning but does not replace full training.
- Mario remains an optional teaching path; the existing Atari teacher remains the workshop's critical data-generation path.
- The work does not upload checkpoints or create external repositories without explicit authorization and credentials.
