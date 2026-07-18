"""Tutorial-mode copy for the playground UI (plan 023).

Every ⓘ card in `page.py` is data here, not markup buried in the page
template -- one place for workshop staff to edit copy without touching
HTML/JS. Each entry is `{"title": ..., "body": ...}`, body <= 80 words,
plain language (attendees are not readers of this codebase).

Where a card cites a measured number, the source yaml/doc comment is named
in a comment next to the entry so future edits (new measurements, retuned
thresholds) can find and update the copy in the same place. This module is
pure data (stdlib only, no imports beyond `__future__`) so it holds on the
8GB tier like the rest of `slm_rl.playground`.

Card coverage (test-asserted by tests/test_playground_tutorial.py):
  - one entry per `Knob.key` in `slm_rl.playground.knobs.KNOBS`
  - "teacher_select", "model_field", "backend_field" (022)
  - "reward_tab" (the shape_reward(ctx) doctrine + ctx-keys reference)
  - "evolve_button", "scoreboard_*" (one per column), "watch_link",
    "ab_button", "gens_link", "play_again_button", "publish_button",
    "signup_card"
"""

from __future__ import annotations

CARDS: dict[str, dict[str, str]] = {
    # --- Knobs: game-shape ------------------------------------------------
    "max_turns": {
        "title": "Max turns",
        "body": (
            "Hard cap on decisions per episode. Playground default is 30 for "
            "fast Docker-CPU demos; repo game yamls are much higher "
            "(Boxing 2500, Space Invaders 2000) for full natural games. "
            "Set this in the Run / Theater panels before Start. Atari "
            "matches can still end earlier (clock, lives, score 100)."
        ),
    },
    "eval_episodes": {
        "title": "Eval episodes (gate)",
        "body": (
            "Frozen-suite episodes for the promote/reject gate after train. "
            "Playground default is 2 (workshop speed); repo game configs use "
            "100. Raise toward 20–100 when you want lower gate noise."
        ),
    },
    "action_repeat": {
        "title": "Action repeat",
        "body": (
            "How many raw game frames one decision covers (ALE games only). "
            "Coarser repeat means fewer decisions per episode, not worse "
            "play: Breakout was swept action_repeat in {1,2,3,4} x dead_zone "
            "and action_repeat=1 won on finer paddle control at equal score "
            "(configs/games/breakout.yaml)."
        ),
    },
    "score_scale": {
        "title": "Score scale",
        "body": (
            "Divides raw game points into a per-step shaped reward the "
            "trainer sees, so no single event dominates training signal. "
            "Space Invaders uses 30.0 (its max single-invader value) to keep "
            "shaped reward per step roughly <= 1 (configs/games/"
            "space-invaders.yaml)."
        ),
    },
    "life_loss_penalty": {
        "title": "Life loss penalty",
        "body": (
            "Extra (negative) shaped reward applied the decision a life is "
            "lost, on top of the score-based reward. Space Invaders and "
            "Breakout both use -0.5; Pong sets this to 0.0 since Pong has no "
            "lives -- opponent points already arrive as negative reward "
            "(configs/games/pong.yaml)."
        ),
    },
    "noop_start_max": {
        "title": "No-op start max",
        "body": (
            "Random no-op actions injected at episode reset (Mnih et al. "
            "2015 DQN protocol), seeded per episode -- never wall-clock. "
            "Without this, ALE Space Invaders resets to the identical board "
            "every eval seed, so a warm-started model just replayed one "
            "memorized script (measured: mean score exactly 5.1667 across "
            "two generations, configs/games/space-invaders.yaml)."
        ),
    },
    # --- Knobs: doom-loop monitor -----------------------------------------
    "action_repeat_threshold": {
        "title": "Action repeat threshold",
        "body": (
            "How many identical actions in a row trigger the doom-loop "
            "monitor's first intervention (reflect). Minimum 3. "
            "Game-specific: set ABOVE what competent play produces "
            "(measure-then-double), e.g. Space Invaders' 88 is 2x its "
            "measured max streak of 44 -- too low and it flags good "
            "players, not loops. Values above max_turns disable this "
            "check; do not do that for SLM/RL training."
        ),
    },
    "ngram_loop_threshold": {
        "title": "N-gram loop threshold",
        "body": (
            "How many times a short action pattern (2-4 actions) must "
            "repeat consecutively to trigger a reflect intervention. "
            "Minimum 3. Same measure-then-double convention as "
            "action_repeat_threshold: Space Invaders' 44 is 2x its "
            "measured max 2-gram repeat count of 22."
        ),
    },
    "state_revisit_threshold": {
        "title": "State revisit threshold",
        "body": (
            "How many times the SAME game state can recur before the "
            "monitor intervenes. Atari keepers measure this against "
            "competent play (configs/games/*.yaml)."
        ),
    },
    "reward_stagnation_window": {
        "title": "Reward stagnation window",
        "body": (
            "Decisions allowed with no reward increase before truncation. "
            "Sparse-reward games need this wide: Space Invaders' window of "
            "240 is 2x its measured max scoreless streak of 120 (configs/"
            "games/space-invaders.yaml) -- too narrow and it truncates the "
            "MAJORITY of good episodes, discarding the best demonstrations."
        ),
    },
    # --- Knobs: training -----------------------------------------------
    "selection_quantile": {
        "title": "Selection quantile",
        "body": (
            "reject_sft trains only on the top fraction of episodes by "
            "reward, 0.0-1.0. Default 0.25 means the best quarter of "
            "rollouts becomes training data (configs/default.yaml) -- "
            "raising it is stricter (fewer, better episodes); lowering it "
            "trains on more but noisier data."
        ),
    },
    "max_completion_tokens": {
        "title": "Max tokens per action",
        "body": (
            "Token budget the model gets to emit one action ('ACTION: 3'). "
            "The same cap governs rollout, gate eval, and GRPO generation. "
            "Playground default is 24 (a legal action needs only a few "
            "tokens); repo default is 256. Raise only if prompts need "
            "the model to explain itself."
        ),
    },
    "episodes_per_generation": {
        "title": "Episodes per generation",
        "body": (
            "How many LLM rollout episodes make up one generation before "
            "selection_quantile filters them. Playground default is 2 — "
            "fast enough for laptop workshops; raise toward 20 for stronger "
            "GRPO signal. Repo train default is 200 for full runs."
        ),
    },
    "group_size": {
        "title": "GRPO group size",
        "body": (
            "Completions sampled per prompt during GRPO. Larger groups give "
            "richer relative rewards but multiply Docker-CPU wall time. "
            "Playground default is 2 (~20 min train budget)."
        ),
    },
    "grpo_max_steps": {
        "title": "GRPO max steps",
        "body": (
            "Hard cap on optimizer steps per generation. Playground default "
            "24 keeps Docker-CPU trains near 20 minutes; leave unset / null "
            "in repo configs for full epochs."
        ),
    },
    "grpo_max_prompts": {
        "title": "GRPO max prompts",
        "body": (
            "Max decision-step prompts exported into the GRPO dataset. "
            "Playground default 32; repo default 512 for fuller runs."
        ),
    },
    "replay_generations": {
        "title": "Replay generations",
        "body": (
            "How many recent generations' rollouts are mixed into training. "
            "Playground default 1 (current gen only) so restarts stay fast; "
            "raise for more replay signal on GPU."
        ),
    },
    "warmstart_episodes": {
        "title": "Warm-start episodes",
        "body": (
            "Teacher-played episodes used to bootstrap generation 1, before "
            "the model has any of its own experience to learn from. Default "
            "1000 (configs/default.yaml teacher section) -- expert "
            "iteration with a real expert (an exact solver or tiny MLP), so "
            "this budget doesn't touch the 8GB model-memory floor."
        ),
    },
    "teacher": {
        "title": "Teacher",
        "body": (
            "Which expert plays the warm-start episodes: heuristic (a "
            "hand-written or exact-solver agent, always available) or dqn "
            "(a trained CleanRL checkpoint). Choosing dqn asks for a "
            "Hugging Face repo that contains dqn.pt — downloaded at create "
            "time — or uses a local bake pack under runs/packs/<game>/."
        ),
    },
    # --- Quick-experiment agent select (distinct from the "teacher" knob
    # above: this picks who plays the QUICK screen, not the evolve warm
    # start) --------------------------------------------------------------
    "teacher_select": {
        "title": "agent",
        "body": (
            "Who plays this quick experiment: solver (teacher) uses the "
            "game's built-in expert (heuristic or exact solver) so you can "
            "measure a knob/reward change's effect on expert play; random "
            "samples legal actions uniformly, a floor for comparison. "
            "Neither loads a language model -- that only happens in a real "
            "evolve run."
        ),
    },
    # --- 022: model / backend fields --------------------------------------
    "model_field": {
        "title": "Model (optional)",
        "body": (
            "Pick a tier preset or type any HF repo id / local path -- blank "
            "uses your hardware tier's default. Presets are official org "
            "repos only (LiquidAI, Qwen, google, nvidia). There is no "
            "official small Qwen3.6 -- those open weights start at 27B. "
            "8GB: stick to <=1B. Hub lookup is advisory only."
        ),
    },
    "backend_field": {
        "title": "Backend",
        "body": (
            "Which inference engine runs the model: tier default, "
            "transformers, transformers-4bit, or mlx. "
            "Leave it alone and behavior is unchanged. Every scoreboard row "
            "shows its resolved model and backend (if overridden) so A/B "
            "comparisons across model choices stay legible at a glance."
        ),
    },
    # --- Reward-code tab ---------------------------------------------------
    "reward_tab": {
        "title": "Reward code",
        "body": (
            "Write shape_reward(ctx) to reshape training-time reward -- it "
            "wraps the built-in formula, called once per decision. ctx has "
            "shared keys (default_reward, score, turn, terminated, "
            "truncated) plus per-game ALE keys (see the template's ctx "
            "reference). "
            "Doctrine: Steering must never be counted as model improvement. "
            "The gate always scores the raw, unshaped outcome, so this hook "
            "changes what reject_sft picks as good demonstrations, never "
            "how easy the gate is to pass."
        ),
    },
    # --- Evolve / scoreboard / links ---------------------------------------
    "evolve_button": {
        "title": "▶ evolve",
        "body": (
            "Launches the real rollout -> train -> eval -> gate loop (not "
            "the quick screen): each generation plays episodes with the "
            "current policy, trains on the top-quantile selected data, then "
            "EvalGate scores the new checkpoint on raw (unshaped) score and "
            "only promotes it to champion if it beats the previous "
            "generation without regressing on loop/invalid/entropy metrics. "
            "One evolve run at a time; a second attempt is refused, not "
            "queued."
        ),
    },
    "watch_link": {
        "title": "watch",
        "body": (
            "Embeds the live-play viewer for this experiment right in the "
            "page -- the episode stream (and, for Atari games, a "
            "re-simulated frame replay, capped at 4 concurrent replays "
            "server-wide). Plain click opens it inline; middle-click / "
            "ctrl-click / cmd-click opens the real page in a new tab "
            "instead."
        ),
    },
    "ab_button": {
        "title": "A/B",
        "body": (
            "Launches an exhibition: the SAME seeded episodes played first "
            "by the base model, then by the current champion, shown side "
            "by side with a live score strip. This is the workshop's money "
            "shot -- 'stock vs trained, 0 -> 1 A/B' -- one model in memory "
            "at a time, same 8GB-tier discipline as everywhere else."
        ),
    },
    "gens_link": {
        "title": "gens",
        "body": (
            "Opens one viewer panel per generation this run has produced, "
            "each filtered to that generation's episodes only. Useful for "
            "watching play quality change generation over generation, not "
            "just base-vs-champion. Same 4-concurrent-replay cap as watch/A-B "
            "for Atari games."
        ),
    },
    "play_again_button": {
        "title": "play",
        "body": (
            "Optional: replay one checkpoint (a generation number from "
            "gens/, or the current champion) with episodes / seed / "
            "temperature knobs. Writes under theater/play/ and opens the "
            "live viewer — no new training. Skip this for the default "
            "workshop path; evolve → A/B is enough."
        ),
    },
    "publish_button": {
        "title": "publish",
        "body": (
            "Pushes this experiment's per-generation datasets, and (if a "
            "generation has been promoted) the champion's LoRA adapter plus "
            "a generated model card, to your own Hugging Face account. Your "
            "token never leaves this machine except as an authenticated "
            "call to the Hub itself -- it is never logged, never in a "
            "subprocess command line, never committed. Disabled with a "
            "tooltip until a token is on file."
        ),
    },
    "signup_card": {
        "title": "sign up",
        "body": (
            "Name + optional Hugging Face token, stored locally on THIS "
            "machine only (~/.../playground/profile.json, mode 0600). "
            "Steps: (1) create a free account at huggingface.co "
            "(2) open Settings → Access Tokens "
            "(https://huggingface.co/settings/tokens) and make a write "
            "token (3) paste it here. Skip disables publish only; "
            "rollouts, evolve, and theater work with no token."
        ),
    },
    # --- Scoreboard columns --------------------------------------------
    "scoreboard_name": {
        "title": "name",
        "body": "The experiment's name, as you typed it when creating it -- also its run-id suffix (pg-<name>) and its config directory under runs/playground/.",
    },
    "scoreboard_model": {
        "title": "model",
        "body": (
            "Which model (and backend, in parentheses, if overridden) this "
            "experiment actually ran -- 'tier default' if you left the "
            "model/backend fields blank. Lets you compare experiments "
            "across model choices without opening experiment.json."
        ),
    },
    "scoreboard_episodes": {
        "title": "episodes",
        "body": "How many episodes this experiment has completed so far, counted from its rollout JSONL by grouping decisions under the same episode_id.",
    },
    "scoreboard_mean": {
        "title": "mean",
        "body": "Mean terminal score across this experiment's completed episodes, parsed from each episode's final outcome (\"score:<n>\"). The headline number for score-based games (the ALE pack).",
    },
    "scoreboard_median": {
        "title": "median",
        "body": "Median terminal score across completed episodes -- less sensitive than the mean to one lucky or unlucky outlier episode.",
    },
    "scoreboard_max": {
        "title": "max",
        "body": "The single best episode's terminal score so far -- a ceiling on what this experiment's policy has shown it can do, not a typical outcome.",
    },
    "scoreboard_actions": {
        "title": "top actions",
        "body": "The top 3 actions by share of all decisions made (e.g. \"FIRE 40%, LEFT 30%\"), the action_mix -- a quick tell for a collapsed policy (one action near 100%) versus varied play.",
    },
    "scoreboard_interventions": {
        "title": "interventions",
        "body": "Count of episodes where the doom-loop monitor stepped in at least once (reflect, backtrack, or truncate). A high count relative to episode count is a sign the policy is looping, not just playing badly.",
    },
    "scoreboard_status": {
        "title": "status",
        "body": "Whether the experiment's rollout/evolve subprocess is still running or has finished -- 'running' means the scoreboard numbers above are still partial and will keep updating every 3 seconds.",
    },
}

# --- "How it works" intro panel (design decision 4) ------------------------
# The ROLLOUT -> DATASET -> TRAIN -> EVAL -> GATE diagram + one-liners, lifted
# from README.md's own pipeline section so the two never drift silently out
# of sync (if you edit one, check the other).
INTRO_DIAGRAM = """\
ROLLOUT ──▶ DATASET ──▶ TRAIN ──▶ EVAL ──▶ GATE: promote / rollback
   ▲        (JSONL →   (GRPO or   (frozen         │
   │         parquet)  reject_sft) suites + ELO)  │
   └───────────── champion gen N+1 ◀───────────────"""

INTRO_STAGES: list[tuple[str, str]] = [
    ("ROLLOUT", "The model (or a teacher) plays episodes in a text-native game; every decision is recorded."),
    ("DATASET", "Recorded decisions are written to JSONL, then converted to parquet -- one reusable dataset per generation."),
    ("TRAIN", "The model fine-tunes on its own best experience: reject_sft for teacher/pack warm-start, then GRPO (TRL + LoRA) on transformers across tiers."),
    ("EVAL", "The new checkpoint plays a frozen, unshaped benchmark suite -- the same seeds every generation, for a fair comparison."),
    ("GATE", "A new generation is only promoted to champion if it demonstrably beats the previous one on that frozen suite; otherwise it's rolled back."),
]

INTRO_ANTI_DOOM_LOOP = (
    "Anti-doom-loop by design: a DoomLoopMonitor escalates reflect → backtrack "
    "→ truncate during play, and the EvalGate only promotes a generation that "
    "beats the champion without regressing on loop/invalid/entropy metrics -- "
    "steering (teachers, pruners, monitor interventions) is never counted as "
    "model improvement."
)

# Verbatim doctrine string (docs/HYBRID_RL.md, CODING_GUIDELINE.md invariant
# 2) -- the reward-code card and the coverage test both need the EXACT
# phrasing, so it lives here once rather than being retyped.
GATE_PURITY_DOCTRINE = "Steering must never be counted as model improvement."
