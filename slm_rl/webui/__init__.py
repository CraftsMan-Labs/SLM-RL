"""Read-only live-play viewer: tail a run's rollout JSONLs and stream them to
a browser. Stdlib only (see CODING_GUIDELINE invariant 1 / 5) — never import
torch/transformers/pyarrow/streamlit here. Launch: `slm-rl watch --run <id>`.
"""

from __future__ import annotations
