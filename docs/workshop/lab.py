"""Runtime helpers for the Colab workshop notebook.

Imported by cells generated from ``build_nb.py``. Keep this module free of
torch / slm_rl imports so tests and the notebook can load it before the
heavy stack is ready.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from typing import Any

WORKSHOP_GAMES = ("boxing", "space-invaders", "freeway", "demon-attack")
MODES = ("QUICK", "FULL")
PRECISIONS = ("q4", "fp16", "auto")
PARSE_STATUSES = ("ok", "retry_ok", "fallback_random")
TRAIN_STRATEGIES = ("reject_sft", "grpo", "both")
GATE_GUESSES = ("promote", "reject")

BACKEND_FOR_PRECISION = {
    "q4": "transformers-4bit",
    "fp16": "transformers",
    "auto": None,
}

# QUICK stays workshop-length. FULL is labeled as long-running in the notebook.
BOUNDS = {
    "QUICK": {
        "seed": (0, 99),
        "dqn_decisions": (100, 8_000),
        "generations": (1, 2),
        "episodes_per_generation": (1, 8),
        "temperature": (0.0, 1.5),
        "max_tokens": (8, 64),
        "theater_episodes": (1, 2),
        "eval_limit": (1, 8),
        "replay_every": (1, 8),
        "row_index": (0, 10_000),
        "action_index": (1, 18),
        "max_turns": (8, 64),
        "miss_reward": (-1.0, 0.0),
        "win_reward": (0.1, 2.0),
        "min_improvement": (0.0, 2.0),
    },
    "FULL": {
        "seed": (0, 99_999),
        "dqn_decisions": (1_000, 300_000),
        "generations": (1, 3),
        "episodes_per_generation": (8, 50),
        "temperature": (0.0, 1.5),
        "max_tokens": (8, 128),
        "theater_episodes": (1, 5),
        "eval_limit": (8, 50),
        "replay_every": (1, 8),
        "row_index": (0, 10_000),
        "action_index": (1, 18),
        "max_turns": (32, 2500),
        "miss_reward": (-1.0, 0.0),
        "win_reward": (0.1, 2.0),
        "min_improvement": (0.0, 2.0),
    },
}


def clamp_int(value: Any, lo: int, hi: int, name: str) -> int:
    """Coerce to int and clip to [lo, hi]."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = lo
    clipped = max(lo, min(hi, parsed))
    if clipped != parsed:
        print(f"clamped {name}: {parsed} → {clipped} (allowed {lo}–{hi})")
    return clipped


def clamp_float(value: Any, lo: float, hi: float, name: str) -> float:
    """Coerce to float and clip to [lo, hi]."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = lo
    clipped = max(lo, min(hi, parsed))
    if abs(clipped - parsed) > 1e-12:
        print(f"clamped {name}: {parsed} → {clipped} (allowed {lo}–{hi})")
    return clipped


def bound(mode: str, name: str) -> tuple[float, float]:
    table = BOUNDS.get(mode, BOUNDS["QUICK"])
    return table[name]


def sanitize_run_name(name: str) -> str:
    """Filesystem-safe run id. Empty / junk becomes ``colab``."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip("-._")
    return cleaned[:40] or "colab"


def resolve_backend(precision: str) -> str | None:
    if precision not in BACKEND_FOR_PRECISION:
        print(f"unknown PRECISION={precision!r}; falling back to q4")
        return BACKEND_FOR_PRECISION["q4"]
    return BACKEND_FOR_PRECISION[precision]


def resolve_game(name: str) -> str:
    key = (name or "").strip().lower()
    if key in WORKSHOP_GAMES:
        return key
    print(f"unknown GAME={name!r}; falling back to boxing")
    return "boxing"


def resolve_mode(name: str) -> str:
    key = (name or "").strip().upper()
    if key in MODES:
        return key
    print(f"unknown MODE={name!r}; falling back to QUICK")
    return "QUICK"


def resolve_choice(value: str, allowed: Iterable[str], default: str) -> str:
    key = (value or "").strip()
    lookup = {item.lower(): item for item in allowed}
    if key.lower() in lookup:
        return lookup[key.lower()]
    print(f"unknown choice {value!r}; falling back to {default!r}")
    return default


def pick_action(legal: list[Any], raw: str) -> Any:
    """Resolve a typed id, label, or 1-based index against ``legal`` actions."""
    if not legal:
        raise ValueError("no legal actions")
    text = (raw or "").strip()
    if not text:
        return legal[0]
    for action in legal:
        if str(getattr(action, "id", "")) == text:
            return action
        if str(getattr(action, "label", "")).lower() == text.lower():
            return action
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(legal):
            return legal[idx - 1]
    print(f"action {raw!r} not in the menu — using {legal[0].id!r}")
    return legal[0]


def grade(prediction: str, actual: str) -> str:
    """One-line prediction readout. Case-insensitive."""
    pred = (prediction or "").strip().lower()
    act = (actual or "").strip().lower()
    if pred in {"", "not sure", "skip"}:
        return f"result: {actual}"
    if pred == act:
        return f"you predicted {prediction} — correct."
    return f"you predicted {prediction}; the result was {actual}."


def scorecard(title: str, rows: list[tuple[str, Any]]) -> None:
    """Compact two-column result table."""
    print(f"=== {title} ===")
    if not rows:
        print("  (empty)")
        return
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        print(f"  {str(key):<{width}}  {value}")


def require_names(namespace: dict[str, Any], *names: str) -> None:
    missing = [name for name in names if name not in namespace or namespace[name] is None]
    if missing:
        raise RuntimeError(
            "Run the earlier cells first. Missing: " + ", ".join(missing)
        )


def skip_gates_enabled(flag: bool | None = None) -> bool:
    """True only for the environment-based instructor/automation hatch."""
    if flag is True:
        return True
    env = (os.environ.get("WORKSHOP_SKIP_GATES") or "").strip().lower()
    return env in {"1", "true", "yes"}


def ask(
    prompt: str,
    *,
    allowed: Iterable[str] | None = None,
    default: str = "",
    skip: bool | None = None,
    reader: Callable[[str], str] | None = None,
) -> str:
    """Block Runtime → Run all until the attendee types.

    Colab form values never pause the kernel. ``input()`` does. Blank answers
    never continue in participant mode. When ``skip`` (or the private
    ``WORKSHOP_SKIP_GATES`` environment hatch) is set, return ``default`` so
    instructor automation and CI can fly through.
    """
    options = tuple(allowed) if allowed is not None else None
    if skip_gates_enabled(skip):
        if default:
            if options:
                lookup = {item.lower(): item for item in options}
                return lookup.get(default.lower(), default)
            return default
        return options[0] if options else ""

    read = reader or input
    hint = f" [{' / '.join(options)}]" if options else ""
    suffix = ":" if not prompt.rstrip().endswith("?") else ""
    label = f"{prompt.rstrip()}{hint}{suffix} "
    while True:
        text = (read(label) or "").strip()
        if options:
            lookup = {item.lower(): item for item in options}
            if text.lower() in lookup:
                return lookup[text.lower()]
            print("  type one of: " + ", ".join(options))
            continue
        if text:
            return text
        print("  type something to continue — Runtime → Run all is paused on purpose.")


def ask_hf_token(
    *,
    default: str = "",
    skip: bool | None = None,
    reader: Callable[[str], str] | None = None,
) -> str:
    """Prompt for a Hugging Face token. Empty is allowed. Never print the value.

    A filled ``default`` (Colab form or prior env) is returned without prompting
    so re-runs and instructor Secrets do not block. Otherwise ``getpass``
    pauses Runtime → Run all until the attendee pastes a token or hits Enter.
    """
    seeded = (default or "").strip()
    if skip_gates_enabled(skip):
        return seeded
    if seeded:
        return seeded
    if reader is None:
        try:
            from getpass import getpass

            read: Callable[[str], str] = getpass
        except Exception:
            read = input
    else:
        read = reader
    return (
        read(
            "Hugging Face token (hf_… write scope — "
            "https://huggingface.co/settings/tokens; Enter to skip): "
        )
        or ""
    ).strip()


def new_card(name: str) -> dict[str, Any]:
    cleaned = (name or "").strip() or "anonymous"
    return {"name": cleaned[:40], "guesses": []}


def ensure_card(namespace: dict[str, Any]) -> dict[str, Any]:
    card = namespace.get("CARD")
    if not isinstance(card, dict) or "guesses" not in card:
        card = new_card(str(namespace.get("DISPLAY_NAME") or "anonymous"))
        namespace["CARD"] = card
    return card


def record_guess(card: dict[str, Any], key: str, guess: str, actual: str) -> dict[str, Any]:
    pred = (guess or "").strip().lower()
    act = (actual or "").strip().lower()
    skipped = pred in {"", "not sure", "skip"}
    row = {
        "key": key,
        "guess": guess,
        "actual": actual,
        "correct": (not skipped) and pred == act,
        "skipped": skipped,
    }
    card.setdefault("guesses", []).append(row)
    return row


def show_card(card: dict[str, Any]) -> None:
    """Print the running prediction card. Safe to call with an empty card."""
    name = str(card.get("name") or "anonymous")
    guesses = list(card.get("guesses") or [])
    scored = [row for row in guesses if not row.get("skipped")]
    hits = sum(1 for row in scored if row.get("correct"))
    print(f"=== {name}'s scorecard ===")
    if not guesses:
        print("  (no guesses yet)")
        return
    width = max(len(str(row.get("key") or "")) for row in guesses)
    for row in guesses:
        mark = "skip" if row.get("skipped") else ("hit" if row.get("correct") else "miss")
        print(
            f"  {str(row.get('key') or ''):<{width}}  {mark:4s}  "
            f"guess={row.get('guess')!r}  actual={row.get('actual')!r}"
        )
    print(f"  {hits}/{len(scored)} correct" if scored else "  no scored guesses yet")


# ---------------------------------------------------------------------------
# WorkShopTracker progress (participant API key — never an admin/master key)
# ---------------------------------------------------------------------------

WST_DEFAULT_BASE_URL = "https://workshop.craftsmanlabs.net"
_TRACKER: Any | None = None


class _HTTPStatusError(Exception):
    """Minimal stand-in so ``_http_status`` can read ``response.status_code``."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.response = type("R", (), {"status_code": int(status_code)})()
        super().__init__(detail or f"HTTP {status_code}")


class WorkshopTracker:
    """Tiny participant client (stdlib only — no private-repo pip install)."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        import json
        import urllib.error
        import urllib.request

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    detail = str(
                        parsed.get("message")
                        or parsed.get("detail")
                        or parsed.get("error")
                        or raw
                    )
                else:
                    detail = raw
            except Exception:
                detail = str(exc)
            raise _HTTPStatusError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error contacting WorkShopTracker: {exc}") from exc

        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/me")

    def start(self, key: str) -> dict[str, Any]:
        return self._request(
            "PUT", f"/api/v1/me/checkpoints/{key}", {"status": "in_progress"}
        )

    def complete(self, key: str) -> dict[str, Any]:
        return self._request(
            "PUT", f"/api/v1/me/checkpoints/{key}", {"status": "completed"}
        )


def bind_tracker(tracker: Any) -> Any:
    """Remember the live ``WorkshopTracker`` client for later chapter cells."""
    global _TRACKER
    _TRACKER = tracker
    return tracker


def get_tracker() -> Any | None:
    return _TRACKER


def load_wst_api_key() -> str:
    """Load the participant key from a Colab Secret or ``WST_API_KEY`` env.

    Never prints the secret. Returns ``""`` when unset.
    """
    try:
        from google.colab import userdata  # type: ignore

        secret = (userdata.get("WST_API_KEY") or "").strip()
        if secret:
            return secret
    except Exception:
        pass
    return (os.environ.get("WST_API_KEY") or "").strip()


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if status is not None else None


def _progress_error(key: str, status: str, exc: BaseException) -> RuntimeError:
    code = _http_status(exc)
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if code == 401 or "api key" in lowered or "invalid or revoked" in lowered:
        return RuntimeError(
            "WorkshopTracker rejected the API key. Generate a participant key on "
            f"{WST_DEFAULT_BASE_URL}/workshop (shown once) and store it as Colab "
            "Secret WST_API_KEY — never paste an admin/master key into the notebook."
        )
    if code == 404:
        return RuntimeError(
            f"Checkpoint {key!r} is not on this workshop run. "
            "Ask the facilitator to sync project checkpoints (chapter-0 … chapter-13) "
            "and create a fresh run."
        )
    if code == 403 or "not open yet" in lowered:
        return RuntimeError(
            f"Checkpoint {key!r} is held. Wait for the facilitator to open this "
            "chapter on the run dashboard, then re-run this cell."
        )
    return RuntimeError(f"Could not mark {key} as {status}: {detail}")


def start_chapter(number: int) -> None:
    """Report ``in_progress`` for ``chapter-N``. No-op when no tracker is bound."""
    key = f"chapter-{int(number)}"
    tracker = _TRACKER
    if tracker is None:
        if skip_gates_enabled():
            return
        print(f"progress skipped (no tracker): would start {key}")
        return
    try:
        tracker.start(key)
    except Exception as exc:  # noqa: BLE001 — surface as workshop-facing error
        raise _progress_error(key, "in_progress", exc) from exc
    print(f"progress: {key} → in_progress")


def complete_chapter(number: int) -> None:
    """Report ``completed`` for ``chapter-N``. No-op when no tracker is bound."""
    key = f"chapter-{int(number)}"
    tracker = _TRACKER
    if tracker is None:
        if skip_gates_enabled():
            return
        print(f"progress skipped (no tracker): would complete {key}")
        return
    try:
        tracker.complete(key)
    except Exception as exc:  # noqa: BLE001 — surface as workshop-facing error
        raise _progress_error(key, "completed", exc) from exc
    print(f"progress: {key} → completed")


def connect_workshop_tracker(
    *,
    join_url: str = "",
    base_url: str = WST_DEFAULT_BASE_URL,
    api_key: str | None = None,
    require: bool = True,
) -> Any | None:
    """Bind a participant ``WorkshopTracker`` (stdlib HTTP — no pip install).

    Returns the client, or ``None`` when ``require`` is false and no key is set
    (CI / offline). Raises with join instructions when ``require`` and the key
    is missing or invalid. Never prints the API key.
    """
    key = (api_key if api_key is not None else load_wst_api_key()).strip()
    join = (join_url or "").strip() or f"{base_url.rstrip('/')}/join/<run-slug>"
    if not key:
        if not require or skip_gates_enabled():
            print("progress skipped: no WST_API_KEY (offline / instructor skip).")
            return None
        raise RuntimeError(
            "Missing participant API key.\n"
            f"1. Open {join}\n"
            "2. Sign in, join the run, click Generate key on /workshop\n"
            "3. Save the secret (shown once) as Colab Secret WST_API_KEY\n"
            "   (or export WST_API_KEY in the runtime env).\n"
            "Do not use an admin/master key in this notebook."
        )

    tracker = WorkshopTracker(base_url=base_url.rstrip("/"), api_key=key)
    try:
        me = tracker.me()
    except Exception as exc:  # noqa: BLE001
        code = _http_status(exc)
        if code in {401, 403}:
            raise RuntimeError(
                "WST_API_KEY was rejected. Generate a *participant* key after joining "
                f"the run at {join}, store it as Colab Secret WST_API_KEY, and re-run."
            ) from exc
        raise RuntimeError(
            f"Could not reach WorkShopTracker at {base_url.rstrip('/')}: {exc}"
        ) from exc

    run = me.get("run") if isinstance(me, dict) else None
    if not run:
        raise RuntimeError(
            "This API key is not enrolled in a workshop run. "
            f"Open {join}, join the run, then Generate key on /workshop."
        )
    bind_tracker(tracker)
    run_name = run.get("name") if isinstance(run, dict) else run
    progress = me.get("progress") if isinstance(me, dict) else None
    n = len(progress) if isinstance(progress, list) else "?"
    print(f"tracker connected — run={run_name!r}, checkpoints={n}")
    print("Completion is reported telemetry for the facilitator dashboard, not a grade.")
    return tracker
