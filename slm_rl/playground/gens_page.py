"""The all-generations grid page (plan 020): one panel per generation
present in an experiment's run, each an iframe of the EXISTING live-play
viewer page pointed at `/watch/<name>/?gen=N`. No new viewer routes needed
-- the `gen` query-string forwarding added to `webui/page.py` (plan 020) is
what makes each panel show only that generation's episodes; the `/watch/`
mounting itself is untouched (plan 014).

Rendered server-side (the generation list is discovered by globbing
`generations/gen_*`, see playground/server.py::_gens_list) so the client
never has to know which gens exist in advance.
"""

from __future__ import annotations

import html

_ATARI_REPLAY_CAP_NOTE = (
    "Atari game screens are capped at 4 concurrent live replays "
    "(shared server-wide) — panels beyond that show the text stream only, "
    "which is expected, not a bug."
)


def render_gens_page(name: str, gens: list[int]) -> str:
    if not gens:
        panels = '<p class="empty">no generations yet for this experiment</p>'
    else:
        panels = "\n".join(_panel(name, g) for g in gens)
    return _TEMPLATE.format(name=html.escape(name), panels=panels, note=_ATARI_REPLAY_CAP_NOTE)


def _panel(name: str, gen: int) -> str:
    from urllib.parse import quote

    src = f"/watch/{quote(name)}/?gen={gen}"
    return (
        '<div class="panel">'
        f'<div class="panel-head">gen {gen}</div>'
        f'<iframe src="{src}" title="gen {gen}"></iframe>'
        "</div>"
    )


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SLM-RL — all-gens grid — {name}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    color-scheme: dark;
    --bg: #14161a; --card: #1d2026; --border: #2c313a;
    --text: #e6e8eb; --muted: #8b93a1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1rem; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 1.1rem; margin: 0 0 0.3rem 0; }}
  .note {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem; max-width: 60rem; }}
  .empty {{ color: var(--muted); }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 1rem;
  }}
  .panel {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
  }}
  .panel-head {{
    padding: 0.4rem 0.7rem; font-size: 0.8rem; color: var(--muted);
    border-bottom: 1px solid var(--border);
  }}
  .panel iframe {{
    width: 100%; height: 420px; border: 0; display: block; background: #0f1115;
  }}
</style>
</head>
<body>
<h1>SLM-RL — all-gens grid — {name}</h1>
<p class="note">{note}</p>
<div class="grid">
{panels}
</div>
</body>
</html>
"""
