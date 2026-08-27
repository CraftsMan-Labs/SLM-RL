# Workshop diagrams

Mermaid sources (`.mmd`) are the reviewable graph. `render.py` writes the
dark-theme SVGs Colab and the Vue deck actually display.

```bash
python docs/workshop/diagrams/render.py
python docs/workshop/sync_presentation_assets.py
```

Do not put `%%html` / mermaid.js in the generated notebook. Attendees should
see the image, not the diagram code.
