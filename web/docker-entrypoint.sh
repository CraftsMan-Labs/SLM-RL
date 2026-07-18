#!/bin/sh
set -e
# Named volume can lag behind package-lock after dependency bumps.
if [ ! -x node_modules/.bin/vite ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  echo "[web] syncing node_modules…"
  npm ci
  cp -f package-lock.json node_modules/.package-lock.json 2>/dev/null || true
fi
exec "$@"
