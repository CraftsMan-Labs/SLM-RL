#!/bin/sh
# Runtime entrypoint for playground/cuda images.
# Named volumes mount as root; chown caches then drop to the non-root app user.
set -eu

mkdir -p /home/slm/.cache/uv /home/slm/.cache/huggingface
chown -R slm:slm /home/slm/.cache

if [ "$(id -u)" = "0" ]; then
  # setpriv does not reset HOME; uv/uvx otherwise look under /root and fail.
  export HOME=/home/slm
  export USER=slm
  export LOGNAME=slm
  exec setpriv --reuid=slm --regid=slm --init-groups -- "$@"
fi

exec "$@"
