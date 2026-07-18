# syntax=docker/dockerfile:1
#
# Multi-stage build: two builder stages compile the .venv for each hardware
# tier; two slim runtime stages copy just the venv + source.
#
# Build targets:
#   docker build --target playground -t slm-rl:playground .   (default, CPU)
#   docker build --target cuda       -t slm-rl:cuda .          (GPU host)

ARG PYTHON_BASE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ---------------------------------------------------------------------------
# builder-cpu: atari + cpu-train (CPU torch/transformers/trl/peft) + dev
# ---------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS builder-cpu

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency manifests first so this (slow) layer is cached across
# source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra atari --extra cpu-train --extra dev

COPY . .
RUN uv sync --frozen --extra atari --extra cpu-train --extra dev

# ---------------------------------------------------------------------------
# builder-cuda: atari + cuda (torch, prebuilt wheels) + dev extras
# ---------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS builder-cuda

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra atari --extra cuda --extra dev

COPY . .
RUN uv sync --frozen --extra atari --extra cuda --extra dev

# ---------------------------------------------------------------------------
# playground: default runtime target (CPU tier: quick experiments + evolve)
# ---------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS playground

# libgomp1: torch OpenMP / some wheels expect it on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 slm && useradd --uid 1000 --gid slm --create-home slm

WORKDIR /app
COPY --from=builder-cpu --chown=slm:slm /app /app
# COPY --chown only chowns the copied contents; /app itself (created by
# WORKDIR above) stays root-owned, which breaks in-container .pytest_cache.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chown slm:slm /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /home/slm/.cache/uv /home/slm/.cache/huggingface \
    && chown -R slm:slm /home/slm/.cache

ENV PATH="/app/.venv/bin:$PATH"

# Stay root for the entrypoint so named-volume cache mounts can be chowned,
# then drop to uid 1000 (slm) before exec'ing the CMD.
ENTRYPOINT ["docker-entrypoint.sh"]
EXPOSE 8780
CMD ["slm-rl", "playground", "--host", "0.0.0.0", "--port", "8780"]

# ---------------------------------------------------------------------------
# cuda: GPU runtime target (nvidia container toolkit injects the driver)
# ---------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS cuda

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 slm && useradd --uid 1000 --gid slm --create-home slm

WORKDIR /app
COPY --from=builder-cuda --chown=slm:slm /app /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chown slm:slm /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /home/slm/.cache/uv /home/slm/.cache/huggingface \
    && chown -R slm:slm /home/slm/.cache

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["docker-entrypoint.sh"]
EXPOSE 8780
CMD ["slm-rl", "playground", "--host", "0.0.0.0", "--port", "8780"]
