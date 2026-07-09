"""OpenEnv bridge (optional extra [openenv], version-pinned — OpenEnv is
experimental and breaking; churn stays contained in this module).

Two directions:
- `make_app(game_cls)`: serve any of our games as an OpenEnv FastAPI app
  (openenv.core.env_server.create_app) so TRL's `environment_factory` path
  works (Phase 5).
- Client mode for OpenEnv-hosted envs (e.g. the Atari Docker image from the
  user's reference example `AtariEnv.from_docker_image(...)`) wrapped back
  into our `Game` contract.
"""

from __future__ import annotations

from slm_rl.games.base import Game


def make_app(game_cls: type[Game]):
    raise NotImplementedError("Phase 5")


def wrap_openenv_client(base_url_or_image: str) -> type[Game]:
    raise NotImplementedError("Phase 5")
