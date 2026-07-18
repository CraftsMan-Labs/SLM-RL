"""Atari via ALE (workshop keepers: Boxing, Space Invaders, Freeway,
Demon Attack).

Default execution is in-process `ale-py` (no Docker, 8GB-safe). RAM decoding
is a per-game plugin file under `ram_maps/`. Real game classes register via
their own modules, not here.
"""
