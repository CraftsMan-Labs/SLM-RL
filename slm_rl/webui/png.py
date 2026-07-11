"""Minimal 8-bit RGB PNG encoder — stdlib only (`zlib`, `struct`).

No PIL, no new dependency: the live game screen (plan 010) needs to turn
numpy RGB frames into bytes a `<img>` tag can display, and a baseline PNG
(no filtering, one zlib stream) is ~30 lines. Not a general-purpose encoder:
only handles 8-bit truecolor (color type 2), no interlacing, no palettes.
"""

from __future__ import annotations

import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_rgb(pixels: bytes, width: int, height: int) -> bytes:
    """Encode raw 8-bit RGB pixel bytes (row-major, no padding, 3 bytes/px)
    as a PNG. `len(pixels)` must equal `width * height * 3`.
    """
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(
            f"expected {expected} bytes for {width}x{height} RGB, got {len(pixels)}"
        )

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,  # bit depth
        2,  # color type: truecolor (RGB)
        0,  # compression method
        0,  # filter method
        0,  # interlace method
    )

    # Filter byte 0 ("None") prefixed to every scanline, per the PNG spec.
    stride = width * 3
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        start = row * stride
        raw += pixels[start : start + stride]

    idat = zlib.compress(bytes(raw))

    return (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
