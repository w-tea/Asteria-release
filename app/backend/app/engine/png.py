from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_solid_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(1, int(width))
    height = max(1, int(height))
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw)),
            _chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
