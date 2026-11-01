from __future__ import annotations


def normalized_h3_frame_count(duration_seconds: float, fps: float) -> tuple[int, int]:
    requested = max(1, int(round(duration_seconds * fps)))
    blocks = max(0, int(round((requested - 5) / 17)))
    actual = 17 * blocks + 5
    return requested, actual
