"""ViTMatte JPEG plate cache round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from live_action_aov.passes.matte.vitmatte_plate_cache import (
    build_plate_jpeg_cache,
    read_plate_jpeg,
)


def test_plate_jpeg_cache_roundtrip(tmp_path: Path) -> None:
    h, w = 32, 48
    frame_range = (1001, 1003)

    def read_frame(frame_idx: int) -> tuple[np.ndarray, dict]:
        v = (frame_idx - 1001) / 10.0
        rgb = np.full((h, w, 3), v, dtype=np.float32)
        return rgb, {}

    build_plate_jpeg_cache(read_frame, frame_range, tmp_path, jpeg_quality=90)
    out = read_plate_jpeg(tmp_path, 1002)
    assert out.shape == (h, w, 3)
    assert abs(float(out[0, 0, 0]) - 0.1) < 0.02
