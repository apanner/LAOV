"""MatteFlowTemporal — RAFT keyframe propagation for matte channels."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from live_action_aov.io.channels import CH_MATTE_A
from live_action_aov.post.matte_flow_temporal import MatteFlowTemporal
from live_action_aov.shared.optical_flow.cache import FlowCache


def test_zero_flow_propagates_keyframe_to_middle_frame() -> None:
    h, w = 16, 16
    key = np.full((h, w), 0.9, dtype=np.float32)
    gap = np.zeros((h, w), dtype=np.float32)
    per_frame = {
        1001: {CH_MATTE_A: key.copy()},
        1002: {CH_MATTE_A: gap.copy()},
        1003: {CH_MATTE_A: gap.copy()},
        1004: {CH_MATTE_A: key.copy()},
    }
    cache = FlowCache()
    zero = np.zeros((2, h, w), dtype=np.float32)
    for f in (1001, 1002, 1003, 1004):
        cache.put("s", f, "forward", zero)
        cache.put("s", f, "backward", zero)

    out = MatteFlowTemporal(
        {
            "applied_to": [CH_MATTE_A],
            "keyframe_stride": 4,
            "final_ema_alpha": 0.0,
        }
    ).apply(per_frame, cache, "s")

    assert np.allclose(out[1001][CH_MATTE_A], key)
    assert np.allclose(out[1004][CH_MATTE_A], key)
    assert float(out[1002][CH_MATTE_A].mean()) > 0.7


def test_keyframes_unchanged() -> None:
    h, w = 8, 8
    a = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    b = np.linspace(1, 0, h * w, dtype=np.float32).reshape(h, w)
    per_frame = {
        1: {CH_MATTE_A: a},
        2: {CH_MATTE_A: np.zeros((h, w), dtype=np.float32)},
        3: {CH_MATTE_A: b},
    }
    cache = FlowCache()
    z = np.zeros((2, h, w), dtype=np.float32)
    for f in (1, 2, 3):
        cache.put("s", f, "forward", z)
        cache.put("s", f, "backward", z)

    out = MatteFlowTemporal(
        {"applied_to": [CH_MATTE_A], "keyframe_stride": 2, "final_ema_alpha": 0.0}
    ).apply(per_frame, cache, "s")
    assert np.allclose(out[1][CH_MATTE_A], a)
    assert np.allclose(out[3][CH_MATTE_A], b)
