"""Split sidecar channel bucketing."""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import CH_N_X, CH_Z, CH_MOTION_X, CH_MATTE_R, MASK_PREFIX
from live_action_aov.io.split_channels import categorize_channels, split_sidecar_pattern


def test_categorize_channels_buckets() -> None:
    ch = {
        CH_Z: np.zeros((4, 4), dtype=np.float32),
        CH_N_X: np.zeros((4, 4), dtype=np.float32),
        CH_MOTION_X: np.zeros((4, 4), dtype=np.float32),
        CH_MATTE_R: np.zeros((4, 4), dtype=np.float32),
        f"{MASK_PREFIX}person": np.zeros((4, 4), dtype=np.float32),
    }
    split = categorize_channels(ch)
    assert CH_Z in split["depth"]
    assert CH_N_X in split["normals"]
    assert CH_MOTION_X in split["flow"]
    assert CH_MATTE_R in split["matte"]
    assert f"{MASK_PREFIX}person" in split["matte"]


def test_split_sidecar_pattern_jpg() -> None:
    pat = split_sidecar_pattern("TB_005_010.####.jpg", "depth")
    assert pat == "TB_005_010.depth.{frame:04d}.exr"
