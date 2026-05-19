# LiveActionAOV — split sidecar output helpers (Desk / Colab lane).

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from live_action_aov.io.channels import (
    AO_CHANNELS,
    DEPTH_CHANNELS,
    FLOW_CHANNELS,
    MATTE_CHANNELS,
    NORMAL_CHANNELS,
    POSITION_CHANNELS,
    is_mask_channel,
)

OutputLayout = Literal["combined", "split_folders"]

SPLIT_FOLDER_NAMES: tuple[str, ...] = ("depth", "normals", "flow", "matte", "vitmatte")

_DEPTH = frozenset(DEPTH_CHANNELS) | frozenset(POSITION_CHANNELS)
_NORMALS = frozenset(NORMAL_CHANNELS) | frozenset(AO_CHANNELS)
_FLOW = frozenset(FLOW_CHANNELS)
_MATTE = frozenset(MATTE_CHANNELS)


def categorize_channels(channels: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Bucket per-frame channels into depth / normals / flow / matte folders."""
    out: dict[str, dict[str, np.ndarray]] = {name: {} for name in SPLIT_FOLDER_NAMES}
    for name, arr in channels.items():
        if name.startswith("vitmatte."):
            out["vitmatte"][name] = arr
        elif is_mask_channel(name) or name in _MATTE:
            out["matte"][name] = arr
        elif name in _FLOW:
            out["flow"][name] = arr
        elif name in _NORMALS:
            out["normals"][name] = arr
        elif name in _DEPTH:
            out["depth"][name] = arr
        else:
            out["depth"][name] = arr
    return {k: v for k, v in out.items() if v}


def split_sidecar_pattern(sequence_pattern: str, category: str) -> str:
    """``hero.####.jpg`` → ``hero.depth.{frame:04d}.exr`` (sidecars are always EXR)."""
    token = f".{category}."
    tail = ".exr"
    m = re.search(r"#+", sequence_pattern)
    if m:
        width = len(m.group(0))
        base = sequence_pattern[: m.start()]
        if base.endswith("."):
            base_ = base[:-1] + token
        else:
            base_ = base + token
        return f"{base_}{{frame:0{width}d}}{tail}"
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    if m:
        width = int(m.group(1)) if m.group(1) else 4
        base = sequence_pattern[: m.start()]
        if base.endswith("."):
            base_ = base[:-1] + token
        else:
            base_ = base + token
        return f"{base_}{{frame:0{width}d}}{tail}"
    stem = sequence_pattern
    for ext in (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
        if stem.lower().endswith(ext):
            return stem[: -len(ext)] + f".{category}{tail}"
    return stem + f".{category}{tail}"


__all__ = [
    "OutputLayout",
    "SPLIT_FOLDER_NAMES",
    "categorize_channels",
    "split_sidecar_pattern",
]
