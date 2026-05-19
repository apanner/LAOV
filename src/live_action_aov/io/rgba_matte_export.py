# LiveActionAOV — AI Matte stage EXRs: one RGBA file per hero slot (Nuke-friendly).

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.channels import MATTE_CHANNELS
from live_action_aov.io.writers.exr import ExrSidecarWriter

_log = logging.getLogger(__name__)

RGBA_CHANNELS = ("R", "G", "B", "A")
METADATA_NAMESPACE = "liveaov"

_MATTE_PASSES = frozenset({"sam3_matte", "birefnet_refiner", "vitmatte_refiner", "rvm_refiner"})

_SLOT_TO_CHANNEL: dict[str, tuple[str, ...]] = {
    "sam3_matte": MATTE_CHANNELS,
    "birefnet_refiner": MATTE_CHANNELS,
    "vitmatte_refiner": ("vitmatte.r", "vitmatte.g", "vitmatte.b", "vitmatte.a"),
    "rvm_refiner": MATTE_CHANNELS,
}


def pack_alpha_to_rgba(alpha: np.ndarray) -> dict[str, np.ndarray]:
    """Grayscale matte in RGB; A holds the matte (standard comp delivery)."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return {"R": a, "G": a.copy(), "B": a.copy(), "A": a.copy()}


def rgba_sidecar_pattern(sequence_pattern: str, slot: str) -> str:
    """``plate.####.exr`` → ``plate_matte_r.{frame:04d}.exr``."""
    token = f"_matte_{slot}."
    tail = ".exr"
    m = re.search(r"#+", sequence_pattern)
    if m:
        width = len(m.group(0))
        base = sequence_pattern[: m.start()]
        if base.endswith("."):
            base = base[:-1]
        return f"{base}{token}{{frame:0{width}d}}{tail}"
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    if m:
        width = int(m.group(1)) if m.group(1) else 4
        base = sequence_pattern[: m.start()]
        if base.endswith("."):
            base = base[:-1]
        return f"{base}{token}{{frame:0{width}d}}{tail}"
    stem = sequence_pattern
    for ext in (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return f"{stem}{token}{{frame:04d}}{tail}"


def _slot_channel_map(pass_name: str, channels: dict[str, np.ndarray]) -> dict[str, str]:
    """Map hero slot letter → source channel name on this pass."""
    names = _SLOT_TO_CHANNEL.get(pass_name, MATTE_CHANNELS)
    slot_letters = ("r", "g", "b", "a")
    out: dict[str, str] = {}
    for slot, ch in zip(slot_letters, names, strict=False):
        if ch in channels:
            out[slot] = ch
    return out


def write_rgba_matte_stage_export(
    *,
    pass_name: str,
    export_subdir: str,
    output_root: Path,
    sequence_pattern: str,
    per_frame_channels: dict[int, dict[str, np.ndarray]],
    shot_name: str,
    pixel_aspect: float = 1.0,
    attrs_extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write ``matte_<stage>/plate_matte_{r|g|b|a}.####.exr`` with channels R,G,B,A only."""
    if pass_name not in _MATTE_PASSES:
        return None

    stage_dir = (output_root / export_subdir).resolve()
    writer = ExrSidecarWriter()
    attrs_base: dict[str, Any] = {
        f"{METADATA_NAMESPACE}/stage": export_subdir,
        f"{METADATA_NAMESPACE}/pass": pass_name,
        f"{METADATA_NAMESPACE}/shot": shot_name,
        f"{METADATA_NAMESPACE}/format": "rgba_matte",
    }
    if attrs_extra:
        attrs_base.update(attrs_extra)

    files_written = 0
    for frame_idx, channels in sorted(per_frame_channels.items()):
        slot_map = _slot_channel_map(pass_name, channels)
        if not slot_map:
            continue
        for slot, ch_name in slot_map.items():
            alpha = channels.get(ch_name)
            if alpha is None:
                continue
            rgba = pack_alpha_to_rgba(alpha)
            attrs = dict(attrs_base)
            attrs[f"{METADATA_NAMESPACE}/frame"] = frame_idx
            attrs[f"{METADATA_NAMESPACE}/slot"] = slot
            attrs[f"{METADATA_NAMESPACE}/source_channel"] = ch_name
            out_path = stage_dir / rgba_sidecar_pattern(sequence_pattern, slot).format(
                frame=frame_idx
            )
            writer.write_frame(
                out_path,
                rgba,
                attrs=attrs,
                pixel_aspect=pixel_aspect,
            )
            files_written += 1

    if files_written == 0:
        _log.warning("RGBA stage export %s: no hero slots with data", export_subdir)
        return None

    _log.info(
        "RGBA stage export %s: %d EXR(s) (R,G,B,A per slot) → %s",
        pass_name,
        files_written,
        stage_dir,
    )
    return stage_dir


__all__ = [
    "RGBA_CHANNELS",
    "pack_alpha_to_rgba",
    "rgba_sidecar_pattern",
    "write_rgba_matte_stage_export",
]
