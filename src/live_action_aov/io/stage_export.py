# LiveActionAOV — write per-pass matte stage EXRs (AI Matte QC lane).

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.channels import (
    MASK_PREFIX,
    MATTE_CHANNELS,
    is_mask_channel,
)
from live_action_aov.io.split_channels import split_sidecar_pattern
from live_action_aov.io.writers.exr import ExrSidecarWriter

_log = logging.getLogger(__name__)

METADATA_NAMESPACE = "liveaov"

_PASS_CHANNEL_FILTER: dict[str, Callable[[str], bool]] = {
    "sam3_matte": is_mask_channel,
    "birefnet_refiner": lambda n: n in MATTE_CHANNELS,
    "vitmatte_refiner": lambda n: n.startswith("vitmatte."),
    "rvm_refiner": lambda n: n in MATTE_CHANNELS,
}


def filter_channels_for_pass(pass_name: str, channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred = _PASS_CHANNEL_FILTER.get(pass_name)
    if pred is None:
        return dict(channels)
    return {k: v for k, v in channels.items() if pred(k)}


def write_pass_stage_export(
    *,
    pass_name: str,
    export_subdir: str,
    output_root: Path,
    sequence_pattern: str,
    per_frame_channels: dict[int, dict[str, np.ndarray]],
    shot_name: str,
    pixel_aspect: float = 1.0,
    attrs_extra: dict[str, Any] | None = None,
    rgba_matte: bool = True,
) -> Path | None:
    """Write one pass snapshot under ``output_root / export_subdir /``."""
    if rgba_matte and pass_name in (
        "sam3_matte",
        "birefnet_refiner",
        "vitmatte_refiner",
        "rvm_refiner",
    ):
        from live_action_aov.io.rgba_matte_export import write_rgba_matte_stage_export

        heroes = None
        if attrs_extra and "heroes" in attrs_extra:
            heroes = attrs_extra.get("heroes")
        return write_rgba_matte_stage_export(
            pass_name=pass_name,
            export_subdir=export_subdir,
            output_root=output_root,
            sequence_pattern=sequence_pattern,
            per_frame_channels=per_frame_channels,
            shot_name=shot_name,
            pixel_aspect=pixel_aspect,
            attrs_extra=attrs_extra,
            heroes=heroes,
        )

    stage_dir = (output_root / export_subdir).resolve()
    filtered_frames: dict[int, dict[str, np.ndarray]] = {}
    for frame_idx, channels in per_frame_channels.items():
        filtered = filter_channels_for_pass(pass_name, channels)
        if filtered:
            filtered_frames[frame_idx] = filtered
    if not filtered_frames:
        _log.warning("Stage export %s: no channels for pass %s", export_subdir, pass_name)
        return None

    template = split_sidecar_pattern(sequence_pattern, export_subdir.replace("/", "_"))
    writer = ExrSidecarWriter()
    attrs_base: dict[str, Any] = {
        f"{METADATA_NAMESPACE}/stage": export_subdir,
        f"{METADATA_NAMESPACE}/pass": pass_name,
        f"{METADATA_NAMESPACE}/shot": shot_name,
    }
    if attrs_extra:
        attrs_base.update(attrs_extra)

    first_path: Path | None = None
    for frame_idx, channels in sorted(filtered_frames.items()):
        attrs = dict(attrs_base)
        attrs[f"{METADATA_NAMESPACE}/frame"] = frame_idx
        out_path = stage_dir / template.format(frame=frame_idx)
        writer.write_frame(
            out_path,
            channels,
            attrs=attrs,
            pixel_aspect=pixel_aspect,
        )
        if first_path is None:
            first_path = out_path

    _log.info(
        "Stage export %s: %d frames → %s",
        pass_name,
        len(filtered_frames),
        stage_dir,
    )
    return stage_dir


__all__ = [
    "filter_channels_for_pass",
    "write_pass_stage_export",
]
