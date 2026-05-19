#!/usr/bin/env python3
"""Build QC MP4 previews from AI Matte EXR sidecars (Colab / local)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger("ai_matte_qc_mp4")

_FRAME_RE = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)

STAGE_DIRS: tuple[tuple[str, str, str], ...] = (
    ("matte_sam3", "sam3", "rgba"),
    ("matte_birefnet", "birefnet", "rgba"),
    ("matte_vitmatte", "vitmatte", "rgba"),
    ("matte", "final", "rgba"),
    ("vitmatte", "final_vitmatte", "rgba"),
)

_COMBINED_MATTE_RE = re.compile(
    r"_(sam3_matte|birefnet_matte|vitmatte_matte|matte)\.(\d+)\.exr$",
    re.IGNORECASE,
)


def _sorted_exrs(folder: Path) -> list[tuple[int, Path]]:
    if not folder.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in sorted(folder.glob("*.exr")):
        m = _COMBINED_MATTE_RE.search(path.name)
        if m:
            found.append((int(m.group(2)), path))
            continue
        m = _FRAME_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    # One combined file per frame — drop duplicate paths for same frame.
    by_frame: dict[int, Path] = {}
    for frame, path in found:
        by_frame.setdefault(frame, path)
    return sorted(by_frame.items(), key=lambda x: x[0])


def _read_exr_channels(path: Path) -> tuple[np.ndarray, list[str]]:
    from live_action_aov.io.oiio_io import read_plate

    pixels, attrs = read_plate(path)
    names = [str(n) for n in (attrs.get("channelnames") or [])]
    if len(names) != pixels.shape[-1]:
        names = [f"ch{i}" for i in range(pixels.shape[-1])]
    return pixels, names


def _channel_plane(pixels: np.ndarray, names: list[str], channel: str) -> np.ndarray | None:
    if channel not in names:
        return None
    idx = names.index(channel)
    return np.clip(pixels[..., idx].astype(np.float32), 0.0, 1.0)


def _union_masks(pixels: np.ndarray, names: list[str], prefix: str) -> np.ndarray | None:
    planes: list[np.ndarray] = []
    for i, name in enumerate(names):
        if name.startswith(prefix):
            planes.append(np.clip(pixels[..., i], 0.0, 1.0))
    if not planes:
        return None
    out = planes[0].copy()
    for plane in planes[1:]:
        np.maximum(out, plane, out=out)
    return out


def _hero_slots_rgb(pixels: np.ndarray, names: list[str], prefix: str) -> np.ndarray | None:
    """One color per hero slot (r=red, g=green, b=blue, a=white) — avoids false 'doubling'."""
    slot_colors = {
        f"{prefix}.r": (255, 0, 0),
        f"{prefix}.g": (0, 255, 0),
        f"{prefix}.b": (0, 0, 255),
        f"{prefix}.a": (255, 255, 255),
    }
    h, w = pixels.shape[0], pixels.shape[1]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    any_slot = False
    for ch, color in slot_colors.items():
        plane = _channel_plane(pixels, names, ch)
        if plane is None:
            continue
        any_slot = True
        for c, val in enumerate(color):
            rgb[..., c] = np.maximum(rgb[..., c], plane * (val / 255.0))
    if not any_slot:
        return None
    return np.clip(rgb, 0.0, 1.0)


def _overlay_plate(plate_rgb: np.ndarray, rgb: np.ndarray, *, alpha_scale: float = 0.55) -> np.ndarray:
    plate = np.clip(plate_rgb[..., :3], 0.0, 1.0).astype(np.float32)
    tint = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    a = np.max(tint, axis=-1, keepdims=True)
    out = plate * (1.0 - alpha_scale * a) + tint * (alpha_scale * a)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_mp4(frames_rgb: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("no frames to encode")
    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio  # type: ignore[no-redef]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(frames_rgb, axis=0),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )


def _export_stage_mp4(
    stage_dir: Path,
    qc_dir: Path,
    shot_label: str,
    suffix: str,
    channel_kind: str,
    *,
    plate_reader: Any | None,
    fps: float,
) -> Path | None:
    entries = _sorted_exrs(stage_dir)
    if not entries:
        return None
    frames_rgb: list[np.ndarray] = []
    for frame_idx, exr_path in entries:
        pixels, names = _read_exr_channels(exr_path)
        if channel_kind == "rgba":
            r = _channel_plane(pixels, names, "R")
            g = _channel_plane(pixels, names, "G")
            b = _channel_plane(pixels, names, "B")
            if r is not None or g is not None or b is not None:
                h, w = pixels.shape[0], pixels.shape[1]
                rgb = np.zeros((h, w, 3), dtype=np.float32)
                if r is not None:
                    rgb[..., 0] = r
                if g is not None:
                    rgb[..., 1] = g
                if b is not None:
                    rgb[..., 2] = b
            else:
                vis = _channel_plane(pixels, names, "A")
                if vis is None:
                    continue
                rgb = np.stack([vis, vis, vis], axis=-1)
        elif channel_kind == "mask":
            vis = _union_masks(pixels, names, "mask.")
            if vis is None:
                continue
            rgb = np.stack([vis, vis, vis], axis=-1)
        else:
            rgb = _hero_slots_rgb(pixels, names, channel_kind)
            if rgb is None:
                continue
        if plate_reader is not None:
            try:
                plate_rgb, _ = plate_reader.read_frame(frame_idx)
                frames_rgb.append(_overlay_plate(plate_rgb, rgb))
            except Exception:
                frames_rgb.append((rgb * 255.0).astype(np.uint8))
        else:
            frames_rgb.append((rgb * 255.0).astype(np.uint8))

    if not frames_rgb:
        return None
    out = qc_dir / f"{shot_label}_{suffix}_qc.mp4"
    _write_mp4(frames_rgb, out, fps)
    _log.info("QC MP4 (%s): %s", suffix, out)
    return out


def export_all_stage_qc_mp4s(
    output_shot_dir: Path,
    *,
    plate_dir: Path | None = None,
    sequence_pattern: str | None = None,
    frame_range: tuple[int, int] | None = None,
    fps: float = 24.0,
    refiner: str = "birefnet_refiner",
) -> dict[str, Path]:
    """Write QC MP4 for each stage folder that exists under the shot output."""
    qc_dir = output_shot_dir / "qc"
    shot_label = output_shot_dir.name
    plate_reader: Any | None = None
    if plate_dir and sequence_pattern and frame_range:
        try:
            from live_action_aov.io.readers.oiio_exr import OIIOExrReader

            plate_reader = OIIOExrReader(plate_dir, sequence_pattern)
        except Exception as exc:
            _log.warning("Plate overlay disabled: %s", exc)

    written: dict[str, Path] = {}
    for subdir, suffix, kind in STAGE_DIRS:
        if suffix == "final_vitmatte" and refiner != "vitmatte_refiner":
            continue
        if suffix == "final" and refiner == "vitmatte_refiner":
            continue
        stage_path = output_shot_dir / subdir
        if not stage_path.is_dir():
            continue
        out = _export_stage_mp4(
            stage_path,
            qc_dir,
            shot_label,
            suffix,
            kind,
            plate_reader=plate_reader,
            fps=fps,
        )
        if out:
            written[suffix] = out
    return written


def export_ai_matte_qc_mp4s(
    output_shot_dir: Path,
    *,
    plate_dir: Path | None = None,
    sequence_pattern: str | None = None,
    frame_range: tuple[int, int] | None = None,
    fps: float = 24.0,
    overlay_plate: bool = True,
) -> dict[str, Path]:
    """Backward-compatible wrapper — exports all stage folders found."""
    return export_all_stage_qc_mp4s(
        output_shot_dir,
        plate_dir=plate_dir,
        sequence_pattern=sequence_pattern,
        frame_range=frame_range,
        fps=fps,
    )
