#!/usr/bin/env python3
"""Build QC MP4 previews from AI Matte EXR sidecars (Colab / local).

Reads ``<shot>/matte/*.matte.<frame>.exr`` and writes:

- ``qc/sam3_matte_qc.mp4`` — union of SAM3 ``mask.<concept>`` channels (hard masks)
- ``qc/final_matte_qc.mp4`` — hero soft mattes ``matte.r/g/b/a`` (BiRefNet + temporal post)

Optional plate overlay when ``plate_dir`` is provided.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

_log = logging.getLogger("ai_matte_qc_mp4")

_FRAME_RE = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)


def _sorted_matte_exrs(matte_dir: Path) -> list[tuple[int, Path]]:
    if not matte_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in sorted(matte_dir.glob("*.exr")):
        m = _FRAME_RE.search(path.name)
        if not m:
            continue
        found.append((int(m.group(1)), path))
    return sorted(found, key=lambda x: x[0])


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
    plane = pixels[..., idx]
    return np.clip(plane.astype(np.float32), 0.0, 1.0)


def _sam3_union_alpha(pixels: np.ndarray, names: list[str]) -> np.ndarray | None:
    planes: list[np.ndarray] = []
    for i, name in enumerate(names):
        if name.startswith("mask."):
            planes.append(np.clip(pixels[..., i], 0.0, 1.0))
    if not planes:
        return None
    out = planes[0].copy()
    for plane in planes[1:]:
        np.maximum(out, plane, out=out)
    return out


def _final_hero_alpha(pixels: np.ndarray, names: list[str]) -> np.ndarray | None:
    slots = ("matte.a", "matte.r", "matte.g", "matte.b")
    planes: list[np.ndarray] = []
    for ch in slots:
        plane = _channel_plane(pixels, names, ch)
        if plane is not None:
            planes.append(plane)
    if not planes:
        return None
    out = planes[0].copy()
    for plane in planes[1:]:
        np.maximum(out, plane, out=out)
    return out


def _to_rgb_vis(alpha: np.ndarray, *, green: bool = False) -> np.ndarray:
    """Grayscale or green-on-black preview (uint8 RGB)."""
    a = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    if green:
        return np.stack([np.zeros_like(a), a, np.zeros_like(a)], axis=-1)
    return np.stack([a, a, a], axis=-1)


def _overlay_plate(plate_rgb: np.ndarray, alpha: np.ndarray, *, green: bool = True) -> np.ndarray:
    plate = np.clip(plate_rgb[..., :3], 0.0, 1.0).astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    if green:
        tint = np.zeros_like(plate)
        tint[..., 1] = 1.0
    else:
        tint = np.ones_like(plate)
    blend = 0.55
    out = plate * (1.0 - blend * a) + tint * (blend * a)
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


def export_ai_matte_qc_mp4s(
    output_shot_dir: Path,
    *,
    plate_dir: Path | None = None,
    sequence_pattern: str | None = None,
    frame_range: tuple[int, int] | None = None,
    fps: float = 24.0,
    overlay_plate: bool = True,
) -> dict[str, Path]:
    """Write SAM3 + final matte QC MP4s under ``output_shot_dir/qc/``."""
    matte_dir = output_shot_dir / "matte"
    qc_dir = output_shot_dir / "qc"
    entries = _sorted_matte_exrs(matte_dir)
    if not entries:
        raise FileNotFoundError(f"No matte EXRs in {matte_dir}")

    sam3_frames: list[np.ndarray] = []
    final_frames: list[np.ndarray] = []
    plate_reader = None
    if overlay_plate and plate_dir and frame_range and sequence_pattern:
        try:
            from live_action_aov.io.readers.oiio_exr import OIIOExrReader

            plate_reader = OIIOExrReader(plate_dir, sequence_pattern)
        except Exception as exc:
            _log.warning("Plate overlay disabled: %s", exc)
            plate_reader = None

    for frame_idx, exr_path in entries:
        pixels, names = _read_exr_channels(exr_path)
        sam3_a = _sam3_union_alpha(pixels, names)
        final_a = _final_hero_alpha(pixels, names)
        if sam3_a is not None:
            if plate_reader is not None:
                try:
                    plate_rgb, _ = plate_reader.read_frame(frame_idx)
                    sam3_frames.append(_overlay_plate(plate_rgb, sam3_a, green=True))
                except Exception:
                    sam3_frames.append(_to_rgb_vis(sam3_a, green=True))
            else:
                sam3_frames.append(_to_rgb_vis(sam3_a, green=True))
        if final_a is not None:
            if plate_reader is not None:
                try:
                    plate_rgb, _ = plate_reader.read_frame(frame_idx)
                    final_frames.append(_overlay_plate(plate_rgb, final_a, green=True))
                except Exception:
                    final_frames.append(_to_rgb_vis(final_a, green=False))
            else:
                final_frames.append(_to_rgb_vis(final_a, green=False))

    written: dict[str, Path] = {}
    shot_label = output_shot_dir.name
    if sam3_frames:
        out = qc_dir / f"{shot_label}_sam3_matte_qc.mp4"
        _write_mp4(sam3_frames, out, fps)
        written["sam3"] = out
        _log.info("QC MP4 (SAM3 masks): %s", out)
    else:
        _log.warning("No mask.* channels in %s — skip SAM3 QC MP4", matte_dir)

    if final_frames:
        out = qc_dir / f"{shot_label}_final_matte_qc.mp4"
        _write_mp4(final_frames, out, fps)
        written["final"] = out
        _log.info("QC MP4 (BiRefNet+temporal matte): %s", out)
    else:
        _log.warning("No matte.r/g/b/a in %s — skip final QC MP4", matte_dir)

    return written
