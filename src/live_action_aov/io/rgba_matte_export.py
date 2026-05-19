# LiveActionAOV — one multi-channel EXR per frame (R,G,B,A + named hero layers).

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

_PASS_STAGE_SUFFIX: dict[str, str] = {
    "sam3_matte": "sam3_matte",
    "birefnet_refiner": "birefnet_matte",
    "vitmatte_refiner": "vitmatte_matte",
    "rvm_refiner": "rvm_matte",
}

_SLOT_SOURCE: dict[str, tuple[str, ...]] = {
    "sam3_matte": MATTE_CHANNELS,
    "birefnet_refiner": MATTE_CHANNELS,
    "vitmatte_refiner": ("vitmatte.r", "vitmatte.g", "vitmatte.b", "vitmatte.a"),
    "rvm_refiner": MATTE_CHANNELS,
}

_EXR_SLOT = {"r": "R", "g": "G", "b": "B", "a": "A"}
_SKIP_LAYER_PREFIXES = ("mask.", "matte.", "vitmatte.")


def _slug_label(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(label).strip().lower()).strip("_")
    return s or "hero"


def normalize_heroes_metadata(heroes: list[Any]) -> list[dict[str, Any]]:
    """``HeroSlot`` (SAM3) or ``dict`` (refiners from ``sam3_instances`` artifact)."""
    out: list[dict[str, Any]] = []
    for h in heroes:
        if isinstance(h, dict):
            out.append(
                {
                    "track_id": int(h["track_id"]),
                    "slot": str(h.get("slot", "")),
                    "label": str(h.get("label", "")),
                }
            )
        else:
            out.append(
                {
                    "track_id": int(h.track_id),
                    "slot": str(h.slot),
                    "label": str(getattr(h, "label", "")),
                }
            )
    return out


def combined_matte_pattern(sequence_pattern: str, stage_suffix: str) -> str:
    """``plate.####.exr`` → ``plate_sam3_matte.{frame:04d}.exr``."""
    token = f"_{stage_suffix}."
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


def _zeros_like(reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32)
    return np.zeros(ref.shape[:2], dtype=np.float32)


def _alpha_plane(channels: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    arr = channels.get(name)
    if arr is None:
        return None
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def build_combined_matte_channels(
    pass_name: str,
    channels: dict[str, np.ndarray],
    heroes: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """One EXR: R,G,B,A = slot mattes; extra float layers per hero label (person1, car, …)."""
    source_names = _SLOT_SOURCE.get(pass_name, MATTE_CHANNELS)
    slot_letters = ("r", "g", "b", "a")
    ref = next(iter(channels.values()), None)
    if ref is None:
        return {}
    z = _zeros_like(ref)
    out: dict[str, np.ndarray] = {k: z.copy() for k in RGBA_CHANNELS}

    hero_by_slot: dict[str, dict[str, Any]] = {}
    if heroes:
        for h in heroes:
            hero_by_slot[str(h.get("slot", ""))] = h

    for slot, src in zip(slot_letters, source_names, strict=False):
        alpha = _alpha_plane(channels, src)
        if alpha is None:
            continue
        exr_name = _EXR_SLOT[slot]
        out[exr_name] = alpha
        hero = hero_by_slot.get(slot)
        if hero:
            slug = _slug_label(str(hero.get("label", f"hero_{slot}")))
            if slug not in RGBA_CHANNELS:
                out[slug] = alpha.copy()

    # Named layers for heroes without a standard slot mapping in this frame.
    if heroes:
        for hero in heroes:
            slot = str(hero.get("slot", ""))
            src = source_names[slot_letters.index(slot)] if slot in slot_letters else ""
            if not src:
                continue
            alpha = _alpha_plane(channels, src)
            if alpha is None:
                continue
            slug = _slug_label(str(hero.get("label", f"track_{hero.get('track_id', 0)}")))
            if slug in out:
                continue
            out[slug] = alpha

    # Instance layers emitted by SAM3 (and refiners that keep the same names).
    for name in sorted(channels):
        if name in out or name in RGBA_CHANNELS:
            continue
        if any(name.startswith(p) for p in _SKIP_LAYER_PREFIXES):
            continue
        alpha = _alpha_plane(channels, name)
        if alpha is not None:
            out[name] = alpha

    return _order_combined_channels(out)


def _order_combined_channels(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """R,G,B,A first so EXR viewers default to a 3-up matte preview."""
    ordered: dict[str, np.ndarray] = {}
    for key in RGBA_CHANNELS:
        if key in channels:
            ordered[key] = channels[key]
    for key in sorted(channels):
        if key not in ordered:
            ordered[key] = channels[key]
    return ordered


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
    heroes: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Write ``{stem}_sam3_matte.####.exr`` (one file, R/G/B/A + named layers)."""
    if pass_name not in _MATTE_PASSES:
        return None

    stage_suffix = _PASS_STAGE_SUFFIX.get(pass_name, "matte")
    stage_dir = (output_root / export_subdir).resolve()
    writer = ExrSidecarWriter()
    template = combined_matte_pattern(sequence_pattern, stage_suffix)

    attrs_base: dict[str, Any] = {
        f"{METADATA_NAMESPACE}/stage": export_subdir,
        f"{METADATA_NAMESPACE}/pass": pass_name,
        f"{METADATA_NAMESPACE}/shot": shot_name,
        f"{METADATA_NAMESPACE}/format": "combined_matte",
        f"{METADATA_NAMESPACE}/channel_layout": "R,G,B,A=hero_slots; named=hero_label",
    }
    if attrs_extra:
        attrs_base.update(attrs_extra)

    frames_written = 0
    for frame_idx, channels in sorted(per_frame_channels.items()):
        combined = build_combined_matte_channels(pass_name, channels, heroes)
        if not combined:
            continue
        attrs = dict(attrs_base)
        attrs[f"{METADATA_NAMESPACE}/frame"] = frame_idx
        attrs[f"{METADATA_NAMESPACE}/layers"] = ",".join(sorted(combined.keys()))
        out_path = stage_dir / template.format(frame=frame_idx)
        writer.write_frame(
            out_path,
            combined,
            attrs=attrs,
            pixel_aspect=pixel_aspect,
        )
        frames_written += 1

    if frames_written == 0:
        _log.warning("Combined matte export %s: no frames", export_subdir)
        return None

    _log.info(
        "Combined matte export %s: %d frames → %s (one EXR per frame, R/G/B/A + named layers)",
        pass_name,
        frames_written,
        stage_dir,
    )
    return stage_dir


def pack_alpha_to_rgba(alpha: np.ndarray) -> dict[str, np.ndarray]:
    """Legacy helper — single subject preview (R=G=B=A)."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return {"R": a, "G": a.copy(), "B": a.copy(), "A": a.copy()}


__all__ = [
    "RGBA_CHANNELS",
    "build_combined_matte_channels",
    "combined_matte_pattern",
    "normalize_heroes_metadata",
    "pack_alpha_to_rgba",
    "write_rgba_matte_stage_export",
]
