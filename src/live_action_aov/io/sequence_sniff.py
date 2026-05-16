# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Discover a numbered plate sequence in a folder (EXR, JPEG, common stills).

Used by CLI, GUI, and Colab batch runners. Picks the longest sequence per
extension, preferring EXR when present, then JPEG, then other still formats.
"""

from __future__ import annotations

import re
from pathlib import Path

from live_action_aov.io.oiio_io import read_plate

_SIDECAR_TOKENS = (".utility.", ".hero.", ".mask.")

# Order when multiple plate types exist in the same folder
_EXTENSION_PRIORITY: tuple[str, ...] = (
    ".exr",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)


def sniff_plate_sequence(
    folder: Path,
) -> tuple[str, tuple[int, int], tuple[int, int], float, Path]:
    """Return `(pattern, frame_range, (w,h), pixel_aspect, first_frame_path)`.

    `pattern` uses `#` padding like ``hero.####.exr`` or ``plate.####.jpg``.
    """
    last_exc: FileNotFoundError | None = None
    for ext in _EXTENSION_PRIORITY:
        try:
            return _sniff_plate_with_extension(folder, ext)
        except FileNotFoundError as e:
            last_exc = e
            continue
    msg = (
        f"No plate sequence ({', '.join(_EXTENSION_PRIORITY)}) found in {folder}"
    )
    if last_exc is not None:
        raise FileNotFoundError(msg) from last_exc
    raise FileNotFoundError(msg)


def _sniff_plate_with_extension(
    folder: Path,
    ext: str,
) -> tuple[str, tuple[int, int], tuple[int, int], float, Path]:
    ext_l = ext.lower()
    if not ext_l.startswith("."):
        ext_l = "." + ext_l
    candidates = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ext_l
        and not any(tok in p.name for tok in _SIDECAR_TOKENS)
    ]
    if not candidates:
        raise FileNotFoundError(f"No *{ext_l} plate files in {folder}")

    tail_digits_re = re.compile(
        rf"(\d+)(?={re.escape(ext_l)}$)", re.IGNORECASE
    )
    groups: dict[str, list[tuple[int, Path]]] = {}
    for p in candidates:
        m = tail_digits_re.search(p.name)
        if not m:
            continue
        digits = m.group(1)
        width = len(digits)
        pattern = p.name[: m.start()] + ("#" * width) + p.name[m.end() :]
        groups.setdefault(pattern, []).append((int(digits), p))

    if not groups:
        raise FileNotFoundError(
            f"No sequenced *{ext_l} files in {folder} "
            "(names must end with digits before the extension)."
        )

    best_pattern = max(
        groups.keys(),
        key=lambda k: (len(groups[k]), -ord(k[0]) if k else 0),
    )
    entries = sorted(groups[best_pattern], key=lambda t: t[0])
    frame_numbers = [f for f, _ in entries]
    frame_range = (min(frame_numbers), max(frame_numbers))
    first_frame_path = entries[0][1]

    pixels, attrs = read_plate(first_frame_path)
    h, w = pixels.shape[:2]
    par = float(attrs.get("pixelAspectRatio", 1.0))
    return best_pattern, frame_range, (w, h), par, first_frame_path


__all__ = ["sniff_plate_sequence", "_EXTENSION_PRIORITY"]
