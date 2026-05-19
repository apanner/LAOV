# LiveActionAOV — batch EXR→JPEG plate cache for ViTMatte (one frame in RAM at a time).

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

_log = logging.getLogger(__name__)

ReadFrameFn = Callable[[int], tuple[np.ndarray, dict[str, Any]]]


def _rgb_to_u8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _u8_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)


def _cache_path(cache_dir: Path, frame_idx: int) -> Path:
    return cache_dir / f"plate_{frame_idx:06d}.jpg"


def build_plate_jpeg_cache(
    read_frame: ReadFrameFn,
    frame_range: tuple[int, int],
    cache_dir: Path,
    *,
    jpeg_quality: int = 92,
    log_every: int = 25,
) -> tuple[int, int]:
    """Write one full-res JPEG per plate frame. Returns ``(height, width)``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    first, last = frame_range
    plate_h, plate_w = 0, 0
    n = last - first + 1
    written = 0
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for ViTMatte plate JPEG cache") from exc

    for i, frame_idx in enumerate(range(first, last + 1)):
        out_path = _cache_path(cache_dir, frame_idx)
        if out_path.is_file():
            if plate_h == 0:
                probe = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
                if probe is not None:
                    plate_h, plate_w = int(probe.shape[0]), int(probe.shape[1])
            continue
        rgb, _attrs = read_frame(frame_idx)
        rgb_u8 = _rgb_to_u8(rgb)
        plate_h, plate_w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(out_path),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 50, 100))],
        )
        written += 1
        if log_every > 0 and (i == 0 or i == n - 1 or (i + 1) % log_every == 0):
            _log.info("ViTMatte plate cache: %d/%d frames (%s)", i + 1, n, cache_dir)

    if plate_h == 0 or plate_w == 0:
        raise RuntimeError(f"ViTMatte plate cache empty under {cache_dir}")
    _log.info(
        "ViTMatte plate cache ready: %s (%dx%d, %d new JPEGs)",
        cache_dir,
        plate_w,
        plate_h,
        written,
    )
    return plate_h, plate_w


def read_plate_jpeg(cache_dir: Path, frame_idx: int) -> np.ndarray:
    """Load one cached plate frame as float RGB ``(H,W,3)`` in ``[0,1]``."""
    path = _cache_path(cache_dir, frame_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ViTMatte plate cache: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for ViTMatte plate JPEG cache") from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read plate cache JPEG: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _u8_to_rgb(rgb)


__all__ = ["build_plate_jpeg_cache", "read_plate_jpeg"]
