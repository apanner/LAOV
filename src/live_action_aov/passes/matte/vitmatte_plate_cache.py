# LiveActionAOV — parallel EXR→JPEG plate cache (SAM3 + BiRefNet + ViTMatte).

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def cache_path(cache_dir: Path, frame_idx: int) -> Path:
    return cache_dir / f"plate_{frame_idx:06d}.jpg"


def plate_cache_frame_count(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return len(list(cache_dir.glob("plate_*.jpg")))


def plate_cache_complete(cache_dir: Path, frame_range: tuple[int, int]) -> bool:
    first, last = frame_range
    n = last - first + 1
    return plate_cache_frame_count(cache_dir) >= n


def _write_one_jpeg(
    frame_idx: int,
    read_frame: ReadFrameFn,
    out_path: Path,
    jpeg_quality: int,
) -> tuple[int, int, int]:
    import cv2

    rgb, _attrs = read_frame(frame_idx)
    rgb_u8 = _rgb_to_u8(rgb)
    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        str(out_path),
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 50, 100))],
    )
    return h, w, frame_idx


def build_plate_jpeg_cache(
    read_frame: ReadFrameFn,
    frame_range: tuple[int, int],
    cache_dir: Path,
    *,
    jpeg_quality: int = 92,
    log_every: int = 25,
    log_label: str = "Plate",
    workers: int = 1,
) -> tuple[int, int]:
    """Write one full-res JPEG per plate frame. Returns ``(height, width)``."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for plate JPEG cache") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    first, last = frame_range
    plate_h, plate_w = 0, 0
    n = last - first + 1
    written = 0
    todo: list[int] = []

    for frame_idx in range(first, last + 1):
        out_path = cache_path(cache_dir, frame_idx)
        if out_path.is_file():
            if plate_h == 0:
                probe = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
                if probe is not None:
                    plate_h, plate_w = int(probe.shape[0]), int(probe.shape[1])
            continue
        todo.append(frame_idx)

    if not todo and plate_h > 0:
        _log.info(
            "%s plate cache hit: %d/%d frames in %s",
            log_label,
            n,
            n,
            cache_dir,
        )
        return plate_h, plate_w

    workers = max(1, int(workers))
    if workers > 1 and len(todo) > 1:
        _log.info(
            "%s plate cache: encoding %d/%d frames with %d workers → %s",
            log_label,
            len(todo),
            n,
            workers,
            cache_dir,
        )
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _write_one_jpeg,
                    frame_idx,
                    read_frame,
                    cache_path(cache_dir, frame_idx),
                    jpeg_quality,
                ): frame_idx
                for frame_idx in todo
            }
            for fut in as_completed(futures):
                h, w, _fid = fut.result()
                if plate_h == 0:
                    plate_h, plate_w = h, w
                written += 1
                done += 1
                if log_every > 0 and (
                    done == 1 or done == len(todo) or done % log_every == 0
                ):
                    _log.info("%s plate cache: %d/%d encoded", log_label, done, len(todo))
    else:
        for i, frame_idx in enumerate(todo):
            h, w, _ = _write_one_jpeg(
                frame_idx,
                read_frame,
                cache_path(cache_dir, frame_idx),
                jpeg_quality,
            )
            if plate_h == 0:
                plate_h, plate_w = h, w
            written += 1
            if log_every > 0 and (i == 0 or i == len(todo) - 1 or (i + 1) % log_every == 0):
                _log.info(
                    "%s plate cache: %d/%d frames (%s)",
                    log_label,
                    i + 1,
                    len(todo),
                    cache_dir,
                )

    if plate_h == 0 or plate_w == 0:
        raise RuntimeError(f"{log_label} plate cache empty under {cache_dir}")
    _log.info(
        "%s plate cache ready: %s (%dx%d, %d new JPEGs)",
        log_label,
        cache_dir,
        plate_w,
        plate_h,
        written,
    )
    return plate_h, plate_w


def read_plate_jpeg(cache_dir: Path, frame_idx: int) -> np.ndarray:
    """Load one cached plate frame as float RGB ``(H,W,3)`` in ``[0,1]``."""
    path = cache_path(cache_dir, frame_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Missing plate cache JPEG: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for plate JPEG cache") from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read plate cache JPEG: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _u8_to_rgb(rgb)


__all__ = [
    "ReadFrameFn",
    "build_plate_jpeg_cache",
    "cache_path",
    "plate_cache_complete",
    "plate_cache_frame_count",
    "read_plate_jpeg",
]
