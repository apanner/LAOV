# LiveActionAOV — pick ViTMatte inference size from GPU VRAM (Colab-friendly).

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Long-edge caps by *total* device VRAM (GiB). ViTDet attention is heavy — stay conservative.
_TIER_BY_TOTAL_GB: tuple[tuple[float, int], ...] = (
    (38.0, 2048),  # A100 40GB, A6000 48GB
    (22.0, 1536),  # RTX 4090 24GB (with headroom)
    (14.0, 1024),  # T4 16GB class
    (10.0, 768),
    (0.0, 512),
)

# Further cap when *free* VRAM is low (e.g. BiRefNet not released yet).
_FREE_GB_CAP: tuple[tuple[float, int], ...] = (
    (30.0, 2048),
    (22.0, 1536),
    (14.0, 1024),
    (8.0, 768),
    (0.0, 512),
)

MIN_LONG_EDGE = 384
MAX_LONG_EDGE = 2560


@dataclass(frozen=True)
class GpuVramInfo:
    device_name: str
    total_gb: float
    free_gb: float | None


def query_cuda_vram(device_index: int = 0) -> GpuVramInfo | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        snap = None
        try:
            from live_action_aov.executors.gpu_release import cuda_vram_snapshot

            snap = cuda_vram_snapshot(device_index)
        except Exception:
            pass
        free_gb: float | None = None
        if snap is not None:
            free_gb = snap["free_gb"]
        else:
            try:
                free_b, _total_b = torch.cuda.mem_get_info(device_index)
                free_gb = float(free_b) / (1024**3)
            except Exception:
                pass
        return GpuVramInfo(
            device_name=str(props.name),
            total_gb=float(props.total_memory) / (1024**3),
            free_gb=free_gb,
        )
    except Exception:
        return None


def _tier_from_table(gb: float, table: tuple[tuple[float, int], ...]) -> int:
    for threshold, edge in table:
        if gb >= threshold:
            return edge
    return table[-1][1]


def resolve_vitmatte_long_edge(
    requested: int | float | str | None = None,
    *,
    vram: GpuVramInfo | None = None,
) -> int:
    """Resolve ViTMatte ``max_inference_long_edge`` (0 / auto = from GPU).

    Override order: ``AI_MATTE_VITMATTE_MAX_EDGE`` env → explicit ``requested`` > 0 → GPU tiers.
    """
    env_raw = os.environ.get("AI_MATTE_VITMATTE_MAX_EDGE", "").strip()
    if env_raw:
        try:
            fixed = int(float(env_raw))
            if fixed > 0:
                return int(min(MAX_LONG_EDGE, max(MIN_LONG_EDGE, fixed)))
        except ValueError:
            pass

    if requested is not None and str(requested).strip().lower() not in ("", "0", "auto", "none"):
        try:
            fixed = int(float(requested))
            if fixed > 0:
                return int(min(MAX_LONG_EDGE, max(MIN_LONG_EDGE, fixed)))
        except (TypeError, ValueError):
            pass

    info = vram if vram is not None else query_cuda_vram()
    if info is None:
        _log.info("ViTMatte VRAM: no CUDA — using long_edge=768")
        return 768

    tier = _tier_from_table(info.total_gb, _TIER_BY_TOTAL_GB)
    if info.free_gb is not None:
        free_cap = _tier_from_table(info.free_gb, _FREE_GB_CAP)
        tier = min(tier, free_cap)

    tier = int(min(MAX_LONG_EDGE, max(MIN_LONG_EDGE, tier)))
    free_s = f"{info.free_gb:.1f}" if info.free_gb is not None else "?"
    _log.info(
        "ViTMatte VRAM: %s total=%.1f GiB free=%s GiB → max_inference_long_edge=%d (auto)",
        info.device_name,
        info.total_gb,
        free_s,
        tier,
    )
    return tier


__all__ = [
    "GpuVramInfo",
    "query_cuda_vram",
    "resolve_vitmatte_long_edge",
]
