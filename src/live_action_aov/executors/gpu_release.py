# LiveActionAOV — release GPU weights and clear CUDA cache between pipeline stages.

from __future__ import annotations

import gc
import logging
from typing import Any

_log = logging.getLogger(__name__)

_MODULE_ATTRS = (
    "_det_model",
    "_trk_model",
    "_det_processor",
    "_trk_processor",
    "_model",
    "_track_predictor",
    "_predictor",
    "_video_session",
)

_CPU_BULK_ATTRS = (
    "_instances",
    "_forward_flow",
    "_refined",
)


def cuda_vram_snapshot(device_index: int = 0) -> dict[str, float] | None:
    """Allocated / reserved / free GiB on the current CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        allocated_b = torch.cuda.memory_allocated(device_index)
        reserved_b = torch.cuda.memory_reserved(device_index)
        gib = 1024**3
        return {
            "total_gb": total_b / gib,
            "free_gb": free_b / gib,
            "allocated_gb": allocated_b / gib,
            "reserved_gb": reserved_b / gib,
        }
    except Exception:
        return None


def clear_vram_cache(*, device_index: int = 0, sync: bool = True) -> None:
    """Aggressive CUDA cache flush (safe to call between frames or stages)."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if sync:
            torch.cuda.synchronize(device_index)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _move_to_cpu(obj: Any) -> None:
    if obj is None:
        return
    try:
        import torch

        if isinstance(obj, torch.nn.Module):
            obj.cpu()
        elif hasattr(obj, "cpu"):
            obj.cpu()
    except Exception:
        pass


def _release_session(session: Any) -> None:
    if session is None:
        return
    _move_to_cpu(getattr(session, "_model", None))
    if hasattr(session, "_model"):
        session._model = None
    if hasattr(session, "_processor"):
        session._processor = None
    if hasattr(session, "release"):
        try:
            session.release()
        except Exception:
            pass


def _clear_pass_cpu_bulk(pass_instance: Any) -> None:
    """Drop large CPU buffers once artifacts are exported to the executor."""
    for attr in _CPU_BULK_ATTRS:
        if not hasattr(pass_instance, attr):
            continue
        val = getattr(pass_instance, attr)
        if attr == "_instances" and isinstance(val, list):
            setattr(pass_instance, attr, [])
        elif isinstance(val, dict):
            val.clear()
        elif isinstance(val, list):
            val.clear()


def release_pass_gpu(
    pass_instance: Any,
    *,
    stage_name: str | None = None,
    log_stats: bool = True,
) -> None:
    """Unload pass models and clear VRAM so the next stage gets a clean GPU."""
    name = stage_name or getattr(pass_instance, "name", type(pass_instance).__name__)
    before = cuda_vram_snapshot() if log_stats else None

    if hasattr(pass_instance, "release_gpu") and callable(pass_instance.release_gpu):
        try:
            pass_instance.release_gpu()
        except Exception as exc:
            _log.warning("release_gpu() failed for %s: %s", name, exc)

    _release_session(getattr(pass_instance, "_session", None))
    if hasattr(pass_instance, "_session"):
        pass_instance._session = None

    for attr in _MODULE_ATTRS:
        if not hasattr(pass_instance, attr):
            continue
        obj = getattr(pass_instance, attr, None)
        if obj is not None:
            _move_to_cpu(obj)
        setattr(pass_instance, attr, None)

    if hasattr(pass_instance, "_device"):
        pass_instance._device = None
    if hasattr(pass_instance, "_dtype"):
        pass_instance._dtype = None

    _clear_pass_cpu_bulk(pass_instance)
    clear_vram_cache()

    if log_stats and before is not None:
        after = cuda_vram_snapshot()
        if after is not None:
            _log.info(
                "Stage VRAM [%s]: allocated %.2f→%.2f GiB, reserved %.2f→%.2f GiB, "
                "free %.2f→%.2f GiB",
                name,
                before["allocated_gb"],
                after["allocated_gb"],
                before["reserved_gb"],
                after["reserved_gb"],
                before["free_gb"],
                after["free_gb"],
            )
        else:
            _log.info("GPU memory released after stage %s", name)
    else:
        _log.info("GPU memory released after stage %s", name)


__all__ = [
    "clear_vram_cache",
    "cuda_vram_snapshot",
    "release_pass_gpu",
]
