# LiveActionAOV — temporal hold / interpolate between sparse refiner keyframes.

from __future__ import annotations

import numpy as np


def fill_alpha_between_keyframes(
    t_count: int,
    refined_keys: dict[int, np.ndarray],
    *,
    fill_between: bool,
    fallback_stack: np.ndarray | None = None,
) -> np.ndarray:
    """Build a full (T,H,W) alpha stack from sparse keyframe predictions.

    - ``fill_between=True``: linear blend between neighbouring keyframe alphas.
    - ``fill_between=False``: only keyframes are set; other frames stay zero unless
      ``fallback_stack`` is provided (e.g. SAM3 hard mask per frame).
    """
    if t_count <= 0:
        return np.zeros((0, 1, 1), dtype=np.float32)
    if not refined_keys:
        if fallback_stack is not None:
            return np.asarray(fallback_stack, dtype=np.float32)
        return np.zeros((t_count, 1, 1), dtype=np.float32)

    sample = next(iter(refined_keys.values()))
    h, w = sample.shape
    out = np.zeros((t_count, h, w), dtype=np.float32)
    if fallback_stack is not None:
        fb = np.asarray(fallback_stack, dtype=np.float32)
        if fb.shape[0] == t_count:
            out = fb.copy()

    for t, alpha in refined_keys.items():
        if 0 <= t < t_count:
            out[t] = alpha

    if not fill_between:
        return out

    key_list = sorted(refined_keys.keys())
    for t in range(t_count):
        if t in refined_keys:
            continue
        prev_k = max((k for k in key_list if k <= t), default=key_list[0])
        next_k = min((k for k in key_list if k >= t), default=key_list[-1])
        if prev_k == next_k:
            out[t] = refined_keys[prev_k]
        else:
            w = (t - prev_k) / max(next_k - prev_k, 1)
            out[t] = (1.0 - w) * refined_keys[prev_k] + w * refined_keys[next_k]
    return out
