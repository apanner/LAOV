# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""RAFT-guided temporal propagation for BiRefNet keyframe mattes.

BiRefNet runs on sparse keyframes only (``fill_between_keyframes: false``).
This post-processor fills every frame by warping matte channels along RAFT
forward/backward flow with forward–backward occlusion rejection, then
optionally applies a light flow-guided EMA polish.

Designed for the AI Matte lane: ``passes_csv: flow,matte`` + this post step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.channels import MATTE_CHANNELS
from live_action_aov.post.temporal_smooth import (
    TemporalSmoother,
    _fb_occlusion,
    _warp_backward,
)
from live_action_aov.shared.optical_flow.cache import FlowCache


class MatteFlowTemporal:
    """Propagate sparse BiRefNet keyframe mattes with bidirectional RAFT warps."""

    name = "matte_flow_temporal"
    algorithm = "raft_keyframe_bidir_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "applied_to": list(MATTE_CHANNELS),
        "keyframe_stride": 4,
        "fb_threshold_px": 1.0,
        "blend_forward_backward": 0.5,
        "final_ema_alpha": 0.25,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: FlowCache,
        shot_id: str,
        **_kwargs: Any,
    ) -> dict[int, dict[str, np.ndarray]]:
        applied_to: list[str] = list(
            self.params.get("applied_to") or list(MATTE_CHANNELS)
        )
        if not applied_to:
            return per_frame_channels

        frames = sorted(per_frame_channels.keys())
        if len(frames) < 2:
            return per_frame_channels

        stride = max(1, int(self.params.get("keyframe_stride", 4)))
        threshold = float(self.params["fb_threshold_px"])
        blend_fb = float(self.params.get("blend_forward_backward", 0.5))
        ema_alpha = float(self.params.get("final_ema_alpha", 0.0))

        key_indices = _keyframe_frame_numbers(frames, stride)
        out: dict[int, dict[str, np.ndarray]] = {
            f: dict(channels) for f, channels in per_frame_channels.items()
        }

        for ch_name in applied_to:
            if not any(ch_name in out[f] for f in frames):
                continue
            key_alpha = {
                f: out[f][ch_name].astype(np.float32, copy=False)
                for f in key_indices
                if ch_name in out[f]
            }
            if not key_alpha:
                continue

            fwd = _propagate_from_keys(
                frames,
                key_indices,
                key_alpha,
                flow_cache,
                shot_id,
                direction="forward",
                fb_threshold_px=threshold,
            )
            bwd = _propagate_from_keys(
                frames,
                key_indices,
                key_alpha,
                flow_cache,
                shot_id,
                direction="backward",
                fb_threshold_px=threshold,
            )
            for f in frames:
                if ch_name not in out[f]:
                    continue
                if f in key_indices:
                    out[f][ch_name] = key_alpha[f].astype(np.float32, copy=False)
                    continue
                prev_k = max(k for k in key_indices if k <= f)
                next_k = min(k for k in key_indices if k >= f)
                if prev_k == next_k:
                    out[f][ch_name] = key_alpha[prev_k].astype(np.float32, copy=False)
                    continue
                w = (f - prev_k) / max(next_k - prev_k, 1)
                fa = fwd.get(f)
                ba = bwd.get(f)
                if fa is None and ba is None:
                    continue
                if fa is None:
                    out[f][ch_name] = ba.astype(np.float32, copy=False)  # type: ignore[union-attr]
                elif ba is None:
                    out[f][ch_name] = fa.astype(np.float32, copy=False)
                else:
                    mixed = (1.0 - w) * fa + w * ba
                    if blend_fb != 0.5:
                        mixed = (1.0 - blend_fb) * fa + blend_fb * ba
                    out[f][ch_name] = mixed.astype(np.float32, copy=False)

        if ema_alpha > 0.0:
            smoother = TemporalSmoother(
                {
                    "applied_to": applied_to,
                    "alpha": ema_alpha,
                    "fb_threshold_px": threshold,
                }
            )
            out = smoother.apply(out, flow_cache, shot_id)
        return out


def _keyframe_frame_numbers(frames: list[int], stride: int) -> set[int]:
    if not frames:
        return set()
    last = frames[-1]
    keys = {f for i, f in enumerate(frames) if i % stride == 0}
    keys.add(last)
    keys.add(frames[0])
    return keys


def _propagate_from_keys(
    frames: list[int],
    key_indices: set[int],
    key_alpha: dict[int, np.ndarray],
    flow_cache: FlowCache,
    shot_id: str,
    *,
    direction: str,
    fb_threshold_px: float,
) -> dict[int, np.ndarray]:
    """Return per-frame alpha estimates warped from nearest keyframe anchors."""
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    result: dict[int, np.ndarray] = dict(key_alpha)

    if direction == "forward":
        for f in frames:
            if f in key_indices:
                continue
            prev_k = max(k for k in key_indices if k <= f)
            start_i = frame_to_idx[prev_k]
            end_i = frame_to_idx[f]
            cur = key_alpha[prev_k].copy()
            for j in range(start_i + 1, end_i + 1):
                cur = _warp_step(
                    cur,
                    frames[j - 1],
                    frames[j],
                    flow_cache,
                    shot_id,
                    fb_threshold_px,
                )
            result[f] = cur
        return result

    if direction == "backward":
        for f in frames:
            if f in key_indices:
                continue
            next_k = min(k for k in key_indices if k >= f)
            start_i = frame_to_idx[next_k]
            end_i = frame_to_idx[f]
            cur = key_alpha[next_k].copy()
            for j in range(start_i - 1, end_i - 1, -1):
                cur = _warp_step_backward(
                    cur,
                    frames[j],
                    frames[j + 1],
                    flow_cache,
                    shot_id,
                    fb_threshold_px,
                )
            result[f] = cur
        return result

    raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")


def _warp_step(
    alpha_prev: np.ndarray,
    f_prev: int,
    f_cur: int,
    flow_cache: FlowCache,
    shot_id: str,
    fb_threshold_px: float,
) -> np.ndarray:
    bwd = flow_cache.get(shot_id, f_cur, "backward")
    fwd_prev = flow_cache.get(shot_id, f_prev, "forward")
    if bwd is None or fwd_prev is None:
        return alpha_prev
    occ = _fb_occlusion(fwd_prev, bwd, threshold_px=fb_threshold_px)
    warped = _warp_backward(alpha_prev, bwd)
    keep = occ > 0.5
    out = warped.copy()
    out[keep] = alpha_prev[keep]
    return out.astype(np.float32, copy=False)


def _warp_step_backward(
    alpha_next: np.ndarray,
    f_cur: int,
    f_next: int,
    flow_cache: FlowCache,
    shot_id: str,
    fb_threshold_px: float,
) -> np.ndarray:
    """Warp alpha from ``f_next`` onto the grid at ``f_cur``."""
    bwd_next = flow_cache.get(shot_id, f_next, "backward")
    fwd_cur = flow_cache.get(shot_id, f_cur, "forward")
    if bwd_next is None or fwd_cur is None:
        return alpha_next
    occ = _fb_occlusion(fwd_cur, bwd_next, threshold_px=fb_threshold_px)
    warped = _warp_backward(alpha_next, bwd_next)
    keep = occ > 0.5
    out = warped.copy()
    out[keep] = alpha_next[keep]
    return out.astype(np.float32, copy=False)


__all__ = ["MatteFlowTemporal"]
