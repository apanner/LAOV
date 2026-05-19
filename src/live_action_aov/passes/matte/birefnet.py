# LiveActionAOV — BiRefNet soft-alpha refiner (AI Matte / colab_ai lane)

"""BiRefNet refiner — per-frame alpha using the official HF inference path.

Sequence feeding (Colab / ``ai_matte_colab_run.py``):

1. ``OIIOExrReader`` (+ display transform) reads each plate frame → (H,W,3) [0,1]
2. SAM3 ``sam3_matte`` produces per-track hard masks (T,H,W)
3. This pass reads the **same** plate stack and, per hero track:
   - ``full_frame`` (default): BiRefNet on the **entire** plate (matches README/handler)
     then multiply by SAM3 mask to keep the tracked instance
   - ``crop``: BiRefNet on SAM3 bbox crop only (less VRAM, can miss context)

BiRefNet is **not** temporal. With ``fill_between_keyframes: false`` (AI Matte default),
only keyframes carry alpha; ``matte_flow_temporal`` post fills the shot using RAFT.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.passes.matte.birefnet_infer import BiRefNetSession

_SLOT_TO_CHANNEL: dict[str, str] = {
    "r": CH_MATTE_R,
    "g": CH_MATTE_G,
    "b": CH_MATTE_B,
    "a": CH_MATTE_A,
}


def _resolve_birefnet_source(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = params.get("model_path") or os.environ.get("AI_MATTE_BIREFNET_MODEL_PATH")
    if raw:
        path = Path(str(raw)).expanduser().resolve()
        if path.is_dir() and (path / "config.json").is_file():
            return str(path), {"local_files_only": True, "trust_remote_code": True}
        raise FileNotFoundError(f"BiRefNet model_path missing config.json: {path}")
    return str(params.get("model_id", "ZhengPeng7/BiRefNet")), {"trust_remote_code": True}


def _mask_bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.25)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad)
    x1 = min(w, x1 + pad)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    return x0, y0, x1, y1


class BiRefNetRefinerPass(UtilityPass):
    name = "birefnet_refiner"
    version = "0.2.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=False,
        notes="BiRefNet (ZhengPeng7/BiRefNet) — verify upstream license for commercial use.",
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_MATTE_R, description="Hero matte slot R (BiRefNet soft alpha)"),
        ChannelSpec(name=CH_MATTE_G, description="Hero matte slot G (BiRefNet soft alpha)"),
        ChannelSpec(name=CH_MATTE_B, description="Hero matte slot B (BiRefNet soft alpha)"),
        ChannelSpec(name=CH_MATTE_A, description="Hero matte slot A (BiRefNet soft alpha)"),
    ]
    requires_artifacts = ["sam3_hard_masks", "sam3_instances"]
    provides_artifacts = ["matte_heroes"]
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "ZhengPeng7/BiRefNet",
        "model_path": None,
        # full_frame = official HF path (whole plate, then * SAM3 mask)
        # crop = legacy bbox crop (faster, less like README)
        "inference_mode": "full_frame",
        "crop_pad": 32,
        "inference_size": 1024,
        "keyframe_stride": 4,
        "hard_mask_dilate": 5,
        "refine_foreground": True,
        "refine_radius": 90,
        "precision": "fp16",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._session: BiRefNetSession | None = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []
        self._refined: list[dict[str, Any]] = []

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        if hard:
            self._hard_masks = next(iter(hard.values())) or {}
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            self._heroes = list(next(iter(heroes.values())) or [])

    def _get_session(self) -> BiRefNetSession:
        if self._session is None:
            repo, load_kw = _resolve_birefnet_source(self.params)
            self._session = BiRefNetSession.from_pretrained(
                repo,
                load_kw=load_kw,
                precision=str(self.params.get("precision", "fp16")),
            )
        return self._session

    def _dilate_stack(self, hard_stack: np.ndarray) -> np.ndarray:
        dilate = max(0, int(self.params.get("hard_mask_dilate", 5)))
        if dilate <= 0:
            return (hard_stack > 0.5).astype(np.float32)
        import cv2

        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        return np.stack(
            [
                cv2.dilate((hard_stack[t] > 0.5).astype(np.uint8), kernel).astype(
                    np.float32
                )
                for t in range(hard_stack.shape[0])
            ],
            axis=0,
        )

    def _alpha_full_frame(
        self,
        session: BiRefNetSession,
        plate_rgb: np.ndarray,
        instance_mask: np.ndarray,
    ) -> np.ndarray:
        hint = instance_mask if float(instance_mask.sum()) > 0 else None
        return session.predict_alpha(
            plate_rgb,
            inference_size=int(self.params.get("inference_size", 1024)),
            instance_mask=hint,
            refine_foreground=bool(self.params.get("refine_foreground", True)),
            refine_radius=int(self.params.get("refine_radius", 90)),
        )

    def _alpha_crop(
        self,
        session: BiRefNetSession,
        plate_rgb: np.ndarray,
        instance_mask: np.ndarray,
        pad: int,
    ) -> np.ndarray:
        h, w = plate_rgb.shape[:2]
        x0, y0, x1, y1 = _mask_bbox(instance_mask, pad)
        crop_rgb = plate_rgb[y0:y1, x0:x1, :]
        crop_hint = instance_mask[y0:y1, x0:x1]
        alpha_crop = session.predict_alpha(
            crop_rgb,
            inference_size=int(self.params.get("inference_size", 1024)),
            instance_mask=crop_hint,
            refine_foreground=bool(self.params.get("refine_foreground", True)),
            refine_radius=int(self.params.get("refine_radius", 90)),
        )
        full = np.zeros((h, w), dtype=np.float32)
        full[y0:y1, x0:x1] = alpha_crop
        return full

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        T, H, W, _ = plate_stack.shape
        session = self._get_session()
        mode = str(self.params.get("inference_mode", "full_frame")).strip().lower()
        pad = int(self.params.get("crop_pad", 32))
        stride = max(1, int(self.params.get("keyframe_stride", 4)))
        hard_proc = self._dilate_stack(hard_stack)

        key_indices = sorted(
            {t for t in range(T) if t % stride == 0} | {T - 1}
        )
        refined_keys: dict[int, np.ndarray] = {}

        for t in key_indices:
            hard_t = hard_proc[t]
            if float(hard_t.sum()) < 1.0:
                refined_keys[t] = np.zeros((H, W), dtype=np.float32)
                continue
            rgb = plate_stack[t]
            if mode == "crop":
                alpha = self._alpha_crop(session, rgb, hard_t, pad)
            else:
                alpha = self._alpha_full_frame(session, rgb, hard_t)
            refined_keys[t] = alpha

        out = np.zeros((T, H, W), dtype=np.float32)
        if not refined_keys:
            return out
        fill_between = bool(self.params.get("fill_between_keyframes", False))
        for t, alpha in refined_keys.items():
            out[t] = alpha
        if not fill_between:
            return out
        key_list = sorted(refined_keys.keys())
        for t in range(T):
            if t in refined_keys:
                continue
            prev_k = max(k for k in key_list if k <= t)
            next_k = min(k for k in key_list if k >= t)
            if prev_k == next_k:
                out[t] = refined_keys[prev_k]
            else:
                w = (t - prev_k) / max(next_k - prev_k, 1)
                out[t] = (1.0 - w) * refined_keys[prev_k] + w * refined_keys[next_k]
        return out

    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        return {}

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        n_frames = last - first + 1
        # Plate sequence: same OIIO path as SAM3 (EXR/JPG, display transform, proxy).
        frames = np.stack(
            [reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0
        ).astype(np.float32, copy=False)
        if frames.ndim != 4 or frames.shape[-1] < 3:
            raise ValueError(f"Plate stack must be (T,H,W,3+), got {frames.shape}")
        frames = frames[..., :3]
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])

        channel_stacks: dict[str, np.ndarray] = {
            ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
            for ch in _SLOT_TO_CHANNEL.values()
        }
        self._refined = []
        for hero in self._heroes:
            slot = str(hero.get("slot", ""))
            channel = _SLOT_TO_CHANNEL.get(slot)
            if channel is None:
                continue
            track_id = int(hero["track_id"])
            entry = self._hard_masks.get(track_id)
            if not entry:
                continue
            stack = entry.get("stack")
            if stack is None:
                continue
            hard_stack = np.asarray(stack, dtype=np.float32)
            if hard_stack.shape[0] != n_frames:
                continue
            soft = self._refine_instance(frames, hard_stack)
            channel_stacks[channel] = soft
            self._refined.append(
                {
                    "track_id": track_id,
                    "slot": slot,
                    "label": entry.get("label", hero.get("label", "")),
                }
            )

        per_frame: dict[int, dict[str, np.ndarray]] = {}
        for i in range(n_frames):
            f = first + i
            per_frame[f] = {ch: channel_stacks[ch][i] for ch in channel_stacks}
        return per_frame

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._refined:
            return {}
        return {"matte_heroes": {0: list(self._refined)}}


__all__ = ["BiRefNetRefinerPass"]
