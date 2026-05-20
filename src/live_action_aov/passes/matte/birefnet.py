# LiveActionAOV — BiRefNet soft-alpha refiner (AI Matte / colab_ai lane)

"""BiRefNet refiner — per-frame alpha using the official HF inference path.

Sequence feeding (Colab / ``ai_matte_colab_run.py``):

1. ``OIIOExrReader`` (+ display transform) reads each plate frame → (H,W,3) [0,1]
2. SAM3 ``sam3_matte`` produces per-track hard masks (T,H,W)
3. This pass reads plates **one frame at a time** (JPEG cache on disk for long 4K shots)
   and, per hero track, runs BiRefNet on keyframes then fills between.

BiRefNet is **not** temporal. With ``keyframe_stride`` > 1, inference runs on sparse
keyframes only; ``fill_between_keyframes: true`` (AI Matte stage default) linearly
interpolates alpha on in-between frames. Use ``matte_flow_temporal`` + RAFT for
motion-aware fill instead of linear blend.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
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
from live_action_aov.passes.matte.keyframe_fill import fill_alpha_between_keyframes
from live_action_aov.passes.matte.vitmatte_plate_cache import (
    build_plate_jpeg_cache,
    read_plate_jpeg,
)

_log = logging.getLogger(__name__)

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
    version = "0.3.0"
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
        "inference_mode": "full_frame",
        "crop_pad": 32,
        "inference_size": 1024,
        "keyframe_stride": 1,
        "fill_between_keyframes": True,
        "hard_mask_dilate": 5,
        "refine_foreground": True,
        "refine_radius": 90,
        "precision": "fp16",
        "use_plate_jpeg_cache": True,
        "plate_cache_dir": None,
        "plate_cache_keep": False,
        "plate_jpeg_quality": 92,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._session: BiRefNetSession | None = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []
        self._refined: list[dict[str, Any]] = []
        self._plate_cache_dir: Path | None = None
        self._plate_cache_temp: Path | None = None

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

    def _dilate_frame(self, hard_t: np.ndarray) -> np.ndarray:
        dilate = max(0, int(self.params.get("hard_mask_dilate", 5)))
        if dilate <= 0:
            return (hard_t > 0.5).astype(np.float32)
        import cv2

        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        return cv2.dilate((hard_t > 0.5).astype(np.uint8), kernel).astype(np.float32)

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

    def _predict_frame(
        self,
        session: BiRefNetSession,
        rgb: np.ndarray,
        hard_t: np.ndarray,
    ) -> np.ndarray:
        mode = str(self.params.get("inference_mode", "full_frame")).strip().lower()
        pad = int(self.params.get("crop_pad", 32))
        hard_proc = self._dilate_frame(hard_t)
        if float(hard_proc.sum()) < 1.0:
            return np.zeros(rgb.shape[:2], dtype=np.float32)
        if mode == "crop":
            return self._alpha_crop(session, rgb, hard_proc, pad)
        return self._alpha_full_frame(session, rgb, hard_proc)

    def _read_plate(
        self,
        frame_idx: int,
        *,
        reader: Any,
        cache_dir: Path | None,
    ) -> np.ndarray:
        if cache_dir is not None:
            return read_plate_jpeg(cache_dir, frame_idx)
        rgb, _attrs = reader.read_frame(frame_idx)
        return np.clip(np.asarray(rgb, dtype=np.float32)[..., :3], 0.0, 1.0)

    def _setup_plate_cache(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> Path | None:
        if not bool(self.params.get("use_plate_jpeg_cache", True)):
            return None
        explicit = self.params.get("plate_cache_dir")
        if explicit:
            cache_dir = Path(str(explicit)).expanduser().resolve()
            self._plate_cache_dir = cache_dir
            self._plate_cache_temp = None
        else:
            cache_dir = Path(tempfile.mkdtemp(prefix="birefnet_plate_"))
            self._plate_cache_dir = cache_dir
            self._plate_cache_temp = cache_dir

        first, last = frame_range
        n = last - first + 1
        log_every = max(10, n // 20)
        _log.info("BiRefNet: building plate JPEG cache → %s", cache_dir)
        build_plate_jpeg_cache(
            reader.read_frame,
            frame_range,
            cache_dir,
            jpeg_quality=int(self.params.get("plate_jpeg_quality", 92)),
            log_every=log_every,
            log_label="BiRefNet",
        )
        return cache_dir

    def _cleanup_plate_cache(self) -> None:
        if bool(self.params.get("plate_cache_keep", False)):
            return
        if self._plate_cache_temp is not None and self._plate_cache_temp.is_dir():
            shutil.rmtree(self._plate_cache_temp, ignore_errors=True)
        self._plate_cache_temp = None

    def _refine_instance_streaming(
        self,
        read_plate_at: Callable[[int], np.ndarray],
        hard_stack: np.ndarray,
        *,
        first_frame: int,
        n_frames: int,
    ) -> np.ndarray:
        T, H, W = int(hard_stack.shape[0]), int(hard_stack.shape[1]), int(hard_stack.shape[2])
        session = self._get_session()
        stride = max(1, int(self.params.get("keyframe_stride", 1)))
        hard_proc = self._dilate_stack(hard_stack)
        key_indices = sorted({t for t in range(T) if t % stride == 0} | {T - 1})
        refined_keys: dict[int, np.ndarray] = {}
        n_keys = len(key_indices)
        log_step = max(1, n_keys // 10)

        for ki, t in enumerate(key_indices):
            if ki == 0 or ki == n_keys - 1 or (ki % log_step) == 0:
                _log.info(
                    "BiRefNet: keyframe %d/%d (frame %s / %s)",
                    ki + 1,
                    n_keys,
                    first_frame + t,
                    first_frame + n_frames - 1,
                )
            hard_t = hard_proc[t]
            if float(hard_t.sum()) < 1.0:
                refined_keys[t] = np.zeros((H, W), dtype=np.float32)
                continue
            rgb = read_plate_at(first_frame + t)
            refined_keys[t] = self._predict_frame(session, rgb, hard_stack[t])
            from live_action_aov.executors.gpu_release import clear_vram_cache

            clear_vram_cache(sync=False)

        fill_between = bool(self.params.get("fill_between_keyframes", True))
        return fill_alpha_between_keyframes(
            T,
            refined_keys,
            fill_between=fill_between,
            fallback_stack=hard_proc if not fill_between else None,
        )

    def release_gpu(self) -> None:
        if self._session is not None:
            self._session.release()
            self._session = None

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
        from live_action_aov.executors.gpu_release import clear_vram_cache

        clear_vram_cache()
        _log.info(
            "BiRefNet: streaming %d frames (%s-%s), stride=%s, mode=%s",
            n_frames,
            first,
            last,
            self.params.get("keyframe_stride", 1),
            self.params.get("inference_mode", "full_frame"),
        )

        cache_dir = self._setup_plate_cache(reader, frame_range)
        try:
            probe = self._read_plate(first, reader=reader, cache_dir=cache_dir)
            plate_h, plate_w = int(probe.shape[0]), int(probe.shape[1])

            channel_stacks: dict[str, np.ndarray] = {
                ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
                for ch in _SLOT_TO_CHANNEL.values()
            }
            self._refined = []

            def read_plate_at(frame_idx: int) -> np.ndarray:
                return self._read_plate(frame_idx, reader=reader, cache_dir=cache_dir)

            _log.info("BiRefNet: refining %d hero matte(s)", len(self._heroes))
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
                _log.info("BiRefNet: hero track_id=%s slot=%s", track_id, slot)
                channel_stacks[channel] = self._refine_instance_streaming(
                    read_plate_at,
                    hard_stack,
                    first_frame=first,
                    n_frames=n_frames,
                )
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
        finally:
            self._cleanup_plate_cache()

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._refined:
            return {}
        return {"matte_heroes": {0: list(self._refined)}}


__all__ = ["BiRefNetRefinerPass"]
