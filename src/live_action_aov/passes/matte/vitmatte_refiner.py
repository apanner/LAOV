# LiveActionAOV — ViTMatte refiner (trimap from SAM3, HF transformers).

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.passes.matte.keyframe_fill import fill_alpha_between_keyframes
from live_action_aov.passes.matte.vitmatte_infer import ViTMatteSession, sam3_mask_to_trimap
from live_action_aov.passes.matte.vitmatte_plate_cache import (
    build_plate_jpeg_cache,
    read_plate_jpeg,
)

_log = logging.getLogger(__name__)

CH_VITMATTE_R = "vitmatte.r"
CH_VITMATTE_G = "vitmatte.g"
CH_VITMATTE_B = "vitmatte.b"
CH_VITMATTE_A = "vitmatte.a"
VITMATTE_CHANNELS = (CH_VITMATTE_R, CH_VITMATTE_G, CH_VITMATTE_B, CH_VITMATTE_A)

_SLOT_TO_CHANNEL: dict[str, str] = {
    "r": CH_VITMATTE_R,
    "g": CH_VITMATTE_G,
    "b": CH_VITMATTE_B,
    "a": CH_VITMATTE_A,
}


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


def _resolve_vitmatte_source(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = params.get("model_path") or os.environ.get("AI_MATTE_VITMATTE_MODEL_PATH")
    if raw:
        path = Path(str(raw)).expanduser().resolve()
        if path.is_dir() and (path / "config.json").is_file():
            return str(path), {"local_files_only": True}
        raise FileNotFoundError(f"ViTMatte model_path missing config.json: {path}")
    return str(params.get("model_id", "hustvl/vitmatte-base-composition-1k")), {}


class ViTMatteRefinerPass(UtilityPass):
    name = "vitmatte_refiner"
    version = "0.2.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=False,
        notes="ViTMatte (hustvl) via transformers — verify license for production.",
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_VITMATTE_R, description="Hero matte R (ViTMatte)"),
        ChannelSpec(name=CH_VITMATTE_G, description="Hero matte G (ViTMatte)"),
        ChannelSpec(name=CH_VITMATTE_B, description="Hero matte B (ViTMatte)"),
        ChannelSpec(name=CH_VITMATTE_A, description="Hero matte A (ViTMatte)"),
    ]
    requires_artifacts = ["sam3_hard_masks", "sam3_instances"]
    provides_artifacts = ["vitmatte_heroes"]
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "hustvl/vitmatte-base-composition-1k",
        "model_path": None,
        "keyframe_stride": 1,
        "fill_between_keyframes": True,
        "hard_mask_dilate": 5,
        "trimap_erode_px": 10,
        "trimap_dilate_px": 25,
        "trimap_erode_iterations": 1,
        "trimap_dilate_iterations": 1,
        "trimap_fg_threshold": 0.5,
        "precision": "fp16",
        # 0 = auto from GPU VRAM (A100 40GB → ~2048, T4 → ~1024). Set explicit int to override.
        "max_inference_long_edge": 0,
        "inference_mode": "crop",
        "crop_pad": 32,
        # Batch: EXR → full-res JPEG on disk, one frame in RAM per infer step.
        "use_plate_jpeg_cache": True,
        "plate_cache_dir": None,
        "plate_cache_keep": False,
        "plate_jpeg_quality": 92,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._session: ViTMatteSession | None = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []
        self._plate_cache_dir: Path | None = None
        self._plate_cache_temp: Path | None = None

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        if hard:
            self._hard_masks = next(iter(hard.values())) or {}
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            self._heroes = list(next(iter(heroes.values())) or [])

    def _dilate_frame(self, hard_t: np.ndarray) -> np.ndarray:
        dilate = max(0, int(self.params.get("hard_mask_dilate", 5)))
        if dilate <= 0 or cv2 is None:
            return (hard_t > 0.5).astype(np.float32)
        import cv2 as _cv2

        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
        )
        return _cv2.dilate((hard_t > 0.5).astype(np.uint8), kernel).astype(np.float32)

    def _trimap_from_hard(self, hard_t: np.ndarray) -> np.ndarray:
        return sam3_mask_to_trimap(
            hard_t,
            erode_px=int(self.params.get("trimap_erode_px", 10)),
            dilate_px=int(self.params.get("trimap_dilate_px", 25)),
            erode_iterations=int(self.params.get("trimap_erode_iterations", 1)),
            dilate_iterations=int(self.params.get("trimap_dilate_iterations", 1)),
            fg_threshold=float(self.params.get("trimap_fg_threshold", 0.5)),
        )

    def _resolved_long_edge(self) -> int:
        from live_action_aov.shared.gpu_inference_budget import resolve_vitmatte_long_edge

        return resolve_vitmatte_long_edge(self.params.get("max_inference_long_edge"))

    def _get_session(self) -> ViTMatteSession:
        if self._session is None:
            repo, load_kw = _resolve_vitmatte_source(self.params)
            long_edge = self._resolved_long_edge()
            self.params["max_inference_long_edge"] = long_edge
            self._session = ViTMatteSession.from_pretrained(
                repo,
                load_kw=load_kw,
                precision=str(self.params.get("precision", "fp16")),
                max_inference_long_edge=long_edge,
            )
        return self._session

    def _predict_frame(
        self,
        session: ViTMatteSession,
        rgb: np.ndarray,
        hard_t: np.ndarray,
    ) -> np.ndarray:
        h, w = rgb.shape[:2]
        if float(hard_t.sum()) < 1.0:
            return np.zeros((h, w), dtype=np.float32)
        mode = str(self.params.get("inference_mode", "crop")).strip().lower()
        pad = int(self.params.get("crop_pad", 32))
        if mode == "crop":
            x0, y0, x1, y1 = _mask_bbox(hard_t, pad)
            crop_rgb = rgb[y0:y1, x0:x1, :]
            crop_hard = hard_t[y0:y1, x0:x1]
            trimap = self._trimap_from_hard(crop_hard)
            alpha_crop = session.predict_alpha(
                crop_rgb,
                trimap,
                instance_mask=crop_hard,
            )
            full = np.zeros((h, w), dtype=np.float32)
            full[y0:y1, x0:x1] = alpha_crop
            return full
        trimap = self._trimap_from_hard(hard_t)
        return session.predict_alpha(rgb, trimap, instance_mask=hard_t)

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
            cache_dir = Path(tempfile.mkdtemp(prefix="vitmatte_plate_"))
            self._plate_cache_dir = cache_dir
            self._plate_cache_temp = cache_dir

        _log.info("ViTMatte: building full-res JPEG plate cache → %s", cache_dir)
        first, last = frame_range
        n = last - first + 1
        build_plate_jpeg_cache(
            reader.read_frame,
            frame_range,
            cache_dir,
            jpeg_quality=int(self.params.get("plate_jpeg_quality", 92)),
            log_every=max(10, n // 20),
            log_label="ViTMatte",
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
        key_indices = sorted({t for t in range(T) if t % stride == 0} | {T - 1})
        refined_keys: dict[int, np.ndarray] = {}
        n_keys = len(key_indices)
        log_step = max(1, n_keys // 10)

        for ki, t in enumerate(key_indices):
            if ki == 0 or ki == n_keys - 1 or (ki % log_step) == 0:
                _log.info(
                    "ViTMatte: keyframe %d/%d (local %d / %d)",
                    ki + 1,
                    n_keys,
                    t,
                    T,
                )
            hard_t = self._dilate_frame(hard_stack[t])
            if float(hard_t.sum()) < 1.0:
                refined_keys[t] = np.zeros((H, W), dtype=np.float32)
                continue
            frame_idx = first_frame + t
            rgb = read_plate_at(frame_idx)
            refined_keys[t] = self._predict_frame(session, rgb, hard_t)
            from live_action_aov.executors.gpu_release import clear_vram_cache

            clear_vram_cache(sync=False)

        fill_between = bool(self.params.get("fill_between_keyframes", True))
        fallback = None
        if not fill_between:
            fallback = np.stack(
                [self._dilate_frame(hard_stack[t]) for t in range(T)],
                axis=0,
            )
        return fill_alpha_between_keyframes(
            T,
            refined_keys,
            fill_between=fill_between,
            fallback_stack=fallback,
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
        long_edge = self._resolved_long_edge()
        self.params["max_inference_long_edge"] = long_edge
        _log.info(
            "ViTMatte: streaming %d frames (%s-%s), infer long_edge<=%s, mode=%s",
            n_frames,
            first,
            last,
            long_edge,
            self.params.get("inference_mode", "crop"),
        )

        cache_dir = self._setup_plate_cache(reader, frame_range)
        try:
            probe = self._read_plate(first, reader=reader, cache_dir=cache_dir)
            plate_h, plate_w = int(probe.shape[0]), int(probe.shape[1])

            channel_stacks: dict[str, np.ndarray] = {
                ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
                for ch in _SLOT_TO_CHANNEL.values()
            }

            def read_plate_at(frame_idx: int) -> np.ndarray:
                return self._read_plate(frame_idx, reader=reader, cache_dir=cache_dir)

            _log.info("ViTMatte: refining %d hero matte(s)", len(self._heroes))
            for hero in self._heroes:
                slot = str(hero.get("slot", ""))
                channel = _SLOT_TO_CHANNEL.get(slot)
                if channel is None:
                    continue
                track_id = int(hero["track_id"])
                entry = self._hard_masks.get(track_id)
                if not entry or entry.get("stack") is None:
                    continue
                hard_stack = np.asarray(entry["stack"], dtype=np.float32)
                if hard_stack.shape[0] != n_frames:
                    continue
                _log.info("ViTMatte: hero track_id=%s slot=%s", track_id, slot)
                channel_stacks[channel] = self._refine_instance_streaming(
                    read_plate_at,
                    hard_stack,
                    first_frame=first,
                    n_frames=n_frames,
                )

            per_frame: dict[int, dict[str, np.ndarray]] = {}
            for i in range(n_frames):
                f = first + i
                per_frame[f] = {ch: channel_stacks[ch][i] for ch in channel_stacks}
            return per_frame
        finally:
            self._cleanup_plate_cache()

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._heroes:
            return {}
        return {"vitmatte_heroes": {0: list(self._heroes)}}


__all__ = ["ViTMatteRefinerPass", "VITMATTE_CHANNELS"]
