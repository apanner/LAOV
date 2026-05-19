# LiveActionAOV — ViTMatte refiner (trimap from SAM3, HF transformers).

from __future__ import annotations

import logging
import os
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
from live_action_aov.passes.matte.vitmatte_infer import ViTMatteSession, sam3_mask_to_trimap

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
    version = "0.1.0"
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
        "keyframe_stride": 4,
        # Pre-dilate SAM3 hard mask before trimap (expands fg seed; separate from trimap dilate).
        "hard_mask_dilate": 5,
        "trimap_erode_px": 10,
        "trimap_dilate_px": 25,
        "trimap_erode_iterations": 1,
        "trimap_dilate_iterations": 1,
        "trimap_fg_threshold": 0.5,
        "precision": "fp16",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._session: ViTMatteSession | None = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        if hard:
            self._hard_masks = next(iter(hard.values())) or {}
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            self._heroes = list(next(iter(heroes.values())) or [])

    def _dilate_stack(self, hard_stack: np.ndarray) -> np.ndarray:
        dilate = max(0, int(self.params.get("hard_mask_dilate", 5)))
        if dilate <= 0 or cv2 is None:
            return (hard_stack > 0.5).astype(np.float32)
        import cv2 as _cv2

        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)
        )
        return np.stack(
            [
                _cv2.dilate((hard_stack[t] > 0.5).astype(np.uint8), kernel).astype(np.float32)
                for t in range(hard_stack.shape[0])
            ],
            axis=0,
        )

    def _trimap_from_hard(self, hard_t: np.ndarray) -> np.ndarray:
        return sam3_mask_to_trimap(
            hard_t,
            erode_px=int(self.params.get("trimap_erode_px", 10)),
            dilate_px=int(self.params.get("trimap_dilate_px", 25)),
            erode_iterations=int(self.params.get("trimap_erode_iterations", 1)),
            dilate_iterations=int(self.params.get("trimap_dilate_iterations", 1)),
            fg_threshold=float(self.params.get("trimap_fg_threshold", 0.5)),
        )

    def _get_session(self) -> ViTMatteSession:
        if self._session is None:
            repo, load_kw = _resolve_vitmatte_source(self.params)
            self._session = ViTMatteSession.from_pretrained(
                repo, load_kw=load_kw, precision=str(self.params.get("precision", "fp16"))
            )
        return self._session

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        T, H, W, _ = plate_stack.shape
        session = self._get_session()
        stride = max(1, int(self.params.get("keyframe_stride", 4)))
        hard_proc = self._dilate_stack(hard_stack)
        key_indices = sorted({t for t in range(T) if t % stride == 0} | {T - 1})
        out = np.zeros((T, H, W), dtype=np.float32)
        for t in key_indices:
            hard_t = hard_proc[t]
            if float(hard_t.sum()) < 1.0:
                continue
            trimap = self._trimap_from_hard(hard_t)
            alpha = session.predict_alpha(
                plate_stack[t],
                trimap,
                instance_mask=hard_t,
            )
            out[t] = alpha
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
        _log.info("ViTMatte: reading %d plate frames (%s-%s)…", n_frames, first, last)
        frames = np.stack(
            [reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0
        ).astype(np.float32)[..., :3]
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        channel_stacks = {
            ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
            for ch in _SLOT_TO_CHANNEL.values()
        }
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
            channel_stacks[channel] = self._refine_instance(frames, hard_stack)

        per_frame: dict[int, dict[str, np.ndarray]] = {}
        for i in range(n_frames):
            f = first + i
            per_frame[f] = {ch: channel_stacks[ch][i] for ch in channel_stacks}
        return per_frame

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._heroes:
            return {}
        return {"vitmatte_heroes": {0: list(self._heroes)}}


__all__ = ["ViTMatteRefinerPass", "VITMATTE_CHANNELS"]
