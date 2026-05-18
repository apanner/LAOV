# LiveActionAOV — BiRefNet soft-alpha refiner (AI Matte / colab_ai lane)

"""BiRefNet refiner — high-quality per-frame alpha from SAM3 hard masks.

Uses a local Hugging Face snapshot (``VDA_models/ZhengPeng7/BiRefNet`` on Drive).
Runs on bbox crops around each tracked instance; optional keyframe_stride
interpolates soft mattes on in-between frames.
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
    repo = str(params.get("model_id", "ZhengPeng7/BiRefNet"))
    return repo, {"trust_remote_code": True}


def _mask_bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    """Return x0, y0, x1, y1 inclusive-exclusive with padding."""
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
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=False,
        notes=(
            "BiRefNet weights (ZhengPeng7/BiRefNet) — verify upstream license "
            "for commercial deliverables before production ship."
        ),
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
        "crop_pad": 32,
        "inference_size": 1024,
        "keyframe_stride": 1,
        "hard_mask_dilate": 5,
        "multiply_sam3_hint": True,
        "precision": "fp16",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
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

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageSegmentation

        repo, load_kw = _resolve_birefnet_source(self.params)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = (
            self.params.get("precision") == "fp16" and self._device.type == "cuda"
        )
        model = AutoModelForImageSegmentation.from_pretrained(repo, **load_kw)
        model.to(self._device)
        if use_fp16:
            model.half()
        model.eval()
        self._model = model
        self._use_fp16 = use_fp16

    def _run_birefnet_on_crop(
        self,
        rgb_crop: np.ndarray,
        hint_crop: np.ndarray | None,
    ) -> np.ndarray:
        """RGB crop (H,W,3) float [0,1] -> alpha (H,W) float [0,1]."""
        import torch
        from PIL import Image
        from torchvision import transforms

        self._load_model()
        assert self._model is not None
        device = self._device
        size = int(self.params.get("inference_size", 1024))

        arr_u8 = (np.clip(rgb_crop, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8, "RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(pil).unsqueeze(0)
        if self._use_fp16:
            tensor = tensor.half()
        tensor = tensor.to(device)

        with torch.no_grad():
            preds = self._model(tensor)[-1].sigmoid().cpu()
        alpha = preds[0].squeeze().float().numpy()
        alpha_pil = transforms.ToPILImage()(alpha)
        alpha_pil = alpha_pil.resize((rgb_crop.shape[1], rgb_crop.shape[0]), Image.BILINEAR)
        out = np.array(alpha_pil, dtype=np.float32) / 255.0
        if hint_crop is not None and self.params.get("multiply_sam3_hint", True):
            out = out * np.clip(hint_crop.astype(np.float32), 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        """BiRefNet on padded bbox crops; keyframe_stride + hold/interp."""
        T, H, W, _ = plate_stack.shape
        pad = int(self.params.get("crop_pad", 32))
        stride = max(1, int(self.params.get("keyframe_stride", 1)))
        dilate = max(0, int(self.params.get("hard_mask_dilate", 5)))

        if dilate > 0:
            import cv2

            kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
            hard_proc = np.stack(
                [cv2.dilate((hard_stack[t] > 0.5).astype(np.uint8), kernel) for t in range(T)],
                axis=0,
            ).astype(np.float32)
        else:
            hard_proc = (hard_stack > 0.5).astype(np.float32)

        key_indices = [t for t in range(T) if t % stride == 0]
        if (T - 1) not in key_indices:
            key_indices.append(T - 1)
        key_indices = sorted(set(key_indices))

        refined_keys: dict[int, np.ndarray] = {}
        for t in key_indices:
            hard_t = hard_proc[t]
            if float(hard_t.sum()) < 1.0:
                refined_keys[t] = np.zeros((H, W), dtype=np.float32)
                continue
            x0, y0, x1, y1 = _mask_bbox(hard_t, pad)
            crop_rgb = plate_stack[t, y0:y1, x0:x1, :]
            crop_hint = hard_t[y0:y1, x0:x1]
            alpha_crop = self._run_birefnet_on_crop(crop_rgb, crop_hint)
            full = np.zeros((H, W), dtype=np.float32)
            full[y0:y1, x0:x1] = alpha_crop
            refined_keys[t] = full

        out = np.zeros((T, H, W), dtype=np.float32)
        if not refined_keys:
            return out
        key_list = sorted(refined_keys.keys())
        for t in range(T):
            if t in refined_keys:
                out[t] = refined_keys[t]
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
        frames = np.stack(
            [reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0
        ).astype(np.float32, copy=False)
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
            per_frame[f] = {
                ch: channel_stacks[ch][i] for ch in channel_stacks
            }
        return per_frame

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._refined:
            return {}
        return {"matte_heroes": {0: list(self._refined)}}


__all__ = ["BiRefNetRefinerPass"]
