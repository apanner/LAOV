# LiveActionAOV — ViTMatte inference (HF transformers).

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def hard_mask_to_trimap(mask: np.ndarray) -> np.ndarray:
    """SAM3 hard mask [0,1] → ViTMatte trimap (0=bg, 128=unknown, 255=fg)."""
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    trimap = np.full(m.shape, 128, dtype=np.uint8)
    trimap[m < 0.05] = 0
    trimap[m > 0.85] = 255
    return trimap


class ViTMatteSession:
    """Lazy-loaded ViTMatte (image + trimap). See HF ViTMatte docs."""

    def __init__(
        self,
        model_id: str,
        *,
        load_kw: dict[str, Any] | None = None,
        precision: str = "fp16",
    ) -> None:
        self._model_id = model_id
        self._load_kw = dict(load_kw or {})
        self._precision = precision
        self._model = None
        self._processor = None
        self._device = None

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        load_kw: dict[str, Any] | None = None,
        precision: str = "fp16",
    ) -> ViTMatteSession:
        return cls(repo_or_path, load_kw=load_kw, precision=precision)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = VitMatteImageProcessor.from_pretrained(
            self._model_id, **{k: v for k, v in self._load_kw.items() if k != "trust_remote_code"}
        )
        self._model = VitMatteForImageMatting.from_pretrained(self._model_id, **self._load_kw)
        self._model.eval()
        self._model.to(self._device)
        if self._precision == "fp16" and self._device.type == "cuda":
            self._model.half()

    def predict_alpha(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray,
        *,
        instance_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return alpha (H,W) float32 in [0,1] at plate resolution."""
        import torch
        from PIL import Image

        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        rgb_u8 = (np.clip(rgb[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        if trimap.shape[:2] != (h, w):
            from PIL import Image as PILImage

            trimap = np.array(
                PILImage.fromarray(trimap).resize((w, h), PILImage.NEAREST),
                dtype=np.uint8,
            )
        pil_rgb = Image.fromarray(rgb_u8, "RGB")
        pil_tri = Image.fromarray(trimap, "L")

        inputs = self._processor(images=pil_rgb, trimaps=pil_tri, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if self._precision == "fp16" and self._device.type == "cuda":
            inputs = {
                k: v.half() if v.is_floating_point() else v for k, v in inputs.items()
            }

        with torch.no_grad():
            alphas = self._model(**inputs).alphas

        from PIL import Image as PILImage

        alpha = alphas.detach().float().cpu()[0, 0].numpy()
        alpha_pil = PILImage.fromarray(
            (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
        )
        alpha = (
            np.array(alpha_pil.resize((w, h), PILImage.BILINEAR), dtype=np.float32) / 255.0
        )
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        if instance_mask is not None:
            m = np.clip(instance_mask.astype(np.float32), 0.0, 1.0)
            if m.shape[:2] != (h, w):
                from PIL import Image as PILImage

                m = (
                    np.array(
                        PILImage.fromarray((m * 255).astype(np.uint8)).resize(
                            (w, h), PILImage.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                    / 255.0
                )
            alpha = alpha * m
        return alpha


__all__ = ["ViTMatteSession", "hard_mask_to_trimap"]
