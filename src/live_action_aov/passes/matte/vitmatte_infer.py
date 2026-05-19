# LiveActionAOV — ViTMatte inference (HF transformers).

"""ViTMatte expects a trimap (0=bg, 128=unknown, 255=fg) built from SAM3 hard masks.

We erode the SAM3 mask for definite foreground and dilate for the unknown band —
same idea as hustvl/Matte-Anything and the ViTMatte README trimap recipes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def sam3_mask_to_trimap(
    mask: np.ndarray,
    *,
    erode_px: int = 10,
    dilate_px: int = 25,
    erode_iterations: int = 1,
    dilate_iterations: int = 1,
    fg_threshold: float = 0.5,
) -> np.ndarray:
    """Build a ViTMatte trimap from a SAM3 hard mask (float or uint8, H×W).

    Morphology (elliptical kernels, Matte-Anything style):

    - **255 (fg):** eroded SAM3 core — definite foreground
    - **128 (unknown):** dilated ring minus eroded core — where ViTMatte refines edges
    - **0 (bg):** outside the dilated mask

    ``erode_px`` / ``dilate_px`` are kernel radii in pixels (at plate resolution).
  """
    if cv2 is None:
        raise ImportError("opencv-python-headless required for ViTMatte trimap morphology")

    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    hard = (m > fg_threshold).astype(np.uint8) * 255
    h, w = hard.shape[:2]
    trimap = np.zeros((h, w), dtype=np.uint8)

    if int(hard.sum()) < 1:
        return trimap

    if dilate_px > 0:
        k_d = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        dilated = cv2.dilate(hard, k_d, iterations=max(1, dilate_iterations))
    else:
        dilated = hard

    if erode_px > 0:
        k_e = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        eroded = cv2.erode(hard, k_e, iterations=max(1, erode_iterations))
    else:
        eroded = hard

    # Unknown band first, then overwrite core with definite fg (Matte-Anything order).
    trimap[dilated > 0] = 128
    trimap[eroded > 0] = 255
    return trimap


# Backward-compatible alias
hard_mask_to_trimap = sam3_mask_to_trimap


def _fit_size(h: int, w: int, max_long_edge: int) -> tuple[int, int]:
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return h, w
    scale = max_long_edge / float(long_edge)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _resize_plane(
    arr: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    nearest: bool = False,
) -> np.ndarray:
    from PIL import Image

    resample = Image.NEAREST if nearest else Image.BILINEAR
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_in = Image.fromarray(rgb.astype(np.uint8), "RGB")
        return np.asarray(pil_in.resize((out_w, out_h), resample))
    plane = arr[..., 0] if arr.ndim == 3 else arr
    if plane.dtype != np.uint8:
        pil_in = Image.fromarray((np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8), "L")
    else:
        pil_in = Image.fromarray(plane.astype(np.uint8), "L")
    return np.asarray(pil_in.resize((out_w, out_h), resample))


class ViTMatteSession:
    """Lazy-loaded ViTMatte (image + trimap). See HF ViTMatte docs."""

    def __init__(
        self,
        model_id: str,
        *,
        load_kw: dict[str, Any] | None = None,
        precision: str = "fp16",
        max_inference_long_edge: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._load_kw = dict(load_kw or {})
        self._precision = precision
        self._max_inference_long_edge = max(256, int(max_inference_long_edge))
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
        max_inference_long_edge: int = 1024,
    ) -> ViTMatteSession:
        return cls(
            repo_or_path,
            load_kw=load_kw,
            precision=precision,
            max_inference_long_edge=max_inference_long_edge,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proc_kw = {k: v for k, v in self._load_kw.items() if k != "trust_remote_code"}
        self._processor = VitMatteImageProcessor.from_pretrained(self._model_id, **proc_kw)
        self._model = VitMatteForImageMatting.from_pretrained(self._model_id, **self._load_kw)
        self._model.eval()
        self._model.to(self._device)
        if self._precision == "fp16" and self._device.type == "cuda":
            self._model.half()

    def release(self) -> None:
        """Drop weights/processor so the next pipeline stage can use VRAM."""
        if self._model is not None:
            try:
                import torch

                self._model.cpu()
                del self._model
            except Exception:
                pass
            self._model = None
        self._processor = None

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

        infer_h, infer_w = _fit_size(h, w, self._max_inference_long_edge)
        if (infer_h, infer_w) != (h, w):
            rgb_u8 = _resize_plane(rgb_u8, infer_h, infer_w, nearest=False).astype(np.uint8)
            trimap = _resize_plane(trimap, infer_h, infer_w, nearest=True).astype(np.uint8)

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
        del inputs

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


__all__ = ["ViTMatteSession", "sam3_mask_to_trimap", "hard_mask_to_trimap"]
