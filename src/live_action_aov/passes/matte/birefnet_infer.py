# LiveActionAOV — BiRefNet inference aligned with HF snapshot (colab_ai/BiRefNet).

"""Official BiRefNet usage (ZhengPeng7/BiRefNet README + handler.py):

1. RGB image (PIL), convert to RGB
2. Resize to (1024, 1024), ImageNet normalize
3. ``preds = model(tensor)[-1].sigmoid()``
4. Resize alpha back to **original image size** (bilinear)
5. Optional: ``refine_foreground`` (FB blur fusion) for cleaner edges

BiRefNet is **single-image** salient segmentation (DIS), not video. For sequences,
call this once per frame (or keyframe) on plates from ``OIIOExrReader``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def refine_foreground_rgb(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    radius: int = 90,
) -> np.ndarray:
    """Photoroom-style foreground refinement from HF ``handler.py``."""
    if cv2 is None:
        return alpha
    from PIL import Image

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image_rgb.shape}")
    h, w = image_rgb.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha_u8 = np.array(
            Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    else:
        alpha_u8 = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    image = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)
    mask = alpha_u8[:, :, None]
    estimated = _fb_blur_fusion_foreground_estimator_2(image, mask, r=radius)
    return np.clip(estimated.mean(axis=2) if estimated.ndim == 3 else estimated, 0.0, 1.0)


def _fb_blur_fusion_foreground_estimator_2(
    image: np.ndarray, alpha: np.ndarray, *, r: int = 90
) -> np.ndarray:
    f_est, blur_b = _fb_blur_fusion_foreground_estimator(image, image, image, alpha, r)
    return _fb_blur_fusion_foreground_estimator(image, f_est, blur_b, alpha, r=6)[0]


def _fb_blur_fusion_foreground_estimator(
    image: np.ndarray,
    f: np.ndarray,
    b: np.ndarray,
    alpha: np.ndarray,
    *,
    r: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        return image, image
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
    blurred_fa = cv2.blur(f * alpha, (r, r))
    blurred_f = blurred_fa / (blurred_alpha + 1e-5)
    blurred_b1a = cv2.blur(b * (1.0 - alpha), (r, r))
    blurred_b = blurred_b1a / ((1.0 - blurred_alpha) + 1e-5)
    f_out = blurred_f + alpha * (image - alpha * blurred_f - (1.0 - alpha) * blurred_b)
    return np.clip(f_out, 0.0, 1.0), blurred_b


class BiRefNetSession:
    """Loaded model + preprocess matching ``handler.py`` / README."""

    def __init__(self, model: Any, *, device: Any, use_fp16: bool) -> None:
        import torch
        from torchvision import transforms

        torch.set_float32_matmul_precision("high")
        self._model = model
        self._device = device
        self._use_fp16 = use_fp16 and device.type == "cuda"
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        load_kw: dict[str, Any] | None = None,
        precision: str = "fp16",
    ) -> BiRefNetSession:
        import torch
        from transformers import AutoModelForImageSegmentation

        load_kw = dict(load_kw or {})
        load_kw.setdefault("trust_remote_code", True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForImageSegmentation.from_pretrained(repo_or_path, **load_kw)
        model.to(device)
        use_fp16 = precision == "fp16" and device.type == "cuda"
        if use_fp16:
            model.half()
        model.eval()
        return cls(model, device=device, use_fp16=use_fp16)

    def predict_alpha(
        self,
        rgb: np.ndarray,
        *,
        inference_size: int = 1024,
        instance_mask: np.ndarray | None = None,
        refine_foreground: bool = True,
        refine_radius: int = 90,
    ) -> np.ndarray:
        """Run BiRefNet on one RGB plate (H,W,3) float in [0,1].

        ``instance_mask`` (optional): SAM3 hard mask same resolution — after DIS
        saliency, alpha is multiplied so only the tracked instance remains.
        """
        import torch
        from PIL import Image
        from torchvision import transforms

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"rgb must be HxWx3+, got {rgb.shape}")
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        rgb3 = np.clip(rgb[..., :3], 0.0, 1.0)
        arr_u8 = (rgb3 * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8, "RGB")

        size = int(inference_size)
        pil_in = pil.resize((size, size), Image.BILINEAR)
        tensor = self._transform(pil_in).unsqueeze(0)
        if self._use_fp16:
            tensor = tensor.half()
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)
            # Training returns nested tuples; inference: list of maps, take last
            if isinstance(out, (list, tuple)):
                pred = out[-1]
            else:
                pred = out
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            alpha = pred.sigmoid().detach().cpu().float()

        alpha = alpha[0].squeeze().numpy()
        alpha_pil = transforms.ToPILImage()(alpha)
        alpha_full = np.array(
            alpha_pil.resize((w, h), Image.BILINEAR), dtype=np.float32
        ) / 255.0

        if instance_mask is not None:
            m = np.clip(instance_mask.astype(np.float32), 0.0, 1.0)
            if m.shape[:2] != (h, w):
                m = np.array(
                    Image.fromarray((m * 255).astype(np.uint8)).resize(
                        (w, h), Image.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
            alpha_full = alpha_full * m

        # HF handler uses FB fusion for RGBA preview; EXR matte uses DIS alpha (pred_pil).
        # Optional light edge soften on alpha only (not the foreground-color path).
        if refine_foreground and cv2 is not None:
            a_u8 = (alpha_full * 255.0).astype(np.uint8)
            a_u8 = cv2.bilateralFilter(a_u8, 5, 50, 50)
            alpha_full = a_u8.astype(np.float32) / 255.0

        return np.clip(alpha_full, 0.0, 1.0)


__all__ = ["BiRefNetSession", "refine_foreground_rgb"]
