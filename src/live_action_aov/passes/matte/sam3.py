# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""SAM 3 concept detector + video tracker (spec §13.1 Phase 3).

Backend: `facebookresearch/sam3` on HuggingFace. Takes a list of concept
prompts (e.g. `["person", "vehicle", ...]`), detects every instance
matching any concept on a seed frame, then tracks each instance across
the clip with SAM 3's own memory bank.

License: `SAM-License-1.0` (Meta's custom license). Commercial use is
permitted with a prohibition on military / ITAR applications — we ship
with `commercial_use=True` and surface the carve-out in `notes`. The
CLI license gate does not block this pass; users who ship for defense
must consult the upstream license themselves.

Outputs (spec §5.1, channels.py):
- Dynamic `mask.<concept>` channels — **union of all instances** of that
  concept per frame. The ExrSidecarWriter appends them after the canonical
  channel order.

Artifacts:
- `sam3_hard_masks`: `dict[int, {"label": str, "stack": (T, H, W) float32}]`
  keyed by `track_id`. Raw per-instance hard masks across the clip. This
  is the same structure CorridorKey (v2c) will consume, so we ship it in
  the shape the spec §21.8 locks in — don't simplify later.
- `sam3_instances`: a list of `HeroSlot` objects from `rank.py`, sorted
  by slot (r, g, b, a). The refiner consumes this to drive per-slot
  refinement.
- `matte_concepts`: the list of concept names actually detected on this
  clip (so the executor can stamp the list into sidecar metadata).

Temporal mode: VIDEO_CLIP — SAM 3 carries state across frames internally.
Per-clip slot lock (brainstorm decision #3): slots never change mid-clip.

Test plumbing mirrors DepthCrafter: `_load_model` + `_detect_seed` +
`_track_instance` are subclass-override hooks so CI can inject deterministic
rectangles without downloading 2 GB of weights.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]

from live_action_aov.core.pass_base import (
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import MASK_PREFIX, MATTE_CHANNELS
from live_action_aov.passes.matte.rank import (
    HeroOverride,
    Instance,
    RankWeights,
    rank_and_assign,
)

_log = logging.getLogger(__name__)


def _slug_label(label: str) -> str:
    """Safe mask channel suffix from a user matte note (e.g. 'bg yellow guy' → 'bg_yellow_guy')."""
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(label).strip().lower()).strip("_")
    return s or "subject"


def _resolve_model_source(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """HF hub id or local snapshot path + ``from_pretrained`` kwargs."""
    raw_path = params.get("model_path") or os.environ.get("LAOV_SAM3_MODEL_PATH")
    if raw_path:
        path = Path(str(raw_path)).expanduser().resolve()
        if path.is_dir():
            if not (path / "config.json").is_file():
                raise FileNotFoundError(
                    f"SAM3 model_path {path} is missing config.json. "
                    "Upload a full Hugging Face snapshot (see COLAB.md → SAM3 on Drive)."
                )
            return str(path), {"local_files_only": True}
        raise FileNotFoundError(f"SAM3 model_path not found: {path}")
    return str(params.get("model_id", "facebook/sam3")), {}


def _wrap_if_gated_repo(repo: str, exc: BaseException) -> RuntimeError | None:
    """Translate a Hugging Face gated-repo / 401 error into actionable guidance.

    Returns a ``RuntimeError`` with setup steps if ``exc`` is recognisable
    as an HF gated-repo failure, or ``None`` if it's some other exception
    (network blip, bad model_id, torch issue) that should propagate unchanged.

    Why both a typed check and a string match: ``transformers.from_pretrained``
    catches ``huggingface_hub.errors.GatedRepoError`` and re-raises it as
    ``OSError``, so the typed isinstance check rarely fires through the
    transformers path. The substring fallback is what handles real-world
    cases. The typed check still helps for direct ``huggingface_hub`` calls.
    """
    is_gated = False
    try:
        from huggingface_hub.errors import GatedRepoError

        if isinstance(exc, GatedRepoError):
            is_gated = True
    except ImportError:
        pass
    if not is_gated:
        msg = str(exc).lower()
        if (
            "gated repo" in msg
            or "401 client error" in msg
            or ("access" in msg and "restricted" in msg)
        ):
            is_gated = True
    if not is_gated:
        return None
    return RuntimeError(
        f"Hugging Face model '{repo}' is gated — you need to request access "
        "and authenticate before it can download.\n\n"
        f"  1. Request access at https://huggingface.co/{repo}\n"
        "  2. Create a token at https://huggingface.co/settings/tokens "
        "('Read' scope is sufficient).\n"
        "  3. Authenticate locally:\n"
        "         uv run hf auth login\n"
        "     (or `uv run huggingface-cli login` on older huggingface_hub).\n"
        "     Paste your token when prompted.\n\n"
        "Full instructions: docs/install.md → "
        "'Hugging Face authentication for gated models'."
    )


def _patch_sam3_transformers_compat() -> None:
    """SAM3 tracker vs vision encoder field-name drift in some transformers builds.

    Tracker ``_prepare_vision_features`` reads ``fpn_position_embeddings``;
    ``Sam3VisionEncoderOutput`` exposes ``fpn_position_encoding`` (fixed upstream
    in huggingface/transformers#43487). Alias on the output class so both work.
    """
    try:
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
    except ImportError:
        return
    if getattr(Sam3VisionEncoderOutput, "_laov_fpn_alias_patch", False):
        return
    fields = getattr(Sam3VisionEncoderOutput, "__dataclass_fields__", {})
    if "fpn_position_embeddings" in fields:
        Sam3VisionEncoderOutput._laov_fpn_alias_patch = True
        return
    if not hasattr(Sam3VisionEncoderOutput, "fpn_position_encoding"):
        return

    # property alias — safe for dataclass ModelOutput instances
    Sam3VisionEncoderOutput.fpn_position_embeddings = property(  # type: ignore[attr-defined]
        lambda self: self.fpn_position_encoding
    )
    Sam3VisionEncoderOutput._laov_fpn_alias_patch = True


@dataclass
class _DetectedInstance:
    """Internal scratch structure — one detected + tracked instance.

    Kept separate from `rank.Instance` because this one carries the full
    per-frame mask stack (heavy), while `rank.Instance` only carries the
    reduced scalars (light, serializable).
    """

    track_id: int
    label: str
    masks: dict[int, np.ndarray]  # frame_idx -> (H, W) float32 in [0, 1]


def _instance_layer_name(inst: _DetectedInstance, instances: list[_DetectedInstance]) -> str:
    """Named EXR layer for one track (disambiguate duplicate labels)."""
    base = _slug_label(inst.label)
    if sum(1 for o in instances if _slug_label(o.label) == base) > 1:
        return f"{base}_{inst.track_id}"
    if base.upper() in {"R", "G", "B", "A"}:
        return f"{base}_layer"
    return base


class SAM3MattePass(UtilityPass):
    name = "sam3_matte"
    version = "0.1.0"
    license = License(
        spdx="SAM-License-1.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "SAM 3 uses Meta's custom license. Commercial use is permitted "
            "but military / ITAR applications are prohibited. Consult the "
            "upstream LICENSE before any defense-adjacent deployment."
        ),
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    # Channels are dynamic (one `mask.<concept>` per detected concept). We
    # declare none statically — the writer accepts unknown channels as long
    # as they follow the `mask.*` / `matte.*` naming convention.
    produces_channels: list[Any] = []
    provides_artifacts = ["sam3_hard_masks", "sam3_instances", "matte_concepts"]

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "facebook/sam3",
        # Full HF snapshot directory (e.g. Drive …/VDA_models/facebook/sam3).
        # When set, loads with local_files_only=True — no Hub download on Colab.
        "model_path": None,
        "concepts": ["person", "vehicle", "tree", "building", "sky", "water", "animal"],
        "confidence_threshold": 0.4,
        "min_area_fraction": 0.005,  # drop instances smaller than 0.5% of plate
        "sample_frame": "middle",  # "first" | "middle" | "last" | int
        "redetect_stride": None,  # None = seed-and-track only (v1 default)
        # Ranking weights (RankWeights schema).
        "ranking": {
            "area": 0.4,
            "centrality": 0.2,
            "motion": 0.2,
            "duration": 0.2,
            "user_priority": 0.0,
        },
        "max_heroes": 4,
        "heroes": [],  # user overrides: [{"track_id": 17, "slot": "r"}]
        # AI Matte / Desk: rectangular prompts on seed frame (normalized 0–1 xywh).
        "box_prompts": [],
        "matte_mode": "auto",  # auto | notes | people_fg | bbox | concepts
        # Per-shot matte notes: [{"label": "bg yellow guy", "prompt": "...", "slot": "r"}]
        "matte_notes": [],
        # Auto downscale SAM3 working plates when float32 stack would exceed RAM (Colab).
        "sam3_max_plate_stack_gb": 14.0,
        # Optional SAM3-only long edge (deliverable masks are still upscaled to full plate).
        "sam3_proxy_long_edge": None,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Populated by `run_shot`; consumed by `emit_artifacts`.
        self._instances: list[_DetectedInstance] = []
        self._heroes: list[Any] = []  # list[rank.HeroSlot]
        self._concepts_found: list[str] = []
        self._plate_shape: tuple[int, int] = (0, 0)
        # Flow snapshot (optional) — used for motion-energy feature when
        # present, ignored otherwise. Populated via ingest_artifacts.
        self._forward_flow: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Artifact ingestion — consume flow if the flow pass ran first.
    # ------------------------------------------------------------------

    # SAM 3 doesn't *require* flow, but if the user put `flow` in their job
    # the ranker can use it. We declare the soft dep by reading it in
    # `ingest_artifacts` and tolerating absence.
    requires_artifacts: list[str] = []  # soft dep; no DAG enforcement

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        self._forward_flow = dict(artifacts.get("forward_flow") or {})

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load BOTH the detector and the tracker.

        SAM 3 ships two models that live under the same HF repo:

        - `Sam3Model` / `Sam3Processor`  — single-image, concept-conditioned
          instance segmentation. Used by `_detect_seed`. `AutoProcessor` on
          `facebook/sam3` resolves to the *video* processor instead of this
          one (the checkpoint's `model_type` is `sam3_video`), so we import
          the image classes explicitly to force the correct path.
        - `Sam3TrackerVideoModel` / `Sam3TrackerVideoProcessor` — inference
          session API with `input_masks=` seeding and
          `propagate_in_video_iterator`. Used by `_track_instance`.

        Tracker runs in bf16 on CUDA (per the SAM 3 model card); the
        detector stays in float32 because it fires once per concept and
        is not the hot loop.
        """
        if self._model is not None:
            return
        _patch_sam3_transformers_compat()
        import torch
        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

        repo, load_kw = _resolve_model_source(self.params)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        # Hub path: SAM 3 is gated — wrap 401 / gated-repo errors with setup steps.
        # Local snapshot (model_path / LAOV_SAM3_MODEL_PATH): no Hub call on Colab.
        try:
            self._det_processor = Sam3Processor.from_pretrained(repo, **load_kw)
            det_model = Sam3Model.from_pretrained(repo, **load_kw)
            det_model.to(self._device).eval()
            self._det_model = det_model

            self._trk_processor = Sam3TrackerVideoProcessor.from_pretrained(repo, **load_kw)
            trk_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
            trk_model = Sam3TrackerVideoModel.from_pretrained(repo, dtype=trk_dtype, **load_kw)
            trk_model.to(self._device).eval()
            self._trk_model = trk_model
            self._trk_dtype = trk_dtype
        except Exception as exc:
            wrapped = _wrap_if_gated_repo(repo, exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

        # Sentinel that the other guard checks — any of the four attrs
        # set above would work; pick one that's definitely non-None.
        self._model = det_model

    # ------------------------------------------------------------------
    # Detection + tracking — split so tests can override either.
    # ------------------------------------------------------------------

    def _seeds_from_box_prompts(
        self,
        seed_frame: np.ndarray,
        box_prompts: list[dict[str, Any]],
        plate_h: int,
        plate_w: int,
    ) -> list[tuple[int, str, np.ndarray]]:
        """Build tracker seeds from normalized or pixel xywh boxes."""
        seeds: list[tuple[int, str, np.ndarray]] = []
        for idx, box in enumerate(box_prompts, start=1):
            label = str(box.get("label") or box.get("concept") or f"object_{idx}")
            norm = box.get("normalized", True)
            if norm:
                x = float(box.get("x", 0.0)) * plate_w
                y = float(box.get("y", 0.0)) * plate_h
                w = float(box.get("w", 0.1)) * plate_w
                h = float(box.get("h", 0.1)) * plate_h
            else:
                x = float(box.get("x", 0.0))
                y = float(box.get("y", 0.0))
                w = float(box.get("w", 100.0))
                h = float(box.get("h", 100.0))
            x0 = int(max(0, min(plate_w - 1, round(x))))
            y0 = int(max(0, min(plate_h - 1, round(y))))
            x1 = int(max(x0 + 1, min(plate_w, round(x + w))))
            y1 = int(max(y0 + 1, min(plate_h, round(y + h))))
            mask = np.zeros((plate_h, plate_w), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            track_id = int(box.get("track_id", idx))
            seeds.append((track_id, label, mask))
        return seeds

    def _detect_seed(
        self,
        seed_frame: np.ndarray,
        concepts: list[str],
    ) -> list[tuple[int, str, np.ndarray]]:
        """Run SAM 3 on one frame, return a list of (track_id, label, mask).

        Single-image, open-vocabulary instance segmentation: for each
        concept in `concepts`, run one forward pass and collect every
        detected instance whose score exceeds `confidence_threshold`.
        Track IDs are assigned sequentially starting at 1 — the tracker
        uses them as SAM 3 `obj_ids` in the downstream session.

        SAM 3's classification head is single-concept per forward, so we
        loop over concepts. For the "person + vehicle + tree" style job
        this means ~3 forward passes on the seed frame — cheap.
        """
        import torch
        from PIL import Image

        self._load_model()
        assert self._det_model is not None
        processor = self._det_processor
        model = self._det_model
        device = self._device

        H, W = int(seed_frame.shape[0]), int(seed_frame.shape[1])
        # Processor expects PIL RGB uint8 (or a torch/numpy tensor the
        # image processor can rescale). PIL is the least-surprising path.
        arr_u8 = (np.clip(seed_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8, "RGB")

        threshold = float(self.params.get("confidence_threshold", 0.4))
        mask_threshold = 0.5
        area_floor = float(self.params["min_area_fraction"]) * H * W

        seeds: list[tuple[int, str, np.ndarray]] = []
        next_track_id = 1
        for concept in concepts:
            inputs = processor(images=pil, text=concept, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # target_sizes is (H, W) not (W, H) — the post-processor
            # asserts this ordering when upsampling the mask logits.
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(H, W)],
            )
            if not results:
                continue
            result = results[0]
            masks = result.get("masks")
            scores = result.get("scores")
            if masks is None or scores is None:
                continue
            n = int(masks.shape[0]) if hasattr(masks, "shape") else 0
            for i in range(n):
                m = masks[i]
                mask_np = (
                    m.float().cpu().numpy()
                    if hasattr(m, "float")
                    else np.asarray(m, dtype=np.float32)
                ).astype(np.float32)
                # Belt-and-braces area floor — the ranker will drop them
                # too, but skipping here saves a tracker session.
                if float(mask_np.sum()) < area_floor:
                    continue
                seeds.append((next_track_id, concept, mask_np))
                next_track_id += 1
        return seeds

    def _detect_seed_note(
        self,
        seed_frame: np.ndarray,
        prompt: str,
    ) -> np.ndarray | None:
        """Single open-vocabulary prompt → best instance mask on the seed frame."""
        import torch
        from PIL import Image

        prompt = str(prompt).strip()
        if not prompt:
            return None

        self._load_model()
        assert self._det_model is not None
        processor = self._det_processor
        model = self._det_model
        device = self._device

        H, W = int(seed_frame.shape[0]), int(seed_frame.shape[1])
        arr_u8 = (np.clip(seed_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8, "RGB")

        threshold = float(self.params.get("confidence_threshold", 0.4))
        mask_threshold = 0.5
        area_floor = float(self.params["min_area_fraction"]) * H * W

        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(H, W)],
        )
        if not results:
            return None
        result = results[0]
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or scores is None:
            return None
        n = int(masks.shape[0]) if hasattr(masks, "shape") else 0
        best_mask: np.ndarray | None = None
        best_score = -1.0
        for i in range(n):
            score = float(scores[i])
            m = masks[i]
            mask_np = (
                m.float().cpu().numpy()
                if hasattr(m, "float")
                else np.asarray(m, dtype=np.float32)
            ).astype(np.float32)
            if float(mask_np.sum()) < area_floor:
                continue
            if score > best_score:
                best_score = score
                best_mask = mask_np
        return best_mask

    def _seeds_from_matte_notes(
        self,
        seed_frame: np.ndarray,
        notes: list[dict[str, Any]],
    ) -> list[tuple[int, str, np.ndarray]]:
        """User matte notes → one SAM3 track seed each (max 4 slots)."""
        slots = ("r", "g", "b", "a")
        seeds: list[tuple[int, str, np.ndarray]] = []
        for i, note in enumerate(notes[:4]):
            if not isinstance(note, dict):
                continue
            label = str(note.get("label") or note.get("name") or f"note_{i + 1}").strip()
            prompt = str(note.get("prompt") or label).strip()
            if not prompt:
                continue
            mask = self._detect_seed_note(seed_frame, prompt)
            if mask is None:
                _log.warning("SAM3: no detection for matte note %r (prompt=%r)", label, prompt)
                continue
            track_id = int(note.get("track_id", i + 1))
            slot = str(note.get("slot", slots[i]))
            channel_label = _slug_label(label)
            _log.info(
                "SAM3: matte note %r (prompt=%r) → track %s slot=%s channel=mask.%s",
                label,
                prompt,
                track_id,
                slot,
                channel_label,
            )
            seeds.append((track_id, channel_label, mask))
        return seeds

    def _track_instance(
        self,
        frames: np.ndarray,
        seed_frame_idx: int,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        """Propagate `seed_mask` across all frames with SAM 3's tracker.

        Uses SAM 3's inference-session API: seed the mask at
        `seed_frame_idx` via `add_inputs_to_inference_session(input_masks=)`,
        then propagate forward + backward with
        `propagate_in_video_iterator`. Each yielded step carries the logits
        for its frame, which `post_process_masks` upsamples + binarizes.

        Input: `frames` is (N, H, W, 3) float32 in [0, 1]; `seed_frame_idx`
        is the local index within `frames`; `seed_mask` is (H, W) float32
        in [0, 1] at plate resolution. Output: (N, H, W) float32 stack of
        hard-ish masks.
        """
        from PIL import Image

        self._load_model()
        assert self._trk_model is not None
        processor = self._trk_processor
        model = self._trk_model
        device = self._device
        dtype = self._trk_dtype

        N = int(frames.shape[0])
        H, W = int(frames.shape[1]), int(frames.shape[2])

        # Processor accepts a list of PIL images (or a 4D torch tensor);
        # PIL matches the detector path and dodges dtype surprises.
        frame_pils = [
            Image.fromarray(
                (np.clip(frames[i], 0.0, 1.0) * 255.0).astype(np.uint8),
                "RGB",
            )
            for i in range(N)
        ]

        session = processor.init_video_session(
            video=frame_pils,
            inference_device=device,
            dtype=dtype,
        )

        # SAM 3 wants a bool / {0,1} mask. Threshold at 0.5.
        seed_bool = (seed_mask > 0.5).astype(np.uint8)
        processor.add_inputs_to_inference_session(
            session,
            frame_idx=int(seed_frame_idx),
            obj_ids=1,  # single-object track; we repeat per instance
            input_masks=seed_bool,
            original_size=(H, W),
        )

        out = np.zeros((N, H, W), dtype=np.float32)

        def _consume(step_iter: Any) -> None:
            """Drain a propagate_in_video_iterator into `out`.

            Each `step` is a Sam3TrackerVideoSegmentationOutput. We try the
            two field names transformers has used historically
            (`frame_idx` / `frame_index`) and skip the step if neither is
            present — defensive against minor API drift between versions.
            """
            for step in step_iter:
                frame_idx = getattr(step, "frame_idx", None)
                if frame_idx is None:
                    frame_idx = getattr(step, "frame_index", None)
                pred_masks = getattr(step, "pred_masks", None)
                if frame_idx is None or pred_masks is None:
                    continue
                # Step yields pred_masks of shape (batch, obj_ids, h, w) — 4D.
                # post_process_masks expects a 5D input (list-of-batches
                # style: batch, frames_per_batch, obj_ids, h, w) and upsamples
                # the trailing 2D to original_sizes. Add the leading dim here.
                if pred_masks.ndim == 4:
                    pred_masks_5d = pred_masks.unsqueeze(0)
                else:
                    pred_masks_5d = pred_masks
                post = processor.post_process_masks(
                    pred_masks_5d,
                    original_sizes=[(H, W)],
                    mask_threshold=0.0,
                    binarize=True,
                )
                # post_process_masks returns list-per-batch; we have one batch.
                if not post:
                    continue
                m = post[0]
                if hasattr(m, "float"):
                    arr = m.float().cpu().numpy()
                else:
                    arr = np.asarray(m, dtype=np.float32)
                # Collapse leading non-spatial dims — post is either
                # (frames_in_batch, obj_ids, H, W) or (obj_ids, H, W).
                # We asked for obj_ids=[1], so the first slice along any
                # non-spatial dim is the single object.
                while arr.ndim > 2:
                    arr = arr[0]
                out[int(frame_idx)] = arr.astype(np.float32, copy=False)

        # Forward from seed to end; then backward from seed to start. The
        # seed frame itself is emitted by forward propagation.
        _consume(model.propagate_in_video_iterator(session, start_frame_idx=int(seed_frame_idx)))
        _consume(
            model.propagate_in_video_iterator(
                session,
                start_frame_idx=int(seed_frame_idx),
                reverse=True,
            )
        )

        return out

    # ------------------------------------------------------------------
    # Per-window lifecycle — SAM 3 is VIDEO_CLIP-native, so `preprocess` /
    # `infer` / `postprocess` are convenience paths that delegate to the
    # shot-level path for a single window.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"SAM3MattePass preprocess expects (N, H, W, 3), got {frames.shape}")
        self._load_model()
        return {"video": frames.astype(np.float32, copy=False)}

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        # Single-window call path is rarely used for VIDEO_CLIP; run_shot is
        # the canonical entry. Return an empty dict so per-frame iterators
        # don't crash.
        return {}

    # ------------------------------------------------------------------
    # Shot-level: detect → track → union → stats → rank
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        n_frames = last - first + 1
        seed_rgb_full, _attrs0 = reader.read_frame(first)
        plate_h, plate_w = int(seed_rgb_full.shape[0]), int(seed_rgb_full.shape[1])
        self._plate_shape = (plate_h, plate_w)
        work_h, work_w = _sam3_working_size(
            plate_h,
            plate_w,
            n_frames,
            max_stack_gb=float(self.params.get("sam3_max_plate_stack_gb", 14.0)),
            proxy_long_edge=self.params.get("sam3_proxy_long_edge"),
        )
        est_gib = _estimate_plate_stack_gib(n_frames, plate_h, plate_w)
        if (work_h, work_w) != (plate_h, plate_w):
            _log.warning(
                "SAM3: plate stack would be ~%.1f GiB at %dx%d — tracking at %dx%d, "
                "masks upscaled to full plate for BiRefNet/ViTMatte",
                est_gib,
                plate_w,
                plate_h,
                work_w,
                work_h,
            )
        else:
            _log.info(
                "SAM3: reading %d plate frames (%s-%s) at full res (~%.1f GiB stack)…",
                n_frames,
                first,
                last,
                est_gib,
            )
        frames = _build_sam3_plate_stack(
            reader, first, last, work_h=work_h, work_w=work_w
        )
        _log.info("SAM3: plate stack %s — detect & track", frames.shape)

        seed_local = _pick_seed_frame(n_frames, self.params["sample_frame"])
        seed_rgb = frames[seed_local]
        work_h, work_w = int(frames.shape[1]), int(frames.shape[2])

        box_prompts = list(self.params.get("box_prompts") or [])
        matte_mode = str(self.params.get("matte_mode", "auto")).strip().lower()
        matte_notes = list(self.params.get("matte_notes") or [])
        if matte_mode == "people_fg":
            concepts = ["person"]
        elif matte_mode == "notes" and matte_notes:
            concepts = []
        elif matte_mode == "bbox" and box_prompts:
            concepts = []
        elif matte_mode == "concepts":
            concepts = [str(c) for c in self.params["concepts"]]
        else:
            # Smart auto — discover common foreground types when user gave no notes.
            concepts = [str(c) for c in self.params.get("concepts") or ["person", "vehicle", "animal"]]

        if box_prompts:
            seeds = self._seeds_from_box_prompts(seed_rgb, box_prompts, work_h, work_w)
        elif matte_mode == "notes" and matte_notes:
            seeds = self._seeds_from_matte_notes(seed_rgb, matte_notes)
        else:
            seeds = self._detect_seed(seed_rgb, concepts) if concepts else []

        # Track each seed across the clip; accumulate _DetectedInstance.
        self._instances = []
        _log.info("SAM3: tracking %d instance(s) across %d frames", len(seeds), n_frames)
        for track_id, label, seed_mask in seeds:
            _log.info("SAM3: track %s (%s) — video propagate…", track_id, label)
            if seed_mask.shape != (work_h, work_w):
                seed_mask = _resize_mask_plane(seed_mask, work_h, work_w)
            stack = self._track_instance(frames, seed_local, seed_mask)
            if stack.ndim != 3 or stack.shape[1:] != (work_h, work_w):
                raise ValueError(
                    f"Track stack for {track_id} has shape {stack.shape}, "
                    f"expected ({n_frames}, {work_h}, {work_w})"
                )
            masks = {
                first + k: _resize_mask_plane(stack[k], plate_h, plate_w)
                for k in range(stack.shape[0])
            }
            # Drop instances that fall below the area floor everywhere.
            area_floor = float(self.params["min_area_fraction"]) * plate_h * plate_w
            if all(m.sum() < area_floor for m in masks.values()):
                continue
            self._instances.append(_DetectedInstance(track_id, label, masks))

        del frames
        import gc

        gc.collect()

        # Union by concept into per-frame channels.
        per_frame: dict[int, dict[str, np.ndarray]] = {first + i: {} for i in range(n_frames)}
        concepts_found: set[str] = set()
        label_set = concepts if concepts else sorted({inst.label for inst in self._instances})
        for concept in label_set:
            # OR across all instances of this concept, per frame.
            any_nonzero = False
            for f in range(first, last + 1):
                union = np.zeros((plate_h, plate_w), dtype=np.float32)
                for inst in self._instances:
                    if inst.label != concept:
                        continue
                    m = inst.masks.get(f)
                    if m is not None:
                        np.maximum(union, m, out=union)
                if union.any():
                    any_nonzero = True
                    per_frame[f][f"{MASK_PREFIX}{concept}"] = union
            if any_nonzero:
                concepts_found.add(concept)
        # If a concept produced no mask on any frame, emit nothing for it
        # (we don't want zero-valued `mask.sky` layers when there's no sky).
        self._concepts_found = sorted(concepts_found)

        # Build the light Instance objects and rank them.
        rank_weights = RankWeights(**self.params["ranking"])
        hero_list = list(self.params.get("heroes") or [])
        if matte_notes and matte_mode == "notes" and not hero_list:
            slots = ("r", "g", "b", "a")
            for i, note in enumerate(matte_notes[:4]):
                if not isinstance(note, dict):
                    continue
                hero_list.append(
                    {
                        "track_id": int(note.get("track_id", i + 1)),
                        "slot": str(note.get("slot", slots[i])),
                    }
                )
        if box_prompts and not hero_list:
            slots = ("r", "g", "b", "a")
            for i, box in enumerate(box_prompts[:4]):
                tid = int(box.get("track_id", i + 1))
                slot = str(box.get("slot", slots[i]))
                hero_list.append({"track_id": tid, "slot": slot})
        overrides = [
            HeroOverride(track_id=int(h["track_id"]), slot=h["slot"])
            for h in hero_list
            if "track_id" in h and "slot" in h
        ]
        ranked_input = [
            self._to_rank_instance(inst, n_frames, plate_h, plate_w, first)
            for inst in self._instances
        ]
        self._heroes = rank_and_assign(
            ranked_input,
            rank_weights,
            n_clip_frames=n_frames,
            max_heroes=int(self.params["max_heroes"]),
            overrides=overrides,
        )
        # Hero slots as matte.r/g/b/a for RGBA stage EXR export (no mask.* in deliverables).
        _slot_channels = dict(zip(("r", "g", "b", "a"), MATTE_CHANNELS, strict=False))
        inst_by_track = {inst.track_id: inst for inst in self._instances}
        for hero in self._heroes:
            inst = inst_by_track.get(hero.track_id)
            if inst is None:
                continue
            ch = _slot_channels.get(hero.slot)
            if not ch:
                continue
            for f, mask in inst.masks.items():
                per_frame.setdefault(f, {})[ch] = mask.astype(np.float32, copy=False)
        # Named layers inside the combined stage EXR (person1, car, …) — one float channel each.
        for inst in self._instances:
            layer = _instance_layer_name(inst, self._instances)
            for f, mask in inst.masks.items():
                per_frame.setdefault(f, {})[layer] = mask.astype(np.float32, copy=False)
        return per_frame

    def _to_rank_instance(
        self,
        inst: _DetectedInstance,
        n_frames: int,
        plate_h: int,
        plate_w: int,
        first_frame: int,
    ) -> Instance:
        """Reduce per-frame masks (+ optional flow) into rank-friendly scalars."""
        plate_area = float(plate_h * plate_w)
        plate_diag = math.hypot(plate_h, plate_w)

        # Area fraction: mean over frames present.
        area_fractions: list[float] = []
        centralities: list[float] = []
        motion_energies: list[float] = []
        for f, mask in inst.masks.items():
            s = float(mask.sum())
            if s <= 0.0:
                continue
            area_fractions.append(s / plate_area)
            # Centroid -> distance from center -> 1 - (d / half_diag) clipped.
            ys, xs = np.where(mask > 0.5)
            if ys.size == 0:
                ys, xs = np.where(mask > 0)
            if ys.size:
                cy = float(ys.mean())
                cx = float(xs.mean())
                dx = cx - plate_w / 2.0
                dy = cy - plate_h / 2.0
                d = math.hypot(dx, dy)
                centralities.append(max(0.0, 1.0 - d / (plate_diag / 2.0)))
            # Motion energy from forward_flow[f] if available.
            fwd = self._forward_flow.get(f)
            if fwd is not None and fwd.shape[-2:] == (plate_h, plate_w):
                mag = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
                masked = mag[mask > 0.5]
                if masked.size:
                    motion_energies.append(float(masked.mean()) / max(plate_diag, 1.0))

        return Instance(
            track_id=inst.track_id,
            label=inst.label,
            frames=sorted(f for f, m in inst.masks.items() if float(m.sum()) > 0.0),
            area_fraction=_mean_or_zero(area_fractions),
            centrality=_mean_or_zero(centralities),
            motion_energy=_mean_or_zero(motion_energies),
            user_priority=0.0,
        )

    def release_gpu(self) -> None:
        """Unload SAM3 detector/tracker weights after stage export."""
        for attr in ("_det_model", "_trk_model", "_det_processor", "_trk_processor", "_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    import torch

                    if hasattr(obj, "cpu"):
                        obj.cpu()
                except Exception:
                    pass
            setattr(self, attr, None)
        self._device = None
        self._dtype = None
        self._instances = []
        self._forward_flow = {}

    # ------------------------------------------------------------------
    # Artifact emission
    # ------------------------------------------------------------------

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._instances and not self._heroes:
            return {}
        # sam3_hard_masks — stash as a single dict value under frame key 0
        # (the artifact is shot-level, not per-frame). Value is
        # {track_id: {"label": str, "stack": (T, H, W)}} — the shape
        # CorridorKey will consume (design §21.8).
        plate_h, plate_w = self._plate_shape
        hard_masks: dict[int, dict[str, Any]] = {}
        for inst in self._instances:
            frames_sorted = sorted(inst.masks)
            if not frames_sorted:
                continue
            stack = np.stack([inst.masks[f] for f in frames_sorted], axis=0)
            hard_masks[inst.track_id] = {
                "label": inst.label,
                "frames": frames_sorted,
                "stack": stack.astype(np.float32, copy=False),
            }

        # sam3_instances — the ranked+slotted hero list, as serializable dicts
        # so downstream consumers don't have to import rank.py just to unpack.
        hero_dicts: list[dict[str, Any]] = []
        for h in self._heroes:
            hero_dicts.append(
                {
                    "track_id": h.track_id,
                    "slot": h.slot,
                    "label": h.label,
                    "score": h.score,
                    # Reference the frames where the instance is present so
                    # the refiner can skip empty frames cheaply.
                    "frames": list(h.instance.frames),
                }
            )

        # Pick any frame key (artifact is shot-level).
        any_frame = 0
        for inst in self._instances:
            if inst.masks:
                any_frame = next(iter(inst.masks))
                break

        return {
            "sam3_hard_masks": {any_frame: hard_masks},
            "sam3_instances": {any_frame: hero_dicts},
            "matte_concepts": {any_frame: list(self._concepts_found)},
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _estimate_plate_stack_gib(n_frames: int, plate_h: int, plate_w: int) -> float:
    return n_frames * plate_h * plate_w * 3 * 4 / (1024**3)


def _sam3_working_size(
    plate_h: int,
    plate_w: int,
    n_frames: int,
    *,
    max_stack_gb: float,
    proxy_long_edge: Any,
) -> tuple[int, int]:
    """Working H×W for SAM3 plate stack (may be smaller than deliverable plate)."""
    if proxy_long_edge not in (None, "", 0, "0"):
        long_edge = int(proxy_long_edge)
        long_src = max(plate_h, plate_w)
        if long_src <= long_edge:
            return plate_h, plate_w
        scale = long_edge / float(long_src)
        return (
            max(64, int(round(plate_h * scale))),
            max(64, int(round(plate_w * scale))),
        )
    if max_stack_gb <= 0 or n_frames <= 0:
        return plate_h, plate_w
    budget_pixels = max_stack_gb * (1024**3) / (n_frames * 3 * 4)
    plate_pixels = plate_h * plate_w
    if budget_pixels >= plate_pixels:
        return plate_h, plate_w
    scale = (budget_pixels / plate_pixels) ** 0.5
    return (
        max(64, int(round(plate_h * scale))),
        max(64, int(round(plate_w * scale))),
    )


def _resize_rgb_plane(rgb: np.ndarray, work_h: int, work_w: int) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32)[..., :3], 0.0, 1.0)
    if rgb.shape[0] == work_h and rgb.shape[1] == work_w:
        return rgb
    if cv2 is not None:
        return cv2.resize(rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
    from PIL import Image

    arr_u8 = (rgb * 255.0).astype(np.uint8)
    return (
        np.asarray(Image.fromarray(arr_u8, "RGB").resize((work_w, work_h), Image.BILINEAR))
        / 255.0
    ).astype(np.float32)


def _resize_mask_plane(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.shape[0] == out_h and m.shape[1] == out_w:
        return m
    if cv2 is not None:
        return cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image

    return (
        np.asarray(
            Image.fromarray((m * 255.0).astype(np.uint8), "L").resize((out_w, out_h), Image.BILINEAR)
        )
        / 255.0
    ).astype(np.float32)


def _build_sam3_plate_stack(
    reader: Any,
    first: int,
    last: int,
    *,
    work_h: int,
    work_w: int,
) -> np.ndarray:
    """One frame at a time — avoids holding a list of full-res arrays."""
    n_frames = last - first + 1
    stack = np.empty((n_frames, work_h, work_w, 3), dtype=np.float32)
    log_step = max(1, n_frames // 10)
    for i, frame_idx in enumerate(range(first, last + 1)):
        if i == 0 or i == n_frames - 1 or (i + 1) % log_step == 0:
            _log.info("SAM3: loading plate %d/%d (frame %s)", i + 1, n_frames, frame_idx)
        rgb, _attrs = reader.read_frame(frame_idx)
        stack[i] = _resize_rgb_plane(rgb, work_h, work_w)
    return stack


def _pick_seed_frame(n_frames: int, sample_frame: Any) -> int:
    """Resolve the `sample_frame` param to a local index in [0, n_frames)."""
    if isinstance(sample_frame, int):
        return max(0, min(n_frames - 1, sample_frame))
    s = str(sample_frame).lower()
    if s == "first":
        return 0
    if s == "last":
        return n_frames - 1
    # "middle" and any unknown value default to middle.
    return n_frames // 2


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


__all__ = ["SAM3MattePass"]
