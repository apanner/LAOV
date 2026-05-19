"""SAM3 transformers API compat — fpn_position_encoding alias."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from live_action_aov.passes.matte.sam3 import _patch_sam3_transformers_compat


def test_fpn_position_embeddings_alias_when_only_encoding_exists() -> None:
    try:
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
    except ImportError:
        pytest.skip("Sam3VisionEncoderOutput not in this transformers build")

    _patch_sam3_transformers_compat()
    fields = getattr(Sam3VisionEncoderOutput, "__dataclass_fields__", {})
    if "fpn_position_embeddings" in fields:
        pytest.skip("transformers already defines fpn_position_embeddings")
    if "fpn_position_encoding" not in fields:
        pytest.skip("no fpn_position_encoding on Sam3VisionEncoderOutput")

    enc = object.__new__(Sam3VisionEncoderOutput)
    object.__setattr__(enc, "fpn_position_encoding", "mock_pos")
    assert enc.fpn_position_embeddings == "mock_pos"
