"""GPU-tier ViTMatte long-edge resolution."""

from __future__ import annotations

from live_action_aov.shared.gpu_inference_budget import (
    GpuVramInfo,
    resolve_vitmatte_long_edge,
)


def test_resolve_explicit_override() -> None:
    assert resolve_vitmatte_long_edge(1024) == 1024
    assert resolve_vitmatte_long_edge("auto") != 0


def test_resolve_a100_tier() -> None:
    vram = GpuVramInfo(device_name="A100", total_gb=40.0, free_gb=35.0)
    assert resolve_vitmatte_long_edge(0, vram=vram) == 2048


def test_resolve_t4_tier() -> None:
    vram = GpuVramInfo(device_name="T4", total_gb=15.0, free_gb=12.0)
    assert resolve_vitmatte_long_edge(None, vram=vram) == 1024


def test_resolve_low_free_caps_tier() -> None:
    vram = GpuVramInfo(device_name="A100", total_gb=40.0, free_gb=7.0)
    assert resolve_vitmatte_long_edge(0, vram=vram) == 768
