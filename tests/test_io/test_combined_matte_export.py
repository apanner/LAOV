"""Combined matte EXR: one file per frame, R/G/B/A + named layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.rgba_matte_export import (
    build_combined_matte_channels,
    combined_matte_pattern,
    normalize_heroes_metadata,
    write_rgba_matte_stage_export,
)


def test_normalize_heroes_metadata_dict_and_slot() -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeHero:
        track_id: int
        slot: str
        label: str

    meta = normalize_heroes_metadata(
        [
            {"track_id": 2, "slot": "r", "label": "person"},
            FakeHero(5, "g", "car"),
        ]
    )
    assert meta[0]["track_id"] == 2 and meta[1]["label"] == "car"


def test_combined_matte_pattern() -> None:
    pat = combined_matte_pattern("plate_v001.####.exr", "sam3_matte")
    assert pat == "plate_v001_sam3_matte.{frame:04d}.exr"


def test_build_combined_matte_channels_slots_and_named() -> None:
    h, w = 4, 4
    r = np.full((h, w), 1.0, dtype=np.float32)
    g = np.full((h, w), 0.5, dtype=np.float32)
    channels = {
        "matte.r": r,
        "matte.g": g,
        "bg_yellow_guy": r.copy(),
        "mask.person": np.zeros((h, w), dtype=np.float32),
    }
    heroes = [
        {"track_id": 1, "slot": "r", "label": "bg yellow guy"},
        {"track_id": 2, "slot": "g", "label": "car"},
    ]
    out = build_combined_matte_channels("sam3_matte", channels, heroes)
    assert list(out.keys())[:4] == ["R", "G", "B", "A"]
    assert np.allclose(out["R"], r)
    assert np.allclose(out["G"], g)
    assert np.allclose(out["bg_yellow_guy"], r)
    assert "mask.person" not in out


def test_write_combined_matte_one_file_per_frame(tmp_path: Path) -> None:
    h, w = 8, 8
    alpha = np.ones((h, w), dtype=np.float32) * 0.75
    per_frame = {
        1001: {"matte.r": alpha, "person_3": alpha * 0.5},
        1002: {"matte.r": alpha * 0.5, "person_3": alpha},
    }
    out_dir = write_rgba_matte_stage_export(
        pass_name="sam3_matte",
        export_subdir="matte_sam3",
        output_root=tmp_path,
        sequence_pattern="shot.####.exr",
        per_frame_channels=per_frame,
        shot_name="shot",
    )
    assert out_dir is not None
    f1 = tmp_path / "matte_sam3" / "shot_sam3_matte.1001.exr"
    f2 = tmp_path / "matte_sam3" / "shot_sam3_matte.1002.exr"
    assert f1.is_file() and f2.is_file()
    assert not list((tmp_path / "matte_sam3").glob("*_matte_r.*.exr"))
