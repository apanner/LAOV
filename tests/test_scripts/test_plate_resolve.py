"""Colab plate folder resolution (nested EXR subfolders, VDA_input search)."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import helpers from scripts/ without pulling full LAOV package init.
import importlib.util
import sys

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location("laov_colab_run", _SCRIPTS / "laov_colab_run.py")
assert _spec and _spec.loader
lcr = importlib.util.module_from_spec(_spec)
sys.modules["laov_colab_run"] = lcr
_spec.loader.exec_module(lcr)


def _write_exr_seq(
    folder: Path,
    base: str,
    frames: range,
    *,
    separator: str = ".",
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for f in frames:
        (folder / f"{base}{separator}{f:04d}.exr").write_bytes(b"")


def test_nested_exr_subfolder(tmp_path: Path) -> None:
    mount = tmp_path / "drive"
    leaf = "TB_073_020_plate_v001"
    plate_root = mount / "VDA_input" / "oppy" / "073_020" / "Compositions" / leaf
    _write_exr_seq(plate_root / "exr", "TB_073_020_plate", range(1001, 1004))

    rel = f"VDA_input/oppy/073_020/Compositions/{leaf}"
    resolved = lcr._resolve_plate_dir(mount, rel)
    assert resolved is not None
    assert resolved.name == "exr"
    assert lcr._folder_has_plates_flat(resolved)


def test_search_when_compositions_missing(tmp_path: Path) -> None:
    mount = tmp_path / "drive"
    leaf = "TB_073_020_plate_v001"
    actual = mount / "VDA_input" / "oppy" / "073_020" / "renders" / leaf / "plates"
    _write_exr_seq(actual, "TB_073_020_plate", range(1001, 1003))

    rel = f"VDA_input/oppy/073_020/Compositions/{leaf}"
    resolved = lcr._resolve_plate_dir(mount, rel)
    assert resolved is not None
    assert resolved.name == "plates"


def test_shot_plate_from_desk_json_uses_first_frame_path(tmp_path: Path) -> None:
    mount = tmp_path / "drive"
    leaf = "TB_073_020_plate_v001"
    plate_root = mount / "VDA_input" / "oppy" / "073_020" / "Compositions" / leaf
    _write_exr_seq(plate_root, "TB_073_020_plate_v001", range(1001, 1004), separator="_")

    first_rel = (
        f"VDA_input/oppy/073_020/Compositions/{leaf}/TB_073_020_plate_v001_1001.exr"
    )
    seq = {
        "plate_first_frame_relpath": first_rel,
        "plate_input_pattern": (
            f"VDA_input/oppy/073_020/Compositions/{leaf}/"
            "TB_073_020_plate_v001_%04d.exr"
        ),
        "sequence_pattern": "TB_073_020_plate_v001_####.exr",
        "first_frame": 1001,
        "last_frame": 1003,
    }

    plate_dir, pattern, frame_range, _res, _par = lcr.shot_plate_from_desk_json(mount, seq)
    assert plate_dir == plate_root.resolve()
    assert pattern == "TB_073_020_plate_v001_####.exr"
    assert frame_range == (1001, 1003)


def test_pick_best_prefers_path_overlap(tmp_path: Path) -> None:
    mount = tmp_path / "drive"
    leaf = "TB_073_020_plate_v001"
    a = mount / "VDA_input" / "oppy" / "073_020" / leaf
    b = mount / "VDA_input" / "other" / leaf
    _write_exr_seq(a / "exr", "p", range(1001, 1002))
    _write_exr_seq(b / "exr", "p", range(1001, 1002))

    rel = f"VDA_input/oppy/073_020/Compositions/{leaf}"
    resolved = lcr._resolve_plate_dir(mount, rel)
    assert resolved is not None
    assert "oppy" in resolved.as_posix()
