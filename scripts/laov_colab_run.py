#!/usr/bin/env python3
# LiveActionAOV — Colab / headless batch runner (reads Desk batch JSON).
"""Execute LAOV jobs produced by Google Desk (LAOV_STANDALONE).

Expects:
  - ``LAOV_DRIVE_MOUNT`` — absolute path where Google Drive ``MyDrive`` is mounted
    (e.g. ``/content/drive/MyDrive`` on Colab with Drive mounted).
  - ``--job-json`` — path to batch config JSON.

Each sequence must define ``plate_folder_drive_relative`` and
``sidecar_output_drive_subpath`` (relative to the Drive mount).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("laov_colab_run")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LAOV batch from Desk JSON.")
    p.add_argument(
        "--job-json",
        required=True,
        help="Path to batch JSON (LAOV_STANDALONE).",
    )
    return p.parse_args()


def _drive_mount() -> Path:
    raw = os.environ.get("LAOV_DRIVE_MOUNT", "/content/drive/MyDrive")
    return Path(raw).resolve()


def _resolve_sam3_model_dir(mount: Path, shared: dict) -> str | None:
    """Use a Drive-local HF snapshot so Colab does not download gated SAM3."""
    explicit = shared.get("sam3_model_path") or os.environ.get("LAOV_SAM3_MODEL_PATH")
    if explicit:
        p = Path(str(explicit).strip().replace("\\", "/"))
        if not p.is_absolute():
            p = mount / str(p).strip().strip("/")
        if p.is_dir() and (p / "config.json").is_file():
            return str(p.resolve())
        _log.warning("sam3_model_path %s is not a valid HF snapshot; checking defaults", p)

    for rel in (
        "VDA_models/facebook/sam3",
        "VDA_model/facebook/sam3",
        "VDA_models/sam3",
        "VDA_model/sam3",
    ):
        candidate = mount / rel
        if candidate.is_dir() and (candidate / "config.json").is_file():
            _log.info("Using Drive SAM3 snapshot: %s", candidate)
            return str(candidate.resolve())
    return None


def _resolve_birefnet_model_dir(mount: Path, shared: dict) -> str | None:
    explicit = shared.get("birefnet_model_path") or os.environ.get(
        "AI_MATTE_BIREFNET_MODEL_PATH"
    )
    if explicit:
        p = Path(str(explicit).strip().replace("\\", "/"))
        if not p.is_absolute():
            p = mount / str(p).strip().strip("/")
        if p.is_dir() and (p / "config.json").is_file():
            return str(p.resolve())
        _log.warning("birefnet_model_path %s invalid; checking defaults", p)

    for rel in (
        "VDA_models/ZhengPeng7/BiRefNet",
        "VDA_model/ZhengPeng7/BiRefNet",
        "VDA_models/birefnet",
        "VDA_model/birefnet",
    ):
        candidate = mount / rel
        if candidate.is_dir() and (candidate / "config.json").is_file():
            _log.info("Using Drive BiRefNet snapshot: %s", candidate)
            return str(candidate.resolve())
    return None


def _resolve_vitmatte_model_dir(mount: Path, shared: dict) -> str | None:
    explicit = shared.get("vitmatte_model_path") or os.environ.get("AI_MATTE_VITMATTE_MODEL_PATH")
    if explicit:
        p = Path(str(explicit).strip().replace("\\", "/"))
        if not p.is_absolute():
            p = mount / str(p).strip().strip("/")
        if p.is_dir() and (p / "config.json").is_file():
            return str(p.resolve())
    for rel in (
        "VDA_models/hustvl/vitmatte-base-composition-1k",
        "VDA_models/hustvl/vitmatte-small-composition-1k",
        "VDA_models/vitmatte",
        "VDA_model/vitmatte",
    ):
        candidate = mount / rel
        if candidate.is_dir() and (candidate / "config.json").is_file():
            _log.info("Using Drive ViTMatte snapshot: %s", candidate)
            return str(candidate.resolve())
    return None


def _matte_pass_params(
    name: str,
    shared: dict,
    seq: dict,
    *,
    sam3_model_dir: str | None,
    birefnet_model_dir: str | None,
) -> dict:
    """Merge shared + per-shot matte settings for detector/refiner passes."""
    params: dict = {}
    matte = dict(seq.get("matte_config") or {})
    mode = str(
        matte.get("matte_mode") or shared.get("matte_mode", "people_fg")
    ).strip().lower()

    if name == "sam3_matte":
        if sam3_model_dir:
            params["model_path"] = sam3_model_dir
        params["matte_mode"] = mode
        if matte.get("concepts"):
            params["concepts"] = list(matte["concepts"])
        elif shared.get("matte_concepts"):
            params["concepts"] = list(shared["matte_concepts"])
        if matte.get("box_prompts"):
            params["box_prompts"] = list(matte["box_prompts"])
            params["matte_mode"] = "bbox"
        if matte.get("heroes"):
            params["heroes"] = list(matte["heroes"])
        if matte.get("sample_frame") is not None:
            params["sample_frame"] = matte["sample_frame"]
        if matte.get("confidence_threshold") is not None:
            params["confidence_threshold"] = matte["confidence_threshold"]

    if name == "flow":
        for key in (
            "backend",
            "precision",
            "inference_resolution",
            "fb_threshold_px",
            "num_flow_updates",
        ):
            val = shared.get(key)
            if val is None:
                val = shared.get(f"flow_{key}")
            if val is not None:
                params[key] = val
        if shared.get("flow_backend") is not None and "backend" not in params:
            params["backend"] = shared["flow_backend"]
        if shared.get("flow_inference_resolution") is not None and "inference_resolution" not in params:
            params["inference_resolution"] = shared["flow_inference_resolution"]

    if name == "birefnet_refiner":
        if birefnet_model_dir:
            params["model_path"] = birefnet_model_dir
        for key in (
            "keyframe_stride",
            "crop_pad",
            "inference_size",
            "hard_mask_dilate",
            "inference_mode",
            "refine_foreground",
            "refine_radius",
            "precision",
            "fill_between_keyframes",
        ):
            val = matte.get(key, shared.get(key))
            if val is not None:
                params[key] = val
        if shared.get("birefnet_crop_pad") is not None and "crop_pad" not in params:
            params["crop_pad"] = shared["birefnet_crop_pad"]
        passes_csv = str(shared.get("passes_csv", ""))
        model_type = str(shared.get("model_type", ""))
        # Legacy depth jobs: sparse BiRefNet + RAFT temporal fill. AI Matte stage EXRs need
        # every frame filled when flow temporal is off.
        if (
            "fill_between_keyframes" not in params
            and "flow" in passes_csv
            and model_type != "AI_MATTE_STANDALONE"
        ):
            params["fill_between_keyframes"] = False

    if name == "vitmatte_refiner":
        vit_dir = os.environ.get("AI_MATTE_VITMATTE_MODEL_PATH")
        if vit_dir:
            params["model_path"] = vit_dir
        for key in (
            "keyframe_stride",
            "fill_between_keyframes",
            "hard_mask_dilate",
            "trimap_erode_px",
            "trimap_dilate_px",
            "trimap_erode_iterations",
            "trimap_dilate_iterations",
            "trimap_fg_threshold",
            "precision",
            "model_id",
        ):
            val = matte.get(key, shared.get(key))
            if val is not None:
                params[key] = val

    if name == "rvm_refiner":
        for key in ("hard_mask_dilate",):
            val = matte.get(key, shared.get(key))
            if val is not None:
                params[key] = val

    return params


_PLATE_SUFFIXES = (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".dpx", ".tga")
_PLATE_SEARCH_MAX_DEPTH = 14
_PLATE_SUBDIR_DEPTH = 3


def _folder_has_plates_flat(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    try:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in _PLATE_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def _resolve_plate_dir_containing_files(
    folder: Path,
    *,
    max_subdir_depth: int = _PLATE_SUBDIR_DEPTH,
) -> Path | None:
    """Folder that actually holds numbered plates (may be a child like ``exr/``)."""
    if not folder.is_dir():
        return None
    if _folder_has_plates_flat(folder):
        return folder.resolve()
    if max_subdir_depth <= 0:
        return None
    try:
        for child in sorted(folder.iterdir()):
            if not child.is_dir():
                continue
            found = _resolve_plate_dir_containing_files(
                child, max_subdir_depth=max_subdir_depth - 1
            )
            if found is not None:
                return found
    except OSError:
        return None
    return None


def _folder_has_plates(folder: Path) -> bool:
    return _resolve_plate_dir_containing_files(folder) is not None


def _leaf_matches_dir(name: str, leaf: str) -> bool:
    if not leaf:
        return False
    return name == leaf or name.startswith(leaf + "_") or name.startswith(leaf + ".")


def _name_matches_plate_leaf(folder: Path, plate_dir: Path, leaf: str) -> bool:
    if not leaf:
        return True
    if _leaf_matches_dir(folder.name, leaf) or _leaf_matches_dir(plate_dir.name, leaf):
        return True
    try:
        rel = plate_dir.relative_to(folder)
        if rel.parts and _leaf_matches_dir(rel.parts[0], leaf):
            return True
    except ValueError:
        pass
    return False


def _search_subtree_for_plates(
    root: Path,
    leaf: str,
    *,
    max_depth: int = _PLATE_SEARCH_MAX_DEPTH,
) -> list[Path]:
    """Recursive search under ``root`` for plate sequences (subfolders included)."""
    if not root.is_dir() or not leaf:
        return []
    matches: list[Path] = []
    seen: set[str] = set()

    def walk(folder: Path, depth: int) -> None:
        if depth > max_depth:
            return
        plate_dir = _resolve_plate_dir_containing_files(folder)
        if plate_dir is not None and _name_matches_plate_leaf(folder, plate_dir, leaf):
            key = str(plate_dir.resolve())
            if key not in seen:
                seen.add(key)
                matches.append(plate_dir.resolve())
        try:
            children = sorted(folder.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_dir():
                walk(child, depth + 1)

    walk(root, 0)
    return matches


def _existing_ancestor_dirs(mount: Path, rel: str) -> list[Path]:
    """Longest existing directory prefixes of ``rel`` (for subtree search anchors)."""
    rel_path = Path(rel.strip().strip("/").replace("\\", "/"))
    parts = rel_path.parts
    anchors: list[Path] = []
    seen: set[str] = set()
    for end in range(len(parts), 0, -1):
        prefix = Path(*parts[:end]) if end else Path(".")
        candidates: list[Path] = []
        if end == 0:
            candidates = [mount / "VDA_input", mount]
        else:
            candidates = [mount / prefix]
            if parts[0] != "VDA_input":
                candidates.append(mount / "VDA_input" / prefix)
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_dir():
                anchors.append(path)
    return anchors


def _search_under_vda_input(mount: Path, leaf: str, *, max_depth: int = _PLATE_SEARCH_MAX_DEPTH) -> list[Path]:
    """Find plate folders under MyDrive/VDA_input when Desk path omits parents."""
    vda = mount / "VDA_input"
    if not vda.is_dir() or not leaf:
        return []
    return _search_subtree_for_plates(vda, leaf, max_depth=max_depth)


def _pick_best_plate_match(mount: Path, rel_plate: str, matches: list[Path]) -> Path | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0].resolve()
    rel_norm = rel_plate.strip().strip("/").replace("\\", "/")
    rel_parts = [p for p in Path(rel_norm).parts if p and p != "VDA_input"]
    scored: list[tuple[int, int, Path]] = []
    for path in matches:
        try:
            rel_to_mount = path.relative_to(mount).as_posix()
        except ValueError:
            rel_to_mount = str(path)
        parts = [p for p in Path(rel_to_mount).parts if p and p != "VDA_input"]
        overlap = 0
        for a, b in zip(reversed(rel_parts), reversed(parts)):
            if a == b:
                overlap += 1
            else:
                break
        try:
            frame_count = sum(1 for e in path.iterdir() if e.is_file())
        except OSError:
            frame_count = 0
        scored.append((overlap, frame_count, path))
    scored.sort(key=lambda t: (-t[0], -t[1], len(t[2].parts)))
    chosen = scored[0][2].resolve()
    _log.warning(
        "Plate path %r matched %d folder(s) under VDA_input; using %s",
        rel_plate,
        len(matches),
        chosen.relative_to(mount),
    )
    return chosen


def _resolve_plate_dir(mount: Path, rel_plate: str) -> Path | None:
    """Try Desk/Drive path layouts; fuzzy siblings; recursive search under VDA_input."""
    rel = rel_plate.strip().strip("/").replace("\\", "/")
    if not rel:
        return None

    candidates: list[Path] = [mount / rel]
    if rel.startswith("VDA_input/"):
        inner = rel[len("VDA_input/") :]
        if inner:
            candidates.append(mount / "VDA_input" / inner)
    else:
        candidates.append(mount / "VDA_input" / rel)

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_dir():
            continue
        plate_dir = _resolve_plate_dir_containing_files(path)
        if plate_dir is not None:
            if plate_dir != path.resolve():
                _log.info(
                    "Plate path %r: using nested folder %s",
                    rel_plate,
                    plate_dir.relative_to(mount),
                )
            return plate_dir

    rel_path = Path(rel)
    parent_rel = rel_path.parent
    leaf = rel_path.name
    if not leaf:
        return None
    parent = mount / parent_rel
    if not parent.is_dir():
        if str(parent_rel).startswith("VDA_input/"):
            parent = mount / parent_rel
        elif parent_rel.parts:
            parent = mount / "VDA_input" / parent_rel
    if parent.is_dir():
        fuzzy: list[Path] = []
        try:
            for child in sorted(parent.iterdir()):
                if not child.is_dir():
                    continue
                if not _leaf_matches_dir(child.name, leaf):
                    continue
                plate_dir = _resolve_plate_dir_containing_files(child)
                if plate_dir is not None:
                    fuzzy.append(plate_dir)
        except OSError:
            fuzzy = []
        if len(fuzzy) == 1:
            chosen = fuzzy[0].resolve()
            _log.warning(
                "Plate path %r not found; using %s",
                rel_plate,
                chosen.relative_to(mount),
            )
            return chosen
        if len(fuzzy) > 1:
            return _pick_best_plate_match(mount, rel_plate, fuzzy)

    for anchor in _existing_ancestor_dirs(mount, rel):
        nested = _search_subtree_for_plates(anchor, leaf)
        if nested:
            chosen = _pick_best_plate_match(mount, rel_plate, nested)
            if chosen is not None:
                _log.warning(
                    "Plate path %r not found; resolved under %s -> %s",
                    rel_plate,
                    anchor.relative_to(mount),
                    chosen.relative_to(mount),
                )
                return chosen

    nested = _search_under_vda_input(mount, leaf)
    if nested:
        return _pick_best_plate_match(mount, rel_plate, nested)
    return None


def _normalize_drive_rel_path(rel: str) -> str:
    p = str(rel or "").strip().replace("\\", "/")
    for prefix in (
        "/content/drive/MyDrive/",
        "/content/drive/My Drive/",
        "MyDrive/",
        "My Drive/",
    ):
        if p.startswith(prefix):
            p = p[len(prefix) :]
    return p.strip("/")


def _mount_path_candidates(mount: Path, rel_folder: str) -> list[Path]:
    """Absolute folder paths to try under the Drive mount."""
    rel = _normalize_drive_rel_path(rel_folder)
    if not rel:
        return []
    out: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)

    _add(mount / rel)
    if rel.startswith("VDA_input/"):
        inner = rel[len("VDA_input/") :]
        if inner:
            _add(mount / "VDA_input" / inner)
    else:
        _add(mount / "VDA_input" / rel)
    return out


def _printf_pattern_to_hash(filename_fmt: str) -> str:
    return re.sub(
        r"%0(\d+)d",
        lambda m: "#" * int(m.group(1)),
        filename_fmt,
    )


def _resolve_plate_from_desk_input_video(
    mount: Path, seq: dict
) -> tuple[Path, str, Path] | None:
    """Locate plates using Desk ``input_video`` + ``first_frame`` (GUI preview paths)."""
    input_video = str(seq.get("input_video") or "").strip()
    if not input_video:
        return None
    rel = _normalize_drive_rel_path(input_video)
    folder_rel = str(Path(rel).parent).replace("\\", "/")
    fname_fmt = Path(rel).name
    first_frame = int(seq.get("first_frame", 1001))
    try:
        first_fname = fname_fmt % first_frame
    except (TypeError, ValueError):
        return None
    pattern = str(seq.get("sequence_pattern") or _printf_pattern_to_hash(fname_fmt))
    for folder in _mount_path_candidates(mount, folder_rel):
        if not folder.is_dir():
            continue
        for candidate in (folder, _resolve_plate_dir_containing_files(folder)):
            if candidate is None:
                continue
            probe = candidate / first_fname
            if probe.is_file():
                return candidate.resolve(), pattern, probe.resolve()
    return None


def _desk_relpath_to_abs(mount: Path, rel: str) -> Path:
    return (mount / _normalize_drive_rel_path(rel)).resolve()


def _frame_basename_from_pattern(pattern: str, frame: int) -> str:
    """``TB_073_020_plate_v001_####.exr`` + 1353 → ``TB_073_020_plate_v001_1353.exr``."""
    hash_run = re.search(r"#+", pattern)
    if hash_run:
        width = len(hash_run.group(0))
        return f"{pattern[: hash_run.start()]}{frame:0{width}d}{pattern[hash_run.end() :]}"
    printf = re.search(r"%0?(\d*)d", pattern)
    if printf:
        width = int(printf.group(1)) if printf.group(1) else 0
        return pattern[: printf.start()] + (f"{frame:0{width}d}" if width else str(frame)) + pattern[printf.end() :]
    return pattern


def _plate_file_ready(path: Path, *, retries: int = 3) -> bool:
    """Drive mounts can list a name before the bytes are readable on Colab."""
    if not path.exists():
        return False
    for attempt in range(max(1, retries)):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            pass
        if attempt + 1 < retries:
            time.sleep(0.35)
    return False


def _plate_pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Match ``oiio_exr._pattern_to_regex`` without importing the LAOV package."""
    if "#" in pattern:
        rx = re.escape(pattern)
        rx = re.sub(
            r"(?:\\#)+",
            lambda m: f"(?P<frame>\\d{{{(len(m.group(0)) // 2)}}})",
            rx,
        )
        return re.compile("^" + rx + "$")
    m = re.search(r"%0?(\d*)d", pattern)
    if m:
        width = m.group(1)
        width_rx = f"\\d{{{int(width)}}}" if width else r"\d+"
        prefix = re.escape(pattern[: m.start()])
        suffix = re.escape(pattern[m.end() :])
        return re.compile(f"^{prefix}(?P<frame>{width_rx}){suffix}$")
    return re.compile("^" + re.escape(pattern) + "$")


def _enumerate_plate_frame_numbers(plate_dir: Path, pattern: str) -> list[int]:
    ext = Path(pattern).suffix.lower() or ".exr"
    regex = _plate_pattern_to_regex(pattern)
    frames: list[int] = []
    for entry in sorted(plate_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ext:
            continue
        match = regex.match(entry.name)
        if not match:
            continue
        frames.append(int(match.group("frame")))
    if not frames:
        raise FileNotFoundError(
            f"No frames matched pattern {pattern!r} in {plate_dir}"
        )
    return sorted(frames)


def _verify_desk_frame_range(
    plate_dir: Path,
    pattern: str,
    desk_f0: int,
    desk_f1: int,
) -> tuple[int, int]:
    """Require every Desk frame file to exist before SAM3/RAFT (no silent range shrink)."""
    expected = desk_f1 - desk_f0 + 1
    missing: list[tuple[int, Path]] = []
    for frame in range(desk_f0, desk_f1 + 1):
        path = plate_dir / _frame_basename_from_pattern(pattern, frame)
        if not _plate_file_ready(path):
            missing.append((frame, path))

    scanned = [
        f
        for f in _enumerate_plate_frame_numbers(plate_dir, pattern)
        if desk_f0 <= f <= desk_f1
    ]

    if missing:
        lines = [
            f"Plate preflight failed: {len(missing)}/{expected} frames missing under",
            f"  {plate_dir}",
            f"  pattern: {pattern}",
        ]
        for frame, path in missing[:30]:
            lines.append(f"  frame {frame}: {path.name}")
        if len(missing) > 30:
            lines.append(f"  ... and {len(missing) - 30} more")
        lines.append(
            f"  Folder scan (regex): {len(scanned)} files in Desk range "
            f"{desk_f0}-{desk_f1}"
        )
        if len(scanned) >= expected - len(missing):
            lines.append(
                "  Names appear in directory listing but files are not readable yet — "
                "wait for Google Drive sync on Colab, remount Drive, then retry."
            )
        else:
            lines.append(
                "  Re-scan the shot in Desk or upload missing EXRs to Drive."
            )
        raise FileNotFoundError("\n".join(lines))

    if len(scanned) != expected:
        _log.warning(
            "Folder scan count %d != Desk span %d (pattern %r) — per-frame paths OK",
            len(scanned),
            expected,
            pattern,
        )

    _log.info(
        "Plate preflight OK: %d frames %s-%s verified in %s",
        expected,
        desk_f0,
        desk_f1,
        plate_dir,
    )
    return desk_f0, desk_f1


def shot_plate_from_desk_json(
    mount: Path,
    seq: dict,
) -> tuple[Path, str, tuple[int, int], tuple[int, int], float]:
    """AI Matte: use Desk JSON paths only (no folder search / no path rewriting).

    Requires ``plate_first_frame_relpath`` or ``input_video`` + ``first_frame`` from the
    Desk prep / Send to Colab flow — same strings as the GUI Colab preview.
    """
    first_rel = str(
        seq.get("plate_first_frame_relpath") or seq.get("plate_first_frame") or ""
    ).strip()
    input_pattern = str(
        seq.get("plate_input_pattern") or seq.get("input_video") or ""
    ).strip()
    if not first_rel and input_pattern:
        rel = _normalize_drive_rel_path(input_pattern)
        fname_fmt = Path(rel).name
        first_frame = int(seq.get("first_frame", 1001))
        try:
            first_rel = f"{Path(rel).parent}/{fname_fmt % first_frame}".replace("\\", "/")
        except (TypeError, ValueError) as exc:
            raise FileNotFoundError(
                f"Cannot build first frame path from input_video={input_pattern!r}: {exc}"
            ) from exc
    if not first_rel:
        raise FileNotFoundError(
            "Desk batch missing plate_first_frame_relpath / input_video — re-send from AI Matte Desk"
        )

    first_path = _desk_relpath_to_abs(mount, first_rel)
    if not first_path.is_file():
        last_rel = str(seq.get("plate_last_frame_relpath") or "").strip()
        lines = [
            "Desk plate file not found on Colab Drive mount.",
            f"  mount: {mount}",
            f"  expected first frame: {first_path}",
            f"  exists: {first_path.is_file()}",
        ]
        if input_pattern:
            lines.append(f"  plate_input_pattern: {input_pattern!r}")
        if last_rel:
            lines.append(f"  plate_last_frame_relpath: {mount / _normalize_drive_rel_path(last_rel)}")
        lines.append(
            "  Sync this file to Google Drive (My Drive), then re-run Cell 3."
        )
        raise FileNotFoundError("\n".join(lines))

    plate_dir = first_path.parent
    pattern = str(
        seq.get("sequence_pattern")
        or _printf_pattern_to_hash(Path(_normalize_drive_rel_path(input_pattern)).name)
        if input_pattern
        else first_path.name
    )
    desk_f0 = int(seq.get("first_frame", 1001))
    desk_f1 = int(seq.get("last_frame", desk_f0))
    f0, f1 = _verify_desk_frame_range(plate_dir, pattern, desk_f0, desk_f1)

    last_rel = str(seq.get("plate_last_frame_relpath") or "").strip()
    if last_rel:
        last_path = _desk_relpath_to_abs(mount, last_rel)
        if not _plate_file_ready(last_path):
            raise FileNotFoundError(
                f"Desk last frame not readable on mount:\n  {last_path}"
            )

    try:
        from live_action_aov.io.oiio_io import read_plate

        pixels, attrs = read_plate(first_path)
        h, w = pixels.shape[:2]
        resolution = (w, h)
        pixel_aspect = float(attrs.get("pixelAspectRatio", 1.0))
    except Exception as exc:
        _log.warning("EXR header read failed (%s); continuing with Desk frame range", exc)
        resolution = (2048, 1152)
        pixel_aspect = 1.0

    try:
        _log.info(
            "Desk plate: %s | pattern=%s | desk=%s-%s | run=%s-%s",
            first_path.relative_to(mount),
            pattern,
            desk_f0,
            desk_f1,
            f0,
            f1,
        )
    except ValueError:
        _log.info(
            "Desk plate: %s | pattern=%s | desk=%s-%s | run=%s-%s",
            first_path,
            pattern,
            desk_f0,
            desk_f1,
            f0,
            f1,
        )
    return plate_dir, pattern, (f0, f1), resolution, pixel_aspect


def resolve_plate_for_batch_sequence(
    mount: Path,
    seq: dict,
    sniff_fn: Callable[[Path], tuple[str, tuple[int, int], tuple[int, int], float]],
) -> tuple[Path, str, tuple[int, int], tuple[int, int], float] | None:
    """Resolve plate folder + pattern; prefers Desk ``input_video`` over folder-only lookup."""
    rel_plate = (
        str(seq.get("plate_folder_drive_relative", "")).strip().strip("/").replace("\\", "/")
    )
    desk = _resolve_plate_from_desk_input_video(mount, seq)
    plate_dir: Path | None = None
    pattern: str | None = None
    first_frame_path: Path | None = None

    if desk is not None:
        plate_dir, pattern, first_frame_path = desk
        try:
            _log.info(
                "Plate from Desk input_video: %s (pattern=%s)",
                plate_dir.relative_to(mount),
                pattern,
            )
        except ValueError:
            _log.info("Plate from Desk input_video: %s (pattern=%s)", plate_dir, pattern)
    if plate_dir is None:
        plate_dir = _resolve_plate_dir(mount, rel_plate)

    if plate_dir is None:
        return None

    if pattern is None or first_frame_path is None:
        try:
            pattern, sniffed_range, resolution, pixel_aspect = sniff_fn(plate_dir)
        except FileNotFoundError:
            return None
    else:
        sniffed_range = (
            int(seq.get("first_frame", 1001)),
            int(seq.get("last_frame", 1100)),
        )
        try:
            from live_action_aov.io.oiio_io import read_plate

            pixels, attrs = read_plate(first_frame_path)
            h, w = pixels.shape[:2]
            resolution = (w, h)
            pixel_aspect = float(attrs.get("pixelAspectRatio", 1.0))
        except Exception as exc:
            _log.warning(
                "Desk first frame %s found; EXR header read failed (%s) — using job frames",
                first_frame_path.name,
                exc,
            )
            resolution = (2048, 1152)
            pixel_aspect = 1.0

    j0 = int(seq.get("first_frame", sniffed_range[0]))
    j1 = int(seq.get("last_frame", sniffed_range[1]))
    s0, s1 = sniffed_range
    f0, f1 = max(s0, j0), min(s1, j1)
    if f0 > f1:
        f0, f1 = j0, j1
    return plate_dir, pattern, (f0, f1), resolution, pixel_aspect


def _format_plate_hint(mount: Path, rel_plate: str, seq: dict | None = None) -> str:
    lines = [f"  mount: {mount}", f"  plate_folder_drive_relative: {rel_plate!r}"]
    if seq:
        iv = seq.get("input_video")
        if iv:
            lines.append(f"  input_video (Desk): {iv!r}")
            rel = _normalize_drive_rel_path(str(iv))
            first = int(seq.get("first_frame", 1001))
            try:
                first_name = Path(rel).name % first
            except (TypeError, ValueError):
                first_name = None
            if first_name:
                lines.append("  Desk first-frame probes:")
                folder_rel = str(Path(rel).parent)
                for folder in _mount_path_candidates(mount, folder_rel):
                    probe = folder / first_name
                    lines.append(
                        f"    - {probe}  exists={probe.is_file()}"
                    )
    rel = rel_plate.strip().strip("/")
    lines.append("  tried:")
    for p in (mount / rel, mount / "VDA_input" / rel):
        plate = _resolve_plate_dir_containing_files(p) if p.is_dir() else None
        lines.append(
            f"    - {p}  exists={p.exists()} dir={p.is_dir()} plates={plate is not None}"
        )
    rel_path = Path(rel)
    leaf = rel_path.name
    anchors = _existing_ancestor_dirs(mount, rel)
    if anchors:
        lines.append("  existing path prefixes (searched recursively):")
        for a in anchors[:6]:
            lines.append(f"    - {a.relative_to(mount)}")
    if leaf:
        hits = _search_under_vda_input(mount, leaf)
        if hits:
            lines.append(f"  matches under VDA_input/ for leaf {leaf!r}:")
            for h in hits[:8]:
                lines.append(f"    - {h.relative_to(mount)}")
        else:
            lines.append(
                f"  no plate sequence found under VDA_input/ for leaf {leaf!r} "
                f"(searched subfolders depth {_PLATE_SEARCH_MAX_DEPTH})"
            )
    parent = mount / rel_path.parent
    if parent.is_dir():
        try:
            kids = sorted(x.name for x in parent.iterdir() if x.is_dir())[:20]
            lines.append(f"  subfolders of {parent.name}/: {kids}")
        except OSError:
            pass
    vda = mount / "VDA_input"
    if vda.is_dir():
        try:
            kids = sorted(x.name for x in vda.iterdir())[:12]
            lines.append(f"  VDA_input/ (first entries): {kids}")
        except OSError:
            pass
    lines.append(
        "  tip: refresh sequence paths in Desk (Drive folder_id) or fix plate_folder_drive_relative"
    )
    return "\n".join(lines)


def main() -> int:
    try:
        from colab_run_status import configure_flushed_logging

        configure_flushed_logging()
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return _main_impl()
    except Exception:
        _log.error("LAOV batch runner crashed:\n%s", traceback.format_exc())
        return 1


def _main_impl() -> int:
    args = _parse_args()
    job_path = Path(args.job_json).resolve()
    if not job_path.is_file():
        _log.error("Job JSON not found: %s", job_path)
        return 1
    with open(job_path, encoding="utf-8") as f:
        cfg = json.load(f)
    shared = cfg.get("shared_settings") or {}
    sequences = cfg.get("sequences") or []
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    mount = _drive_mount()
    if not mount.is_dir():
        _log.error("Drive mount not found: %s (mount Drive in Cell 1)", mount)
        return 1
    _log.info("Drive mount: %s", mount)

    passes_csv = str(shared.get("passes_csv", "depth,normals,flow,matte"))
    raw_names = [x.strip() for x in passes_csv.split(",") if x.strip()]

    try:
        from live_action_aov.cli.app import (  # type: ignore[attr-defined]
            _resolve_semantic_passes,
            _sniff_sequence,
        )
        from live_action_aov.core.job import Job, PassConfig, Shot
        from live_action_aov.core.registry import get_registry
        from live_action_aov import run as laov_run
        from live_action_aov.io.oiio_io import require_oiio

        require_oiio()
    except Exception as exc:
        _log.error("LAOV import/setup failed: %s", exc)
        _log.error("%s", traceback.format_exc())
        return 1

    depth_backend = str(shared.get("depth_backend", "depth_anything_v2"))
    normals_backend = str(shared.get("normals_backend", "dsine"))
    matte_detector = str(shared.get("matte_detector", "sam3_matte"))
    refiner = str(shared.get("refiner", "rvm_refiner"))
    allow_nc = bool(shared.get("allow_noncommercial", False))
    display_transform = bool(shared.get("display_transform", False))
    colorspace = shared.get("colorspace")
    colorspace_s = str(colorspace) if colorspace else "auto"
    proxy_raw = shared.get("proxy_long_edge")
    proxy_long_edge = int(proxy_raw) if proxy_raw is not None else None
    output_layout = str(shared.get("output_layout", "split_folders")).strip()
    if output_layout not in ("combined", "split_folders"):
        _log.warning("Unknown output_layout %r — using split_folders", output_layout)
        output_layout = "split_folders"

    sam3_model_dir = _resolve_sam3_model_dir(mount, shared)
    if sam3_model_dir:
        os.environ["LAOV_SAM3_MODEL_PATH"] = sam3_model_dir
    birefnet_model_dir = _resolve_birefnet_model_dir(mount, shared)
    if birefnet_model_dir:
        os.environ["AI_MATTE_BIREFNET_MODEL_PATH"] = birefnet_model_dir

    pass_names = _resolve_semantic_passes(
        raw_names,
        depth_backend=depth_backend,
        normals_backend=normals_backend,
        matte_detector=matte_detector,
        refiner=refiner,
    )
    registry = get_registry()
    blocked: list[tuple[str, str]] = []
    for name in pass_names:
        cls = registry.get_pass(name)
        lic = cls.declared_license()
        if not lic.commercial_use and not allow_nc:
            blocked.append((name, lic.spdx))
    if blocked:
        for n, spdx in blocked:
            _log.error("Non-commercial pass %s (%s) — set allow_noncommercial in Desk.", n, spdx)
        return 2

    out_folder_path = str(shared.get("output_folder_path", "VDA_output")).strip("/")
    date_folder = str(os.environ.get("LAOV_RUNTIME_DATE_FOLDER", "")).strip().strip("/")

    status = None
    try:
        from colab_run_status import reporter_from_job_json

        status = reporter_from_job_json(mount, cfg)
        if status:
            status.begin_run("LAOV batch — resolving passes")
            _log.info("Status file: %s", status.status_path_hint())
    except ImportError:
        pass

    any_failed = False
    shot_total = len(sequences)
    for shot_idx, seq in enumerate(sequences):
        rel_plate = str(seq.get("plate_folder_drive_relative", "")).strip().strip("/").replace("\\", "/")
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        if status:
            status.shot_begin(shot_name, shot_idx + 1, shot_total)
        if not rel_plate:
            _log.error("Sequence missing plate_folder_drive_relative")
            return 1

        resolved = resolve_plate_for_batch_sequence(mount, seq, _sniff_sequence)
        if resolved is None:
            _log.error(
                "Plate folder not found.\n%s",
                _format_plate_hint(mount, rel_plate, seq),
            )
            return 1
        plate_dir, pattern, (f0, f1), resolution, pixel_aspect = resolved

        output_dir: Path | None
        if rel_side:
            if date_folder:
                output_dir = (mount / out_folder_path / date_folder / rel_side).resolve()
            else:
                output_dir = (mount / out_folder_path / rel_side).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

        _log.info(
            "Shot %s: plate=%s pattern=%s frames=%s-%s output=%s layout=%s",
            shot_name,
            plate_dir,
            pattern,
            f0,
            f1,
            output_dir,
            output_layout,
        )

        shot = Shot(
            name=shot_name,
            folder=plate_dir,
            apply_display_transform=display_transform,
            colorspace=colorspace_s,
            sequence_pattern=pattern,
            frame_range=(f0, f1),
            resolution=resolution,
            pixel_aspect=pixel_aspect,
            passes_enabled=pass_names,
            output_dir=output_dir,
            proxy_long_edge=proxy_long_edge,
            output_layout=output_layout,  # type: ignore[arg-type]
        )
        pass_configs: list[PassConfig] = []
        for name in pass_names:
            params = _matte_pass_params(
                name,
                shared,
                seq,
                sam3_model_dir=sam3_model_dir,
                birefnet_model_dir=birefnet_model_dir,
            )
            pass_configs.append(PassConfig(name=name, params=params))
        job = Job(shot=shot, passes=pass_configs)
        progress_cb = (
            status.make_laov_callback(shot_name, shot_idx, shot_total) if status else None
        )
        try:
            if status:
                status.stage(f"{shot_name}: engine — passes {passes_csv}", shot_name=shot_name)
            laov_run(job, progress_callback=progress_cb)
        except Exception:
            _log.error("LAOV run failed for shot %s:\n%s", shot_name, traceback.format_exc())
            if status:
                status.shot_end(shot_name, ok=False, message="engine error")
            any_failed = True
            continue

        if shot.status != "done":
            _log.error("Shot %s finished with status=%s (expected done)", shot_name, shot.status)
            if status:
                status.shot_end(shot_name, ok=False, message=str(shot.status))
            any_failed = True
        else:
            _log.info("Done shot=%s status=%s", shot_name, shot.status)
            if status:
                status.shot_end(shot_name, ok=True)

    if any_failed:
        _log.error("One or more shots failed.")
        if status:
            status.finish_run(ok=False)
        return 1

    _log.info("All sequences complete.")
    if status:
        status.finish_run(ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
