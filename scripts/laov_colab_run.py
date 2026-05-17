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
import sys
import traceback
from pathlib import Path

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


_PLATE_SUFFIXES = (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".dpx", ".tga")


def _folder_has_plates(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    try:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in _PLATE_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def _leaf_matches_dir(name: str, leaf: str) -> bool:
    if not leaf:
        return False
    return name == leaf or name.startswith(leaf + "_") or name.startswith(leaf + ".")


def _search_under_vda_input(mount: Path, leaf: str, *, max_depth: int = 8) -> list[Path]:
    """Find plate folders under MyDrive/VDA_input when Desk path omits parents (e.g. LOT_test/)."""
    vda = mount / "VDA_input"
    if not vda.is_dir() or not leaf:
        return []
    matches: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(folder.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if _leaf_matches_dir(child.name, leaf) and _folder_has_plates(child):
                matches.append(child)
            walk(child, depth + 1)

    walk(vda, 0)
    return matches


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
        if path.is_dir() and _folder_has_plates(path):
            return path.resolve()

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
                if _leaf_matches_dir(child.name, leaf) and _folder_has_plates(child):
                    fuzzy.append(child)
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

    nested = _search_under_vda_input(mount, leaf)
    if nested:
        return _pick_best_plate_match(mount, rel_plate, nested)
    return None


def _format_plate_hint(mount: Path, rel_plate: str) -> str:
    lines = [f"  mount: {mount}", f"  plate_folder_drive_relative: {rel_plate!r}"]
    rel = rel_plate.strip().strip("/")
    lines.append("  tried:")
    for p in (mount / rel, mount / "VDA_input" / rel):
        lines.append(f"    - {p}  exists={p.exists()} dir={p.is_dir()}")
    rel_path = Path(rel)
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
    return "\n".join(lines)


def main() -> int:
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

    any_failed = False
    for seq in sequences:
        rel_plate = str(seq.get("plate_folder_drive_relative", "")).strip().strip("/").replace("\\", "/")
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        if not rel_plate:
            _log.error("Sequence missing plate_folder_drive_relative")
            return 1

        plate_dir = _resolve_plate_dir(mount, rel_plate)
        if plate_dir is None:
            _log.error("Plate folder not found.\n%s", _format_plate_hint(mount, rel_plate))
            return 1

        output_dir: Path | None
        if rel_side:
            if date_folder:
                output_dir = (mount / out_folder_path / date_folder / rel_side).resolve()
            else:
                output_dir = (mount / out_folder_path / rel_side).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

        try:
            pattern, sniffed_range, resolution, pixel_aspect = _sniff_sequence(plate_dir)
        except FileNotFoundError as exc:
            _log.error("No plate sequence in %s: %s", plate_dir, exc)
            return 1

        j0 = int(seq.get("first_frame", sniffed_range[0]))
        j1 = int(seq.get("last_frame", sniffed_range[1]))
        s0, s1 = sniffed_range
        f0 = max(s0, j0)
        f1 = min(s1, j1)
        if f0 > f1:
            _log.warning(
                "Job frame range %s-%s outside sniffed %s-%s; using sniffed.",
                j0,
                j1,
                s0,
                s1,
            )
            f0, f1 = s0, s1

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
        job = Job(shot=shot, passes=[PassConfig(name=n) for n in pass_names])
        try:
            laov_run(job)
        except Exception:
            _log.error("LAOV run failed for shot %s:\n%s", shot_name, traceback.format_exc())
            any_failed = True
            continue

        if shot.status != "done":
            _log.error("Shot %s finished with status=%s (expected done)", shot_name, shot.status)
            any_failed = True
        else:
            _log.info("Done shot=%s status=%s", shot_name, shot.status)

    if any_failed:
        _log.error("One or more shots failed.")
        return 1

    _log.info("All sequences complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
