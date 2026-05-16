#!/usr/bin/env python3
# LiveActionAOV — Colab / Docker batch runner (reads Desk batch JSON).
"""Execute LAOV jobs produced by Google Desk (LAOV_STANDALONE).

Expects:
  - ``LAOV_DRIVE_MOUNT`` — absolute path where Google Drive ``MyDrive`` is mounted
    (e.g. ``/data`` in Docker, ``/content/drive/MyDrive`` in Colab fallback).
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
    raw = os.environ.get("LAOV_DRIVE_MOUNT", "/data")
    return Path(raw).resolve()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
    passes_csv = str(shared.get("passes_csv", "depth,normals,flow,matte"))
    raw_names = [x.strip() for x in passes_csv.split(",") if x.strip()]

    from live_action_aov.cli.app import (  # type: ignore[attr-defined]
        _resolve_semantic_passes,
        _sniff_sequence,
    )
    from live_action_aov.core.job import Job, PassConfig, Shot
    from live_action_aov.core.registry import get_registry
    from live_action_aov import run as laov_run

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

    out_folder_path = str(shared.get("output_folder_path", "VDA_Jobs/results")).strip("/")
    date_folder = str(os.environ.get("LAOV_RUNTIME_DATE_FOLDER", "")).strip().strip("/")

    for seq in sequences:
        rel_plate = str(seq.get("plate_folder_drive_relative", "")).strip().strip("/").replace("\\", "/")
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        if not rel_plate:
            _log.error("Sequence missing plate_folder_drive_relative")
            return 1
        plate_dir = (mount / rel_plate).resolve()
        if not plate_dir.is_dir():
            _log.error("Plate folder not found: %s", plate_dir)
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

        pattern, sniffed_range, resolution, pixel_aspect = _sniff_sequence(plate_dir)
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
        )
        job = Job(shot=shot, passes=[PassConfig(name=n) for n in pass_names])
        _log.info("Running shot=%s passes=%s output_dir=%s", shot_name, pass_names, output_dir)
        laov_run(job)
        _log.info("Done shot=%s status=%s", shot_name, job.shot.status)

    _log.info("All sequences complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
