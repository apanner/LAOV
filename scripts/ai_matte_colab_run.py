#!/usr/bin/env python3
"""AI Matte Colab batch runner — SAM3 track + BiRefNet refine (Desk AI_MATTE_STANDALONE).

Separate entry point from ``laov_colab_run.py`` so Colab and git always ship the
matte-only lane with the correct defaults and ``birefnet_refiner`` pass.

Usage on Colab (after clone + pip install LAOV)::

    export LAOV_DRIVE_MOUNT=/content/drive/MyDrive
    export LAOV_RUNTIME_DATE_FOLDER=$(date +%Y%m%d)
    python scripts/ai_matte_colab_run.py --job-json /content/ai_matte_config.json

Expects batch JSON from ``google_desk_app`` (model_type ``AI_MATTE_STANDALONE``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Allow ``import laov_colab_run`` when executed as scripts/ai_matte_colab_run.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import laov_colab_run as lcr  # noqa: E402

_log = logging.getLogger("ai_matte_colab_run")

AI_MATTE_DEFAULTS = {
    "passes_csv": "matte",
    "matte_detector": "sam3_matte",
    "refiner": "birefnet_refiner",
    "display_transform": True,
    "output_layout": "split_folders",
    "matte_mode": "people_fg",
    "matte_concepts": ["person"],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AI Matte batch from Desk JSON.")
    p.add_argument(
        "--job-json",
        required=True,
        help="Path to AI_MATTE_STANDALONE batch JSON.",
    )
    p.add_argument(
        "--probe-first-frame",
        action="store_true",
        help="Log shape/min/max of first plate frame per shot (OIIO read check).",
    )
    return p.parse_args()


def _merge_shared_defaults(shared: dict) -> dict:
    out = dict(shared)
    for key, val in AI_MATTE_DEFAULTS.items():
        if out.get(key) in (None, ""):
            out[key] = val
    if not str(out.get("passes_csv", "")).strip():
        out["passes_csv"] = "matte"
    return out


def _verify_registry(refiner: str) -> int:
    try:
        from live_action_aov.core.registry import get_registry

        reg = get_registry()
        for name in ("sam3_matte", refiner):
            reg.get_pass(name)
        _log.info("Registry OK: sam3_matte + %s", refiner)
        return 0
    except KeyError as exc:
        _log.error(
            "Pass not registered: %s\n"
            "Push LAOV with birefnet_refiner in pyproject.toml entry points, "
            "then re-clone on Colab.",
            exc,
        )
        return 1
    except Exception as exc:
        _log.error("Registry check failed: %s", exc)
        return 1


def _probe_plate_read(plate_dir: Path, pattern: str, frame: int) -> None:
    try:
        from live_action_aov.io.readers.oiio_exr import OIIOExrReader

        reader = OIIOExrReader(plate_dir, pattern)
        px, attrs = reader.read_frame(frame)
        _log.info(
            "Plate probe frame %s: path=%s shape=%s dtype=%s min=%.4f max=%.4f channels=%s",
            frame,
            plate_dir,
            px.shape,
            px.dtype,
            float(px.min()),
            float(px.max()),
            attrs.get("channelnames"),
        )
    except Exception as exc:
        _log.error("Plate probe failed for %s frame %s: %s", plate_dir, frame, exc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    job_path = Path(args.job_json).resolve()
    if not job_path.is_file():
        _log.error("Job JSON not found: %s", job_path)
        return 1

    with open(job_path, encoding="utf-8") as f:
        cfg = json.load(f)

    shared = _merge_shared_defaults(cfg.get("shared_settings") or {})
    cfg["shared_settings"] = shared
    model_type = str(shared.get("model_type", ""))
    if model_type and model_type != "AI_MATTE_STANDALONE":
        _log.warning("Expected model_type AI_MATTE_STANDALONE, got %r", model_type)

    matte_detector = str(shared.get("matte_detector", "sam3_matte"))
    refiner = str(shared.get("refiner", "birefnet_refiner"))
    _log.info("AI Matte batch — detector=%s refiner=%s passes=%s", matte_detector, refiner, shared.get("passes_csv"))

    try:
        from live_action_aov.io.oiio_io import require_oiio

        require_oiio()
    except Exception as exc:
        _log.error("OpenImageIO required for plate read: %s", exc)
        return 1

    if _verify_registry(refiner) != 0:
        return 1

    mount = lcr._drive_mount()
    if not mount.is_dir():
        _log.error("Drive mount not found: %s", mount)
        return 1
    _log.info("Drive mount: %s", mount)

    sam3_model_dir = lcr._resolve_sam3_model_dir(mount, shared)
    if sam3_model_dir:
        os.environ["LAOV_SAM3_MODEL_PATH"] = sam3_model_dir
        _log.info("SAM3 model: %s", sam3_model_dir)
    else:
        _log.warning("No SAM3 snapshot on Drive — Hub download may require HF login")

    birefnet_model_dir = lcr._resolve_birefnet_model_dir(mount, shared)
    if birefnet_model_dir:
        os.environ["AI_MATTE_BIREFNET_MODEL_PATH"] = birefnet_model_dir
        _log.info("BiRefNet model: %s", birefnet_model_dir)
    elif refiner == "birefnet_refiner":
        _log.error(
            "BiRefNet snapshot not found on Drive. Upload MyDrive/VDA_models/ZhengPeng7/BiRefNet/"
        )
        return 1

    sequences = cfg.get("sequences") or []
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    try:
        return _run_sequences(
            cfg,
            mount,
            shared,
            sequences,
            matte_detector=matte_detector,
            refiner=refiner,
            probe_first_frame=args.probe_first_frame,
        )
    except Exception:
        _log.error("AI Matte runner crashed:\n%s", traceback.format_exc())
        return 1


def _run_sequences(
    cfg: dict,
    mount: Path,
    shared: dict,
    sequences: list,
    *,
    matte_detector: str,
    refiner: str,
    probe_first_frame: bool,
) -> int:
    from live_action_aov.cli.app import _resolve_semantic_passes, _sniff_sequence
    from live_action_aov.core.job import Job, PassConfig, Shot
    from live_action_aov.core.registry import get_registry
    from live_action_aov import run as laov_run

    passes_csv = str(shared.get("passes_csv", "matte"))
    raw_names = [x.strip() for x in passes_csv.split(",") if x.strip()]
    allow_nc = bool(shared.get("allow_noncommercial", False))
    display_transform = bool(shared.get("display_transform", True))
    colorspace = shared.get("colorspace")
    colorspace_s = str(colorspace) if colorspace else "auto"
    proxy_raw = shared.get("proxy_long_edge")
    proxy_long_edge = int(proxy_raw) if proxy_raw is not None else None
    output_layout = str(shared.get("output_layout", "split_folders"))

    pass_names = _resolve_semantic_passes(
        raw_names,
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
        matte_detector=matte_detector,
        refiner=refiner,
    )
    registry = get_registry()
    for name in pass_names:
        lic = registry.get_pass(name).declared_license()
        if not lic.commercial_use and not allow_nc:
            _log.error("Non-commercial pass %s — enable allow_noncommercial in Desk", name)
            return 2

    out_folder_path = str(shared.get("output_folder_path", "VDA_output")).strip("/")
    date_folder = str(os.environ.get("LAOV_RUNTIME_DATE_FOLDER", "")).strip().strip("/")
    sam3_model_dir = lcr._resolve_sam3_model_dir(mount, shared)
    birefnet_model_dir = lcr._resolve_birefnet_model_dir(mount, shared)

    any_failed = False
    for seq in sequences:
        rel_plate = str(seq.get("plate_folder_drive_relative", "")).strip().strip("/").replace("\\", "/")
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        if not rel_plate:
            _log.error("Sequence missing plate_folder_drive_relative")
            return 1

        plate_dir = lcr._resolve_plate_dir(mount, rel_plate)
        if plate_dir is None:
            _log.error("Plate folder not found.\n%s", lcr._format_plate_hint(mount, rel_plate))
            return 1

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
        f0, f1 = max(s0, j0), min(s1, j1)
        if f0 > f1:
            f0, f1 = s0, s1

        matte_cfg = seq.get("matte_config") or {}
        _log.info(
            "Shot %s: plate=%s pattern=%s frames=%s-%s matte_mode=%s boxes=%s",
            shot_name,
            plate_dir,
            pattern,
            f0,
            f1,
            matte_cfg.get("matte_mode", shared.get("matte_mode")),
            len(matte_cfg.get("box_prompts") or []),
        )

        if probe_first_frame:
            _probe_plate_read(plate_dir, pattern, f0)

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
        pass_configs = [
            PassConfig(
                name=name,
                params=lcr._matte_pass_params(
                    name,
                    shared,
                    seq,
                    sam3_model_dir=sam3_model_dir,
                    birefnet_model_dir=birefnet_model_dir,
                ),
            )
            for name in pass_names
        ]
        job = Job(shot=shot, passes=pass_configs)
        try:
            laov_run(job)
        except Exception:
            _log.error("Run failed for %s:\n%s", shot_name, traceback.format_exc())
            any_failed = True
            continue

        if shot.status != "done":
            _log.error("Shot %s status=%s (expected done)", shot_name, shot.status)
            any_failed = True
        else:
            _log.info("Done shot=%s", shot_name)

    if any_failed:
        return 1
    _log.info("AI Matte — all sequences complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
