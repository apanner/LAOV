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
import importlib.util
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

# Allow ``import laov_colab_run`` when executed as scripts/ai_matte_colab_run.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import laov_colab_run as lcr  # noqa: E402
from colab_run_status import ColabRunStatus, configure_flushed_logging, reporter_from_job_json  # noqa: E402

_log = logging.getLogger("ai_matte_colab_run")

AI_MATTE_DEFAULTS = {
    "passes_csv": "flow,matte",
    "matte_detector": "sam3_matte",
    "refiner": "birefnet_refiner",
    "display_transform": True,
    "output_layout": "split_folders",
    "matte_mode": "people_fg",
    "matte_concepts": ["person"],
    "fill_between_keyframes": False,
    "flow_backend": "raft_large",
    "flow_inference_resolution": 520,
    "matte_temporal_ema": 0.15,
    "propagation_mode": "nearest_anchor",
    "export_sam3_exr": True,
    "export_birefnet_exr": True,
    "export_vitmatte_exr": True,
    "export_final_exr": False,
    "export_flow_exr": False,
    "run_matte_flow_temporal": False,
    "matte_pipeline_phase": "stages",
    "qc_mp4": True,
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
    p.add_argument(
        "--phase",
        choices=("stages", "sam3", "refine", "temporal", "all"),
        default=None,
        help="Pipeline phase (overrides batch JSON matte_pipeline_phase).",
    )
    return p.parse_args()


def _pipeline_phase(shared: dict, cli_phase: str | None) -> str:
    phase = str(cli_phase or shared.get("matte_pipeline_phase", "stages")).strip().lower()
    if phase not in ("stages", "sam3", "refine", "temporal", "all"):
        return "stages"
    return phase


def _refiners_from_exports(shared: dict) -> list[str]:
    """Refiner passes to run after SAM3 (checkboxes + active refiner)."""
    refiner = str(shared.get("refiner", "birefnet_refiner"))
    names: list[str] = []
    if bool(shared.get("export_birefnet_exr", True)) or refiner == "birefnet_refiner":
        names.append("birefnet_refiner")
    if bool(shared.get("export_vitmatte_exr", False)) or refiner == "vitmatte_refiner":
        names.append("vitmatte_refiner")
    if refiner == "rvm_refiner":
        names.append("rvm_refiner")
    # Dedupe preserve order
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _build_ai_matte_pass_names(shared: dict, *, phase: str) -> list[str]:
    """Schedule passes for one Colab phase."""
    refiner = str(shared.get("refiner", "birefnet_refiner"))

    if phase == "stages":
        names = ["sam3_matte"]
        names.extend(_refiners_from_exports(shared))
        return names

    if phase == "sam3":
        return ["sam3_matte"]

    if phase == "refine":
        names: list[str] = []
        if refiner == "birefnet_refiner":
            names.append("birefnet_refiner")
        elif refiner == "vitmatte_refiner":
            names.append("vitmatte_refiner")
        elif refiner == "rvm_refiner":
            names.append("rvm_refiner")
        else:
            names.append(refiner)
        return names

    if phase == "temporal":
        return ["flow"]

    # all — legacy single job (not recommended on Colab)
    names = ["sam3_matte"]
    need_flow = bool(shared.get("export_flow_exr", False))
    if bool(shared.get("run_matte_flow_temporal", False)) and bool(
        shared.get("export_final_exr", False)
    ):
        need_flow = True
    if refiner == "vitmatte_refiner":
        need_flow = bool(shared.get("export_flow_exr", False))
    if need_flow:
        names.insert(0, "flow")
    if bool(shared.get("export_birefnet_exr", False)) or refiner == "birefnet_refiner":
        if "birefnet_refiner" not in names:
            names.append("birefnet_refiner")
    if bool(shared.get("export_vitmatte_exr", False)) or refiner == "vitmatte_refiner":
        if "vitmatte_refiner" not in names:
            names.append("vitmatte_refiner")
    if refiner == "rvm_refiner" and "rvm_refiner" not in names:
        names.append("rvm_refiner")
    return names


def _stage_export_map(shared: dict, *, phase: str) -> dict[str, str]:
    exports: dict[str, str] = {}
    if phase == "stages":
        if shared.get("export_sam3_exr", True):
            exports["sam3_matte"] = "matte_sam3"
        if shared.get("export_birefnet_exr", True):
            exports["birefnet_refiner"] = "matte_birefnet"
        if shared.get("export_vitmatte_exr", False):
            exports["vitmatte_refiner"] = "matte_vitmatte"
    elif phase == "sam3" and shared.get("export_sam3_exr", True):
        exports["sam3_matte"] = "matte_sam3"
    elif phase == "refine":
        refiner = str(shared.get("refiner", "birefnet_refiner"))
        if refiner == "birefnet_refiner" and shared.get("export_birefnet_exr", True):
            exports["birefnet_refiner"] = "matte_birefnet"
        elif refiner == "vitmatte_refiner" and shared.get("export_vitmatte_exr", True):
            exports["vitmatte_refiner"] = "matte_vitmatte"
        elif refiner == "rvm_refiner":
            exports["rvm_refiner"] = "matte"
    elif phase == "all":
        if shared.get("export_sam3_exr", True):
            exports["sam3_matte"] = "matte_sam3"
        if shared.get("export_birefnet_exr", True):
            exports["birefnet_refiner"] = "matte_birefnet"
        if shared.get("export_vitmatte_exr", True):
            exports["vitmatte_refiner"] = "matte_vitmatte"
    return exports


def _should_run_matte_temporal(shared: dict) -> bool:
    refiner = str(shared.get("refiner", "birefnet_refiner"))
    if refiner == "vitmatte_refiner":
        return False
    if not bool(shared.get("run_matte_flow_temporal", False)):
        return False
    return bool(shared.get("export_final_exr", False))


def _ai_matte_post_configs(shared: dict) -> list:
    from live_action_aov.core.job import PostConfig
    from live_action_aov.io.channels import MATTE_CHANNELS

    stride = int(shared.get("keyframe_stride", 4))
    fb = shared.get("flow_fb_threshold_px", shared.get("fb_threshold_px", 1.0))
    refiner = str(shared.get("refiner", "birefnet_refiner"))
    applied = list(MATTE_CHANNELS)
    if refiner == "vitmatte_refiner":
        from live_action_aov.passes.matte.vitmatte_refiner import VITMATTE_CHANNELS

        applied = list(VITMATTE_CHANNELS)
    return [
        PostConfig(
            name="matte_flow_temporal",
            params={
                "applied_to": applied,
                "keyframe_stride": stride,
                "fb_threshold_px": fb,
                "propagation_mode": shared.get("propagation_mode", "nearest_anchor"),
                "blend_forward_backward": shared.get("matte_flow_blend", 0.5),
                "final_ema_alpha": shared.get("matte_temporal_ema", 0.15),
            },
        ),
    ]


def _ensure_birefnet_colab_deps() -> None:
    """BiRefNet HF snapshot modeling imports kornia (not always on Colab by default)."""
    if importlib.util.find_spec("kornia") is not None:
        return
    _log.info("Installing kornia for BiRefNet (pip install kornia>=0.7)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "kornia>=0.7"],
        check=True,
    )


def _merge_shared_defaults(shared: dict) -> dict:
    out = dict(shared)
    for key, val in AI_MATTE_DEFAULTS.items():
        if out.get(key) in (None, ""):
            out[key] = val
    if not str(out.get("passes_csv", "")).strip():
        out["passes_csv"] = "matte"
    return out


def _verify_registry(pass_names: list[str]) -> int:
    try:
        from live_action_aov.core.registry import get_registry

        reg = get_registry()
        reg.load_all()
        for name in pass_names:
            reg.get_pass(name)
        if "matte_flow_temporal" not in reg._post:
            raise KeyError("matte_flow_temporal post-processor")
        _log.info("Registry OK: %s + matte_flow_temporal", ", ".join(pass_names))
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
    configure_flushed_logging()
    args = _parse_args()
    job_path = Path(args.job_json).resolve()
    if not job_path.is_file():
        _log.error("Job JSON not found: %s", job_path)
        return 1

    with open(job_path, encoding="utf-8") as f:
        cfg = json.load(f)

    shared = _merge_shared_defaults(cfg.get("shared_settings") or {})
    phase = _pipeline_phase(shared, args.phase)
    shared["matte_pipeline_phase"] = phase
    cfg["shared_settings"] = shared
    model_type = str(shared.get("model_type", ""))
    if model_type and model_type != "AI_MATTE_STANDALONE":
        _log.warning("Expected model_type AI_MATTE_STANDALONE, got %r", model_type)

    matte_detector = str(shared.get("matte_detector", "sam3_matte"))
    refiner = str(shared.get("refiner", "birefnet_refiner"))
    _log.info(
        "AI Matte batch — phase=%s detector=%s refiner=%s",
        phase,
        matte_detector,
        refiner,
    )

    pass_names_preview = _build_ai_matte_pass_names(shared, phase=phase)

    if "birefnet_refiner" in pass_names_preview:
        try:
            _ensure_birefnet_colab_deps()
        except subprocess.CalledProcessError as exc:
            _log.error("Failed to install kornia for BiRefNet: %s", exc)
            return 1

    try:
        from live_action_aov.io.oiio_io import require_oiio

        require_oiio()
    except Exception as exc:
        _log.error("OpenImageIO required for plate read: %s", exc)
        return 1

    if _verify_registry(pass_names_preview) != 0:
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
    elif "birefnet_refiner" in pass_names_preview and not birefnet_model_dir:
        _log.error(
            "BiRefNet snapshot not found on Drive. Upload MyDrive/VDA_models/ZhengPeng7/BiRefNet/"
        )
        return 1

    if "vitmatte_refiner" in pass_names_preview:
        vit_dir = lcr._resolve_vitmatte_model_dir(mount, shared)
        if vit_dir:
            os.environ["AI_MATTE_VITMATTE_MODEL_PATH"] = vit_dir
            _log.info("ViTMatte model: %s", vit_dir)
        elif phase in ("refine", "all"):
            _log.error(
                "ViTMatte snapshot not found. Upload MyDrive/VDA_models/hustvl/vitmatte-base-composition-1k/"
            )
            return 1

    if phase == "sam3" and not sam3_model_dir:
        _log.warning("SAM3 phase: no local snapshot — Hub download may require HF login")

    sequences = cfg.get("sequences") or []
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    status = reporter_from_job_json(mount, cfg)
    if status:
        status.begin_run(f"AI Matte phase={phase}")

    try:
        return _run_sequences(
            cfg,
            mount,
            shared,
            sequences,
            phase=phase,
            matte_detector=matte_detector,
            refiner=refiner,
            probe_first_frame=args.probe_first_frame,
            status=status,
        )
    except Exception:
        _log.error("AI Matte runner crashed:\n%s", traceback.format_exc())
        if status:
            status.finish_run(ok=False)
        return 1


def _run_sequences(
    cfg: dict,
    mount: Path,
    shared: dict,
    sequences: list,
    *,
    phase: str,
    matte_detector: str,
    refiner: str,
    probe_first_frame: bool,
    status: ColabRunStatus | None = None,
) -> int:
    from live_action_aov.core.job import Job, PassConfig, PostConfig, Shot
    from live_action_aov.core.registry import get_registry
    from live_action_aov import run as laov_run

    allow_nc = bool(shared.get("allow_noncommercial", False))
    display_transform = bool(shared.get("display_transform", True))
    colorspace = shared.get("colorspace")
    colorspace_s = str(colorspace) if colorspace else "auto"
    proxy_raw = shared.get("proxy_long_edge")
    proxy_long_edge = int(proxy_raw) if proxy_raw is not None else None
    output_layout = str(shared.get("output_layout", "split_folders"))

    pass_names = _build_ai_matte_pass_names(shared, phase=phase)
    stage_exports = _stage_export_map(shared, phase=phase)
    _log.info("Phase %s — passes: %s", phase, ", ".join(pass_names))
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
    vitmatte_model_dir = lcr._resolve_vitmatte_model_dir(mount, shared)

    any_failed = False
    shot_total = len(sequences)
    for shot_idx, seq in enumerate(sequences):
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        if status:
            status.shot_begin(shot_name, shot_idx + 1, shot_total)

        try:
            if status:
                status.stage(f"{shot_name}: plate preflight", shot_name=shot_name)
            plate_dir, pattern, (f0, f1), resolution, pixel_aspect = lcr.shot_plate_from_desk_json(
                mount, seq
            )
        except FileNotFoundError as exc:
            _log.error("%s", exc)
            if status:
                status.shot_end(shot_name, ok=False, message="plate preflight failed")
                status.finish_run(ok=False)
            return 1

        if rel_side:
            if date_folder:
                output_dir = (mount / out_folder_path / date_folder / rel_side).resolve()
            else:
                output_dir = (mount / out_folder_path / rel_side).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

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

        preload_sam3_dir = None
        if phase == "refine" and output_dir is not None:
            from live_action_aov.io.sam3_artifact_io import sam3_artifact_path

            sam3_stage = (output_dir / "matte_sam3").resolve()
            if not sam3_artifact_path(sam3_stage).is_file():
                _log.error(
                    "%s: missing %s — run Colab with matte_pipeline_phase=sam3 first",
                    shot_name,
                    sam3_artifact_path(sam3_stage),
                )
                any_failed = True
                if status:
                    status.shot_end(shot_name, ok=False, message="SAM3 artifacts missing")
                continue
            preload_sam3_dir = sam3_stage

        write_final = bool(shared.get("export_final_exr", False)) or phase == "temporal"
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
            pass_export_subdirs=stage_exports,
            write_final_sidecars=write_final,
            preload_sam3_artifacts_dir=preload_sam3_dir,
        )
        if vitmatte_model_dir:
            os.environ["AI_MATTE_VITMATTE_MODEL_PATH"] = vitmatte_model_dir
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
        post_configs: list[PostConfig] = []
        if phase == "temporal" or (
            phase == "all" and "flow" in pass_names and _should_run_matte_temporal(shared)
        ):
            if phase == "temporal":
                post_configs = _ai_matte_post_configs(shared)
            elif _should_run_matte_temporal(shared):
                post_configs = _ai_matte_post_configs(shared)
        job = Job(shot=shot, passes=pass_configs, post=post_configs)
        progress_cb = (
            status.make_laov_callback(shot_name, shot_idx, shot_total) if status else None
        )
        try:
            if status:
                status.stage(
                    f"{shot_name}: engine — {', '.join(pass_names)}",
                    shot_name=shot_name,
                )
            laov_run(job, progress_callback=progress_cb)
        except Exception:
            _log.error("Run failed for %s:\n%s", shot_name, traceback.format_exc())
            if status:
                status.shot_end(shot_name, ok=False, message="engine error")
            any_failed = True
            continue

        if shot.status != "done":
            _log.error("Shot %s status=%s (expected done)", shot_name, shot.status)
            if status:
                status.shot_end(shot_name, ok=False, message=str(shot.status))
            any_failed = True
        else:
            _log.info("Done shot=%s", shot_name)
            if status:
                status.shot_end(shot_name, ok=True)
            if output_dir and bool(shared.get("qc_mp4", True)):
                try:
                    if status:
                        status.stage(f"{shot_name}: QC MP4 export", shot_name=shot_name)
                    from ai_matte_qc_mp4 import export_all_stage_qc_mp4s

                    fps = float(shared.get("fps") or shared.get("frame_rate") or 24.0)
                    export_all_stage_qc_mp4s(
                        output_dir,
                        plate_dir=plate_dir,
                        sequence_pattern=pattern,
                        frame_range=(f0, f1),
                        fps=fps,
                        refiner=refiner,
                    )
                except Exception as exc:
                    _log.warning("QC MP4 export failed for %s: %s", shot_name, exc)

    if any_failed:
        if status:
            status.finish_run(ok=False)
        return 1
    _log.info("AI Matte — all sequences complete.")
    if status:
        status.finish_run(ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
