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
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

# Allow ``import laov_colab_run`` when executed as scripts/ai_matte_colab_run.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import laov_colab_run as lcr  # noqa: E402
from colab_run_status import ColabRunStatus, configure_flushed_logging, reporter_from_job_json  # noqa: E402

_log = logging.getLogger("ai_matte_colab_run")

AI_MATTE_DEFAULTS = {
    # Stage jobs: matte only. RAFT/flow is optional phase ``temporal`` (separate Colab run).
    "passes_csv": "matte",
    "matte_detector": "sam3_matte",
    "refiner": "birefnet_refiner",
    "display_transform": True,
    "output_layout": "split_folders",
    "matte_mode": "people_fg",
    "matte_concepts": ["person"],
    # 1 = BiRefNet/ViTMatte on every frame (no sparse keyframes + blend).
    "keyframe_stride": 1,
    "fill_between_keyframes": True,
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
    "proxy_long_edge": None,
    # ViTMatte: 0 = auto long_edge from GPU (40GB A100 → 2048, 16GB → 1024, …).
    "max_inference_long_edge": 0,
    "vitmatte_inference_mode": "crop",
    "vitmatte_crop_pad": 32,
    "use_plate_jpeg_cache": True,
    "plate_cache_keep": True,
    "plate_jpeg_quality": 92,
    "plate_cache_workers": 8,
    "sam3_load_workers": 8,
    # SAM3: auto downscale tracking when 4K×long clips would exceed ~14 GiB RAM.
    "sam3_max_plate_stack_gb": 14.0,
    # VDA-style: one Python process per shot (full RAM/GPU reset between shots).
    "batch_one_process_per_shot": True,
    "batch_gap_seconds": 5,
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
    p.add_argument(
        "--shot-index",
        type=int,
        default=None,
        metavar="N",
        help="Run only sequences[N] from the batch JSON (used internally for one-shot-per-process batch).",
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
    if bool(shared.get("export_vitmatte_exr", True)) or refiner == "vitmatte_refiner":
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
        refiners = _refiners_from_exports(shared)
        # Stable order: BiRefNet before ViTMatte (both only need SAM3 artifacts).
        for r in ("birefnet_refiner", "vitmatte_refiner", "rvm_refiner"):
            if r in refiners:
                names.append(r)
        return names

    if phase == "sam3":
        return ["sam3_matte"]

    if phase == "refine":
        # Run every checked refiner (Desk export checkboxes), not only the active refiner combo.
        names = _refiners_from_exports(shared)
        if not names:
            _log.warning("refine phase: no export_*_exr enabled — defaulting to %s", refiner)
            names = [refiner]
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
    if bool(shared.get("export_birefnet_exr", True)) or refiner == "birefnet_refiner":
        if "birefnet_refiner" not in names:
            names.append("birefnet_refiner")
    if bool(shared.get("export_vitmatte_exr", True)) or refiner == "vitmatte_refiner":
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
        if shared.get("export_vitmatte_exr", True):
            exports["vitmatte_refiner"] = "matte_vitmatte"
    elif phase == "sam3" and shared.get("export_sam3_exr", True):
        exports["sam3_matte"] = "matte_sam3"
    elif phase == "refine":
        if shared.get("export_birefnet_exr", True):
            exports["birefnet_refiner"] = "matte_birefnet"
        if shared.get("export_vitmatte_exr", True):
            exports["vitmatte_refiner"] = "matte_vitmatte"
        if shared.get("export_rvm_exr", False):
            exports["rvm_refiner"] = "matte"
    elif phase == "all":
        if shared.get("export_sam3_exr", True):
            exports["sam3_matte"] = "matte_sam3"
        if shared.get("export_birefnet_exr", True):
            exports["birefnet_refiner"] = "matte_birefnet"
        if shared.get("export_vitmatte_exr", True):
            exports["vitmatte_refiner"] = "matte_vitmatte"
    return exports


def _stage_exr_count(output_dir: Path, subdir: str) -> int:
    d = output_dir / subdir
    if not d.is_dir():
        return 0
    return len(list(d.glob("*.exr")))


def _plate_cache_count(output_dir: Path) -> int:
    from live_action_aov.passes.matte.vitmatte_plate_cache import plate_cache_frame_count

    return plate_cache_frame_count(output_dir / "_plate_jpeg_cache")


def _make_plate_reader(
    plate_dir: Path,
    pattern: str,
    frame_range: tuple[int, int],
    *,
    display_transform: bool,
    colorspace_s: str,
    proxy_long_edge: int | None,
) -> Any:
    from live_action_aov.io.readers.display_transform_reader import DisplayTransformedReader
    from live_action_aov.io.readers.oiio_exr import OIIOExrReader
    from live_action_aov.io.readers.proxy import wrap_if_proxy

    raw = OIIOExrReader(plate_dir, pattern)
    raw = wrap_if_proxy(raw, proxy_long_edge)
    if not display_transform:
        return raw
    reader = DisplayTransformedReader(
        raw,
        colorspace_override=colorspace_s if colorspace_s != "auto" else None,
    )
    reader.analyze(frame_range)
    return reader


def _prewarm_plate_jpeg_cache(
    *,
    plate_dir: Path,
    pattern: str,
    frame_range: tuple[int, int],
    cache_dir: Path,
    shared: dict,
    display_transform: bool,
    colorspace_s: str,
    proxy_long_edge: int | None,
) -> None:
    """One parallel EXR→JPEG pass on Drive — SAM3 + refiners read JPEGs (GPU work starts sooner)."""
    from live_action_aov.passes.matte.vitmatte_plate_cache import (
        build_plate_jpeg_cache,
        plate_cache_complete,
    )

    if plate_cache_complete(cache_dir, frame_range):
        from live_action_aov.passes.matte.vitmatte_plate_cache import plate_cache_frame_count

        _log.info(
            "Plate JPEG cache ready (%d frames): %s",
            plate_cache_frame_count(cache_dir),
            cache_dir,
        )
        return
    reader = _make_plate_reader(
        plate_dir,
        pattern,
        frame_range,
        display_transform=display_transform,
        colorspace_s=colorspace_s,
        proxy_long_edge=proxy_long_edge,
    )
    workers = max(1, int(shared.get("plate_cache_workers", 8)))
    _log.info(
        "Pre-building plate JPEG cache (%d parallel workers) — avoids slow SAM3 EXR loop",
        workers,
    )
    build_plate_jpeg_cache(
        reader.read_frame,
        frame_range,
        cache_dir,
        jpeg_quality=int(shared.get("plate_jpeg_quality", 92)),
        log_every=max(10, (frame_range[1] - frame_range[0] + 1) // 15),
        log_label="Plate",
        workers=workers,
    )


def _incomplete_matte_stages(
    output_dir: Path,
    frame_range: tuple[int, int],
    shared: dict,
) -> list[str]:
    """Stage folder names still missing deliverable EXRs."""
    f0, f1 = frame_range
    n_frames = f1 - f0 + 1
    missing: list[str] = []
    if bool(shared.get("export_sam3_exr", True)) and not _stage_export_complete(
        output_dir, "matte_sam3", n_frames
    ):
        missing.append("matte_sam3")
    if bool(shared.get("export_birefnet_exr", True)) and not _stage_export_complete(
        output_dir, "matte_birefnet", n_frames
    ):
        missing.append("matte_birefnet")
    if bool(shared.get("export_vitmatte_exr", True)) and not _stage_export_complete(
        output_dir, "matte_vitmatte", n_frames
    ):
        missing.append("matte_vitmatte")
    return missing


def _stage_export_complete(output_dir: Path, subdir: str, n_frames: int) -> bool:
    return _stage_exr_count(output_dir, subdir) >= n_frames


def _log_stage_scan(output_dir: Path, frame_range: tuple[int, int]) -> None:
    """Print what is already on Drive (no copying — same destination tree)."""
    from live_action_aov.io.sam3_artifact_io import sam3_artifact_path

    f0, f1 = frame_range
    n_frames = f1 - f0 + 1
    sam3_stage = output_dir / "matte_sam3"
    npz = sam3_artifact_path(sam3_stage)
    _log.info(
        "Resume scan — destination: %s  frames=%s-%s (%d)",
        output_dir,
        f0,
        f1,
        n_frames,
    )
    _log.info(
        "  matte_sam3/     %d EXR(s)  NPZ=%s",
        _stage_exr_count(output_dir, "matte_sam3"),
        "yes" if npz.is_file() else "no",
    )
    _log.info(
        "  matte_birefnet/ %d EXR(s)  (need %d to skip BiRefNet)",
        _stage_exr_count(output_dir, "matte_birefnet"),
        n_frames,
    )
    _log.info(
        "  matte_vitmatte/ %d EXR(s)  (need %d to skip ViTMatte)",
        _stage_exr_count(output_dir, "matte_vitmatte"),
        n_frames,
    )
    _log.info(
        "  _plate_jpeg_cache/ %d JPEG(s) (work cache for refiners; rebuilt if incomplete)",
        _plate_cache_count(output_dir),
    )


def _plan_stage_passes(
    output_dir: Path | None,
    pass_names: list[str],
    *,
    phase: str,
    frame_range: tuple[int, int],
) -> tuple[list[str], Path | None]:
    """Skip passes whose outputs already exist under ``output_dir`` on Drive.

    Nothing is copied or moved between folders. Each pass writes in place:

    - ``matte_sam3/`` — combined matte EXRs + ``_sam3_artifacts.npz`` (masks for refiners)
    - ``matte_birefnet/``, ``matte_vitmatte/`` — combined matte EXRs per stage
    - ``_plate_jpeg_cache/`` — optional full-res JPEG plate cache (refiners read plates)

    SAM3 skip uses the **NPZ** (refiners need hard masks). EXR count is logged;
    refiners skip only when their folder has ``>= n_frames`` ``*.exr``.
    """
    if output_dir is None:
        return pass_names, None
    from live_action_aov.io.sam3_artifact_io import sam3_artifact_path

    f0, f1 = frame_range
    n_frames = f1 - f0 + 1
    sam3_stage = (output_dir / "matte_sam3").resolve()
    npz_path = sam3_artifact_path(sam3_stage)
    has_npz = npz_path.is_file()
    planned = list(pass_names)
    preload: Path | None = None

    _log_stage_scan(output_dir, frame_range)

    if phase == "refine":
        if has_npz:
            preload = sam3_stage
            if "sam3_matte" in planned:
                planned.remove("sam3_matte")
        return planned, preload

    if phase != "stages":
        return planned, None

    # --- SAM3: NPZ is required to run BiRefNet/ViTMatte without re-tracking ---
    if "sam3_matte" in planned and has_npz:
        exr_n = _stage_exr_count(output_dir, "matte_sam3")
        if exr_n >= n_frames:
            _log.info(
                "Skip sam3_matte — NPZ + %d/%d EXRs already in matte_sam3/",
                exr_n,
                n_frames,
            )
        else:
            _log.warning(
                "Skip sam3_matte — NPZ present but only %d/%d EXRs in matte_sam3/ "
                "(refiners use NPZ; re-export SAM3 EXRs only if you need them)",
                exr_n,
                n_frames,
            )
        planned.remove("sam3_matte")
        preload = sam3_stage

    if "birefnet_refiner" in planned and _stage_export_complete(
        output_dir, "matte_birefnet", n_frames
    ):
        _log.info(
            "Skip birefnet_refiner — matte_birefnet/ has %d EXRs",
            _stage_exr_count(output_dir, "matte_birefnet"),
        )
        planned.remove("birefnet_refiner")

    if "vitmatte_refiner" in planned and _stage_export_complete(
        output_dir, "matte_vitmatte", n_frames
    ):
        _log.info(
            "Skip vitmatte_refiner — matte_vitmatte/ has %d EXRs",
            _stage_exr_count(output_dir, "matte_vitmatte"),
        )
        planned.remove("vitmatte_refiner")

    if preload and planned:
        _log.info("Next passes (in place on Drive): %s", ", ".join(planned))
    return planned, preload


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


def _ensure_kornia() -> None:
    """BiRefNet always needs kornia on Colab."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "kornia>=0.7"],
        check=True,
    )
    import kornia  # noqa: F401

    _log.info("kornia OK")


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
    print("[ai_matte_colab_run] starting…", flush=True)
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
            _ensure_kornia()
        except Exception as exc:
            _log.error("kornia install failed (BiRefNet needs it): %s", exc)
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

    vitmatte_model_dir = lcr._resolve_vitmatte_model_dir(mount, shared)
    if vitmatte_model_dir:
        os.environ["AI_MATTE_VITMATTE_MODEL_PATH"] = vitmatte_model_dir
        _log.info("ViTMatte model: %s", vitmatte_model_dir)

    if "sam3_matte" in pass_names_preview and not sam3_model_dir:
        _log.error("SAM3 not found: MyDrive/VDA_models/facebook/sam3/")
        return 1

    sequences = cfg.get("sequences") or []
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    if args.shot_index is not None:
        if args.shot_index < 0 or args.shot_index >= len(sequences):
            _log.error("shot-index %s out of range (0..%s)", args.shot_index, len(sequences) - 1)
            return 1
        sequences = [sequences[args.shot_index]]
        _log.info(
            "Single-shot subprocess mode — index %s: %s",
            args.shot_index,
            sequences[0].get("shot_name", "shot"),
        )

    status = reporter_from_job_json(mount, cfg)
    if status:
        status.begin_run(f"AI Matte phase={phase}")
    from colab_run_status import start_colab_keepalive, stop_colab_keepalive

    start_colab_keepalive(status, interval_sec=90)

    use_subprocess_batch = (
        args.shot_index is None
        and len(sequences) > 1
        and bool(shared.get("batch_one_process_per_shot", True))
    )

    try:
        if use_subprocess_batch:
            _log.info(
                "Batch queue: %d shot(s), one fresh Python process each (like VDA engine)",
                len(cfg.get("sequences") or []),
            )
            return _run_batch_one_process_per_shot(
                job_path=job_path,
                cfg=cfg,
                phase=phase,
                probe_first_frame=args.probe_first_frame,
                status=status,
            )
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
    finally:
        stop_colab_keepalive()


def _print_shot_banner(shot_index: int, shot_total: int, shot_name: str) -> None:
    """VDA-style per-shot separator in Colab logs."""
    print("\n" + "=" * 60, flush=True)
    print(f"SHOT {shot_index}/{shot_total}: {shot_name}", flush=True)
    print("=" * 60, flush=True)


def _between_shots_cleanup() -> None:
    """Clear GPU between shots so one failure/OOM does not poison the next."""
    from live_action_aov.executors.gpu_release import clear_vram_cache, cuda_vram_snapshot

    clear_vram_cache()
    snap = cuda_vram_snapshot()
    if snap:
        _log.info(
            "Between shots VRAM: free %.1f GiB, allocated %.1f GiB",
            snap["free_gb"],
            snap["allocated_gb"],
        )


def _run_batch_one_process_per_shot(
    *,
    job_path: Path,
    cfg: dict,
    phase: str,
    probe_first_frame: bool,
    status: ColabRunStatus | None,
) -> int:
    """Feed shots one after another — each shot = new ``ai_matte_colab_run.py`` process."""
    import time

    shared = cfg.get("shared_settings") or {}
    all_sequences: list = cfg.get("sequences") or []
    shot_total = len(all_sequences)
    gap_s = max(0, int(shared.get("batch_gap_seconds", 5)))
    runner = Path(__file__).resolve()
    succeeded: list[str] = []
    failed: list[str] = []

    print("\n" + "=" * 60, flush=True)
    print(f"BATCH QUEUE — {shot_total} shot(s), sequential (one process per shot)", flush=True)
    print("=" * 60, flush=True)

    for shot_idx, seq in enumerate(all_sequences):
        shot_name = str(seq.get("shot_name", "shot"))
        shot_num = shot_idx + 1

        if shot_idx > 0 and gap_s > 0:
            _log.info("Cooling %ds before next shot (VDA-style gap)…", gap_s)
            time.sleep(gap_s)

        _print_shot_banner(shot_num, shot_total, shot_name)
        if status:
            status.shot_begin(shot_name, shot_num, shot_total)

        cmd = [
            sys.executable,
            str(runner),
            "--job-json",
            str(job_path),
            "--phase",
            phase,
            "--shot-index",
            str(shot_idx),
        ]
        if probe_first_frame and shot_idx == 0:
            cmd.append("--probe-first-frame")

        _log.info("Launching shot subprocess: %s", " ".join(cmd))
        env = os.environ.copy()
        env["LAOV_BATCH_CHILD"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if cmd[0] == sys.executable:
            cmd = [sys.executable, "-u", *cmd[1:]]
        result = subprocess.run(cmd, env=env)
        code = int(result.returncode)

        if code == 0:
            succeeded.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=True)
            _log.info("Shot subprocess OK: %s", shot_name)
        else:
            failed.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=False, message=f"exit {code}")
            _log.error("Shot subprocess failed (exit %s): %s", code, shot_name)

    print("\n" + "=" * 60, flush=True)
    print("BATCH SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for name in succeeded:
        print(f"  OK   {name}", flush=True)
    for name in failed:
        print(f"  FAIL {name}", flush=True)
    print(
        f"\n{len(succeeded)}/{shot_total} succeeded, {len(failed)} failed.",
        flush=True,
    )

    if failed:
        if status:
            status.finish_run(
                ok=len(succeeded) > 0,
                message=f"{len(succeeded)}/{shot_total} shots OK",
            )
        return 1 if not succeeded else 0

    if status:
        status.finish_run(ok=True)
    return 0


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
    if proxy_raw in (None, "", 0, "0"):
        proxy_long_edge = None
    else:
        proxy_long_edge = int(proxy_raw)
    output_layout = str(shared.get("output_layout", "split_folders"))

    pass_names = _build_ai_matte_pass_names(shared, phase=phase)
    vit_dir = lcr._resolve_vitmatte_model_dir(mount, shared)
    if "vitmatte_refiner" in pass_names:
        if vit_dir:
            os.environ["AI_MATTE_VITMATTE_MODEL_PATH"] = vit_dir
        elif bool(shared.get("export_vitmatte_exr", True)):
            _log.error(
                "ViTMatte requested (export_vitmatte_exr) but model not on Drive. "
                "Upload MyDrive/VDA_models/hustvl/vitmatte-base-composition-1k/ "
                "or uncheck ViTMatte in Desk and re-send job."
            )
            return 1
        else:
            pass_names = [p for p in pass_names if p != "vitmatte_refiner"]

    stage_exports = _stage_export_map(shared, phase=phase)
    if "vitmatte_refiner" not in pass_names:
        stage_exports.pop("vitmatte_refiner", None)
    if phase == "sam3":
        _log.warning(
            "matte_pipeline_phase=sam3 — ONLY SAM3 runs. Use 'stages' for SAM3+BiRefNet+ViTMatte."
        )
    _log.info("Phase %s — passes: %s", phase, ", ".join(pass_names))
    _log.info(
        "Exports: sam3=%s birefnet=%s vitmatte=%s qc_mp4=%s",
        shared.get("export_sam3_exr", True),
        shared.get("export_birefnet_exr", True),
        shared.get("export_vitmatte_exr", True),
        shared.get("qc_mp4", True),
    )
    _log.info(
        "Refiner timing: keyframe_stride=%s fill_between_keyframes=%s "
        "(neural infer every N frames; gaps filled by blend unless stride=1)",
        shared.get("keyframe_stride", 1),
        shared.get("fill_between_keyframes", True),
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
    vitmatte_model_dir = lcr._resolve_vitmatte_model_dir(mount, shared)

    shot_total = len(sequences)
    succeeded: list[str] = []
    failed: list[str] = []
    _log.info("AI Matte batch — %d shot(s), continue on per-shot errors (VDA-style)", shot_total)

    for shot_idx, seq in enumerate(sequences):
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))
        shot_num = shot_idx + 1

        if shot_idx > 0:
            _between_shots_cleanup()

        _print_shot_banner(shot_num, shot_total, shot_name)
        if status:
            status.shot_begin(shot_name, shot_num, shot_total)

        try:
            if status:
                status.stage(f"{shot_name}: plate preflight", shot_name=shot_name)
            plate_dir, pattern, (f0, f1), resolution, pixel_aspect = lcr.shot_plate_from_desk_json(
                mount, seq
            )
        except FileNotFoundError as exc:
            _log.error("[%s] preflight failed: %s", shot_name, exc)
            failed.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=False, message="plate preflight failed")
            continue

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
            "Shot %s: plate=%s pattern=%s frames=%s-%s res=%s proxy=%s matte_mode=%s",
            shot_name,
            plate_dir,
            pattern,
            f0,
            f1,
            resolution or "full",
            proxy_long_edge or "off (full res)",
            matte_cfg.get("matte_mode", shared.get("matte_mode")),
        )

        if probe_first_frame:
            _probe_plate_read(plate_dir, pattern, f0)

        shot_pass_names = list(pass_names)
        preload_sam3_dir = None
        if output_dir is not None and phase in ("stages", "refine"):
            from live_action_aov.io.sam3_artifact_io import sam3_artifact_path

            shot_pass_names, preload_sam3_dir = _plan_stage_passes(
                output_dir,
                shot_pass_names,
                phase=phase,
                frame_range=(f0, f1),
            )
            if phase == "refine" and preload_sam3_dir is None:
                sam3_stage = (output_dir / "matte_sam3").resolve()
                _log.error(
                    "%s: missing %s — run stages first or re-run Cell 3 (auto-resumes refiners)",
                    shot_name,
                    sam3_artifact_path(sam3_stage),
                )
                failed.append(shot_name)
                if status:
                    status.shot_end(shot_name, ok=False, message="SAM3 artifacts missing")
                continue

        if not shot_pass_names:
            _log.info("[%s] all stage EXRs already on Drive — skip engine", shot_name)
            succeeded.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=True, message="already complete")
            continue

        shot_stage_exports = {
            k: v for k, v in stage_exports.items() if k in shot_pass_names
        }

        if output_dir is not None and any(
            p in shot_pass_names for p in ("sam3_matte", "birefnet_refiner", "vitmatte_refiner")
        ):
            try:
                _prewarm_plate_jpeg_cache(
                    plate_dir=plate_dir,
                    pattern=pattern,
                    frame_range=(f0, f1),
                    cache_dir=(output_dir / "_plate_jpeg_cache").resolve(),
                    shared=shared,
                    display_transform=display_transform,
                    colorspace_s=colorspace_s,
                    proxy_long_edge=proxy_long_edge,
                )
            except Exception as exc:
                _log.error("[%s] plate JPEG cache build failed: %s", shot_name, exc)
                failed.append(shot_name)
                if status:
                    status.shot_end(shot_name, ok=False, message="plate cache failed")
                continue

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
            pass_export_subdirs=shot_stage_exports,
            write_final_sidecars=write_final,
            preload_sam3_artifacts_dir=preload_sam3_dir,
        )
        if vitmatte_model_dir:
            os.environ["AI_MATTE_VITMATTE_MODEL_PATH"] = vitmatte_model_dir
        pass_configs = []
        for name in shot_pass_names:
            params = lcr._matte_pass_params(
                name,
                shared,
                seq,
                sam3_model_dir=sam3_model_dir,
                birefnet_model_dir=birefnet_model_dir,
            )
            if output_dir is not None and name in (
                "sam3_matte",
                "birefnet_refiner",
                "vitmatte_refiner",
            ):
                cache_dir = (output_dir / "_plate_jpeg_cache").resolve()
                params.setdefault("use_plate_jpeg_cache", True)
                params.setdefault("plate_cache_dir", str(cache_dir))
                params.setdefault("plate_cache_keep", True)
                params.setdefault(
                    "plate_cache_workers", int(shared.get("plate_cache_workers", 8))
                )
                if name == "sam3_matte":
                    params.setdefault(
                        "sam3_load_workers", int(shared.get("sam3_load_workers", 8))
                    )
            pass_configs.append(PassConfig(name=name, params=params))
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
                    f"{shot_name}: engine — {', '.join(shot_pass_names)}",
                    shot_name=shot_name,
                )
            laov_run(job, progress_callback=progress_cb)
        except MemoryError:
            _log.error(
                "[%s] out of memory — SAM3 may already be on Drive under %s/matte_sam3/. "
                "Re-run Cell 3 (same JSON): auto-resumes BiRefNet + ViTMatte only.",
                shot_name,
                output_dir or "?",
            )
            failed.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=False, message="out of memory")
            _between_shots_cleanup()
            continue
        except Exception:
            _log.error(
                "Run failed for %s:\n%s\n"
                "If matte_sam3/ exists, re-run Cell 3 — refiners resume automatically.",
                shot_name,
                traceback.format_exc(),
            )
            failed.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=False, message="engine error")
            _between_shots_cleanup()
            continue

        if shot.status != "done":
            _log.error("Shot %s status=%s (expected done)", shot_name, shot.status)
            failed.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=False, message=str(shot.status))
        elif output_dir is not None:
            missing_stages = _incomplete_matte_stages(output_dir, (f0, f1), shared)
            if missing_stages:
                _log.error(
                    "[%s] incomplete matte stages on Drive: %s — re-run Cell 3 to resume",
                    shot_name,
                    ", ".join(missing_stages),
                )
                failed.append(shot_name)
                if status:
                    status.shot_end(shot_name, ok=False, message="incomplete stages")
            else:
                _log.info(
                    "Done shot=%s — matte_sam3 + matte_birefnet + matte_vitmatte complete",
                    shot_name,
                )
                succeeded.append(shot_name)
                if status:
                    status.shot_end(shot_name, ok=True)
        else:
            _log.info("Done shot=%s", shot_name)
            succeeded.append(shot_name)
            if status:
                status.shot_end(shot_name, ok=True)

        if shot_name in succeeded and output_dir and bool(shared.get("qc_mp4", True)):
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

    print("\n" + "=" * 60, flush=True)
    print("BATCH SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for name in succeeded:
        print(f"  OK   {name}", flush=True)
    for name in failed:
        print(f"  FAIL {name}", flush=True)
    print(
        f"\n{len(succeeded)}/{shot_total} succeeded, {len(failed)} failed.",
        flush=True,
    )

    if failed:
        _log.warning(
            "Batch partial — failed: %s. Successful outputs kept on Drive. "
            "Re-run phase=refine for shots that already have matte_sam3/.",
            ", ".join(failed),
        )
        if status:
            status.finish_run(
                ok=len(succeeded) > 0,
                message=f"{len(succeeded)}/{shot_total} shots OK",
            )
        return 1 if not succeeded else 0

    _log.info("AI Matte — all %d sequence(s) complete.", shot_total)
    if status:
        status.finish_run(ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise SystemExit(1) from None
