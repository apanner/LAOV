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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow ``import laov_colab_run`` when executed as scripts/ai_matte_colab_run.py
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import laov_colab_run as lcr  # noqa: E402

_log = logging.getLogger("ai_matte_colab_run")

_PASS_STAGE_LABELS: dict[str, str] = {
    "flow": "RAFT optical flow",
    "sam3_matte": "SAM3 mask tracking",
    "matte": "SAM3 mask tracking",
    "birefnet_refiner": "BiRefNet soft matte refine",
    "rvm_refiner": "RVM matte refine",
    "matte_flow_temporal": "Matte temporal (flow fill)",
    "position_from_depth": "Position from depth",
    "temporal_smoother": "Temporal smoother",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _configure_colab_stdout() -> None:
    """Colab notebooks buffer stdout; line-buffer so stage lines appear live."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


class ColabStageReporter:
    """Flush stage lines to Colab stdout + ``MyDrive/VDA_Jobs/status/{job_id}_status.json``."""

    def __init__(self, mount: Path, cfg: dict) -> None:
        self.mount = mount
        self.job_id = str(cfg.get("job_id") or cfg.get("batch_id") or "ai_matte")
        self.batch_id = str(cfg.get("batch_id") or self.job_id)
        self.status_path = mount / "VDA_Jobs" / "status" / f"{self.job_id}_status.json"
        sequences = cfg.get("sequences") or []
        self._data: dict[str, Any] = {
            "job_id": self.job_id,
            "batch_id": self.batch_id,
            "model_type": "AI_MATTE_STANDALONE",
            "status": "running",
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "total_sequences": len(sequences),
            "successful_sequences": 0,
            "failed_sequences": 0,
            "current_stage": "Starting",
            "progress_percent": 0,
            "progress_fraction": 0.0,
            "shot_index": 0,
            "current_shot": "",
            "frame_range": "",
            "total_frames": 0,
            "sequences": [
                {
                    "shot_name": str(s.get("shot_name", f"shot_{i + 1}")),
                    "status": "pending",
                }
                for i, s in enumerate(sequences)
            ],
        }
        self._shot_index = 0

    def _emit_console(self, message: str, *, fraction: float | None = None) -> None:
        if fraction is not None:
            line = f"[AI_MATTE] [{fraction * 100:5.1f}%] {message}"
        else:
            line = f"[AI_MATTE] {message}"
        print(line, flush=True)

    def _persist(self) -> None:
        self._data["updated_at"] = _utc_now_iso()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def stage(
        self,
        message: str,
        *,
        fraction: float | None = None,
        shot_name: str | None = None,
        frame_range: str | None = None,
        total_frames: int | None = None,
    ) -> None:
        self._data["current_stage"] = message
        if fraction is not None:
            self._data["progress_fraction"] = round(float(fraction), 4)
            self._data["progress_percent"] = int(round(float(fraction) * 100))
        if shot_name is not None:
            self._data["current_shot"] = shot_name
        if frame_range is not None:
            self._data["frame_range"] = frame_range
        if total_frames is not None:
            self._data["total_frames"] = total_frames
        self._persist()
        self._emit_console(message, fraction=fraction)

    def begin_shot(self, shot_index: int, shot_name: str, frame_range: str, total_frames: int) -> None:
        self._shot_index = shot_index
        self._data["shot_index"] = shot_index + 1
        self._data["sequences"][shot_index]["status"] = "running"
        self.stage(
            f"Shot {shot_index + 1}/{self._data['total_sequences']}: {shot_name} "
            f"(frames {frame_range}, {total_frames} frames)",
            shot_name=shot_name,
            frame_range=frame_range,
            total_frames=total_frames,
        )

    def shot_done(self, shot_index: int, shot_name: str, *, output_dir: str | None = None) -> None:
        entry = self._data["sequences"][shot_index]
        entry["status"] = "success"
        if output_dir:
            entry["matte_output_dir"] = output_dir
        self._data["successful_sequences"] = sum(
            1 for s in self._data["sequences"] if s.get("status") == "success"
        )

    def shot_failed(self, shot_index: int, shot_name: str, error: str) -> None:
        entry = self._data["sequences"][shot_index]
        entry["status"] = "failed"
        entry["error"] = error[:2000]
        self._data["failed_sequences"] = sum(
            1 for s in self._data["sequences"] if s.get("status") == "failed"
        )

    def progress(self, fraction: float, label: str) -> None:
        pretty = label
        for key, human in _PASS_STAGE_LABELS.items():
            if key in label:
                pretty = label.replace(f": {key}", f": {human}").replace(key, human)
                break
        self.stage(pretty, fraction=fraction)

    def finish_batch(self, *, success: bool) -> None:
        failed = int(self._data.get("failed_sequences") or 0)
        total = int(self._data.get("total_sequences") or 0)
        ok = int(self._data.get("successful_sequences") or 0)
        if success and failed == 0:
            self._data["status"] = "completed"
        elif ok > 0:
            self._data["status"] = "partial"
        else:
            self._data["status"] = "failed"
        self._data["completed_at"] = _utc_now_iso()
        self.stage(
            f"Batch finished — {ok}/{total} shot(s) OK"
            + (f", {failed} failed" if failed else ""),
            fraction=1.0 if success else self._data.get("progress_fraction", 0.0),
        )


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
    "matte_temporal_ema": 0.25,
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


def _ai_matte_post_configs(shared: dict) -> list:
    from live_action_aov.core.job import PostConfig

    stride = int(shared.get("keyframe_stride", 4))
    fb = shared.get("flow_fb_threshold_px", shared.get("fb_threshold_px", 1.0))
    return [
        PostConfig(
            name="matte_flow_temporal",
            params={
                "keyframe_stride": stride,
                "fb_threshold_px": fb,
                "blend_forward_backward": shared.get("matte_flow_blend", 0.5),
                "final_ema_alpha": shared.get("matte_temporal_ema", 0.25),
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


def _verify_registry(refiner: str) -> int:
    try:
        from live_action_aov.core.registry import get_registry

        reg = get_registry()
        reg.load_all()
        for name in ("flow", "sam3_matte", refiner):
            reg.get_pass(name)
        if "matte_flow_temporal" not in reg._post:
            raise KeyError("matte_flow_temporal post-processor")
        _log.info("Registry OK: flow + sam3_matte + %s + matte_flow_temporal", refiner)
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
    _configure_colab_stdout()
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

    if refiner == "birefnet_refiner":
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

    if _verify_registry(refiner) != 0:
        return 1

    mount = lcr._drive_mount()
    if not mount.is_dir():
        _log.error("Drive mount not found: %s", mount)
        return 1
    _log.info("Drive mount: %s", mount)

    sequences = cfg.get("sequences") or []
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    reporter = ColabStageReporter(mount, cfg)
    out_root = str(shared.get("output_folder_path", "VDA_output")).strip("/")
    date_folder = str(os.environ.get("LAOV_RUNTIME_DATE_FOLDER", "")).strip().strip("/")
    matte_root = str(shared.get("ai_matte_output_root", "AI_MATTE_output"))
    reporter._data["output_location"] = (
        f"{out_root}/{date_folder}/{matte_root}" if date_folder else f"{out_root}/{matte_root}"
    )
    reporter.stage(
        f"AI Matte batch started — {len(sequences)} shot(s), passes={shared.get('passes_csv')}",
        fraction=0.0,
    )
    reporter.stage(f"Status file: {reporter.status_path}", fraction=0.01)

    reporter.stage("Resolving SAM3 model on Drive", fraction=0.02)
    sam3_model_dir = lcr._resolve_sam3_model_dir(mount, shared)
    if sam3_model_dir:
        os.environ["LAOV_SAM3_MODEL_PATH"] = sam3_model_dir
        _log.info("SAM3 model: %s", sam3_model_dir)
        reporter.stage(f"SAM3 model: {sam3_model_dir}", fraction=0.03)
    else:
        _log.warning("No SAM3 snapshot on Drive — Hub download may require HF login")
        reporter.stage("SAM3: Hub download (no Drive snapshot)", fraction=0.03)

    reporter.stage("Resolving BiRefNet model on Drive", fraction=0.04)
    birefnet_model_dir = lcr._resolve_birefnet_model_dir(mount, shared)
    if birefnet_model_dir:
        os.environ["AI_MATTE_BIREFNET_MODEL_PATH"] = birefnet_model_dir
        _log.info("BiRefNet model: %s", birefnet_model_dir)
        reporter.stage(f"BiRefNet model: {birefnet_model_dir}", fraction=0.05)
    elif refiner == "birefnet_refiner":
        _log.error(
            "BiRefNet snapshot not found on Drive. Upload MyDrive/VDA_models/ZhengPeng7/BiRefNet/"
        )
        reporter.stage("BiRefNet model missing on Drive", fraction=0.0)
        reporter.finish_batch(success=False)
        return 1

    try:
        code = _run_sequences(
            cfg,
            mount,
            shared,
            sequences,
            matte_detector=matte_detector,
            refiner=refiner,
            probe_first_frame=args.probe_first_frame,
            reporter=reporter,
        )
        reporter.finish_batch(success=(code == 0))
        return code
    except Exception:
        _log.error("AI Matte runner crashed:\n%s", traceback.format_exc())
        reporter.stage("Batch crashed — see log above", fraction=0.0)
        reporter.finish_batch(success=False)
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
    reporter: ColabStageReporter,
) -> int:
    from live_action_aov.cli.app import _resolve_semantic_passes
    from live_action_aov.core.job import Job, PassConfig, PostConfig, Shot
    from live_action_aov.core.registry import get_registry
    from live_action_aov.executors.local import LocalExecutor

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
    total_shots = len(sequences)
    for shot_idx, seq in enumerate(sequences):
        rel_side = str(seq.get("sidecar_output_drive_subpath", "")).strip().strip("/").replace("\\", "/")
        shot_name = str(seq.get("shot_name", "shot"))

        reporter.stage(
            f"Shot {shot_idx + 1}/{total_shots}: plate preflight ({shot_name})",
            shot_name=shot_name,
        )
        try:
            plate_dir, pattern, (f0, f1), resolution, pixel_aspect = lcr.shot_plate_from_desk_json(
                mount, seq
            )
        except FileNotFoundError as exc:
            _log.error("%s", exc)
            reporter.shot_failed(shot_idx, shot_name, str(exc))
            return 1

        frame_count = f1 - f0 + 1
        reporter.begin_shot(
            shot_idx,
            shot_name,
            f"{f0}-{f1}",
            frame_count,
        )

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
            reporter.stage("Reading first plate frame (OIIO probe)", shot_name=shot_name)
            _probe_plate_read(plate_dir, pattern, f0)

        reporter.stage(
            f"LAOV engine — passes: {', '.join(pass_names)}",
            shot_name=shot_name,
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
        post_configs: list[PostConfig] = _ai_matte_post_configs(shared)
        if "flow" not in pass_names:
            post_configs = []
        job = Job(shot=shot, passes=pass_configs, post=post_configs)
        try:
            LocalExecutor().submit(
                job,
                progress_callback=reporter.progress,
            )
        except Exception as exc:
            _log.error("Run failed for %s:\n%s", shot_name, traceback.format_exc())
            reporter.shot_failed(shot_idx, shot_name, str(exc))
            any_failed = True
            continue

        if shot.status != "done":
            _log.error("Shot %s status=%s (expected done)", shot_name, shot.status)
            reporter.shot_failed(shot_idx, shot_name, f"shot status={shot.status}")
            any_failed = True
        else:
            out_str = str(output_dir) if output_dir else ""
            reporter.shot_done(shot_idx, shot_name, output_dir=out_str or None)
            _log.info("Done shot=%s", shot_name)
            reporter.stage(f"Shot complete: {shot_name}", shot_name=shot_name)

    if any_failed:
        return 1
    _log.info("AI Matte — all sequences complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
