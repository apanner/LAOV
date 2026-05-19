# LiveActionAOV — persist SAM3 artifacts between Colab phases (SAM3 → refine).

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

ARTIFACT_FILENAME = "_sam3_artifacts.npz"
META_FILENAME = "_sam3_artifacts_meta.json"


def sam3_artifact_path(stage_dir: Path) -> Path:
    return Path(stage_dir).resolve() / ARTIFACT_FILENAME


def save_sam3_artifacts(
    stage_dir: Path,
    artifacts: dict[str, dict[int, Any]],
    *,
    frame_range: tuple[int, int],
    shot_name: str,
) -> Path | None:
    """Write SAM3 hard masks + hero list for a later refine-only run."""
    hard = artifacts.get("sam3_hard_masks") or {}
    heroes = artifacts.get("sam3_instances") or {}
    if not hard:
        _log.warning("No sam3_hard_masks to save for shot %s", shot_name)
        return None
    hard_masks = next(iter(hard.values()), None)
    hero_list = next(iter(heroes.values()), None)
    if not isinstance(hard_masks, dict):
        _log.warning("Unexpected sam3_hard_masks shape for %s", shot_name)
        return None

    stage_dir = Path(stage_dir).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = sam3_artifact_path(stage_dir)

    payload: dict[str, Any] = {
        "frame_range": [int(frame_range[0]), int(frame_range[1])],
        "shot_name": shot_name,
    }
    track_keys: list[str] = []
    for track_id, entry in hard_masks.items():
        if not isinstance(entry, dict) or entry.get("stack") is None:
            continue
        key = f"track_{int(track_id)}"
        track_keys.append(key)
        stack = np.asarray(entry["stack"], dtype=np.float32)
        payload[f"{key}_stack"] = stack
        payload[f"{key}_label"] = np.array(str(entry.get("label", "")))
        frames = entry.get("frames")
        if frames is not None:
            payload[f"{key}_frames"] = np.asarray(frames, dtype=np.int32)

    payload["track_keys"] = np.array(track_keys, dtype=object)
    if hero_list is not None:
        meta_path = stage_dir / META_FILENAME
        meta_path.write_text(json.dumps(hero_list, indent=0), encoding="utf-8")
        payload["heroes_json_path"] = np.array(str(meta_path.name))

    np.savez_compressed(out, **payload)
    _log.info("Saved SAM3 artifacts for refine phase → %s", out)
    return out


def load_sam3_artifacts(stage_dir: Path) -> dict[str, dict[int, Any]] | None:
    """Load artifacts dict for ``ingest_artifacts`` on refiners."""
    path = sam3_artifact_path(stage_dir)
    if not path.is_file():
        _log.error("SAM3 artifact file missing: %s (run SAM3 phase first)", path)
        return None

    data = np.load(path, allow_pickle=True)
    track_keys = [str(k) for k in data.get("track_keys", [])]
    hard_masks: dict[int, dict[str, Any]] = {}
    for key in track_keys:
        tid = int(key.replace("track_", "", 1))
        stack = np.asarray(data[f"{key}_stack"], dtype=np.float32)
        label_arr = data.get(f"{key}_label")
        label = str(label_arr.item()) if label_arr is not None else ""
        entry: dict[str, Any] = {"label": label, "stack": stack}
        frames_arr = data.get(f"{key}_frames")
        if frames_arr is not None:
            entry["frames"] = [int(x) for x in np.asarray(frames_arr).tolist()]
        hard_masks[tid] = entry

    hero_list: list[dict[str, Any]] = []
    heroes_name = data.get("heroes_json_path")
    if heroes_name is not None:
        meta_path = path.parent / str(heroes_name.item())
        if meta_path.is_file():
            hero_list = json.loads(meta_path.read_text(encoding="utf-8"))

    any_frame = 0
    if hard_masks:
        first = next(iter(hard_masks.values()))
        fr = first.get("frames")
        if fr:
            any_frame = int(fr[0])

    _log.info(
        "Loaded SAM3 artifacts from %s (%d tracks, %d heroes)",
        path,
        len(hard_masks),
        len(hero_list),
    )
    return {
        "sam3_hard_masks": {any_frame: hard_masks},
        "sam3_instances": {any_frame: hero_list},
    }


__all__ = [
    "ARTIFACT_FILENAME",
    "load_sam3_artifacts",
    "sam3_artifact_path",
    "save_sam3_artifacts",
]
