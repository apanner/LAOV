# LAOV on Google Colab (Desk lane)

This document matches the **LAOV_STANDALONE** flow from `google_desk_app` (`run_app_laov.bat` → Send to Colab).

## What runs on Colab

**Direct install on the Colab VM** (no Docker):

1. **GPU runtime** (Runtime → Change runtime type → **GPU**).
2. Mount Drive and load the batch JSON (your Desk-generated notebook cells).
3. **`git clone`** `LAOV_GIT_URL` (default `https://github.com/apanner/LAOV.git`).
4. **`pip install -e . --no-deps`** then **`python scripts/colab_setup.py`** (installs **only missing** packages; use `--full` to reinstall everything).
5. **`python scripts/laov_colab_run.py --job-json ...`** with `LAOV_DRIVE_MOUNT=/content/drive/MyDrive`.

## Output layout (default: split folders)

Desk / Colab default is **`output_layout: split_folders`** — separate EXR sequences per AOV type (faster writes and easier Nuke reads than one huge multi-channel utility EXR).

```
MyDrive/{output_folder}/{YYYYMMDD}/LAOV_output/{shot_name}/
  depth/
    TB_005_010.depth.1001.exr    # Z, Z_raw, depth.confidence, P.x/y/z
  normals/
    TB_005_010.normals.1001.exr  # N.x/y/z, normals.confidence, ao.a
  flow/
    TB_005_010.flow.1001.exr     # motion, forward/backward flow channels
  matte/
    TB_005_010.matte.1001.exr    # matte.r/g/b/a, mask.<concept>
  laov_run.log
```

Set **Output layout → Combined utility EXR** in Desk if you need the legacy single file per frame (`*.utility.*.exr` with all channels).

Inference time is unchanged; split mode mainly saves **disk write time** and **comp load time**.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `LAOV_DRIVE_MOUNT` | Drive root containing `VDA_input/...` (Colab: `/content/drive/MyDrive`). |
| `LAOV_RUNTIME_DATE_FOLDER` | `YYYYMMDD` under your output path. |
| `LAOV_GIT_URL` | Optional git clone URL. |
| `LAOV_SAM3_MODEL_PATH` | Optional full path to a local SAM3 HF snapshot (see below). |

**Plate path:** batch JSON should use the full Drive-relative folder (e.g. `VDA_input/LOT_test/TB_005_010_1`). If Desk sends only `VDA_input/TB_005_010_1`, `laov_colab_run.py` searches under `VDA_input/` for a matching plate folder.

**Pip warnings** after `pip install -e --no-deps` about `pyside6` are harmless on Colab (GUI-only). **`numpy` must be 1.x** — Colab ships 2.x; `colab_setup.py` downgrades to `numpy>=1.26,<2.0` before the batch run. If you still see `numpy 2.0.2`, re-run `python scripts/colab_setup.py` and check `[OK] numpy 1.x`.

## SAM3 on Google Drive (no Colab login)

Download **once** on any machine where you have accepted [facebook/sam3](https://huggingface.co/facebook/sam3) access:

```bash
cd LAOV
pip install "huggingface_hub>=0.34"
hf auth login
python scripts/download_sam3_for_drive.py
```

Windows: `scripts\download_sam3_for_drive.bat`

This writes `VDA_models/facebook/sam3/` (ready to upload). Verify only: `python scripts/download_sam3_for_drive.py --verify-only`.

Upload the **entire** `sam3_snapshot` folder to Drive (must contain `config.json` and weight files):

```text
MyDrive/VDA_models/facebook/sam3/config.json
MyDrive/VDA_models/facebook/sam3/model.safetensors
… (other processor / config files from the snapshot)
```

Colab auto-detects (first match wins):

- `VDA_models/facebook/sam3`
- `VDA_model/facebook/sam3`
- `VDA_models/sam3` or `VDA_model/sam3`

Or set in batch JSON `shared_settings.sam3_model_path` to e.g. `VDA_models/facebook/sam3` (relative to Drive mount).

## Progress / stage updates (Colab)

During a run you should see flushed **`[STAGE]`** lines in the notebook (pass boundaries, shot preflight, percent). LAOV also writes:

```text
MyDrive/VDA_Jobs/status/{job_id}_status.json
```

Open that JSON on Drive (or re-enable Desk job monitoring) to see `stage`, `progress_percent`, and per-shot `sequences[].status` while a long SAM3 / BiRefNet job is running. Inside each pass, logs include sub-stages (e.g. `SAM3: track …`, `BiRefNet: keyframe 12/101`, `RAFT: pair 200/399`).

After `git pull`, re-run Cell 3 (setup cell optional if deps unchanged).

## AI Matte Desk app

Run `google_desk_app/run_app_ai_matte.bat` — per-shot **People / Auto FG / OpenCV bbox**, then **Send to Colab**.

Colab Cell 3 runs the **dedicated** runner (not the generic LAOV batch script):

```bash
python scripts/ai_matte_colab_run.py --job-json /content/ai_matte_config.json --probe-first-frame
```

Batch uses `passes_csv: flow,matte` (RAFT optical flow → SAM3 → BiRefNet keyframes →
`matte_flow_temporal` post), `refiner: birefnet_refiner`, models from Drive paths below.

### AI Matte output folder structure

Default Desk layout (`output_folder_path` is usually `VDA_output`, date from Colab env):

```text
MyDrive/<output_folder_path>/<YYYYMMDD>/AI_MATTE_output/<shot_name>/
  matte_sam3/      # SAM3 only — mask.<concept> EXRs
  matte_birefnet/  # BiRefNet keyframes only — matte.r/g/b/a (before temporal)
  matte_vitmatte/  # ViTMatte keyframes — vitmatte.r/g/b/a (optional, trimap from SAM3)
  matte/           # Final temporal matte (BiRefNet + RAFT fill) when enabled in Desk
  flow/            # RAFT motion (optional; not a matte)
  qc/              # MP4 per stage (*_sam3_*, *_birefnet_*, *_final_*, …)
  laov_run.log
```

**What is what**

| Output | What it is |
|--------|------------|
| **`flow/`** | Optical flow (motion vectors). Intermediate data for `matte_flow_temporal` — not for comp mattes. |
| **`matte/*.matte.<frame>.exr`** | Multi-channel EXR per frame. |
| → `mask.<concept>` | **SAM3** — hard union masks per concept (e.g. `mask.person`). |
| → `matte.r/g/b/a` | Optional: **RAFT temporal fill** (`export_final_exr`) — off by default; use stage folders instead. |
| **`qc/*_sam3_matte_qc.mp4`** | Preview of SAM3 `mask.*` (green on plate). |
| **`qc/*_final_matte_qc.mp4`** | Preview of final `matte.*` (BiRefNet + temporal). |

**Pipeline (yes — BiRefNet uses SAM3)**

```text
Plate → RAFT (flow/) → SAM3 track (mask.*) → BiRefNet refine (matte.r/g/b/a on keyframes)
      → (optional) matte_flow_temporal → matte/ EXRs
      → stage EXRs: matte_sam3/, matte_birefnet/, matte_vitmatte/ → qc/ MP4s
```

Example EXR:

```text
MyDrive/VDA_output/20260519/AI_MATTE_output/TB_073_020_plate_v001/matte/TB_073_020_plate_v001.matte.1001.exr
```

Example QC MP4:

```text
.../TB_073_020_plate_v001/qc/TB_073_020_plate_v001_sam3_matte_qc.mp4
.../TB_073_020_plate_v001/qc/TB_073_020_plate_v001_final_matte_qc.mp4
```

`shot_name` comes from the sequence / folder leaf in the batch JSON. EXR plates may live in a **subfolder** (e.g. `.../TB_073_020_plate_v001/exr/`); `ai_matte_colab_run.py` searches nested folders under the JSON path and under `VDA_input/` when the exact path is missing.

**Plate paths:** Desk writes exact Colab-relative file paths into the batch JSON (`plate_first_frame_relpath`, `plate_input_pattern`). Colab joins them to `/content/drive/MyDrive/` — no folder search. Re-send from AI Matte Desk after refreshing Drive paths. If the first-frame file is missing on the mount, sync `S:\VDA_input\...` to Google Drive cloud.

**Git:** push LAOV `main` so Colab clone includes `scripts/ai_matte_colab_run.py`, nested plate search in `laov_colab_run.py`, and `birefnet_refiner` in `pyproject.toml`. In Desk, set **LAOV repo** to your fork URL if needed.

See `colab_ai/README.md`.

## BiRefNet on Google Drive (no Colab login)

Download **once** (public model, no HF license gate):

```bash
cd LAOV
python scripts/download_birefnet_for_drive.py
```

Windows: `scripts\download_birefnet_for_drive.bat`

Upload to:

```text
MyDrive/VDA_models/ZhengPeng7/BiRefNet/config.json
MyDrive/VDA_models/ZhengPeng7/BiRefNet/model.safetensors
```

Verify: `python scripts/download_birefnet_for_drive.py --verify-only`

**Colab dependency:** BiRefNet’s Hugging Face modeling code requires **`kornia`**. `scripts/colab_setup.py` installs it automatically. If you see `No module named 'kornia'` after an older setup cell, run:

```bash
pip install -q "kornia>=0.7"
```

or re-run `python scripts/colab_setup.py --full` after `git pull`.

### ViTMatte (high-quality refiner, trimap from SAM3)

Hugging Face only publishes **small** and **base** Composition-1k checkpoints (no separate “large” repo). Use **base** for best quality:

```bash
cd LAOV
python scripts/download_vitmatte_for_drive.py --model base
```

Windows: `scripts\download_vitmatte_for_drive.bat`

Upload to:

```text
MyDrive/VDA_models/hustvl/vitmatte-base-composition-1k/
```

Desk / batch JSON: `vitmatte_model_path`: `VDA_models/hustvl/vitmatte-base-composition-1k`

Verify: `python scripts/download_vitmatte_for_drive.py --verify-only`

**SAM3 → trimap (ViTMatte):** on each keyframe, SAM3 hard mask is optionally **expanded** (`hard_mask_dilate`, default 5px), then a classic trimap is built with elliptical **erode** (definite fg = 255) and **dilate** (unknown band = 128, bg = 0). Tunables: `trimap_erode_px` (10), `trimap_dilate_px` (25). Requires OpenCV (`opencv-python-headless` from `colab_setup.py`).

## SAM3 + transformers on Colab

If SAM3 tracking fails with ``fpn_position_embeddings``, upgrade transformers then re-run Cell 2 setup:

```bash
pip install -q -U "transformers>=4.51.0"
```

LAOV also applies a runtime alias patch in ``sam3_matte`` for older Colab caches.

## Hugging Face login (only if SAM3 is not on Drive)

```python
!pip install -q huggingface_hub
!huggingface-cli login
```

## Colab phases (Desk **Colab phase**)

| Phase | What runs |
|-------|-----------|
| **stages** (default) | SAM3 → then every **checked** refiner (BiRefNet, ViTMatte) in **one** Colab job |
| **sam3** | SAM3 only → `matte_sam3/` + `_sam3_artifacts.npz` |
| **refine** | One refiner only, loads prior `matte_sam3/_sam3_artifacts.npz` |
| **temporal** | RAFT + optional `matte/` fill (off by default) |

CLI: `--phase stages|sam3|refine|temporal`

**numpy on Colab:** `colab_setup.py` always runs `ensure_numpy_colab()` first (downgrades Colab’s numpy 2.x). `ai_matte_colab_run.py` calls it again at startup. **PySide6** is not installed on Colab — ignore pip metadata warnings.

**Exit code 1 with little output:** scroll up for `ERROR` lines — common causes: missing `VDA_models/...` snapshot (SAM3 / BiRefNet / ViTMatte), plate path not on Drive, or `kornia` missing (BiRefNet). Uncheck ViTMatte in Desk if that model is not on Drive.

## Colab notebook: `CODE_FILE_ID` must be AI Matte cellcode

If Cell 1 fails with `ImportError: cannot import name 'setup_ai_matte_cell1'`, the notebook is downloading **VDA/DVD** cellcode, not AI Matte.

1. Re-run **Send to Colab** from `run_app_ai_matte.bat` and copy **both** IDs from the generated notebook.
2. Or on Drive: `VDA_Jobs/code/{job_id}_cellcode.py` from **that** job.
3. After download, verify: `grep setup_ai_matte_cell1 /content/cellcode.py`

Templates live in `LAOV/colab_templates/` (same as `google_desk_app/colab_templates/`).

## Verifying outputs

```bash
liveaov inspect /path/to/TB_005_010.depth.1001.exr
```
