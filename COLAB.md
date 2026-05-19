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

**Pip warnings** after `pip install -e --no-deps` about `pyside6` or `numpy<2` are expected on Colab. PySide6 is for the desktop GUI only (not installed on Colab). Numpy 2.x on Colab is fine for the headless lane — the warning is package metadata, not a failed install.

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
  matte/           # hero mattes — matte.r, matte.g, matte.b, matte.a per frame
  flow/            # RAFT motion sidecars (when flow pass is enabled)
  laov_run.log
```

Example:

```text
MyDrive/VDA_output/20260519/AI_MATTE_output/TB_073_020_plate_v001/matte/TB_073_020_plate_v001.matte.1001.exr
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

## Verifying outputs

```bash
liveaov inspect /path/to/TB_005_010.depth.1001.exr
```
