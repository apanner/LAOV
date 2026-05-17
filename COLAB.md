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
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download facebook/sam3 --local-dir ./sam3_snapshot
```

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

## Hugging Face login (only if SAM3 is not on Drive)

```python
!pip install -q huggingface_hub
!huggingface-cli login
```

## Verifying outputs

```bash
liveaov inspect /path/to/TB_005_010.depth.1001.exr
```
