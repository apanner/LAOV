# LAOV on Google Colab (Desk lane)

This document matches the **LAOV_STANDALONE** flow from `google_desk_app` (`run_app_laov.bat` → Send to Colab).

## What runs on Colab

**Direct install on the Colab VM** (no Docker):

1. **GPU runtime** (Runtime → Change runtime type → **GPU**).
2. Mount Drive and load the batch JSON (your Desk-generated notebook cells).
3. **`git clone`** `LAOV_GIT_URL` (default `https://github.com/apanner/LAOV.git`).
4. **`pip install -e . --no-deps`** then **`python scripts/colab_setup.py`** to install the full Colab dependency bundle.
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

## Hugging Face (SAM3)

```python
!huggingface-cli login
```

## Verifying outputs

```bash
liveaov inspect /path/to/TB_005_010.depth.1001.exr
```
