# LAOV on Google Colab (Desk lane)

This document matches the **LAOV_STANDALONE** flow from `google_desk_app` (`run_app_laov.bat` → Send to Colab).

## What runs on Colab

**Direct install on the Colab VM** (no Docker):

1. **GPU runtime** (Runtime → Change runtime type → **GPU**).
2. Mount Drive and load the batch JSON (your Desk-generated notebook cells).
3. **`git clone`** `LAOV_GIT_URL` (default `https://github.com/apanner/LAOV.git`).
4. **`pip install -e . --no-deps`** then **`python scripts/colab_setup.py`** to install the full Colab dependency bundle (reuses Colab’s `torch` / `torchvision`, does **not** install **PySide6**, does **not** downgrade numpy).
5. **`python scripts/laov_colab_run.py --job-json ...`** with `LAOV_DRIVE_MOUNT=/content/drive/MyDrive`.

If `colab_setup.py` is missing from the clone, the cell falls back to **`pip install -e ".[matte,dsine]"`**.

## Rough download sizes (order-of-magnitude)

| What | Typical download / disk | Notes |
|------|---------------------------|--------|
| **Git clone** | &lt; few MB | Source only. |
| **`colab_setup.py` pip** (first time) | **~1–4 GB** | `transformers`, `oiio-python`, `opencolorio`, etc.; **no second torch** if Colab GPU stack already works. |
| **PySide6** (full `.[matte,dsine]` fallback only) | **~0.3–0.8 GB** | Not installed by the `colab_setup` path. |
| **Hugging Face model cache** (first job) | **~3–15+ GB** | SAM 3 + Depth Anything V2 + tracker weights dominate. |

## Environment variables

| Variable | Meaning |
|----------|---------|
| `LAOV_DRIVE_MOUNT` | Absolute path to the Drive root that contains `VDA_input/...` (Colab: `/content/drive/MyDrive`). |
| `LAOV_RUNTIME_DATE_FOLDER` | `YYYYMMDD` folder segment under your Desk output path (set by Colab cellcode). |
| `LAOV_GIT_URL` | Optional override for the `git clone` URL. |

## Hugging Face (SAM3 / gated weights)

Accept model terms on huggingface.co, then in Colab:

```python
!huggingface-cli login
```

Use a token with **read** access.

## Optional: persist HF cache on Drive

Reduces repeat downloads when sessions restart, for example:

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/.cache/huggingface"
```

(Create the folder once on Drive.)

## Verifying outputs

On any machine with LAOV installed:

```bash
liveaov inspect /path/to/plate.1001.utility.exr
```

Confirm channels and `liveaov/*` metadata.
