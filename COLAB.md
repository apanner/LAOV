# LAOV on Google Colab (Desk lane)

This document matches the **LAOV_STANDALONE** flow from `google_desk_app` (`run_app_laov.bat` → Send to Colab).

## What runs where

1. **Docker (preferred):** Colab pulls `ghcr.io/<owner>/laov-colab:latest` (override in Desk **Docker image** or env `LAOV_DOCKER_IMAGE`). The container mounts `MyDrive` at `/data` and reads the job JSON at `/config/job.json`.
2. **Fallback:** If Docker is unavailable or `docker run` fails, Colab clones `LAOV_GIT_URL`, runs `pip install -e . --no-deps`, then [`scripts/colab_setup.py`](scripts/colab_setup.py) to **pip-install only missing** Python packages (reuses Colab’s existing `torch` / `torchvision`, avoids downloading **PySide6** for the GUI). If `colab_setup.py` is not in the clone, it falls back to a full `pip install -e ".[matte,dsine]"`.

## Rough download sizes (order-of-magnitude)

Numbers vary by wheel index, cache, and model variant. Use this to budget Colab disk and time.

| What | Typical download / disk | Notes |
|------|---------------------------|--------|
| **Docker image** `pytorch/pytorch:2.4.0-cuda12.1…` + LAOV layer | **~6–12 GB pull**, **~15–25 GB** uncompressed | One big pull; includes torch + most pip deps baked in. |
| **Fallback: git clone** | **&lt; few MB** | Source only. |
| **Fallback: `colab_setup.py` pip** (first time) | **~1–4 GB** | `transformers`, `oiio-python`, `opencolorio`, wheels; **no second torch** if Colab GPU stack already works. |
| **PySide6** (old full `pip install -e ".[matte,dsine]"` only) | **~0.3–0.8 GB** | **Not** installed by the `colab_setup` path. |
| **Hugging Face model cache** (first job) | **~3–15+ GB** | SAM 3 + Depth Anything V2 + tracker weights dominate; DSINE/RVM use `torch.hub` caches (often **~0.2–1 GB** each). RAFT uses **torchvision** weights (usually modest, often bundled/cache). |
| **Optional: persist HF cache on Drive** | — | Reduces repeat downloads when sessions restart: e.g. `export HF_HOME=/content/drive/MyDrive/.cache/huggingface` before running (create folder once). |

Upstream docs also cite **~15 GB** venv + **~10–40 GB** models for a full local “everything” install; the Colab **commercial preset** (`depth_anything_v2`, `dsine`, `flow`, `sam3`+`rvm`) is a subset of that but still **model-heavy** because of SAM 3.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `LAOV_DRIVE_MOUNT` | Absolute path to the Drive root that contains `VDA_input/...` (inside Docker: `/data`). |
| `LAOV_RUNTIME_DATE_FOLDER` | `YYYYMMDD` folder segment under your Desk output path (set by Colab cellcode). |
| `LAOV_DOCKER_IMAGE` | Optional override for the container image URL. |
| `LAOV_GIT_URL` | Optional override for fallback git clone. |

## Hugging Face (SAM3 / gated weights)

Accept model terms on huggingface.co, then in Colab:

```python
!huggingface-cli login
```

Use a token with **read** access.

## Publishing the Docker image

From this repository root (as git root for LAOV):

```bash
docker build -t ghcr.io/<your_github_user>/laov-colab:latest .
docker push ghcr.io/<your_github_user>/laov-colab:latest
```

Or use the GitHub Action **LAOV Colab Docker** (workflow_dispatch) if enabled on your fork.

## Verifying outputs

On any machine with LAOV installed:

```bash
liveaov inspect /path/to/plate.1001.utility.exr
```

Confirm channels and `liveaov/*` metadata.
