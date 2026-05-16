#!/usr/bin/env python3
"""Colab dependency fill for LAOV headless runs (`laov_colab_run.py`).

Colab already ships a CUDA `torch` / `torchvision` build. A full
`pip install -e ".[matte,dsine]"` re-resolves core deps and can pull a
second PyTorch wheel and always installs `PySide6` (GUI), which is unused
on Colab.

Workflow (Google Desk `laov_cellcode_template.py` on Colab):

1. `pip install -e /path/to/LAOV --no-deps`
2. `python scripts/colab_setup.py`  — installs only import-missing packages below.

Does **not** install or upgrade torch; use a GPU runtime with working CUDA.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys

# importlib module name -> pip requirement (pin loosely; Colab cache helps repeats)
REQUIRED_IMPORTS: dict[str, str] = {
    "pydantic": "pydantic>=2.5",
    "typer": "typer>=0.12",
    "yaml": "PyYAML>=6.0",
    "numpy": "numpy>=1.26,<2.0",
    "OpenImageIO": "oiio-python>=2.5",
    "PyOpenColorIO": "opencolorio>=2.3",
    "transformers": "transformers>=4.45",
    "huggingface_hub": "huggingface-hub>=0.23",
    "PIL": "Pillow",
    "tqdm": "tqdm>=4.66",
    "rich": "rich>=13.7",
    "platformdirs": "platformdirs>=4.2",
    "geffnet": "geffnet>=1.0",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
    "cv2": "opencv-python-headless>=4.8",
    "scipy": "scipy>=1.11",
}

TORCH_MODULES = ("torch", "torchvision")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _ensure_numpy1() -> None:
    """LAOV pins numpy<2; Colab often ships numpy 2.x."""
    try:
        import numpy as np

        major = int(str(np.__version__).split(".")[0])
    except Exception:
        return
    if major >= 2:
        print("[INSTALL] Colab numpy 2.x detected — installing numpy>=1.26,<2.0 for LAOV...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "numpy>=1.26,<2.0"],
            check=True,
        )


def ensure_laov_colab_dependencies() -> list[str]:
    """Install missing packages; return pip spec strings that were installed."""
    missing_torch = [m for m in TORCH_MODULES if not _has_module(m)]
    if missing_torch:
        raise RuntimeError(
            "Colab runtime missing: "
            + ", ".join(missing_torch)
            + ". Use Runtime → Change runtime type → GPU, then re-run."
        )

    _ensure_numpy1()

    to_install: list[str] = []
    seen: set[str] = set()
    for module_name, pip_spec in REQUIRED_IMPORTS.items():
        if _has_module(module_name):
            continue
        name = pip_spec.split(">")[0].split("=")[0].split("[")[0].strip()
        if name not in seen:
            seen.add(name)
            to_install.append(pip_spec)

    if not to_install:
        print("[OK] LAOV Colab: core Python deps already present.")
    else:
        print("[INSTALL] LAOV Colab missing packages: " + ", ".join(to_install))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *to_install],
            check=True,
        )
        print("[OK] LAOV Colab dependency install finished.")

    _verify_oiio()
    return to_install


def _verify_oiio() -> None:
    try:
        import OpenImageIO as oiio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "OpenImageIO (oiio-python) failed to import after install. "
            "Try: pip install -q oiio-python>=2.5 opencolorio>=2.3"
        ) from exc
    print("[OK] OpenImageIO import check passed.")


if __name__ == "__main__":
    ensure_laov_colab_dependencies()
