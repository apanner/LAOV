#!/usr/bin/env python3
"""Install LAOV Colab dependencies (headless — no PySide6, no torch reinstall).

Run after: ``pip install -e /content/LAOV --no-deps``

Uses Colab's existing ``torch`` / ``torchvision``. Installs every package LAOV
needs for the commercial Desk preset (depth, normals, flow, matte).
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

# Pip specs for Colab headless lane (torch/torchvision/PySide6 excluded on purpose).
COLAB_PIP_DEPS: tuple[str, ...] = (
    "pydantic>=2.5",
    "typer>=0.12",
    "PyYAML>=6.0",
    "oiio-python>=2.5",
    "opencolorio>=2.3",
    "transformers>=4.51.0",
    "huggingface-hub>=0.23",
    "tokenizers",
    "safetensors",
    "Pillow",
    "tqdm>=4.66",
    "rich>=13.7",
    "platformdirs>=4.2",
    "geffnet>=1.0",
    "opencv-python-headless>=4.8",
    "scipy>=1.11",
    "einops>=0.4",
    "easydict>=1.10",
    "kornia>=0.7",
)

# import_name -> pip spec (for post-install import verification)
VERIFY_IMPORTS: dict[str, str] = {
    "pydantic": "pydantic>=2.5",
    "typer": "typer>=0.12",
    "yaml": "PyYAML>=6.0",
    "OpenImageIO": "oiio-python>=2.5",
    "PyOpenColorIO": "opencolorio>=2.3",
    "transformers": "transformers>=4.45",
    "huggingface_hub": "huggingface-hub>=0.23",
    "PIL": "Pillow",
    "geffnet": "geffnet>=1.0",
    "cv2": "opencv-python-headless>=4.8",
    "kornia": "kornia>=0.7",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
}

TORCH_MODULES = ("torch", "torchvision")
_PLATE_SUFFIXES = (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".dpx", ".tga")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(*specs: str, quiet: bool = True) -> None:
    if not specs:
        return
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(specs)
    print("[INSTALL] " + ", ".join(specs))
    subprocess.run(cmd, check=True)


def ensure_laov_colab_dependencies(*, full: bool = False) -> None:
    """Install Colab deps. ``full=False`` (default) installs only missing imports."""
    missing_torch = [m for m in TORCH_MODULES if not _has_module(m)]
    if missing_torch:
        raise RuntimeError(
            "Colab runtime missing: "
            + ", ".join(missing_torch)
            + ". Use Runtime → Change runtime type → GPU, then re-run."
        )

    if full:
        print("[COLAB] Installing LAOV Colab dependency bundle...")
        _pip_install(*COLAB_PIP_DEPS, quiet=False)
    else:
        to_install: list[str] = []
        seen: set[str] = set()
        for mod, spec in VERIFY_IMPORTS.items():
            if mod in TORCH_MODULES:
                continue
            if _has_module(mod):
                continue
            name = spec.split(">")[0].split("=")[0].split("[")[0].strip()
            if name not in seen:
                seen.add(name)
                to_install.append(spec)
        if to_install:
            _pip_install(*to_install, quiet=False)
        else:
            print("[OK] LAOV Colab: imports already satisfied.")

    _verify_imports()
    print("[OK] LAOV Colab dependency install finished.")


def _verify_imports() -> None:
    failed: list[str] = []
    for mod, spec in VERIFY_IMPORTS.items():
        if not _has_module(mod):
            failed.append(f"{mod} (pip: {spec})")
    if failed:
        raise RuntimeError(
            "Missing imports after install: " + ", ".join(failed) + ". Re-run colab_setup.py."
        )
    try:
        import OpenImageIO as oiio  # noqa: F401

        _ = oiio
    except ImportError as exc:
        raise RuntimeError(
            "OpenImageIO import failed. Try: pip install -q --force-reinstall oiio-python>=2.5"
        ) from exc
    print("[OK] Import verification passed (OpenImageIO, torch, transformers, …).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAOV Colab dependency installer")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Reinstall the full COLAB_PIP_DEPS bundle (default: only missing packages).",
    )
    args = parser.parse_args()
    ensure_laov_colab_dependencies(full=args.full)
