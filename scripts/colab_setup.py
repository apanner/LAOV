#!/usr/bin/env python3
"""Install LAOV Colab dependencies (headless — no PySide6, no torch reinstall).

Run after: ``pip install -e /content/LAOV --no-deps``

  python scripts/colab_setup.py          # missing packages only
  python scripts/colab_setup.py --matte  # AI Matte lane (numpy 1.x + kornia + I/O)
  python scripts/colab_setup.py --full   # reinstall full bundle
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

COLAB_PIP_DEPS: tuple[str, ...] = (
    "numpy>=1.26,<2.0",
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

# AI Matte Colab — always install these (BiRefNet needs kornia every run).
MATTE_COLAB_DEPS: tuple[str, ...] = (
    "numpy>=1.26,<2.0",
    "kornia>=0.7",
    "oiio-python>=2.5",
    "opencolorio>=2.3",
    "geffnet>=1.0",
    "opencv-python-headless>=4.8",
    "transformers>=4.51.0",
    "huggingface-hub>=0.23",
    "Pillow",
    "tqdm>=4.66",
)

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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(*specs: str, quiet: bool = False) -> None:
    if not specs:
        return
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(specs)
    print("[INSTALL] " + ", ".join(specs))
    subprocess.run(cmd, check=True)


def _numpy_major() -> int | None:
    try:
        import numpy as np
    except ImportError:
        return None
    parts = getattr(np, "__version__", "0").split(".")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def ensure_numpy_colab() -> None:
    major = _numpy_major()
    if major is not None and major < 2:
        import numpy as np

        print(f"[OK] numpy {np.__version__}")
        return
    print("[INSTALL] numpy>=1.26,<2.0")
    _pip_install("numpy>=1.26,<2.0")
    import numpy as np

    print(f"[OK] numpy {np.__version__}")


def ensure_kornia_colab() -> None:
    """BiRefNet HF code imports kornia — install every AI Matte run."""
    print("[INSTALL] kornia>=0.7 (BiRefNet)")
    _pip_install("kornia>=0.7")
    import kornia  # noqa: F401

    print(f"[OK] kornia {kornia.__version__}")


def ensure_matte_colab_dependencies() -> None:
    """AI Matte lane: numpy 1.x, kornia, plate I/O, transformers."""
    print("[COLAB] AI Matte dependency setup")
    ensure_numpy_colab()
    ensure_kornia_colab()
    rest = tuple(
        s
        for s in MATTE_COLAB_DEPS
        if not s.startswith("numpy") and not s.startswith("kornia")
    )
    _pip_install(*rest, quiet=True)
    _verify_imports(must_have=("kornia", "numpy", "OpenImageIO", "torch", "transformers"))
    print("[OK] AI Matte Colab deps ready.")


def ensure_laov_colab_dependencies(*, full: bool = False) -> None:
    ensure_numpy_colab()
    missing_torch = [m for m in TORCH_MODULES if not _has_module(m)]
    if missing_torch:
        raise RuntimeError(
            "Colab runtime missing: "
            + ", ".join(missing_torch)
            + ". Use Runtime → Change runtime type → GPU, then re-run."
        )

    if full:
        print("[COLAB] Installing full LAOV Colab bundle...")
        _pip_install(*COLAB_PIP_DEPS)
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
            _pip_install(*to_install)
        else:
            print("[OK] LAOV Colab: imports already satisfied.")
        if not _has_module("kornia"):
            ensure_kornia_colab()

    _verify_imports()
    print("[OK] LAOV Colab dependency install finished.")


def _verify_imports(must_have: tuple[str, ...] | None = None) -> None:
    check = must_have or tuple(VERIFY_IMPORTS.keys())
    failed: list[str] = []
    for mod in check:
        spec = VERIFY_IMPORTS.get(mod, mod)
        if not _has_module(mod):
            failed.append(f"{mod} ({spec})")
    if failed:
        raise RuntimeError("Missing imports: " + ", ".join(failed))
    import OpenImageIO as oiio  # noqa: F401

    _ = oiio
    print("[OK] Import check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAOV Colab dependency installer")
    parser.add_argument("--full", action="store_true", help="Reinstall full COLAB_PIP_DEPS")
    parser.add_argument(
        "--matte",
        action="store_true",
        help="AI Matte lane: numpy 1.x + kornia + plate I/O (recommended for Cell 3)",
    )
    args = parser.parse_args()
    if args.matte:
        ensure_matte_colab_dependencies()
    else:
        ensure_laov_colab_dependencies(full=args.full)
