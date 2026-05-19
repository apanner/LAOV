#!/usr/bin/env python3
"""Download ViTMatte Hugging Face snapshot for Google Drive / Colab.

Models (Composition-1k, public on Hugging Face):
  - hustvl/vitmatte-base-composition-1k   ← recommended (best quality on HF)
  - hustvl/vitmatte-small-composition-1k  ← faster, lower VRAM

Usage:
  scripts\\download_vitmatte_for_drive.bat
  python scripts/download_vitmatte_for_drive.py --model base
  python scripts/download_vitmatte_for_drive.py --verify-only

Upload the output folder to Drive as:
  MyDrive/VDA_models/hustvl/vitmatte-base-composition-1k/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODELS: dict[str, str] = {
    "base": "hustvl/vitmatte-base-composition-1k",
    "small": "hustvl/vitmatte-small-composition-1k",
}
DEFAULT_VARIANT = "base"
REQUIRED_FILES = ("config.json",)
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent / ".hf_token"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ViTMatte for VDA_models on Drive.")
    p.add_argument(
        "--model",
        choices=tuple(MODELS.keys()),
        default=DEFAULT_VARIANT,
        help="base = highest quality on HF (default); small = faster.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Local folder (default: VDA_models/hustvl/<repo_name>).",
    )
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--token-file", type=Path, default=None)
    return p.parse_args()


def _default_output(variant: str) -> Path:
    repo = MODELS[variant]
    name = repo.split("/")[-1]
    return Path("VDA_models") / "hustvl" / name


def _read_token_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip()
        if token.startswith("#") or not token:
            continue
        if token.startswith("hf_"):
            return token
    return None


def _verify_snapshot(out: Path) -> bool:
    if not out.is_dir():
        print(f"MISSING: directory {out}")
        return False
    ok = True
    for name in REQUIRED_FILES:
        path = out / name
        if not path.is_file():
            print(f"MISSING: {path}")
            ok = False
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin")) + list(out.glob("*.pth"))
    if not weights:
        print(f"MISSING: no .safetensors/.bin/.pth weights in {out}")
        ok = False
    else:
        total_mb = sum(f.stat().st_size for f in weights) / (1024 * 1024)
        print(f"OK: {len(weights)} weight file(s), ~{total_mb:.0f} MB")
    if ok:
        print(f"OK: snapshot at {out}")
    return ok


def _optional_hf_login(*, token_file: Path | None = None) -> None:
    from huggingface_hub import HfApi, login

    try:
        user = HfApi().whoami()
        name = user.get("name") or user.get("fullname") or "OK"
        print(f"Hugging Face: already logged in as {name}")
        return
    except Exception:
        pass

    token = os.environ.get("HF_TOKEN", "").strip() or None
    if not token and token_file:
        token = _read_token_file(token_file)

    if token:
        login(token=token, add_to_git_credential=False)
        user = HfApi().whoami()
        name = user.get("name") or user.get("fullname") or "OK"
        print(f"Logged in as {name}.")
    else:
        print("No HF token — downloading public model (no login required).")


def _download(repo_id: str, out: Path, *, token_file: Path | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('Install: pip install "huggingface_hub>=0.34"', file=sys.stderr)
        raise SystemExit(1) from None

    _optional_hf_login(token_file=token_file)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {out}")
    if "base" in repo_id:
        print("(~350–450 MB for base — a few minutes on a good connection.)")
    else:
        print("(~100–150 MB for small.)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print("Download finished.")


def main() -> int:
    args = _parse_args()
    repo_id = MODELS[args.model]
    out = (args.output or _default_output(args.model)).resolve()

    if args.verify_only:
        return 0 if _verify_snapshot(out) else 1

    token_file = args.token_file
    if token_file is None and DEFAULT_TOKEN_FILE.is_file():
        token_file = DEFAULT_TOKEN_FILE

    _download(repo_id, out, token_file=token_file)
    if not _verify_snapshot(out):
        print("Download completed but verification failed.", file=sys.stderr)
        return 1

    print()
    print("Next steps:")
    print(f"  1. Upload to Drive: MyDrive/VDA_models/hustvl/{out.name}/")
    print(f"  2. Desk / Colab vitmatte_model_path: VDA_models/hustvl/{out.name}")
    print(f"  3. Local folder: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
