#!/usr/bin/env python3
"""Download the SAM3 Hugging Face snapshot for Google Drive / Colab.

Requires HF access to https://huggingface.co/facebook/sam3 (accept license once).

Usage:
  huggingface-cli login
  python scripts/download_sam3_for_drive.py
  python scripts/download_sam3_for_drive.py --output D:/VDA_models/facebook/sam3

Upload the output folder to Drive as:
  MyDrive/VDA_models/facebook/sam3/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "facebook/sam3"
REQUIRED_FILES = ("config.json",)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SAM3 for VDA_models on Drive.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("VDA_models") / "facebook" / "sam3",
        help="Local folder (upload this tree to MyDrive/VDA_models/facebook/sam3).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check that --output looks like a complete snapshot.",
    )
    return p.parse_args()


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
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    if not weights:
        print(f"MISSING: no .safetensors/.bin weights in {out}")
        ok = False
    else:
        total_mb = sum(f.stat().st_size for f in weights) / (1024 * 1024)
        print(f"OK: {len(weights)} weight file(s), ~{total_mb:.0f} MB")
    if ok:
        print(f"OK: snapshot at {out}")
    return ok


def _ensure_hf_login() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError

    try:
        HfApi().whoami()
        return
    except LocalTokenNotFoundError:
        pass
    except Exception as exc:
        msg = str(exc).lower()
        if "401" not in msg and "403" not in msg and "gated" not in msg:
            raise

    print("No Hugging Face token found on this machine.", file=sys.stderr)
    print("Run ONE of:", file=sys.stderr)
    print("  hf auth login", file=sys.stderr)
    print("  huggingface-cli login", file=sys.stderr)
    print("Or set env HF_TOKEN=hf_...", file=sys.stderr)
    print(f"Also confirm access: https://huggingface.co/{REPO_ID}", file=sys.stderr)
    raise SystemExit(1)


def _download(out: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('Install: pip install "huggingface_hub>=0.34"', file=sys.stderr)
        raise SystemExit(1) from None

    _ensure_hf_login()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {out}")
    print("(This can take several minutes; repo is several GB.)")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print("Download finished.")


def main() -> int:
    args = _parse_args()
    out = args.output.resolve()

    if args.verify_only:
        return 0 if _verify_snapshot(out) else 1

    _download(out)
    if not _verify_snapshot(out):
        print("Download completed but verification failed — check HF login and repo access.", file=sys.stderr)
        return 1

    print()
    print("Next steps:")
    print(f"  1. Upload this folder to Google Drive:")
    print(f"     MyDrive/VDA_models/facebook/sam3/")
    print(f"  2. Local path to upload: {out}")
    print("  3. On Colab, re-run Cell 3 — log should show 'Using Drive SAM3 snapshot: ...'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
