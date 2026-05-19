#!/usr/bin/env python3
"""Download the SAM3 Hugging Face snapshot for Google Drive / Colab.

Requires HF access to https://huggingface.co/facebook/sam3 (accept license once).

Usage:
  scripts\\download_sam3_for_drive.bat
  Uses scripts/.hf_token if present (gitignored), else prompts for paste.

Upload the output folder to Drive as:
  MyDrive/VDA_models/facebook/sam3/
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ID = "facebook/sam3"
REQUIRED_FILES = ("config.json",)
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent / ".hf_token"


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
    p.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="Optional: read token from a gitignored file instead of prompting.",
    )
    return p.parse_args()


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


def _prompt_for_hf_token() -> str:
    print()
    print("=" * 60)
    print("Hugging Face token required (SAM3 is a gated model)")
    print("=" * 60)
    print(f"Confirm access: https://huggingface.co/{REPO_ID}")
    print("Create a Read token: https://huggingface.co/settings/tokens")
    print()
    print("Paste your token below. Characters are hidden. Nothing is saved in git.")
    print("=" * 60)
    token = getpass.getpass("hf token: ").strip()
    if not token.startswith("hf_"):
        print("ERROR: Token should start with hf_", file=sys.stderr)
        raise SystemExit(1)
    return token


def _ensure_hf_login(*, token_file: Path | None = None) -> None:
    from huggingface_hub import HfApi, login
    from huggingface_hub.errors import LocalTokenNotFoundError

    # Already logged in from a previous run on this PC?
    try:
        user = HfApi().whoami()
        name = user.get("name") or user.get("fullname") or "OK"
        print(f"Hugging Face: already logged in as {name}")
        return
    except LocalTokenNotFoundError:
        pass
    except Exception as exc:
        msg = str(exc).lower()
        if "401" not in msg and "403" not in msg and "gated" not in msg:
            raise

    token = os.environ.get("HF_TOKEN", "").strip() or None
    if not token and token_file:
        token = _read_token_file(token_file)
        if token:
            print(f"Using token from {token_file}")

    if not token:
        token = _prompt_for_hf_token()

    login(token=token, add_to_git_credential=False)
    user = HfApi().whoami()
    name = user.get("name") or user.get("fullname") or "OK"
    print(f"Logged in as {name}. Starting download...")
    print()


def _download(out: Path, *, token_file: Path | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('Install: pip install "huggingface_hub>=0.34"', file=sys.stderr)
        raise SystemExit(1) from None

    _ensure_hf_login(token_file=token_file)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {out}")
    print("(Several GB — can take 10–30+ minutes depending on connection.)")
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

    token_file = args.token_file
    if token_file is None and DEFAULT_TOKEN_FILE.is_file():
        token_file = DEFAULT_TOKEN_FILE

    _download(out, token_file=token_file)
    if not _verify_snapshot(out):
        print("Download completed but verification failed — check HF access.", file=sys.stderr)
        return 1

    print()
    print("Next steps:")
    print("  1. Upload to Drive: MyDrive/VDA_models/facebook/sam3/")
    print(f"  2. Local folder: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
