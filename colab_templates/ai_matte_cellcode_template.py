"""
AI Matte Colab cellcode — clone LAOV, run matte-only (SAM3 + BiRefNet + ViTMatte).
Stored in Drive: VDA_Jobs/code/{job_id}_cellcode.py

Canonical copy: keep in sync with google_desk_app/colab_templates/ (Desk upload source).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime


def setup_ai_matte_cell1(drive_service):
    from google.colab import auth, drive

    print("=" * 60)
    print("AI Matte — Colab setup (SAM3 + BiRefNet + ViTMatte)")
    print("=" * 60)
    print("\n[STEP 1.1] Authenticating and mounting Drive...")
    auth.authenticate_user()
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
    else:
        print("[OK] Drive already mounted")
    print("\n[OK] AI Matte setup complete (Cell 1)")
    return drive_service


def load_ai_matte_config_cell2(drive_service, config_file_id):
    from googleapiclient.http import MediaIoBaseDownload

    print("[DOWNLOAD] Loading AI Matte config from Drive...")
    config_path = "/content/ai_matte_config.json"
    request = drive_service.files().get_media(fileId=config_file_id)
    with open(config_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print("   Downloading: " + str(int(status.progress() * 100)) + "%")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    is_batch = "batch_id" in config and "sequences" in config
    drive_base_path = "/content/drive/MyDrive"
    date_folder = datetime.now().strftime("%Y%m%d")
    logger = logging.getLogger("ai_matte_colab")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    print("\n[OK] Config loaded — batch=" + str(is_batch) + ", date=" + date_folder)
    return config, drive_base_path, date_folder, logger, is_batch, config_path


def _run_cmd(label: str, cmd: list[str], env: dict | None = None) -> None:
    print("\n[COLAB] " + label)
    print("       " + " ".join(cmd))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.stderr:
        print(r.stderr.rstrip())
    if r.returncode != 0:
        raise RuntimeError(label + " failed (exit " + str(r.returncode) + ")")


def _run_colab_host_job(config_path: str, laov_git: str, date_folder: str) -> bool:
    laov_root = "/content/LAOV"
    if os.path.exists(laov_root):
        shutil.rmtree(laov_root)
    _run_cmd("git clone LAOV", ["git", "clone", "--depth", "1", laov_git, laov_root])
    _run_cmd(
        "pip install LAOV",
        [sys.executable, "-m", "pip", "install", "-e", laov_root, "--no-deps"],
    )
    setup_py = os.path.join(laov_root, "scripts", "colab_setup.py")
    if os.path.isfile(setup_py):
        _run_cmd("colab_setup.py (numpy pin + deps)", [sys.executable, setup_py])
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import numpy as np; "
                "assert tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 0), "
                "np.__version__",
            ],
            check=True,
        )
        print("[OK] numpy version OK for LAOV")
    else:
        _run_cmd(
            "pip install LAOV[matte]",
            [sys.executable, "-m", "pip", "install", "-e", laov_root + "[matte]"],
        )
    env = os.environ.copy()
    env["LAOV_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    env["LAOV_RUNTIME_DATE_FOLDER"] = date_folder
    script = os.path.join(laov_root, "scripts", "ai_matte_colab_run.py")
    if not os.path.isfile(script):
        print("[WARN] ai_matte_colab_run.py missing — LAOV repo on git is too old.")
        print("       Set laov_git_url in Desk to a branch that includes scripts/ai_matte_colab_run.py")
        script = os.path.join(laov_root, "scripts", "laov_colab_run.py")
    print("\n[COLAB] " + os.path.basename(script) + " (AI Matte batch)...")
    phase = "sam3"
    try:
        with open(config_path, encoding="utf-8") as _jf:
            _cfg = json.load(_jf)
        phase = str((_cfg.get("shared_settings") or {}).get("matte_pipeline_phase", "sam3"))
    except Exception:
        pass
    cmd = [sys.executable, script, "--job-json", config_path, "--probe-first-frame", "--phase", phase]
    print("[COLAB] matte_pipeline_phase=" + phase)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print("\n[ERROR] ai_matte_colab_run.py exited with code " + str(r.returncode))
        print("        Scroll up in this cell for ERROR / Traceback lines above.")
    return r.returncode == 0


def process_ai_matte_cell3(
    config,
    drive_base_path,
    date_folder,
    logger,
    is_batch,
    config_path,
):
    if not is_batch:
        print("[ERROR] Expected batch config with sequences[]")
        return False
    shared = config.get("shared_settings", {})
    git_url = str(
        os.environ.get("LAOV_GIT_URL")
        or shared.get("laov_git_url")
        or "https://github.com/apanner/LAOV.git"
    )
    phase = str(shared.get("matte_pipeline_phase", "sam3"))
    print("\n[COLAB] AI Matte phase=" + phase)
    try:
        ok = _run_colab_host_job(config_path, git_url, date_folder)
    except Exception as exc:
        print("[ERROR] " + str(exc))
        print(traceback.format_exc())
        ok = False
    if ok:
        print("\n[DONE] AI Matte batch finished.")
    else:
        print("\n[ERROR] Batch failed - scroll up for traceback (SAM3 / models / plate paths).")
    root = shared.get("ai_matte_output_root", "AI_MATTE_output")
    out_path = shared.get("output_folder_path", "VDA_output")
    print(
        "   Outputs: MyDrive/"
        + str(out_path)
        + "/"
        + date_folder
        + "/.../<shot>/matte_sam3/ matte_birefnet/ matte_vitmatte/ (qc/)"
    )
    return ok
