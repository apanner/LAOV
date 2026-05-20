"""
AI Matte Colab cellcode — clone LAOV, run SAM3 + BiRefNet (+ optional ViTMatte).
Drive: VDA_Jobs/code/{job_id}_cellcode.py
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
    print("AI Matte — mount Drive")
    print("=" * 60)
    auth.authenticate_user()
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
    else:
        print("[OK] Drive already mounted")
    return drive_service


def load_ai_matte_config_cell2(drive_service, config_file_id):
    from googleapiclient.http import MediaIoBaseDownload

    config_path = "/content/ai_matte_config.json"
    request = drive_service.files().get_media(fileId=config_file_id)
    with open(config_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print("   Config download: " + str(int(status.progress() * 100)) + "%")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    is_batch = "batch_id" in config and "sequences" in config
    date_folder = datetime.now().strftime("%Y%m%d")
    logger = logging.getLogger("ai_matte_colab")
    if not logger.handlers:
        h = logging.StreamHandler()
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    print("[OK] Config loaded — batch=" + str(is_batch) + " date=" + date_folder)
    return config, "/content/drive/MyDrive", date_folder, logger, is_batch, config_path


def _run_cmd(label: str, cmd: list[str], env: dict | None = None) -> None:
    print("\n[STEP] " + label)
    print("       " + " ".join(cmd))
    run_env = (env or os.environ.copy())
    run_env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run(cmd, env=run_env)
    if r.returncode != 0:
        raise RuntimeError(label + " failed (exit " + str(r.returncode) + ")")


def _stream_subprocess(cmd: list[str], env: dict) -> int:
    """Stream child stdout/stderr into the notebook (Colab hides buffered output)."""
    run_env = dict(env)
    run_env["PYTHONUNBUFFERED"] = "1"
    if cmd and cmd[0] == sys.executable:
        cmd = [sys.executable, "-u", *cmd[1:]]
    print("       " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 120:
            tail.pop(0)
    code = int(proc.wait())
    if code != 0 and tail:
        print("\n[RUNNER] last lines before exit " + str(code) + ":", flush=True)
        for line in tail[-40:]:
            print(line, end="", flush=True)
    return code


def _run_colab_host_job(config_path: str, laov_git: str, date_folder: str) -> bool:
    laov_root = "/content/LAOV"
    if os.path.exists(laov_root):
        shutil.rmtree(laov_root)

    # 1) Clone LAOV (must include combined-matte EXR export on main)
    _run_cmd("Clone LAOV", ["git", "clone", "--depth", "1", laov_git, laov_root])
    commit = subprocess.run(
        ["git", "-C", laov_root, "log", "-1", "--oneline"],
        capture_output=True,
        text=True,
        check=False,
    )
    print("[LAOV] " + (commit.stdout or commit.stderr or "").strip())
    export_py = os.path.join(laov_root, "src", "live_action_aov", "io", "rgba_matte_export.py")
    with open(export_py, encoding="utf-8") as f:
        if "combined_matte" not in f.read():
            raise RuntimeError(
                "LAOV on GitHub is too old (still per-slot EXR export). "
                "Push latest LAOV main, then re-run Cell 3."
            )

    # 2) Editable install (no deps — colab_setup installs them)
    _run_cmd(
        "pip install LAOV",
        [sys.executable, "-m", "pip", "install", "-e", laov_root, "--no-deps"],
    )

    # 3) AI Matte deps: numpy 1.x + kornia + OIIO
    setup_py = os.path.join(laov_root, "scripts", "colab_setup.py")
    _run_cmd("colab_setup.py --matte", [sys.executable, setup_py, "--matte"])

    # 4) Run batch
    env = os.environ.copy()
    env["LAOV_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    env["LAOV_RUNTIME_DATE_FOLDER"] = date_folder
    script = os.path.join(laov_root, "scripts", "ai_matte_colab_run.py")
    if not os.path.isfile(script):
        raise RuntimeError(
            "ai_matte_colab_run.py missing — git pull LAOV main or set laov_git_url in Desk"
        )

    phase = "stages"
    with open(config_path, encoding="utf-8") as f:
        phase = str((json.load(f).get("shared_settings") or {}).get("matte_pipeline_phase", "stages"))

    cmd = [
        sys.executable,
        script,
        "--job-json",
        config_path,
        "--probe-first-frame",
        "--phase",
        phase,
    ]
    print("\n[STEP] Run batch — matte_pipeline_phase=" + phase)
    exit_code = _stream_subprocess(cmd, env)
    if exit_code != 0:
        print(
            "\n[STOP] ai_matte_colab_run.py exited with code %s (see output above). "
            "Common fixes: SAM3/BiRefNet under MyDrive/VDA_models/, plate path in JSON, "
            "HF token for gated SAM3. Partial batch: re-run phase=refine if matte_sam3/ exists."
            % exit_code,
            flush=True,
        )
    return exit_code == 0


def process_ai_matte_cell3(
    config,
    drive_base_path,
    date_folder,
    logger,
    is_batch,
    config_path,
):
    if not is_batch:
        print("[ERROR] Batch JSON must contain sequences[]")
        return False

    shared = config.get("shared_settings", {})
    git_url = str(
        os.environ.get("LAOV_GIT_URL")
        or shared.get("laov_git_url")
        or "https://github.com/apanner/LAOV.git"
    )
    print("\n" + "=" * 60)
    print("AI Matte batch — clone → deps (kornia) → SAM3 → BiRefNet → QC")
    print("=" * 60)

    try:
        ok = _run_colab_host_job(config_path, git_url, date_folder)
    except Exception as exc:
        print("[ERROR] " + str(exc))
        print(traceback.format_exc())
        ok = False

    out_path = shared.get("output_folder_path", "VDA_output")
    print(
        "\nOutputs: MyDrive/"
        + str(out_path)
        + "/"
        + date_folder
        + "/<shot>/matte_sam3/ matte_birefnet/ (qc/)"
    )
    return ok
