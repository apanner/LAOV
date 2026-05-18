@echo off
REM Local test for AI Matte Colab runner (path to batch JSON on disk).
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage: scripts\ai_matte_colab_run.bat path\to\ai_matte_batch.json
  echo Optional: set LAOV_DRIVE_MOUNT=G:\My Drive
  exit /b 1
)

python scripts\ai_matte_colab_run.py --job-json "%~1" --probe-first-frame %*
endlocal
