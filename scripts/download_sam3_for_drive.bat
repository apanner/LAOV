@echo off
setlocal
cd /d "%~dp0.."

echo SAM3 download for VDA_models\facebook\sam3
echo.
echo Uses scripts\.hf_token if present, else prompts for token.
echo .hf_token is gitignored - never commit it.
echo.

python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
  echo Installing huggingface_hub...
  python -m pip install "huggingface_hub>=0.34"
)

python scripts\download_sam3_for_drive.py %*
endlocal
