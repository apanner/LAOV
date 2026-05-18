@echo off
setlocal
cd /d "%~dp0.."

echo BiRefNet download for VDA_models\ZhengPeng7\BiRefNet
echo.
echo Model is public on Hugging Face (no license gate).
echo Optional: scripts\.hf_token for faster downloads.
echo.

python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
  echo Installing huggingface_hub...
  python -m pip install "huggingface_hub>=0.34"
)

python scripts\download_birefnet_for_drive.py %*
endlocal
