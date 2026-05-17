@echo off
setlocal
cd /d "%~dp0.."

echo SAM3 download for VDA_models (facebook/sam3)
echo Requires: HF access to facebook/sam3 + login once (hf auth login)
echo.

python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
  echo Installing huggingface_hub...
  python -m pip install "huggingface_hub>=0.34"
)

python scripts\download_sam3_for_drive.py %*
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 exit /b %EXITCODE%

echo.
echo Upload folder VDA_models\facebook\sam3 to MyDrive\VDA_models\facebook\sam3
endlocal
exit /b 0
