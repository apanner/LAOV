@echo off
setlocal
cd /d "%~dp0.."
python scripts\download_vitmatte_for_drive.py %*
exit /b %ERRORLEVEL%
