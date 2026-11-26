@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo VitaFormer is not set up yet. Run install_windows.bat first.
  pause
  exit /b 1
)
echo Starting VitaFormer (local only). Your browser will open; keep this window open while you work.
"%~dp0runtime\python.exe" -m vitaformer app
pause
