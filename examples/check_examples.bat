@echo off
setlocal
rem Check a VitaFormer run of a bundled example set.
rem How to use: drag the run folder (for example C:\Users\<you>\VitaFormer_results\20261124-141553_run)
rem onto this file, or run:  check_examples.bat "<run folder>"
chcp 65001 >nul
set "PY=%LOCALAPPDATA%\Programs\VitaFormer\runtime\python.exe"
if not exist "%PY%" set "PY=%ProgramFiles%\VitaFormer\runtime\python.exe"
if not exist "%PY%" (
  echo Could not find the Python runtime installed with VitaFormer.
  echo Expected: %LOCALAPPDATA%\Programs\VitaFormer\runtime\python.exe
  echo If you installed VitaFormer elsewhere, run:
  echo   "<install folder>\runtime\python.exe" "%~dp0check_examples.py" "<run folder>"
  pause
  exit /b 2
)
if "%~1"=="" (
  echo Drag a VitaFormer run folder onto this file, or type its path below.
  set /p RUN="Run folder: "
) else (
  set "RUN=%~1"
)
"%PY%" "%~dp0check_examples.py" "%RUN%"
echo.
pause
