@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================================
echo  VitaFormer - first-time setup (Windows 10/11, 64-bit)
echo  Downloads: Python 3.11.9 embeddable (~11 MB) and pinned packages
echo  (PyTorch CPU ~200 MB). Nothing else on your computer is changed;
echo  everything is installed inside this folder ("runtime").
echo  Images are never uploaded anywhere.
echo ============================================================
set PYVER=3.11.9
set RT=%~dp0runtime
set REQ=requirements_cpu.txt
if /I "%1"=="gpu" set REQ=requirements_gpu_cu124.txt
if exist "%RT%\python.exe" goto pip
mkdir "%RT%" 2>nul
echo [1/3] Downloading Python %PYVER% ...
powershell -NoProfile -Command "$ProgressPreference='Continue'; Invoke-WebRequest -Uri https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip -OutFile '%RT%\py.zip'" || goto fail
powershell -NoProfile -Command "Expand-Archive -Force '%RT%\py.zip' '%RT%'" || goto fail
del "%RT%\py.zip"
powershell -NoProfile -Command "(Get-Content '%RT%\python311._pth') -replace '#import site','import site' | Set-Content '%RT%\python311._pth'"
echo ..\>> "%RT%\python311._pth"
echo [2/3] Installing pip ...
powershell -NoProfile -Command "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile '%RT%\get-pip.py'" || goto fail
"%RT%\python.exe" "%RT%\get-pip.py" --no-warn-script-location || goto fail
:pip
echo [3/3] Installing pinned packages from %REQ% (this can take several minutes) ...
"%RT%\python.exe" -m pip install --no-warn-script-location -r "%~dp0%REQ%" || goto fail
"%RT%\python.exe" -c "import torch, timm, scipy, pandas, PIL; print('packages OK, torch', torch.__version__)" || goto fail
echo.
echo Setup finished. Start VitaFormer with start_VitaFormer.bat
pause
exit /b 0
:fail
echo.
echo SETUP FAILED. Check the internet connection (or proxy) and run install_windows.bat again;
echo finished steps are skipped. To install without internet, copy a "runtime" folder prepared
echo on another Windows computer with this script into this folder.
pause
exit /b 1
