@echo off
REM FedLEASE one-command setup for Windows (cmd.exe / PowerShell).
REM Usage:
REM    setup.bat

setlocal enabledelayedexpansion

echo ==========================================
echo   FedLEASE setup (Windows)
echo ==========================================

REM 1. Python check
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python 3.10+ from python.org first.
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/4] Using Python %PYVER%

REM 2. Virtual env
if not exist ".venv" (
    echo [2/4] Creating virtual environment .venv
    python -m venv .venv
) else (
    echo [2/4] Reusing existing .venv
)

call .venv\Scripts\activate.bat

REM 3. Install dependencies
echo [3/4] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip wheel setuptools >nul

REM Detect NVIDIA GPU
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo     No NVIDIA GPU detected - installing default torch wheels
    pip install torch torchvision
) else (
    echo     CUDA GPU detected - installing torch with CUDA 12.1 wheels
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

pip install -r requirements.txt

REM 4. .env
if not exist ".env" (
    if exist ".env.example" (
        copy /Y .env.example .env >nul
        echo [4/4] Created .env from .env.example - please edit it and add your HF_TOKEN
    )
) else (
    echo [4/4] .env already exists - skipping
)

echo.
echo ==========================================
echo   Setup complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Edit .env and set HF_TOKEN=hf_xxxxxxxx
echo   2. Activate the venv:   .venv\Scripts\activate
echo   3. Run the experiment:  python run.py --preset full
echo.

endlocal
