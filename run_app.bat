@echo off
REM =====================================================
REM  Predict ETA Project - Automated Training & Prediction
REM =====================================================

REM Change directory to the project root (this script's folder)
cd /d "%~dp0"

REM Activate the virtual environment
echo [i] Activating virtual environment...
call .venv\Scripts\activate

REM Train a model (Decision Tree)
echo [i] Training Decision Tree model...
python -m app.main train --algo decision_tree --out models\dt.pkl
if errorlevel 1 (
    echo [!] Training failed.
    exit /b %errorlevel%
)

REM Run prediction
echo [i] Running prediction...
python -m app.main predict --model models\dt.pkl
if errorlevel 1 (
    echo [!] Prediction failed.
    exit /b %errorlevel%
)

echo.
echo [✓] All steps completed successfully.
pause
