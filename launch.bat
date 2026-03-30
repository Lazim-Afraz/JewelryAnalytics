@echo off
title Jewelry Analytics
color 0A

echo.
echo  ============================================================
echo   💎  Jewelry Portfolio Analytics
echo  ============================================================
echo.

:: ── Change to project directory (same folder as this .bat file) ──────────
cd /d "%~dp0"

:: ── Activate venv if it exists ────────────────────────────────────────────
if exist "%~dp0venv\Scripts\activate.bat" (
    echo  [*] Activating virtual environment...
    call "%~dp0venv\Scripts\activate.bat"
) else if exist "%~dp0.venv\Scripts\activate.bat" (
    echo  [*] Activating virtual environment...
    call "%~dp0.venv\Scripts\activate.bat"
) else (
    echo  [*] No venv found, using system Python
)

:: ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [!] Python not found. Please add Python to PATH.
    pause
    exit /b 1
)

:: ── Start Ollama if not already running ───────────────────────────────────
echo  [*] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo  [*] Starting Ollama in background...
    start "" /min ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo  [*] Ollama already running
)

:: ── Open browser then launch Streamlit ───────────────────────────────────
echo  [*] Opening http://localhost:8501
echo  [*] Press Ctrl+C to stop the server
echo.
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

:: ── Pause on exit so you can read any error messages ─────────────────────
echo.
echo  [!] Server stopped.
pause
