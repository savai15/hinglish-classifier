@echo off
title Hinglish Complaint Classifier
cd /d "%~dp0"

echo.
echo ================================================
echo   Hinglish Complaint Classifier v1.0
echo ================================================
echo.

:: --------------------------------------------------
::  Step 1: Check Python
:: --------------------------------------------------
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo        ERROR: Python not found!
    echo        Install Python 3.10+ from https://python.org
    echo        Make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo        %%v

:: --------------------------------------------------
::  Step 2: Install Python dependencies
:: --------------------------------------------------
echo [2/6] Installing Python libraries (first time takes ~2 min)...
pip install -r requirements.txt --quiet --disable-pip-version-check 2>nul
if errorlevel 1 (
    echo        Retrying with --user flag...
    pip install -r requirements.txt --user --quiet --disable-pip-version-check 2>nul
)
echo        Python libraries ready!

:: --------------------------------------------------
::  Step 3: Check Node.js
:: --------------------------------------------------
echo [3/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo        ERROR: Node.js not found!
    echo        Install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo        %%v

:: --------------------------------------------------
::  Step 4: Install frontend dependencies
:: --------------------------------------------------
echo [4/6] Installing frontend libraries (first time takes ~1 min)...
cd frontend
if not exist node_modules (
    npm install --silent 2>nul
) else (
    npm install --silent 2>nul
)
cd ..
echo        Frontend libraries ready!

:: --------------------------------------------------
::  Step 5: Kill old processes on our ports
:: --------------------------------------------------
echo [5/6] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING 2^>nul') do taskkill /f /pid %%a >nul 2>&1
timeout /t 1 /nobreak >nul

:: --------------------------------------------------
::  Step 6: Start servers
:: --------------------------------------------------
echo [6/6] Starting servers...
echo.

echo        Starting backend...
start "Backend" /min cmd /c "cd /d "%~dp0" && python -m uvicorn api.main:app --reload --port 8000"

echo        Waiting for backend...
:waitbe
timeout /t 1 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto waitbe
echo        Backend ready!

echo        Starting frontend...
start "Frontend" /min cmd /c "cd /d "%~dp0frontend" && npm run dev"

echo        Waiting for frontend...
:waitfe
timeout /t 1 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 goto waitfe
echo        Frontend ready!

echo.
start http://localhost:5173
echo ================================================
echo   All done! Browser opening...
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API docs:  http://localhost:8000/docs
echo ================================================
echo.
echo   Press any key to stop servers...
pause >nul

:: Kill servers on exit
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING 2^>nul') do taskkill /f /pid %%a >nul 2>&1
echo   Servers stopped.
