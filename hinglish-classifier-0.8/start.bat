@echo off
title HinglishAI
cd /d "%~dp0"

echo.
echo ================================================
echo   HinglishAI - Complaint Classifier
echo ================================================
echo.

:: Step 1: Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo        ERROR: Python not found!
    echo        Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
python --version
echo.

:: Step 2: Install Python deps
echo [2/6] Installing Python libraries...
pip install -r requirements.txt --quiet --disable-pip-version-check 2>nul
if errorlevel 1 (
    echo        Retrying with --user...
    pip install -r requirements.txt --user --quiet --disable-pip-version-check 2>nul
)
echo        Done!
echo.

:: Step 3: Check Node
echo [3/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo        ERROR: Node.js not found!
    echo        Install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)
node --version
echo.

:: Step 4: Install frontend deps
echo [4/6] Installing frontend libraries...
cd frontend
call npm install --silent 2>nul
cd ..
echo        Done!
echo.

:: Step 5: Kill old servers
echo [5/6] Cleaning up old processes...
taskkill /f /fi "WINDOWTITLE eq Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Frontend*" >nul 2>&1
timeout /t 1 /nobreak >nul
echo        Done!
echo.

:: Step 6: Start servers
echo [6/6] Starting servers...
echo.

echo        Starting backend on port 8000...
start "Backend" /min python -m uvicorn api.main:app --reload --port 8000

echo        Waiting for backend to start...
:waitbe
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto waitbe
echo        Backend ready!
echo.

echo        Starting frontend on port 5173...
start "Frontend" /min cmd /c "cd frontend && npm run dev"

echo        Waiting for frontend to start...
:waitfe
timeout /t 2 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 goto waitfe
echo        Frontend ready!
echo.

echo ================================================
echo   All done! Opening browser...
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API docs:  http://localhost:8000/docs
echo ================================================
echo.

start http://localhost:5173

echo   Press any key to stop servers...
pause >nul

:: Cleanup
taskkill /f /fi "WINDOWTITLE eq Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Frontend*" >nul 2>&1
echo   Servers stopped.
