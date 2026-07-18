@echo off
setlocal EnableExtensions
title Trading Simulator
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8000"
set "URL=http://127.0.0.1:8000/"
set "PY=%~dp0venv\Scripts\python.exe"

echo ========================================
echo   Trading Simulator - Local Server
echo ========================================
echo.

if not exist "%PY%" (
    echo [ERROR] Virtual environment not found:
    echo   %PY%
    echo.
    echo Create it once with:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0manage.py" (
    echo [ERROR] manage.py not found in %~dp0
    pause
    exit /b 1
)

:: If server is already running, just open the browser
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Server already running. Opening browser...
    powershell -NoProfile -Command "Start-Process 'http://127.0.0.1:8000/'"
    echo.
    echo You can close this window.
    timeout /t 3 >nul
    exit /b 0
)

echo Starting Django at %URL%
echo Keep this window open while you use the app.
echo Close this window ^(or press Ctrl+C^) to stop the server.
echo.
echo TIP: If the browser does not open, visit: %URL%
echo.

:: Open browser after server has a moment to boot (separate process)
start "Open Trading Simulator" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:8000/'"

"%PY%" manage.py runserver %HOST%:%PORT%

echo.
echo Server stopped.
pause
