@echo off
REM deploy.bat — push local code to the Pi and restart FINbot.
REM
REM Usage: just double-click this, or run from any terminal in the repo root.
REM
REM Requires:
REM   - Pi reachable at finbot.local (set via Pi Imager hostname)
REM   - SSH key auth set up (run `ssh-copy-id finbot@finbot.local` once,
REM     or accept the password prompt each time)
REM   - rsync available on Windows — comes with Git for Windows / cygwin
REM
REM What it does:
REM   1. rsync the entire repo (excluding .venv, __pycache__, *.db) to ~/quant-pi
REM   2. SSH in and restart the systemd service
REM   3. Tail the last 20 log lines so you can see boot messages

setlocal

REM Override these via environment variables, e.g.:  set PI_HOST=you@192.168.1.50
if "%PI_HOST%"=="" set PI_HOST=pi@raspberrypi.local
if "%PI_PATH%"=="" set PI_PATH=/home/pi/quant-pi

echo.
echo === Deploying to %PI_HOST%:%PI_PATH% ===
echo.

rsync -avz --delete ^
    --exclude=".venv" ^
    --exclude="__pycache__" ^
    --exclude="*.pyc" ^
    --exclude="*.db" ^
    --exclude="*.db-journal" ^
    --exclude=".git" ^
    --exclude="quant_pi.db*" ^
    ./ %PI_HOST%:%PI_PATH%/

if %ERRORLEVEL% neq 0 (
    echo.
    echo === rsync failed. Check that Pi is reachable: ping %PI_HOST% ===
    pause
    exit /b 1
)

echo.
echo === Restarting finbot service on Pi ===
echo.

ssh %PI_HOST% "sudo systemctl restart finbot && sleep 2 && journalctl -u finbot -n 20 --no-pager"

echo.
echo === Done. Check the DSI screen for the new dashboard. ===
echo.

endlocal
