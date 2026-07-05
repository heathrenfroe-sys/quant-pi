@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
py -m quant_pi.main
pause
