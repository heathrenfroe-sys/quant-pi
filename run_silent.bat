@echo off
cd /d "%~dp0"
start "" /b ".venv\Scripts\pythonw.exe" -m quant_pi.main
exit
