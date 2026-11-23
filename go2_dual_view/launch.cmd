@echo off
setlocal
set "GO2_PYTHON=%USERPROFILE%\Desktop\Projects\Go2-Learning\.venv-win\Scripts\python.exe"
if not exist "%GO2_PYTHON%" set "GO2_PYTHON=python"
"%GO2_PYTHON%" "%~dp0viewer.py"
if errorlevel 1 pause
