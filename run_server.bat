@echo off
setlocal
cd /d %~dp0
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)
if not exist ".venv\Scripts\text2dialog-server.exe" (
  ".venv\Scripts\python.exe" -m pip install -e .
  if errorlevel 1 exit /b 1
)
if "%TEXT2DIALOG_HOST%"=="" set "TEXT2DIALOG_HOST=127.0.0.1"
if "%TEXT2DIALOG_PORT%"=="" set "TEXT2DIALOG_PORT=8000"
".venv\Scripts\text2dialog-server.exe" --host "%TEXT2DIALOG_HOST%" --port "%TEXT2DIALOG_PORT%"
