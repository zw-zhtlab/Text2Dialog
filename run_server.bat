@echo off
setlocal
cd /d %~dp0
if not exist ".venv" (
  py -3 -m venv .venv
)
call ".venv\Scripts\activate"
python -m pip install --upgrade pip
python -m pip install -r "text2dialog\requirements.txt"
cd text2dialog
if "%TEXT2DIALOG_HOST%"=="" set "TEXT2DIALOG_HOST=127.0.0.1"
if "%TEXT2DIALOG_PORT%"=="" set "TEXT2DIALOG_PORT=8000"
python -m uvicorn server:app --host %TEXT2DIALOG_HOST% --port %TEXT2DIALOG_PORT%
