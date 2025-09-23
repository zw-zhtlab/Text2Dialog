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
python -m uvicorn server:app --host 0.0.0.0 --port 8000
