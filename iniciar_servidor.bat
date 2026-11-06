@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.10 -m venv .venv
  if errorlevel 1 goto failure
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failure
.venv\Scripts\python.exe server.py --open
goto end
:failure
echo Nao foi possivel iniciar. Instale Python 3.10 de 64 bits e confira a conexao.
:end
pause
