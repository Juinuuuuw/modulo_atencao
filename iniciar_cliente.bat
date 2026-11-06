@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-client\Scripts\python.exe (
  py -3 -m venv .venv-client
  if errorlevel 1 goto failure
)
.venv-client\Scripts\python.exe -m pip install -r requirements-client.txt
if errorlevel 1 goto failure
.venv-client\Scripts\python.exe camera_client.py
goto end
:failure
echo Nao foi possivel iniciar. Confira a instalacao do Python e a conexao.
:end
pause
