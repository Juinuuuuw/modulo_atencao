@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  call :find_python
  if not defined PYTHON_EXE goto failure
  echo Criando ambiente com:
  "%PYTHON_EXE%" --version
  "%PYTHON_EXE%" -m venv .venv
  if errorlevel 1 goto failure
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failure
.venv\Scripts\python.exe server.py --open
goto end

:find_python
for %%V in (311 310) do (
  if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python%%V\python.exe"
    goto :eof
  )
)
for %%V in (3.11 3.10) do (
  py -%%V -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
    goto :eof
  )
)
goto :eof

:failure
echo Nao foi possivel iniciar. Instale Python 3.10 ou 3.11 de 64 bits e confira a conexao.
:end
pause
