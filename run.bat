@echo off
REM Script para executar o aplicativo YOLO com o ambiente virtual
REM Uso: run.bat

echo Iniciando FEI Vision Studio...
venv\Scripts\python.exe main.py

if errorlevel 1 (
    echo.
    echo Erro ao executar o aplicativo!
    pause
)
