@echo off
setlocal
chcp 65001 > nul

rem UTF-8
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found
    pause
    exit /b 1
)

echo [INFO] Building...

venv\Scripts\python.exe -m PyInstaller --name kgb --onefile --windowed --clean --paths . --collect-all tiktoken --collect-all langgraph --collect-all langchain --collect-all langchain_openai --collect-all langchain_google_genai --collect-all langchain_anthropic --collect-all anthropic --collect-all qtawesome --collect-submodules tools --collect-submodules ui --hidden-import=PySide6.QtCore --hidden-import=PySide6.QtGui --hidden-import=PySide6.QtWidgets --hidden-import=PySide6.QtSvg --hidden-import=tiktoken_ext --hidden-import=tiktoken_ext.openai_public --icon=icon.ico --add-data "icon.ico;." main.py

set "BUILD_EXIT=%ERRORLEVEL%"

if %BUILD_EXIT% neq 0 (
    echo [ERROR] Build failed %BUILD_EXIT%
) else (
    echo [SUCCESS] Done
)

pause
exit /b %BUILD_EXIT%