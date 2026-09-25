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

echo [INFO] Priming tiktoken cache for offline token counting...
set "TIKTOKEN_CACHE_DIR=%CD%\tiktoken_cache"
if not exist "%TIKTOKEN_CACHE_DIR%" mkdir "%TIKTOKEN_CACHE_DIR%"
venv\Scripts\python.exe -c "import tiktoken; [tiktoken.get_encoding(name) for name in ('cl100k_base','o200k_base')]"
if %ERRORLEVEL% neq 0 (
    echo [WARN] Failed to prime tiktoken cache; the exe may fall back to the character heuristic when offline.
)

echo [INFO] Building optimized EXE with PyInstaller...

venv\Scripts\python.exe -m PyInstaller ^
    --name kgb ^
    --onefile ^
    --windowed ^
    --clean ^
    --paths . ^
    --collect-submodules tools ^
    --collect-submodules ui ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=PySide6.QtSvg ^
    --hidden-import=tiktoken_ext ^
    --hidden-import=tiktoken_ext.openai_public ^
    --exclude-module pytest ^
    --exclude-module _pytest ^
    --exclude-module pluggy ^
    --exclude-module iniconfig ^
    --exclude-module unittest ^
    --exclude-module tkinter ^
    --exclude-module tcl ^
    --exclude-module tk ^
    --exclude-module boto3 ^
    --exclude-module botocore ^
    --exclude-module s3transfer ^
    --exclude-module ast_grep_cli ^
    --exclude-module pyinstaller ^
    --exclude-module pyinstaller_hooks_contrib ^
    --exclude-module altgraph ^
    --exclude-module pefile ^
    --exclude-module matplotlib ^
    --exclude-module scipy ^
    --exclude-module IPython ^
    --exclude-module jupyter ^
    --exclude-module PySide6.QtWebEngine ^
    --exclude-module PySide6.QtWebEngineCore ^
    --exclude-module PySide6.QtWebEngineWidgets ^
    --exclude-module PySide6.QtQml ^
    --exclude-module PySide6.QtQuick ^
    --exclude-module PySide6.Qt3D ^
    --exclude-module PySide6.QtSql ^
    --exclude-module PySide6.QtMultimedia ^
    --exclude-module PySide6.QtPdf ^
    --icon=icon.ico ^
    --add-data "icon.ico;." ^
    --add-data "tiktoken_cache;tiktoken_cache" ^
    main.py

set "BUILD_EXIT=%ERRORLEVEL%"

if %BUILD_EXIT% neq 0 (
    echo [ERROR] Build failed %BUILD_EXIT%
) else (
    echo [SUCCESS] Done! Your EXE is in the 'dist' folder.
)

pause
exit /b %BUILD_EXIT%