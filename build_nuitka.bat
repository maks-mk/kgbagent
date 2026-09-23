@echo off
setlocal
chcp 65001 > nul

rem UTF-8
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

rem Проверка: использовать venv при наличии, иначе системный python (для GitHub Actions)
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [INFO] Building optimized EXE with Nuitka...

%PYTHON_EXE% -m nuitka ^
    --onefile ^
    --windows-console-mode=disable ^
    --enable-plugin=pyside6 ^
    --output-filename=kgb.exe ^
    --windows-icon-from-ico=icon.ico ^
    --include-package=tools ^
    --include-package=ui ^
    --include-package=tiktoken_ext ^
    --include-data-files=icon.ico=icon.ico ^
    --assume-yes-for-downloads ^
    --show-progress ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=_pytest ^
    --nofollow-import-to=pluggy ^
    --nofollow-import-to=iniconfig ^
    --nofollow-import-to=unittest ^
    --nofollow-import-to=tkinter ^
    --nofollow-import-to=boto3 ^
    --nofollow-import-to=botocore ^
    --nofollow-import-to=s3transfer ^
    --nofollow-import-to=ast_grep_cli ^
    --nofollow-import-to=pyinstaller ^
    --nofollow-import-to=matplotlib ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=PySide6.QtWebEngine ^
    --nofollow-import-to=PySide6.QtWebEngineCore ^
    --nofollow-import-to=PySide6.QtWebEngineWidgets ^
    --nofollow-import-to=PySide6.QtQml ^
    --nofollow-import-to=PySide6.QtQuick ^
    --nofollow-import-to=PySide6.Qt3D ^
    --nofollow-import-to=PySide6.QtSql ^
    --nofollow-import-to=PySide6.QtMultimedia ^
    --nofollow-import-to=PySide6.QtPdf ^
    main.py

set "BUILD_EXIT=%ERRORLEVEL%"

if %BUILD_EXIT% neq 0 (
    echo [ERROR] Build failed %BUILD_EXIT%
) else (
    echo [SUCCESS] Done
)

exit /b %BUILD_EXIT%