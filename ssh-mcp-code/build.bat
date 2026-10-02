@echo off
setlocal
pushd "%~dp0" || exit /b 1
uv run python -m PyInstaller --onefile --name ssh-mcp --collect-all winpty --collect-all ssh_mcp run.py
set "build_exit=%errorlevel%"
popd
exit /b %build_exit%
