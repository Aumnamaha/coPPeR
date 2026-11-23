@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%build\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [coPPeR] Error: Virtual environment not found at "%VENV_PY%"
    exit /b 1
)

echo [coPPeR] Launching WebView2 workspace via PowerShell...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%VENV_PY%' -ArgumentList 'app/workspace_app.py' -WorkingDirectory '%ROOT%'"

echo [coPPeR] Workspace launched. Backend at http://127.0.0.1:8088 | WebView2 window opened.

