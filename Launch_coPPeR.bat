@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%build\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [coPPeR] Error: Virtual environment not found at "%VENV_PY%"
    exit /b 1
)

echo [coPPeR] Starting local inference server...
start "coPPeR Server" /min "%VENV_PY%" -m uvicorn app.inference_server:app --host 127.0.0.1 --port 8088 --log-level info

echo [coPPeR] Waiting for server readiness...
timeout /t 2 /nobreak >nul

echo [coPPeR] Launching WebView2 workspace...
start "coPPeR Workspace" "%VENV_PY%" "%ROOT%app\workspace_app.py"

echo [coPPeR] Done. Server: http://127.0.0.1:8088 | Workspace: Native WebView2 window

