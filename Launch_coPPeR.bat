@echo off
REM coPPeR v2 - One-Click Windows Launcher
cd /d "%~dp0"
echo [coPPeR] Starting local inference server...
start "coPPeR Server" cmd /c "build\.venv\Scripts\python.exe -m uvicorn app.inference_server:app --host 127.0.0.1 --port 8088 --log-level info"
timeout /t 3 >nul
echo [coPPeR] Launching WebView2 workspace...
start "coPPeR Workspace" cmd /c "build\.venv\Scripts\python.exe app\workspace_app.py"
echo [coPPeR] Done. Server: http://127.0.0.1:8088 | Workspace: Native WebView2 window
