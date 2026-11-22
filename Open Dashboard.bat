@echo off
REM Anime Pipeline - opens the render dashboard in your browser.
REM Starts the dashboard server first if it is not running.
setlocal
set PROJ=%~dp0
set PROJ=%PROJ:~0,-1%

echo Checking dashboard server...
curl -sf --max-time 5 http://localhost:8000/api/chapters >nul 2>&1
if errorlevel 1 (
  echo Starting dashboard server...
  start /min "anime-dashboard" cmd /c "cd /d "%PROJ%\hermes-skill\scripts" && python pipeline_server.py"
  echo Waiting for it to warm up...
  timeout /t 30 /nobreak >nul
  
  echo Verifying server is ready...
  curl -sf --max-time 10 http://localhost:8000/api/chapters >nul 2>&1
  if errorlevel 1 (
    echo WARNING: Server may not be fully ready yet.
  )
)

echo Opening http://localhost:8000/ ...
start "" "http://localhost:8000/"
endlocal
