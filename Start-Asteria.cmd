@echo off
setlocal
set "ASTERIA_SAFE_MODE="
if /I "%~1"=="--safe-mode" set "ASTERIA_SAFE_MODE=-SafeMode"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\Start-Asteria.ps1" -AppRoot "%~dp0." -TargetMode packaged %ASTERIA_SAFE_MODE%
set "ASTERIA_EXIT=%ERRORLEVEL%"
if not "%ASTERIA_EXIT%"=="0" pause
exit /b %ASTERIA_EXIT%
