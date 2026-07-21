@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\Stop-Asteria.ps1" -AppRoot "%~dp0."
set "ASTERIA_EXIT=%ERRORLEVEL%"
if not "%ASTERIA_EXIT%"=="0" pause
exit /b %ASTERIA_EXIT%
