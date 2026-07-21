@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\Start-Asteria.ps1" -AppRoot "%~dp0." -TargetMode packaged
set "ASTERIA_EXIT=%ERRORLEVEL%"
if not "%ASTERIA_EXIT%"=="0" pause
exit /b %ASTERIA_EXIT%
