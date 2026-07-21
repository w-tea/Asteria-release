@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\Stop-Asteria.ps1" -AppRoot "%~dp0."
if not "%ERRORLEVEL%"=="0" (
  pause
  exit /b %ERRORLEVEL%
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\Repair-Asteria.ps1" -AppRoot "%~dp0." -ConfirmDownload
set "ASTERIA_EXIT=%ERRORLEVEL%"
if not "%ASTERIA_EXIT%"=="0" pause
exit /b %ASTERIA_EXIT%
