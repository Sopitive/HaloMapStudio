@echo off
rem Thin wrapper so build-windows.ps1 can be double-clicked or run from cmd.exe.
rem All arguments are passed through (for example:  build-windows.cmd -Package).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-windows.ps1" %*
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" echo Build failed (exit code %RC%).
rem keep the window open when started by double-click (cmd /c puts "/c" in the command line)
echo %cmdcmdline% | find /i "/c" >nul && pause
exit /b %RC%
