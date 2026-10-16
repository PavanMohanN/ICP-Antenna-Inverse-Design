@echo off
REM Windows launcher: double-click, then type the target frequency in GHz.
python "%~dp0design_antenna.py" %*
pause
