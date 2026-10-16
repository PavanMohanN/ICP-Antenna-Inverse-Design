@echo off
REM Builds a standalone Windows executable (dist\design_antenna.exe) - run once on a Windows PC with Python.
python -m pip install --upgrade pyinstaller numpy scipy matplotlib
python -m PyInstaller --onefile --name design_antenna --add-data "model;model" --hidden-import icp_engine --hidden-import rf_circuit design_antenna.py
echo.
echo Done: dist\design_antenna.exe   (usage: design_antenna.exe --band 5 6 --rl 10 --circuit)
pause
