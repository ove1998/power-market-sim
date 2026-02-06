@echo off
REM Strommarkt-Simulation Dashboard Launcher
echo.
echo ========================================
echo   Strommarkt-Simulation Dashboard
echo ========================================
echo.
echo Starte Streamlit Dashboard...
echo.

cd /d "%~dp0"
streamlit run dashboard\app.py

pause
