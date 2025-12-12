@echo off
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo.
    echo Dependencies installed successfully!
) else (
    echo.
    echo Error installing dependencies.
)
pause
