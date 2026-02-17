@echo off
chcp 65001 >nul

REM 检查虚拟环境
if not exist "venv" (
    echo [错误] 请先运行 install.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo ========================================
echo   Grok2API
echo   http://localhost:8000
echo   管理后台: http://localhost:8000/admin
echo ========================================
echo.

python main.py
pause
