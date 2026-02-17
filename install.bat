@echo off
chcp 65001 >nul

echo ========================================
echo   Grok2API 安装
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/2] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo        已存在，跳过
)

call venv\Scripts\activate.bat

echo [2/2] 安装依赖...
pip install -r requirements.txt -q

echo.
echo ========================================
echo   安装完成，运行 start.bat 启动服务
echo ========================================
echo.
pause
