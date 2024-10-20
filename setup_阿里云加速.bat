@echo off

REM 设置虚拟环境的名称
set VENV_NAME=venv

REM 检查虚拟环境是否已经存在
if not exist %VENV_NAME% (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM 激活虚拟环境
call %VENV_NAME%\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM 安装 requirements.txt 中的依赖项
echo Installing dependencies...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)


echo Setup complete.
pause