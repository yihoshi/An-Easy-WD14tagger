@echo off
chcp 65001 >nul
setlocal

REM 设置变量
set outputDir=.\tagged images
set scriptPath=.\tagger.py
set modelDir=.\models
set batchSize=1
set threshold=0.27
set captionExtension=.txt
set venvPath=.\venv
set replaceUnderscores=true
set maxDataLoaderWorkers=4

REM 激活虚拟环境
echo Activating virtual environment...
if not exist "%venvPath%\Scripts\activate" (
    echo Virtual environment activation script not found.
    exit /b 1
)
call %venvPath%\Scripts\activate

REM 获取用户输入的待打标文件夹路径
set /p inputDir=Enter the directory containing images to tag: 

REM 检查用户输入的路径是否存在
if not exist "%inputDir%" (
    echo The specified input directory does not exist: %inputDir%
    deactivate
    exit /b 1
)

REM 运行Python脚本
echo Running the Python script...
python %scriptPath% ^
    %inputDir% ^
    --model_dir %modelDir% ^
    --batch_size %batchSize% ^
    --thresh %threshold% ^
    --caption_extension %captionExtension% ^
    --max_data_loader_n_workers %maxDataLoaderWorkers% ^
    --replace_underscores

REM 检查脚本运行是否成功
if %errorlevel% neq 0 (
    echo Failed to run the Python script.
    goto cleanup
)

echo Script completed successfully!
echo.
echo If you do not want to move the input directory, press Ctrl+C to exit now.
pause


REM 移动整个输入文件夹到指定的输出文件夹
echo Moving the entire input directory to %outputDir%...
move "%inputDir%" "%outputDir%"

echo Input directory moved successfully.
pause

:cleanup
REM 退出虚拟环境
deactivate

endlocal
