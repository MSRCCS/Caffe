echo off

set SOLUTION_DIR=%~1%
set CONFIGURATION=%~2% 
set OUTPUT_DIR=%~3%

echo PyCaffePostBuild.cmd : copy _caffe.pyd to output
robocopy "%SOLUTION_DIR%bin\%CONFIGURATION%" "%OUTPUT_DIR%" _caffe.p* /xo /xn >nul
if %errorlevel% geq 8 exit /b 1
exit /b 0
