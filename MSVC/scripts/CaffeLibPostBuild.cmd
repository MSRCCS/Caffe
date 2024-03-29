echo off

set SOLUTION_DIR=%~1%
set CONFIGURATION=%~2% 
set OUTPUT_DIR=%~3%

IF /I %CONFIGURATION% == Release (
echo CaffeLibPostBuild.cmd : copy caffe RELEASE dependencies to output.
robocopy "%SOLUTION_DIR%libraries\bin" "%OUTPUT_DIR%" caffe*.dll lib*.dll glog.dll snappy.dll vcruntime140.dll /xo /xf *d.dll *d1.dll >nul
if %errorlevel% geq 8 exit /b 1
robocopy "%SOLUTION_DIR%libraries\lib" "%OUTPUT_DIR%" gflags.dll boost_system*.dll boost_thread*.dll boost_filesystem*.dll boost_python*.dll boost_chrono*.dll /xo /xf *gd*.dll >nul
if %errorlevel% geq 8 exit /b 1
robocopy "%SOLUTION_DIR%libraries\x64\vc14\bin" "%OUTPUT_DIR%" opencv_core*.dll opencv_imgproc*.dll opencv_imgcodecs*.dll opencv_highgui*.dll opencv_videoio*.dll /xo /xf *d.dll >nul
if %errorlevel% geq 8 exit /b 1
) ELSE (
echo CaffeLibPostBuild.cmd : copy caffe DEBUG dependencies to output.
robocopy "%SOLUTION_DIR%libraries\bin" "%OUTPUT_DIR%" caffe*d.dll caffe*d1.dll lib*.dll glogd.dll snappyd.dll vcruntime140.dll /xo >nul
if %errorlevel% geq 8 exit /b 1
robocopy "%SOLUTION_DIR%libraries\lib" "%OUTPUT_DIR%" gflagsd.dll boost_system*gd*.dll boost_thread*gd*.dll boost_filesystem*gd*.dll boost_python*gd*.dll boost_chrono*gd*.dll /xo >nul
if %errorlevel% geq 8 exit /b 1
robocopy "%SOLUTION_DIR%libraries\x64\vc14\bin" "%OUTPUT_DIR%" opencv_core*d.dll opencv_imgproc*d.dll opencv_imgcodecs*d.dll opencv_highgui*d.dll opencv_videoio*d.dll /xo >nul
if %errorlevel% geq 8 exit /b 1
)
robocopy "%SOLUTION_DIR%cudnn\cuda\bin" "%OUTPUT_DIR%" cudnn*.dll /xo >nul
if %errorlevel% geq 8 exit /b 1
robocopy "%SOLUTION_DIR%nccl\bin" "%OUTPUT_DIR%" *.dll /xo >nul
if %errorlevel% geq 8 exit /b 1
exit /b 0
