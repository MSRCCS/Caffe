echo off

set SOLUTION_DIR=%~1%
set CONFIGURATION=%~2% 
set OUTPUT_DIR=%~3%

IF /I %CONFIGURATION% == Release (
echo CaffeLibPostBuild.cmd : copy caffe RELEASE dependencies to output.
robocopy "%SOLUTION_DIR%libraries\bin" "%OUTPUT_DIR%" caffe*.dll lib*.dll glog.dll snappy.dll vcruntime140.dll /xo /xn /xf *d.dll *d1.dll >nul
robocopy "%SOLUTION_DIR%libraries\lib" "%OUTPUT_DIR%" gflags.dll boost_system*.dll boost_thread*.dll boost_filesystem*.dll boost_python*.dll boost_chrono*.dll /xo /xn /xf *gd*.dll >nul
robocopy "%SOLUTION_DIR%libraries\x64\vc14\bin" "%OUTPUT_DIR%" opencv_core*.dll opencv_imgproc*.dll opencv_imgcodecs*.dll opencv_highgui*.dll opencv_videoio*.dll /xo /xn /xf *d.dll >nul
robocopy "%SOLUTION_DIR%cudnn\cuda\bin" "%OUTPUT_DIR%" cudnn*.dll /xo /xn >nul
) ELSE (
echo CaffeLibPostBuild.cmd : copy caffe DEBUG dependencies to output.
robocopy "%SOLUTION_DIR%libraries\bin" "%OUTPUT_DIR%" caffe*d.dll caffe*d1.dll lib*.dll glogd.dll snappyd.dll vcruntime140.dll /xo /xn /xf  >nul
robocopy "%SOLUTION_DIR%libraries\lib" "%OUTPUT_DIR%" gflagsd.dll boost_system*gd*.dll boost_thread*gd*.dll boost_filesystem*gd*.dll boost_python*gd*.dll boost_chrono*gd*.dll /xo /xn >nul
robocopy "%SOLUTION_DIR%libraries\x64\vc14\bin" "%OUTPUT_DIR%" opencv_core*d.dll opencv_imgproc*d.dll opencv_imgcodecs*d.dll opencv_highgui*d.dll opencv_videoio*d.dll /xo /xn >nul
robocopy "%SOLUTION_DIR%cudnn\cuda\bin" "%OUTPUT_DIR%" cudnn*.dll /xo /xn  >nul
)
