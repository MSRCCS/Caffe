set SOLUTION_DIR=%~1%
set PROTO_DIR=%~2%

SET SRC_PROTO_DIR=%SOLUTION_DIR%src\caffe\proto
set PROTO_TEMP_DIR=%SRC_PROTO_DIR%\temp

echo ProtoCompile.cmd : Create proto temp directory "%PROTO_TEMP_DIR%"
mkdir "%PROTO_TEMP_DIR%"

echo ProtoCompile.cmd : Generating "%PROTO_TEMP_DIR%\caffe.pb.h" and "%PROTO_TEMP_DIR%\caffe.pb.cc"
"%PROTO_DIR%protoc" --proto_path="%SRC_PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%SRC_PROTO_DIR%\caffe.proto"

echo ProtoCompile.cmd : Compare newly compiled caffe.pb.h with existing one
fc /b "%PROTO_TEMP_DIR%\caffe.pb.h" "%SRC_PROTO_DIR%\caffe.pb.h" > NUL

if errorlevel 1 (
    echo ProtoCompile.cmd : Move newly generated caffe.pb.cc/h to "%SRC_PROTO_DIR%"
    move /y "%PROTO_TEMP_DIR%\caffe.pb.h" "%SRC_PROTO_DIR%\caffe.pb.h"
    move /y "%PROTO_TEMP_DIR%\caffe.pb.cc" "%SRC_PROTO_DIR%\caffe.pb.cc"
)

rmdir /S /Q "%PROTO_TEMP_DIR%"