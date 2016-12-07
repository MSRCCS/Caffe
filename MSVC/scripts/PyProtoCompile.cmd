set SOLUTION_DIR=%~1%
set PROTO_DIR=%~2%

SET SRC_PROTO_DIR=%SOLUTION_DIR%src\caffe\proto
set PROTO_TEMP_DIR=%SRC_PROTO_DIR%\temp
set PROTO_OUT_DIR=%SOLUTION_DIR%python\caffe\proto

echo PyProtoCompile.cmd : Create proto temp directory "%PROTO_TEMP_DIR%"
mkdir "%PROTO_TEMP_DIR%"

echo PyProtoCompile.cmd : Generating "%PROTO_TEMP_DIR%\caffe_pb2.py"
"%PROTO_DIR%protoc" --proto_path="%SRC_PROTO_DIR%" --python_out="%PROTO_TEMP_DIR%" "%SRC_PROTO_DIR%\caffe.proto"

echo PyProtoCompile.cmd : Compare newly compiled caffe_pb2.py with existing one
fc /b "%PROTO_TEMP_DIR%\caffe_pb2.py" "%PROTO_OUT_DIR%\caffe_pb2.py" > NUL

if errorlevel 1 (
    echo ProtoCompile-py.cmd : Move newly generated caffe_pb2.py to "%PROTO_OUT_DIR%"
    move /y "%PROTO_TEMP_DIR%\caffe_pb2.py" "%PROTO_OUT_DIR%\caffe_pb2.py"
)

rmdir /S /Q "%PROTO_TEMP_DIR%"