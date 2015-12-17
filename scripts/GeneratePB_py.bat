if exist "../python/caffe/proto/caffe_pb2.py" (
    echo caffe_pb2.py remains the same as before
) else (
    echo caffe_pb2.py is being generated
	..\3rdparty\tools\protoc -I="../src/caffe/proto" --python_out="../src/caffe/proto" "../src/caffe/proto/caffe.proto"
    copy "..\src\caffe\proto\caffe_pb2.py" "..\python\caffe\proto\caffe_pb2.py"
)

