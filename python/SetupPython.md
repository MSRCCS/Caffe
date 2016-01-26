#Setup Python Environment for WinCaffe

To use the Python wrapper for Caffe, please follow the below instructions:

1. Install Python 2.7.9 or above, assuming the installed folder is `c:\python27`.
2. Add `c:\python27` and `c:\python27\scripts` to the system environment variable `PATH`, so that `pip.exe` (under `c:\python27\scripts`) can be run easily.
3. Set system environment variable PYTHON_ROOT = c:\python27, which is needed by Visual Studio to find python libraries.
4. From http://www.lfd.uci.edu/~gohlke/pythonlibs, download and install the following latest packages:

	```
	Pip install numpy-1.9.2+mkl-cp27-none-win_amd64.whl
	Pip install scipy-0.16.0b2-cp27-none-win_amd64.whl
	Pip install scikit_image-0.11.3-cp27-none-win_amd64.whl
	```
5. Install protobuf: `pip install protobuf`
6. Run `scripts\GeneratePB_py.bat` to generate `python\caffe\proto\caffe_pb2.py`
7. Then built `caffe.python` in `caffe.sln`.

To test if the python wrapper setup is successful, go to `$CaffeRoot$\python` and run `Python.exe`. Then in Python, try `import caffe`. If the `caffe` module is imported without any exception, the setup is done.
