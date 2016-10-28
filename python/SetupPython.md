#Setup Python Environment for WinCaffe

To use the Python wrapper for Caffe, please follow the below instructions:

1. Install Python 2.7.9 or above, assuming the installed folder is `c:\python27`.
2. Add `c:\python27` and `c:\python27\scripts` to the system environment variable `PATH`, so that `pip.exe` (under `c:\python27\scripts`) can be run easily.
3. Upgrade pip so that new Python packages can be installed correctly.

	```
	python -m pip install --upgrade pip
	```
4. Set system environment variable PYTHON_ROOT = c:\python27, which is needed by Visual Studio to find python libraries.
5. From http://www.lfd.uci.edu/~gohlke/pythonlibs, download and install the following packages:

	```
	pip install numpy-1.11.1+mkl-cp27-cp27m-win_amd64.whl
	pip install scikit_image-0.12.3-cp27-cp27m-win_amd64.whl
	pip install scipy-0.17.1-cp27-cp27m-win_amd64.whl
	pip install matplotlib
	pip install protobuf
	```
   Please notice the specified package versions, which have been tested working correctly with Caffe.
   To check the installed packages and their versions, run

	```
	pip freeze
	```
6. Run `scripts\GeneratePB_py.bat` to generate `python\caffe\proto\caffe_pb2.py`
7. Then built `caffe.python` in `caffe.sln`.

To test if the python wrapper setup is successful, go to `$CaffeRoot$\python` and run `Python.exe`. Then in Python, try `import caffe`. If the `caffe` module is imported without any exception, the setup is done.
