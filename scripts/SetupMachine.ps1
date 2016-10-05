########################################################################################
# instructions for setup caffe training environment on Windows Server 2012 R2
# You can also try it on windows 10, and report issues
# written by Yuxiao Hu (yuxiaohu@microsoft.com)
# last update: 5/16/2016
########################################################################################


# install chocolatey first
Set-ExecutionPolicy Unrestricted
iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))

# confirm installation automatically
choco feature enable -n allowGlobalConfirmation

# install visual studio 2013 from
# \\products\PUBLIC\Archives\USEnglish\Developers\"Visual Studio 2013 (All)"\Ultimate\vs_ultimate.exe

# install Cuda cuda_7.5.18_windows from https://developer.nvidia.com/cuda-downloads, or
# \\ccs-z840-02\Tools\install\cuda_7.5.18_windows.exe

# download cudaDnn SDK v3.0 from https://developer.nvidia.com/cudnn, or \\ccs-z840-02\Tools\install\cudnn-7.0-win-x64-v3.0-prod.zip
# refer to https://github.com/MSRCCS/Caffe/tree/WinCaffe about installation, ping ccsg-msr if you don't have access
# run WinCaffe\scripts\installCuDNN.ps1 $downloadedZipFile  in PowerShell where $downloadedZipFile  is the path to your downloaded cuDNN file. Example:  .\scripts\installCuDNN.ps1 ~\Downloads\cudnn-7.0-win-x64-v3.0-prod.zip 
# (if you haven't got access to https://github.com/MSRCCS/Caffe,) you can also get installCuDNN.ps1 from \\ccs-t7600\src\caffe\scripts  

# (optional)in case Cuda7.5 is installed before Visual Studio, copy files:
# from: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\visual_studio_integration\MSBuildExtensions 
# to: C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\BuildCustomizations
# to make sure Visual Studio is aware of Cuda stuffs

# install misc tools
choco install classic-shell 
choco install conemu
choco install notepadplusplus
choco install cygwin
choco install irfanview 
choco install procexp
choco install dependencywalker  
choco install 7zip.install
choco install as-ssd 

# install build environment
choco install python2 
choco install sourcetree 
choco install github
choco install git-lfs

# optional: to use the python tools in Caffe, refer to https://github.com/MSRCCS/Caffe/blob/WinCaffe/python/SetupPython.md
# the required packages can be found from \\ccs-z840-02\Tools\install
# first add pythonFolder and pythonFolder\scripts to $PATH$
# then add environment variable: PYTHON_ROOT=PythonFolder
Pip install \\ccs-z840-02\tools\install\numpy-1.9.2+mkl-cp27-none-win_amd64.whl
Pip install \\ccs-z840-02\tools\install\scikit_image-0.11.3-cp27-none-win_amd64.whl
Pip install \\ccs-z840-02\tools\install\scipy-0.16.0b2-cp27-none-win_amd64.whl

# enlist WinCaffe branch from https://github.com/MSRCCS/Caffe, 
# if you don't have access to this repo, ping Jinl@microsoft.com 
# e.g. go to e:\src folder, launch gitHub for weindows, make sure you are logged in to github, since Caffe is currently a private repo, then open git-shell and run:
git clone git@github.com:MSRCCS/caffe.git --recursive
# you will need git lfs for the folder caffe\3rdparty, or you can simply copy it directly from \\ccs-t7600\src\caffe\3rdparty to your src\caffe\3rdparty
#install protocal Buffer
pip install protobuf 
# go to your src\caffe folder, Run  scripts\GeneratePB_py.bat  to generate  python\caffe\proto\caffe_pb2.py 
# Then built caffe.sln in Visual Studio 2013, which will take about 20 minutes for a full build, some warnings are expected

# (if there is no build errors) now you are all set to start coding ... :)
