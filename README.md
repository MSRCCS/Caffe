# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

**Note**: This repo is ported from `git@github.com:MSRDL/caffe.git` for research development. The **master** branch follows BVLC/Caffe master, and the **WinCaffe** branch will merge the latest changes from **master** and keep it compilable in Windows.

## Prerequisite
1. Visual Studio 2015
2. Python 2.7 - Anaconda is recommended: https://www.continuum.io/downloads
3. Cuda 8.0
 
## Windows Setup
1. Clone the repository:

   ```
   git clone https://github.com/MSRCCS/Caffe.git
   ```
2. Download 3rd party dependencies - under the caffe root folder, run:

   ```
   python .\scripts\download_prebuilt_dependencies.py
   ```
3. Download `cuDNN v5.0` [from nVidia website](https://developer.nvidia.com/cudnn). Please select v5 of CuDnn, which is the version that verifies to build with this WinCaffe package. 
   Then under the caffe root folder, run:
   
   ```
   python .\scripts\install_cudnn.py $downloadedZipFile
   ```
   where `$downloadedZipFile` is the path to your downloaded cuDNN file.
4. Set system environment variable `PYTHON_ROOT = path\to\your\python\root`, which is needed by Visual Studio to find python libraries.

Now, you should be able to build `caffe.sln` in Visual Studio 2015.

### Common issues when pulling new commits from BVLC's branch
- If linking fails: it's likely that there are new `cpp` files that need to be added to the `caffelib` project.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
