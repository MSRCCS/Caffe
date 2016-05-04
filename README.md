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

## Windows Setup
**Requirements**: Visual Studio 2013 and CUDA 7.5

Once the requirements are satisfied, run these commands
```
git clone git@github.com:MSRCCS/caffe.git
cd caffe
git clone git@github.com:leizhangcn/wincaffe-3rdparty.git 3rdparty
```

Download `cuDNN` [from nVidia website](https://developer.nvidia.com/cudnn). Please specifically select v3 of CuDnn, which is the version that verifies to build with this WinCaffe package. 
Then run `.\scripts\installCuDNN.ps1 $downloadedZipFile` in PowerShell where `$downloadedZipFile` is the path to your downloaded cuDNN file. Example: `.\scripts\installCuDNN.ps1 ~\Downloads\cudnn-7.0-win-x64-v3.0-prod.zip`

Now, you should be able to build `caffe.sln`, except for the caffe.python project. Please build using Microsoft Visual Studio 2013. 

## Python Setup
To build caffe.python and use the python wrapper on Windows, please follow [Python Setup](python/SetupPython.md) to setup the python environment. 

## Development

### Common issues when pulling new commits from BVLC's branch
- If compilation fails: regenerate `caffe.pb.h` and `caffe.pb.cc` files. This can be done by removing `src\caffe\proto\caffe.pb.h` file. The build process will regenerate if this file is missing.
- If linking fails: it's likely that there are new `cpp` files that need to be added to the `caffelib` project.

## Development

### Common issues when syncing from main branch
- If compilation fails: regenerate `caffe.pb.h` and `caffe.pb.cc` files. This can be done by removing `src\caffe\proto\caffe.pb.h` file. The build process will regenerate if this file is missing.
- If linking fails: it's likely that there are new `cpp` files that need to be added to the `caffelib` project so that they are compiled.

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
