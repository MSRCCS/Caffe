#!/usr/bin/python
#
# copyright Guillaume Dumont (2016)

import os
import sys
import hashlib
import argparse
import tarfile
import zipfile

from six.moves import urllib
from download_model_binary import reporthook

WIN_DEPENDENCIES_URLS = {
    ('v120', '2.7'):("https://github.com/willyd/caffe-builder/releases/download/v1.0.1/libraries_v120_x64_py27_1.0.1.tar.bz2",
                  "3f45fe3f27b27a7809f9de1bd85e56888b01dbe2"),
    ('v140', '2.7'):("https://github.com/willyd/caffe-builder/releases/download/v1.0.1/libraries_v140_x64_py27_1.0.1.tar.bz2",
                  "427faf33745cf8cd70c7d043c85db7dda7243122"),
    ('v140', '3.5'):("https://github.com/willyd/caffe-builder/releases/download/v1.0.1/libraries_v140_x64_py35_1.0.1.tar.bz2",
                  "1f55dac54aeab7ae3a1cda145ca272dea606bdf9"),
}

NCCL_URL = ("https://github.com/leizhangcn/nccl-windows/releases/download/v1.3.2-1/nccl_vc140_x64_1.3.2.zip",
            "fa14204e3b92117a9c4a8e2099f95f187d76c194")

# function for checking SHA1.
def model_checks_out(filename, sha1):
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1

def download(url, sha1):
    dep_filename = os.path.split(url)[1]
    print("Downloading package ({}). Please wait...".format(dep_filename))
    urllib.request.urlretrieve(url, dep_filename, reporthook)
    if not model_checks_out(dep_filename, sha1):
        print('ERROR: package did not download correctly! Run this again.')
        sys.exit(1)
    print("\nDone.")
    return dep_filename

def download_nccl():
    # Download NCCL library
    url, sha1 = NCCL_URL
    nccl_filename = download(url, sha1)
    # Extract the binaries from the zip file
    with zipfile.ZipFile(nccl_filename, 'r') as zip:
        print("Extracting nccl. Please wait...")
        zip.extractall()
        print("Done.")
    os.remove(nccl_filename)

def main(args_list):
    parser = argparse.ArgumentParser(
        description='Download prebuilt dependencies for windows.')
    parser.add_argument('--msvc_version', default='v140', choices=['v120', 'v140'])

    args = parser.parse_args(args_list)

    assert args.msvc_version == 'v140', 'Only Visual Studio 2015 is supported!'

    # get the appropriate url
    pyver = '{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor)
    assert pyver == '2.7', 'Only Python 2.7 is supported!'
    try:
        url, sha1 = WIN_DEPENDENCIES_URLS[(args.msvc_version, pyver)]
    except KeyError:
        print('ERROR: Could not find url for MSVC version = {} and Python version = {}.\n{}'
              .format(args.msvc_version, pyver,
              'Available combinations are: {}'.format(list(WIN_DEPENDENCIES_URLS.keys()))))
        sys.exit(1)

    # Download dependencies
    dep_filename = download(url, sha1)
    # Extract the binaries from the tar file
    with tarfile.open(dep_filename, 'r:bz2') as tar:
        print("Extracting dependencies. Please wait...")
        tar.extractall()
        print("Done.")
    os.remove(dep_filename)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    download_nccl()