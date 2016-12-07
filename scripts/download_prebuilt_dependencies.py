#!/usr/bin/python
#
# copyright Guillaume Dumont (2016)

import os
import sys
import hashlib
import argparse
import tarfile

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

# function for checking SHA1.
def model_checks_out(filename, sha1):
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download prebuilt dependencies for windows.')
    parser.add_argument('--msvc_version', default='v140', choices=['v120', 'v140'])
    args = parser.parse_args()

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

    dep_filename = os.path.split(url)[1]
    # Download binaries
    print("Downloading dependencies ({}). Please wait...".format(dep_filename))
    urllib.request.urlretrieve(url, dep_filename, reporthook)
    if not model_checks_out(dep_filename, sha1):
        print('ERROR: dependencies did not download correctly! Run this again.')
        sys.exit(1)
    print("\nDone.")

    # Extract the binaries from the tar file
    tar = tarfile.open(dep_filename, 'r:bz2')
    print("Extracting dependencies. Please wait...")
    tar.extractall()
    print("Done.")
    tar.close()
    os.remove(dep_filename)
