import argparse
import os
import zipfile

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='install cudnn')
    parser.add_argument('zipfile', help='downloaded cudnn zip file')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    with zipfile.ZipFile(args.zipfile, 'r') as zf:
        zf.extractall('cudnn')

