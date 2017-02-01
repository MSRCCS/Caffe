import argparse
import os
import sys
import zipfile

def parse_args(args_list):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='install cudnn')
    parser.add_argument('zipfile', help='downloaded cudnn zip file')
    args = parser.parse_args(args_list)

    return args

def main(args_list):
    args = parse_args(args_list)

    print('Installing cudnn...')
    with zipfile.ZipFile(args.zipfile, 'r') as zf:
        zf.extractall('cudnn')
    print('Done.')
	
if __name__ == '__main__':
	main(sys.argv[1:])
