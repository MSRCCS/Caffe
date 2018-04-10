import argparse
import numpy as np
import os
import base64
import cv2

def calc_kl(cmd):
    img_count = 0
    pixel_count = 0.
    cov = np.zeros((3,3))
    outtsv = os.path.splitext(cmd.tsv)[0] + '.norm.tsv'
    f_out = open(outtsv, 'w') 
    with open(cmd.tsv, 'r') as f_tsv:
        for line in f_tsv:
            cols = line.rstrip().split('\t')
            vecstring = cols[cmd.vec_col]
            featureVectorBytes = base64.b64decode(vecstring)
            featureVectorArray = np.frombuffer(featureVectorBytes, dtype=np.float32)
            featureVectorArray = np.asarray(bytearray(featureVectorBytes), dtype=np.float)
            featureVectorNorm = np.linalg.norm(featureVectorArray)
            f_out.write(line.rstrip()+'\t' + str(featureVectorNorm)+'\n');
            print 'images processed: %d\r' % img_count,
            if img_count >= cmd.max_num:
                break
    f_out.close()
    print('\ndone.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', required=True, help='input tsv file')
    parser.add_argument('--vec_col', required=True, type=int, help='image column index')
    parser.add_argument('--max_num', default=10000, type=int, help='max number of images to calc')
    return parser.parse_args()

if __name__ == "__main__":
    cmd = parse_args()

    calc_kl(cmd)
