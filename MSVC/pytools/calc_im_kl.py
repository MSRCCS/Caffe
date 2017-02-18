import argparse
import numpy as np
import os
import base64
import cv2

def calc_kl(cmd):
    img_count = 0
    pixel_count = 0.
    cov = np.zeros((3,3))
    with open(cmd.tsv, 'r') as f_tsv:
        for line in f_tsv:
            cols = line.rstrip().split('\t')
            imagestring = cols[cmd.img_col]
            jpgbytestring = base64.b64decode(imagestring)
            img_array = np.asarray(bytearray(jpgbytestring), dtype=np.uint8)
            im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            data = im.astype(np.float32).reshape(-1, im.shape[2])
            cov += np.cov(data.transpose()) * data.shape[0]
            img_count += 1
            pixel_count += data.shape[0]
            print 'images processed: %d\r' % img_count,
            if img_count >= cmd.max_num:
                break

    cov /= pixel_count
    eigval, eigvec = np.linalg.eig(cov)

    outtsv = os.path.splitext(cmd.tsv)[0] + '.kl.txt'
    with open(outtsv, 'w') as f_out:
        f_out.write('%s\n' % ','.join(['%f'%x for x in eigval]))
        f_out.write('%s\n' % ','.join(['%f'%x for x in eigvec.reshape(-1)]))

    print('\ndone.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', required=True, help='input tsv file')
    parser.add_argument('--img_col', required=True, type=int, help='image column index')
    parser.add_argument('--max_num', default=10000, type=int, help='max number of images to calc')
    return parser.parse_args()

if __name__ == "__main__":
    cmd = parse_args()

    calc_kl(cmd)
