#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/yolo_bbs_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_yolo_bbs(int batches, int num_anchor, int height, int width,
                                const Dtype* blob_xy_data, const Dtype* blob_wh_data,
                                const Dtype* biases_data,
                                Dtype* bbs_data) {
    CUDA_KERNEL_LOOP(index, batches * num_anchor * height * width) {
        int b = index / (num_anchor * height * width);
        int left = index % (num_anchor * height * width);
        int n = left / (height * width);
        left = left % (height * width);
        int j = left / width;
        int i = left % width;
        Dtype* curr_bbs_data = bbs_data + index * 4;
        int offset_double_bnji = b * (2 * num_anchor) * height * width + n * height * width + j * width + i;
        int offset_double_bnji_next = offset_double_bnji + num_anchor * height * width;
        *(curr_bbs_data + 0) = (*(blob_xy_data + offset_double_bnji) + i) / width;
        *(curr_bbs_data + 1) = (*(blob_xy_data + offset_double_bnji_next) + j) / height;
        double w = *(blob_wh_data + offset_double_bnji);
        double h = *(blob_wh_data + offset_double_bnji_next);
        *(curr_bbs_data + 2) = exp(w) * biases_data[2 * n] / width;
        *(curr_bbs_data + 3) = exp(h) * biases_data[2 * n + 1] / height;
    }
}

template <typename Dtype>
__global__ void kernel_correct_bbs(int total, 
                                   int im_w, int im_h, int netw, int neth,
                                   int new_w, int new_h,
                                   Dtype* bbs_data) {
    CUDA_KERNEL_LOOP(i, total) {
        Dtype x = bbs_data[4 * i + 0];
        Dtype y = bbs_data[4 * i + 1];
        Dtype w = bbs_data[4 * i + 2];
        Dtype h = bbs_data[4 * i + 3];

        x = (x - (netw - new_w) / 2. / netw) / ((Dtype)new_w / netw);
        y = (y - (neth - new_h) / 2. / neth) / ((Dtype)new_h / neth);
        w *= (Dtype)netw / new_w;
        h *= (Dtype)neth / new_h;
        x *= im_w;
        w *= im_w;
        y *= im_h;
        h *= im_h;
        bbs_data[4 * i + 0] = x;
        bbs_data[4 * i + 1] = y;
        bbs_data[4 * i + 2] = w;
        bbs_data[4 * i + 3] = h;
    }
}

template <typename Dtype>
void YoloBBsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];

    auto bbs = top[0];

    int batches = blob_xy->num();
    int height = blob_xy->height();
    int width = blob_xy->width();
    int num_anchor = blob_xy->channels() / 2;

    kernel_yolo_bbs<Dtype> << <CAFFE_GET_BLOCKS(batches * num_anchor * height * width), CAFFE_CUDA_NUM_THREADS >> >(
        batches, num_anchor, height, width,
        blob_xy->gpu_data(), blob_wh->gpu_data(), 
        biases_.gpu_data(),
        bbs->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    if (bottom.size() < 3)
        return;

    auto blob_imageinfo = bottom[2];

    int net_h = feat_stride_ * height;
    int net_w = feat_stride_ * width;

    auto im_info = blob_imageinfo->cpu_data();
    int im_h = im_info[0];
    int im_w = im_info[1];
    // when used for Caffe timing, im_w and im_h might be 0 and we need to give them valid values.
    if (im_w == 0)
        im_w = net_w;
    if (im_h == 0)
        im_h = net_h;

    int new_w = 0;
    int new_h = 0;
    if (((Dtype)net_w / im_w) < ((Dtype)net_h / im_h)) {
        new_w = net_w;
        new_h = (im_h * net_w) / im_w;
    } else {
        new_h = net_h;
        new_w = (im_w * net_h) / im_h;
    }

    kernel_correct_bbs<Dtype> << <CAFFE_GET_BLOCKS(bbs->count() / 4), CAFFE_CUDA_NUM_THREADS >> > (
        bbs->count() / 4,
        im_w, im_h, net_w, net_h,
        new_w, new_h,
        bbs->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloBBsLayer);

}  // namespace caffe
