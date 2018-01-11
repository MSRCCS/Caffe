#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/yolo_eval_compat_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_compat_yolo_map(int outer_num, int classes, int inner_num,
                                        float threshold,
                                        const Dtype* bottom_data,
                                        Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
        // index == n * inner_num + s
        const int n = index / inner_num;
        const int s = index % inner_num;
        Dtype maxval = -FLT_MAX;
        for (int c = 0; c < classes; ++c) {
            auto p = bottom_data[(n * classes + c) * inner_num + s];
            if (p <= threshold)
                p = 0;
            if (p > maxval)
                maxval = p;
            top_data[(n * inner_num + s) * (classes + 1) + c] = p;
        }
        top_data[(n * inner_num + s) * (classes + 1) + classes] = maxval;
    }
}

template <typename Dtype>
__global__ void kernel_compat_yolo_top(int count, int classes, 
                                       float threshold,
                                       const Dtype* bottom_data,
                                       const Dtype* class_data,
                                       Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, count) {
        auto p = bottom_data[index];
        if (p <= threshold)
            p = 0;
        int c = (int)class_data[index];
        top_data[index * (classes + 1) + c] = p;
        top_data[index * (classes + 1) + classes] = p;
    }
}
template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto bottom_data = bottom[0]->gpu_data();
    auto top_data = top[0]->mutable_gpu_data();
    if (bottom.size() == 1) {
        kernel_compat_yolo_map << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> > (
            outer_num_, classes_, inner_num_,
            threshold_,
            bottom_data,
            top_data);
        return;
    }

    auto class_data = bottom[1]->gpu_data();
    int count = bottom[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);

    kernel_compat_yolo_top << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        count, classes_,
        threshold_,
        bottom_data,
        class_data,
        top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloEvalCompatLayer);

}  // namespace caffe
