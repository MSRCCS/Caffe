#include <algorithm>
#include <vector>
#include <cfloat>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/yolo_eval_compat_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_compat_yolo_append(int outer_num, int classes, int inner_num, int channels,
                                          bool move_axis, 
                                          const Dtype* bottom_data,
                                          Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
        // index == n * inner_num + s
        const int n = index / inner_num;
        const int s = index % inner_num;
        Dtype maxval = -FLT_MAX;
        for (int c = 0; c < classes; ++c) {
            auto p = bottom_data[(n * classes + c) * inner_num + s];
            if (p > maxval)
                maxval = p;
            if (move_axis)
                top_data[(n * inner_num + s) * channels + c] = p;
            else
                top_data[(n * channels + c) * inner_num + s] = p;
        }
        if (move_axis)
            top_data[(n * inner_num + s) * channels + classes] = maxval;
        else
            top_data[(n * channels + classes) * inner_num + s] = maxval;
    }
}

template <typename Dtype>
__global__ void kernel_compat_yolo_move(int count, int classes, int inner_num, int channels,
                                        const Dtype* bottom_data,
                                        Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, count) {
        const int s = index % inner_num;
        const int c = (index / inner_num) % classes;
        const int n = (index / inner_num) / classes;

        auto p = bottom_data[(n * classes + c) * inner_num + s];
        top_data[(n * inner_num + s) * channels + c] = p;
    }
}

template <typename Dtype>
__global__ void kernel_compat_yolo_top(int count, int classes, int inner_num, int channels,
                                       bool move_axis, bool append_max,
                                       const Dtype* bottom_data,
                                       const Dtype* class_data,
                                       Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, count) {
        auto p = bottom_data[index];
        int c = (int)class_data[index];
        if (move_axis) {
            top_data[index * channels + c] = p;
            if (append_max)
                top_data[index * channels + classes] = p;
        } else {
            // index == n * inner_num_ + s
            const int n = index / inner_num;
            const int s = index % inner_num;

            top_data[(n * channels + c) * inner_num + s] = p;
            if (append_max)
                top_data[(n * channels + classes) * inner_num + s] = p;
        }
    }
}
template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto bottom_data = bottom[0]->gpu_data();
    auto top_data = top[0]->mutable_gpu_data();
    int count = bottom[0]->count();
    int channels = classes_;
    if (append_max_)
        channels = classes_ + 1;

    if (bottom.size() == 1) {
        if (!append_max_) {
            if (!move_axis_) {
                // Do not move the axis, nor append max column
                caffe_copy(count, bottom_data, top_data);
                return;
            }
            kernel_compat_yolo_move << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
                count, classes_, inner_num_, channels,
                bottom_data,
                top_data);
            return;
        }
        kernel_compat_yolo_append << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_), CAFFE_CUDA_NUM_THREADS >> > (
            outer_num_, classes_, inner_num_, channels,
            move_axis_,
            bottom_data,
            top_data);
        return;
    }

    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

    kernel_compat_yolo_top << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        count, classes_, inner_num_, channels,
        move_axis_, append_max_,
        bottom_data,
        bottom[1]->gpu_data(),
        top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloEvalCompatLayer);

}  // namespace caffe
