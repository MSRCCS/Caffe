#include <algorithm>
#include <vector>
#include <cfloat>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/yolo_eval_compat_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_compat_yolo_append(int outer_num, int classes, int inner_num,
                                          int channels, int sum_inner_num, int s_offset, int c_offset,
                                          bool move_axis, bool update_max,
                                          const Dtype* bottom_data,
                                          Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
        // index == n * inner_num + s
        const int n = index / inner_num;
        const int s = index % inner_num;
        auto s2 = s + s_offset;
        auto sum_classes = channels - 1;
        Dtype maxval = -FLT_MAX;
        if (update_max) {
            if (move_axis)
                maxval = top_data[(n * sum_inner_num + s2) * channels + sum_classes];
            else
                maxval = top_data[(n * channels + sum_classes) * sum_inner_num + s2];
        }
        for (int c = 0; c < classes; ++c) {
            auto p = bottom_data[(n * classes + c) * inner_num + s];
            if (p > maxval)
                maxval = p;
            auto c2 = c + c_offset;
            if (move_axis)
                top_data[(n * sum_inner_num + s2) * channels + c2] = p;
            else
                top_data[(n * channels + c2) * sum_inner_num + s2] = p;
        }
        if (move_axis)
            top_data[(n * sum_inner_num + s2) * channels + sum_classes] = maxval;
        else
            top_data[(n * channels + sum_classes) * sum_inner_num + s2] = maxval;
    }
}

template <typename Dtype>
__global__ void kernel_compat_yolo(int outer_num, int classes, int inner_num, 
                                   int channels, int sum_inner_num, int s_offset, int c_offset,
                                   bool move_axis,
                                   const Dtype* bottom_data,
                                   Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * classes * inner_num) {
        int s = index % inner_num;
        int c = (index / inner_num) % classes;
        const int n = (index / inner_num) / classes;
        auto sum_classes = channels - 1;

        auto p = bottom_data[(n * classes + c) * inner_num + s];
        s += s_offset;
        // concatenate objectness
        if (c == classes - 1)
            c = sum_classes;
        else
            c += c_offset;
        if (move_axis)
            top_data[(n * sum_inner_num + s) * channels + c] = p;
        else
            top_data[(n * channels + c) * sum_inner_num + s] = p;
    }
}

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto top_data = top[0]->mutable_gpu_data();
    const int bottom_count = bottom.size();
    if (bottom_count > 1)
        caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

    int channels = sum_classes_ + 1;
    int c_offset = 0;
    int s_offset = 0;
    for (int i = 0; i < bottom_count; ++i) {
        auto bottom_data = bottom[i]->gpu_data();
        auto inner_num = bottom[i]->count(2);
        auto classes = bottom[i]->shape(1);
        if (!append_max_) {
            kernel_compat_yolo << <CAFFE_GET_BLOCKS(outer_num_ * classes * inner_num), CAFFE_CUDA_NUM_THREADS >> > (
                outer_num_, classes, inner_num,
                channels, sum_inner_num_, s_offset, c_offset,
                move_axis_,
                bottom_data,
                top_data);
            classes--;
        } else {
            kernel_compat_yolo_append << <CAFFE_GET_BLOCKS(outer_num_ * sum_inner_num_), CAFFE_CUDA_NUM_THREADS >> > (
                outer_num_, classes, inner_num,
                channels, sum_inner_num_, s_offset, c_offset,
                move_axis_, i > 0,
                bottom_data,
                top_data);
        }
        s_offset += inner_num;
        c_offset += classes;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloEvalCompatLayer);

}  // namespace caffe
