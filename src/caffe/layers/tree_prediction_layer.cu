#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/tree_prediction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_hierarchical_prob(
    const int outer_num, const int channels, const int inner_num,
    const int label_count, const int* label_data,
    const Dtype* prob_data, const int* parent_data,
    Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * label_count * inner_num) {
        int s = index % inner_num;
        int i = (index / inner_num) % label_count;
        int n = (index / inner_num) / label_count;

        int label_value = label_data[i];
        double p = 1;
        while (label_value >= 0) {
            p *= prob_data[(n * channels + label_value) * inner_num + s];
            label_value = parent_data[label_value];
        }
        top_data[(n * label_count + i) * inner_num + s] = static_cast<Dtype>(p);
    }
}

template <typename Dtype>
__global__ void kernel_argmax(
    const int outer_num, const int inner_num,
    const int label_count, const int* label_data,
    const Dtype* top_data,
    Dtype* argmax_data,
    Dtype* max_data) {
    CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
        // index == n * inner_num + s
        const int n = index / inner_num;
        const int s = index % inner_num;

        int argmax = 0;
        Dtype maxval = -FLT_MAX;
        for (int i = 0; i < label_count; ++i) {
            Dtype prob = top_data[(n * label_count + i) * inner_num + s];
            if (prob > maxval) {
                argmax = label_data[i];
                maxval = prob;
            }
        }

        argmax_data[n * inner_num + s] = argmax;
        if (max_data)
            max_data[n * inner_num + s] = maxval;
    }
}

template <typename Dtype>
__global__ void kernel_top_prediction(
    const int outer_num, const int channels, const int inner_num,
    const Dtype* prob_data, const int* group_offset_data, const int* group_size_data, const int* child_data,
    const float threshold,
    Dtype* top_data, Dtype* argmax_data) {
    CUDA_KERNEL_LOOP(index, outer_num * inner_num) {
        // index == n * inner_num + s
        const int n = index / inner_num;
        const int s = index % inner_num;

        int g = 0; // start from the root
        double p = 1;
        int parent_argmax = 0;
        Dtype parent_p = 0;
        int argmax = 0;
        // Tree search
        do {
            auto offset = group_offset_data[g];
            auto size = group_size_data[g];
            Dtype maxval = -FLT_MAX;
            for (int j = 0; j < size; ++j) {
                Dtype prob = prob_data[(n * channels + offset + j) * inner_num + s];
                if (prob > maxval) {
                    argmax = offset + j;
                    maxval = prob;
                }
            }
            p *= maxval;
            g = child_data[argmax];
            if (p <= threshold) {
                argmax = parent_argmax;
                p = parent_p;
                break;
            }
            parent_p = p;
            parent_argmax = argmax;
        } while (g > 0);

        top_data[n * inner_num + s] = static_cast<Dtype>(p);
        argmax_data[n * inner_num + s] = argmax;
    }
}

template <typename Dtype>
void TreePredictionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    auto argmax_data = top[0]->mutable_gpu_data();
    auto top_data = prob_.mutable_gpu_data();
    auto prob_data = bottom[0]->gpu_data();
    int channels = bottom[0]->shape(axis_);

    if (has_map_) {
        Dtype* max_data = NULL;
        if (top.size() == 3)
            max_data = top[2]->mutable_cpu_data();

        auto parent_data = tree_.parent_.gpu_data();
        auto label_count = label_map_.count();
        auto label_data = label_map_.gpu_data();

        // Find hierarchical probabilities for labels in the map
        kernel_hierarchical_prob<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * label_count * inner_num_),
            CAFFE_CUDA_NUM_THREADS >> >(outer_num_, channels, inner_num_,
                                        label_count, label_data,
                                        prob_data, parent_data,
                                        top_data);
        // Find the argmax
        kernel_argmax<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
            CAFFE_CUDA_NUM_THREADS >> >(outer_num_, inner_num_,
                                        label_count, label_data,
                                        top_data,
                                        argmax_data,
                                        max_data);

        if (top.size() >= 2)
            top[1]->ShareData(prob_);
        return;
    }

    //---------------------------------------------------------------------------
    //                          Top Prediction
    //---------------------------------------------------------------------------
    auto child_data = tree_.child_.gpu_data();
    auto group_size_data = tree_.group_size_.gpu_data();
    auto group_offset_data = tree_.group_offset_.gpu_data();

    kernel_top_prediction<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels, inner_num_,
                                     prob_data, group_offset_data, group_size_data, child_data,
                                     threshold_,
                                     top_data, argmax_data);

    if (top.size() >= 2)
        top[1]->ShareData(prob_);
}

INSTANTIATE_LAYER_GPU_FUNCS(TreePredictionLayer);


}  // namespace caffe
