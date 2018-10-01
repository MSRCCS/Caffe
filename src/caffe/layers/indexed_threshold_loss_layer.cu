#include <vector>

#include "caffe/layers/indexed_threshold_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_threshold_positive(const int num, const int dim,
                                          const float threshold, const float scale, 
                                          const Dtype* index_data, const Dtype* bottom_data,
                                          Dtype* diff_data, Dtype* weight_data) {
    CUDA_KERNEL_LOOP(i, num) {
        auto index = static_cast<int>(index_data[i]);
        if (bottom_data[dim * i + index] < threshold) {
            diff_data[dim * i + index] = scale * (bottom_data[dim * i + index] - threshold);
            weight_data[dim * i + index] = scale;
        } else {
            diff_data[dim * i + index] = 0;
        }
    }
}

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    auto index_data = bottom[1]->gpu_data();
    auto bottom_data = bottom[0]->gpu_data();
    auto diff_data = diff_.mutable_gpu_data();
    auto weights_data = weights_.mutable_gpu_data();
    auto dim = bottom[0]->count(index_axis_ + 1);
    const Dtype alpha = sqrt(null_scale_);
    caffe_gpu_set(count, alpha, weights_data);
    if (null_scale_ == 1) {
        caffe_copy(count, bottom_data, diff_data);
    } else {
        caffe_gpu_axpy(
            count,        // count
            alpha,        // alpha
            bottom_data,  // a
            diff_data);   // b
    }

    kernel_threshold_positive<Dtype> << <CAFFE_GET_BLOCKS(outer_num_),
        CAFFE_CUDA_NUM_THREADS >> >(outer_num_, dim,
                                    threshold_, sqrt(positive_scale_),
                                    index_data, bottom_data, 
                                    diff_data, weights_data);

    Dtype dot;
    caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << type()
            << " Layer cannot backpropagate to index inputs.";
    }
    if (!propagate_down[0])
        return;

    int count = bottom[0]->count();
    caffe_gpu_mul(count, weights_.gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
    // Scale gradiant by loss
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_gpu_axpy(
        count,                           // count
        alpha,                           // alpha
        diff_.gpu_data(),                // a
        bottom[0]->mutable_gpu_diff());  // b
}

INSTANTIATE_LAYER_GPU_FUNCS(IndexedThresholdLossLayer);

}  // namespace caffe
