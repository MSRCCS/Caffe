#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmaxtree_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KernelHierarchicalObjectness(
    const int nthreads,
    const int* parent_data, const Dtype* prob_data,
    const Dtype* label, const Dtype* object_data,
    const int dim, const int spatial_dim, const int label_stride,
    const bool has_ignore_label_, const int ignore_label_,
    double* label_prob_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index == n * spatial_dim + s
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;

        int label_value = static_cast<int>(label[n * label_stride]);
        if (has_ignore_label_ && label_value == ignore_label_)
            continue;

        double p = object_data[index];  // Scale by objectness
        while (label_value >= 0) {
            p *= prob_data[n * dim + label_value * spatial_dim + s];
            label_value = parent_data[label_value];
        }

        label_prob_data[index] = p;
    }
}

template <typename Dtype>
__global__ void SoftmaxTreeWithLossForwardGPUWithObjectness(
    const int num,
    const int* parent_data, const Dtype* prob_data,
    const Dtype* label, const double* label_prob_data, int* label_index_data,
    const int dim, const int spatial_dim, const int label_stride,
    const bool has_ignore_label_, const int ignore_label_,
    Dtype* loss_data, Dtype* counts) {
    CUDA_KERNEL_LOOP(n, num) {
        counts[n] = 0;
        loss_data[n] = 0;
        int label_value = static_cast<int>(label[n * label_stride]);
        if (has_ignore_label_ && label_value == ignore_label_)
            continue;
        double max_prob = -1;
        int max_idx = 0;
        // TODO: devise a reduction plan for max
        for (int s = 0; s < spatial_dim; ++s) {
            if (label_prob_data[n * spatial_dim + s] > max_prob) {
                max_prob = label_prob_data[n * spatial_dim + s];
                max_idx = s;
            }
        }
        label_index_data[n] = max_idx;
        while (label_value >= 0) {
            loss_data[n] -= log(max(prob_data[n * dim + label_value * spatial_dim + max_idx], Dtype(FLT_MIN)));
            counts[n]++;
            label_value = parent_data[label_value];
        }
    }
}

template <typename Dtype>
__global__ void SoftmaxTreeWithLossForwardGPU(
    const int nthreads,
    const int* parent_data, const Dtype* prob_data, const Dtype* label, 
    const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    Dtype* loss_data, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index == n * spatial_dim + s
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;
        counts[index] = 0;
        loss_data[index] = 0;
        int label_value = static_cast<int>(label[index]);
        if (has_ignore_label_ && label_value == ignore_label_)
            continue;

        while (label_value >= 0) {
            loss_data[index] -= log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
            counts[index]++;
            label_value = parent_data[label_value];
        }
    }
}

template <typename Dtype>
__global__ void SoftmaxTreeWithLossBackwardGPU(
    const int nthreads,
    const int* parent_data, const int* group_offset_data, const int* group_size_data, const int* group_data,
    const Dtype* label, const Dtype* prob_data, Dtype* bottom_diff,
    const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index == n * spatial_dim + s
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;
        counts[index] = 0;
        int label_value = static_cast<int>(label[index]);
        if (has_ignore_label_ && label_value == ignore_label_)
            continue;
        while (label_value >= 0) {
            int g = group_data[label_value];
            int offset = group_offset_data[g];
            // TODO: Use dynamic parallelism for devices with 3.5 compute capability
            for (int c = 0; c < group_size_data[g]; ++c)
                bottom_diff[n * dim + (offset + c) * spatial_dim + s] = prob_data[n * dim + (offset + c) * spatial_dim + s];

            bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
            counts[index]++;
            label_value = parent_data[label_value];
        }
    }
}

template <typename Dtype>
__global__ void SoftmaxTreeWithLossBackwardGPUWithObjectness(
    const int num,
    const int* parent_data, const int* group_offset_data, const int* group_size_data, const int* group_data,
    const Dtype* label, const int* label_index_data, const Dtype* prob_data, Dtype* bottom_diff,
    const int dim, const int spatial_dim, const int label_stride,
    const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {

    CUDA_KERNEL_LOOP(n, num) {
        int label_value = static_cast<int>(label[n * label_stride]);

        counts[n] = 0;
        if (has_ignore_label_ && label_value == ignore_label_)
            continue;

        int label_spatial_idx = label_index_data[n];
        while (label_value >= 0) {
            int g = group_data[label_value];
            int offset = group_offset_data[g];
            // TODO: Use dynamic parallelism for devices with 3.5 compute capability
            for (int c = 0; c < group_size_data[g]; ++c)
                bottom_diff[n * dim + (offset + c) * spatial_dim + label_spatial_idx] = prob_data[n * dim + (offset + c) * spatial_dim + label_spatial_idx];

            bottom_diff[n * dim + label_value * spatial_dim + label_spatial_idx] -= 1;
            counts[n]++;
            label_value = parent_data[label_value];
        }
    }
}

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    softmaxtree_layer_->Forward(softmaxtree_bottom_vec_, softmaxtree_top_vec_);
    auto prob_data = prob_.gpu_data();
    auto label = bottom[1]->gpu_data();
    auto parent_data = softmaxtree_layer_->softmax_tree_.parent_.gpu_data();

    const int dim = prob_.count() / outer_num_;
    int nthreads = outer_num_ * inner_num_;
    auto loss_data = loss_.mutable_gpu_data();
    auto counts = loss_.mutable_gpu_diff();

    if (with_objectness_) {
        auto object_data = bottom[2]->cpu_data();
        KernelHierarchicalObjectness<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS >> > (nthreads,
                                         parent_data, prob_data,
                                         label, object_data,
                                         dim, inner_num_, objectness_label_stride_,
                                         has_ignore_label_, ignore_label_,
                                         label_prob_.mutable_gpu_data());

        nthreads = outer_num_;
        SoftmaxTreeWithLossForwardGPUWithObjectness<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS >> > (nthreads,
                                         parent_data, prob_data,
                                         label, label_prob_.gpu_data(), label_index_.mutable_gpu_data(),
                                         dim, inner_num_, objectness_label_stride_,
                                         has_ignore_label_, ignore_label_,
                                         loss_data, counts);

    } else {
        SoftmaxTreeWithLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS >> > (nthreads, 
                                         parent_data, prob_data, label, 
                                         dim, inner_num_, 
                                         has_ignore_label_, ignore_label_, 
                                         loss_data, counts);
    }

    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);
    }
}

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }
    if (with_objectness_) {
        if (propagate_down[2]) {
            LOG(FATAL) << type()
                << " Layer cannot backpropagate to objectness inputs.";
        }
    }
    if (!propagate_down[0])
        return;

    auto bottom_diff = bottom[0]->mutable_gpu_diff();
    auto prob_data = prob_.gpu_data();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    auto label = bottom[1]->gpu_data();
    auto parent_data = softmaxtree_layer_->softmax_tree_.parent_.gpu_data();
    auto group_offset_data = softmaxtree_layer_->softmax_tree_.group_offset_.gpu_data();
    auto group_size_data = softmaxtree_layer_->softmax_tree_.group_size_.gpu_data();
    auto group_data = softmaxtree_layer_->softmax_tree_.group_.gpu_data();
    const int dim = prob_.count() / outer_num_;
    int nthreads = outer_num_ * inner_num_;
    auto counts = loss_.mutable_gpu_diff();
    if (with_objectness_) {
        nthreads = outer_num_;
        SoftmaxTreeWithLossBackwardGPUWithObjectness<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS >> > (nthreads,
                                         parent_data, group_offset_data, group_size_data, group_data,
                                         label, label_index_.gpu_data(), prob_data, bottom_diff,
                                         dim, inner_num_, objectness_label_stride_,
                                         has_ignore_label_, ignore_label_, counts);
    } else {
        SoftmaxTreeWithLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS >> > (nthreads,
                                         parent_data, group_offset_data, group_size_data, group_data,
                                         label, prob_data, bottom_diff,
                                         dim, inner_num_,
                                         has_ignore_label_, ignore_label_, counts);
    }
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxTreeWithLossLayer);

}  // namespace caffe
