#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/softmaxtree_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_subtract_max(const int num, const int channels, const int spatial_dim, const int groups,
                                    const int* group_offset_data, const int* group_size_data, Dtype* data) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        Dtype maxval = -FLT_MAX;
        for (int j = 0; j < size; ++j) {
            if (data[(n * channels + offset + j) * spatial_dim + s] > maxval)
                maxval = data[(n * channels + offset + j) * spatial_dim + s];
        }
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // Subtract the max
        for (int j = 0; j < size; ++j)
            data[(n * channels + offset + j) * spatial_dim + s] -= maxval;
    }
}

template <typename Dtype>
__global__ void kernel_div_sum(const int num, const int channels, const int spatial_dim, const int groups,
                               const int* group_offset_data, const int* group_size_data, Dtype* data) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        Dtype sum = 0;
        for (int j = 0; j < size; ++j)
            sum += data[(n * channels + offset + j) * spatial_dim + s];
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // divide by sum
        for (int j = 0; j < size; ++j)
            data[(n * channels + offset + j) * spatial_dim + s] /= sum;
    }
}

template <typename Dtype>
__global__ void kernel_subtract_dot(const int num, const int channels, const int spatial_dim, const int groups,
                                    const int* group_offset_data, const int* group_size_data, 
                                    const Dtype* data_1, const Dtype* data_2, Dtype* out) {
    CUDA_KERNEL_LOOP(index, num * groups * spatial_dim) {
        int s = index % spatial_dim;
        int g = (index / spatial_dim) % groups;
        int n = (index / spatial_dim) / groups;
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        Dtype dot = 0;
        for (int j = 0; j < size; ++j) {
            dot += (data_1[(n * channels + offset + j) * spatial_dim + s]
                    * data_2[(n * channels + offset + j) * spatial_dim + s]);
        }
        // TODO: Use dynamic parallelism for devices with 3.5 compute capability
        // subtract the dot
        for (int j = 0; j < size; ++j)
            out[(n * channels + offset + j) * spatial_dim + s] -= dot;
    }
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  auto group_offset_data = softmax_tree_.group_offset_.gpu_data();
  auto group_size_data = softmax_tree_.group_size_.gpu_data();
  const auto groups = softmax_tree_.groups();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the per-group max to avoid numerical issues, compute the exp,
  // and then per-group normalize.
  kernel_subtract_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * groups * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, groups,
                                group_offset_data, group_size_data, top_data);
  // exponentiate
  caffe_gpu_exp<Dtype>(count, top_data, top_data);
  // per-group sum after exp, and divide
  kernel_div_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * groups * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, groups, 
                                group_offset_data, group_size_data, top_data);
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  auto group_offset_data = softmax_tree_.group_offset_.gpu_data();
  auto group_size_data =softmax_tree_.group_size_.gpu_data();
  const auto groups = softmax_tree_.groups();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  // Compute per-group inner1d(top_diff, top_data) and subtract them from the bottom diff.
  kernel_subtract_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * groups * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, groups,
                                group_offset_data, group_size_data, 
                                top_diff, top_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(count, bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxTreeLayer);


}  // namespace caffe
