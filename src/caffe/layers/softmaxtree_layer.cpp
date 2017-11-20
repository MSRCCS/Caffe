#include <algorithm>
#include <vector>
#include <float.h>

#include "caffe/layers/softmaxtree_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);
    const SoftmaxTreeParameter &softmaxtree_param = this->layer_param().softmaxtree_param();
    softmax_tree_.read(softmaxtree_param.tree().c_str());
    if (softmax_tree_.groups() == 1)
        LOG(WARNING) << "With only a single group in the tree, it is more efficient to use SoftmaxLayer instead of SoftmaxTreeLayer";

#ifndef CPU_ONLY
    // Pre-fetch data
    if (Caffe::mode() == Caffe::GPU) {
        softmax_tree_.group_size_.mutable_gpu_data();
        softmax_tree_.group_offset_.mutable_gpu_data();
    }
#endif
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  int channels = bottom[0]->shape(softmax_axis_);

  // This may requires a reshape layer to reshape to CxA before softmaxtree
  CHECK(channels == softmax_tree_.nodes()) << "Channel count: " << channels << " must match tree node count: " << softmax_tree_.nodes();

  top[0]->ReshapeLike(*bottom[0]);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  if (Caffe::mode() == Caffe::CPU) {
      vector<int> mult_dims(1, channels);
      sum_multiplier_.Reshape(mult_dims);
      caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  auto group_offset_data = softmax_tree_.group_offset_.cpu_data();
  auto group_size_data = softmax_tree_.group_size_.cpu_data();
  auto groups = softmax_tree_.groups();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_; // == channels * inner_num_
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top_data);

  // We need to subtract the per-group max to avoid numerical issues, compute the exp,
  // and then normalize per-group.
  for (int i = 0; i < outer_num_; ++i) {

#pragma omp parallel for
    for (int g = 0; g < groups; ++g) {
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        for (int k = 0; k < inner_num_; ++k) {
            Dtype maxval = -FLT_MAX;
            for (int j = 0; j < size; ++j) {
                if (top_data[(offset + j) * inner_num_ + k] > maxval)
                    maxval = top_data[(offset + j) * inner_num_ + k];
            }
            // TODO: cblas_[s|d]gemm with correct strides can accelerate this, when size is large
            // Subtract the max
            for (int j = 0; j < size; ++j)
                top_data[(offset + j) * inner_num_ + k] -= maxval;
        }
    }

    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);

    // per-group sum after exp, and divide
#pragma omp parallel for
    for (int g = 0; g < groups; ++g) {
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        for (int k = 0; k < inner_num_; ++k) {
            auto sum = caffe_cpu_strided_dot(size, sum_multiplier_.cpu_data(), 1, &top_data[offset * inner_num_ + k], inner_num_);
            // divide by sum
            for (int j = 0; j < size; ++j)
                top_data[(offset + j) * inner_num_ + k] /= sum;
        }
    }

    top_data += dim;
  }
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  auto group_offset_data = softmax_tree_.group_offset_.cpu_data();
  auto group_size_data = softmax_tree_.group_size_.cpu_data();
  auto groups = softmax_tree_.groups();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_; // == channels * inner_num_
  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  for (int i = 0; i < outer_num_; ++i) {
    // compute per-group dot(top_diff, top_data) and subtract them from the bottom diff
#pragma omp parallel for
      for (int g = 0; g < groups; ++g) {
          auto offset = group_offset_data[g];
          auto size = group_size_data[g];
          for (int k = 0; k < inner_num_; ++k) {
              auto dot = caffe_cpu_strided_dot<Dtype>(size, 
                                                      bottom_diff + i * dim + offset * inner_num_ + k, inner_num_, 
                                                      top_data + i * dim + offset * inner_num_ + k, inner_num_);
              // TODO: cblas_[s|d]gemm with correct strides can accelerate this, when size is large
              // Subtract the dot
              for (int j = 0; j < size; ++j)
                  bottom_diff[i * dim + (offset + j) * inner_num_ + k] -= dot;
          }
      }
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxTreeLayer);
#endif

INSTANTIATE_CLASS(SoftmaxTreeLayer);
REGISTER_LAYER_CLASS(SoftmaxTree);

}  // namespace caffe
