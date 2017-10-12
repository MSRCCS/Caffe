#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  if (bottom.size() == 2) {
      caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  } else if (bottom.size() == 3) {
      caffe_gpu_powx(count, diff_.gpu_data(), Dtype(2), diff2_.mutable_gpu_data());
      caffe_gpu_dot(count, diff2_.gpu_data(), bottom[2]->gpu_data(), &dot);
  }
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      if (bottom.size() == 2) {
          caffe_gpu_axpby(
                  bottom[i]->count(),              // count
                  alpha,                              // alpha
                  diff_.gpu_data(),                   // a
                  Dtype(0),                           // beta
                  bottom[i]->mutable_gpu_diff());  // b
      } else if (bottom.size() == 3) {
          caffe_gpu_mul(bottom[i]->count(), bottom[2]->gpu_data(), diff_.gpu_data(), 
                  diff2_.mutable_gpu_data());
          caffe_gpu_axpby(
                  bottom[i]->count(),              // count
                  alpha,                              // alpha
                  diff2_.gpu_data(),                   // a
                  Dtype(0),                           // beta
                  bottom[i]->mutable_gpu_diff());  // b
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
