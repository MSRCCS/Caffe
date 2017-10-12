#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  if (bottom.size() == 3) {
      diff2_.ReshapeLike(*bottom[0]);
      CHECK_EQ(bottom[0]->count(), bottom[2]->count());
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  Dtype dot = 0;
  if (bottom.size() == 2) {
    dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  } else if (bottom.size() == 3) {
    caffe_sqr(count, diff_.cpu_data(), diff2_.mutable_cpu_data());
    dot = caffe_cpu_dot(count, diff2_.cpu_data(), bottom[2]->cpu_data());
  }
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      if (bottom.size() == 2) {
          caffe_cpu_axpby(
                  bottom[i]->count(),              // count
                  alpha,                              // alpha
                  diff_.cpu_data(),                   // a
                  Dtype(0),                           // beta
                  bottom[i]->mutable_cpu_diff());  // b
      } else if (bottom.size() == 3) {
          caffe_mul(bottom[i]->count(), bottom[2]->cpu_data(), diff_.cpu_data(), 
                  diff2_.mutable_cpu_data());
          caffe_cpu_axpby(
                  bottom[i]->count(),              // count
                  alpha,                              // alpha
                  diff2_.cpu_data(),                   // a
                  Dtype(0),                           // beta
                  bottom[i]->mutable_cpu_diff());  // b
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
