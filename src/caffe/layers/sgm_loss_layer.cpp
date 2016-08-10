#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/sgm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SgmLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void SgmLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void SgmLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sgm prob values.
  const Dtype* fc7 = bottom[0]->cpu_data();
  Dtype loss = 0;
  Dtype batch_size = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);
  for (int i = 0; i < batch_size; ++i) {
    Blob<Dtype> square(1, num_channels, 1, 1);
    Dtype* square_data = square.mutable_cpu_data();
    caffe_sqr(num_channels, fc7 + bottom[0]->offset(i), square_data);
    Dtype normsqr = caffe_cpu_asum<Dtype>(num_channels, square_data);
    loss += normsqr;
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
void SgmLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* fc7_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* fc7 = bottom[0]->cpu_data();
  int batch_size = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);
  Dtype loss_weight = (top[0]->cpu_diff()[0] * 2) / batch_size; // 2 is from dxa = 2*xa
  caffe_copy(batch_size* num_channels, fc7, fc7_diff);
  caffe_scal(batch_size* num_channels, loss_weight, fc7_diff);

}

#ifdef CPU_ONLY
STUB_GPU(SgmLossLayer);
#endif
INSTANTIATE_CLASS(SgmLossLayer);
REGISTER_LAYER_CLASS(SgmLoss);
}  // namespace caffe
