#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/wsgm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WSgmLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  WSgmParameter param = this->layer_param_.wsgm_param();
  num_classes_ = param.num_classes();
}

template <typename Dtype>
void WSgmLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void WSgmLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sgm prob values.
  const Dtype* fc7 = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* prob = bottom[2]->cpu_data();
  Dtype loss = 0;
  Dtype probsqr = 0;
  Dtype batch_size = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);

  for (int i = 0; i < batch_size; ++i) {
    Blob<Dtype> sq(1, num_classes_, 1, 1);
    Dtype* sq_data = sq.mutable_cpu_data();
    caffe_sqr(num_classes_, prob + bottom[2]->offset(i), sq_data);
    Dtype sum_pksqr = caffe_cpu_asum<Dtype>(num_classes_, sq_data);
    Blob<Dtype> square(1, num_channels, 1, 1);
    Dtype* square_data = square.mutable_cpu_data();
    caffe_sqr(num_channels, fc7 + bottom[0]->offset(i), square_data);
    Dtype normsqr = caffe_cpu_asum<Dtype>(num_channels, square_data);
    Dtype alpha_i = sum_pksqr + Dtype(1.0) - 2 * prob[i*num_classes_ + int(label[i])];  //(*(prob + bottom[2]->offset(label[i])));
    loss = loss + alpha_i*normsqr;
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
void WSgmLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* fc7_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* fc7 = bottom[0]->cpu_data();
  Dtype batch_size = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);
  Dtype loss_weight = (top[0]->cpu_diff()[0] * 2) / batch_size; // 2 is from dxa = 2*xa
  caffe_copy(int(batch_size)* num_channels, fc7, fc7_diff);
  caffe_scal(int(batch_size)* num_channels, loss_weight, fc7_diff);
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* prob = bottom[2]->cpu_data();
  Dtype* prob_diff = bottom[2]->mutable_cpu_diff();
  for (int i = 0; i < batch_size; ++i) {

    caffe_axpy(num_classes_, Dtype(2.0), prob + bottom[2]->offset(i), prob_diff + bottom[2]->offset(i));
    prob_diff[i*num_classes_ + int(label[i])] = prob_diff[i*num_classes_ + int(label[i])] - 2;


    Blob<Dtype> sq(1, num_classes_, 1, 1);
    Dtype* sq_data = sq.mutable_cpu_data();
    caffe_sqr(num_classes_, prob + bottom[2]->offset(i), sq_data);
    Dtype sum_pksqr = caffe_cpu_asum<Dtype>(num_classes_, sq_data);
    Blob<Dtype> square(1, num_channels, 1, 1);
    Dtype* square_data = square.mutable_cpu_data();
    caffe_sqr(num_channels, fc7 + bottom[0]->offset(i), square_data);
    Dtype normsqr = caffe_cpu_asum<Dtype>(num_channels, square_data);
    Dtype alpha_i = sum_pksqr + Dtype(1.0) - 2 * prob[i*num_classes_ + int(label[i])];

    caffe_scal(num_channels, alpha_i, fc7_diff + bottom[0]->offset(i));
    caffe_scal(num_classes_, normsqr, prob_diff + bottom[2]->offset(i));

  }
  loss_weight = (top[0]->cpu_diff()[0]) / batch_size;
  caffe_scal(int(batch_size)* num_classes_, loss_weight, prob_diff);
}

#ifdef CPU_ONLY
  STUB_GPU(WSgmLossLayer);
#endif
  INSTANTIATE_CLASS(WSgmLossLayer);
  REGISTER_LAYER_CLASS(WSgmLoss);
}  // namespace caffe
