#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/dense_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// Bottom is fc7 and labels. Top is dense loss
template <typename Dtype>
void DenseLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DenseLossParameter param = this->layer_param_.dense_loss_param();
  num_classes_ = param.num_classes();
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {

    this->blobs_.resize(1);
    // Initialize the weights
    int num_channels = bottom[0]->shape(1);
    vector<int> weight_shape(2);
    weight_shape[0] = num_classes_;
    weight_shape[1] = num_channels;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.dense_loss_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void DenseLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> sz;
  DenseLossParameter param = this->layer_param_.dense_loss_param();
  num_classes_ = param.num_classes();
  int num_channels = bottom[0]->shape(1);
  sz.push_back(num_classes_);
  sz.push_back(num_channels);
  sz.push_back(1);
  sz.push_back(1);
  mean_.Reshape(sz);
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  const Dtype* mean_data = this->blobs_[0]->cpu_data();
  const Dtype* fc7 = bottom[0]->cpu_data();
  Dtype loss = 0;
  Dtype batch_size = bottom[0]->shape(0);
  const Dtype* label = bottom[1]->cpu_data();
  int num_channels = bottom[0]->shape(1);
  for (int i = 0; i < batch_size; i++)
  {
    Blob<Dtype> sub_blob(1, num_channels, 1, 1);
    Dtype* sub_data = sub_blob.mutable_cpu_data();
    int m_offset = label[i];
    caffe_sub(num_channels, mean_data + this->blobs_[0]->offset(m_offset), fc7 + bottom[0]->offset(i), sub_data);
    Blob<Dtype> padded_square(1, num_channels, 1, 1);
    Dtype* padded_square_data = padded_square.mutable_cpu_data();
    caffe_sqr(num_channels, sub_data, padded_square_data);
    Dtype normsqr = caffe_cpu_asum<Dtype>(num_channels, padded_square_data);
    loss = loss + normsqr;
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
void DenseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* fc7_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* mean_data = this->blobs_[0]->cpu_data();
  Dtype* mean_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* fc7 = bottom[0]->cpu_data();
  int batch_size = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);
  const Dtype* label = bottom[1]->cpu_data();

  Dtype loss_weight = (top[0]->cpu_diff()[0] * 2) / batch_size; // 2 is from dxa = 2*xa
  // dfc7
  for (int i = 0; i < batch_size; i++)
  {
    int m_offset = label[i];
    Blob<Dtype> temp_diff(1, num_channels, 1, 1);
    Dtype* temp_data = temp_diff.mutable_cpu_data();
    caffe_sub(num_channels, fc7 + bottom[0]->offset(i), mean_data + this->blobs_[0]->offset(m_offset), fc7_diff + bottom[0]->offset(i));

    //dmean
    caffe_sub(num_channels, mean_data + this->blobs_[0]->offset(m_offset), fc7 + bottom[0]->offset(i), temp_data);
    caffe_axpy(num_channels, loss_weight, temp_data, mean_diff + this->blobs_[0]->offset(m_offset));

  }
  caffe_scal(batch_size* num_channels, loss_weight, fc7_diff);


}

#ifdef CPU_ONLY
  STUB_GPU(DenseLossLayer);
#endif

INSTANTIATE_CLASS(DenseLossLayer);
REGISTER_LAYER_CLASS(DenseLoss);

}  // namespace caffe
