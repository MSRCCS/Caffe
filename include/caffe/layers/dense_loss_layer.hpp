#ifndef CAFFE_DENSE_LOSS_LAYER_HPP_
#define CAFFE_DENSE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/* DenseLossLayer
*/
template <typename Dtype>
class DenseLossLayer : public LossLayer<Dtype> {
public:
  explicit DenseLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseLoss"; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  int num_classes_;
  Blob<Dtype> mean_;
};

}  // namespace caffe

#endif  // CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_
