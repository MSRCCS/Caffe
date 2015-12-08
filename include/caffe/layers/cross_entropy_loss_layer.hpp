#ifndef CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_
#define CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/* CrossEntropyLossLayer
*/
template <typename Dtype>
class CrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit CrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_CONTRASTIVE_LOSS_LAYER_HPP_
