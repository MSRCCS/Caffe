#ifndef CAFFE_XCOV_LOSS_LAYER_HPP_
#define CAFFE_XCOV_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

//#include "caffe/neuron_layers.hpp"

namespace caffe {
/**
 * @brief Cross-Covariance loss layer.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class XCovLossLayer : public LossLayer<Dtype> {
 public:
  explicit XCovLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "XCovLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<Blob<Dtype>*> mean_vec_, temp_vec_;
  Blob<Dtype> mean_0_, mean_1_;
  Blob<Dtype> temp_0_, temp_1_;
  Blob<Dtype> xcov_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> batch_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_XCOV_LOSS_LAYER_HPP_
