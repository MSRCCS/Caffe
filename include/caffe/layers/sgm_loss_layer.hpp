#ifndef CAFFE_SGM_LOSS_LAYER_HPP_
#define CAFFE_SGM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Squared Gradient Magnitude(SGM) loss. 
 * This loss is equal to the L2 Norm of the fc7 features
 * Note: The sample specific multiplier for loss as described 
 * in Ross's paper on lowshot learning is yet to be implemented 
 */
template <typename Dtype>
class SgmLossLayer : public LossLayer<Dtype> {
 public:
	 explicit SgmLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SgmLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	  */
  /**
   * @brief Computes the softmax loss error gradient w.r.t. the predictions.
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  */
};

}  // namespace caffe

#endif  // CAFFE_SGM_LOSS_LAYER_HPP_
