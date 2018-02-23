#ifndef CAFFE_CCA_LOSS_LAYER_HPP_
#define CAFFE_CCA_LOSS_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  /*
    @brief Classification Center Angular Loss implementation.

    Example:
    layer {
      name: "fc_w"
      type: "Parameter"
      top: "fc_w"
      param { name: "fc_w" }
      parameter_param { shape { dim: 1024 dim: ??? } }
    }
    layer {
      name: "loss"
      type: "CCALoss"
      bottom: "prev_fc"
      bottom: "fc_w"
      bottom: "label"
      top: "loss"
    }
   */
  template <typename Dtype>
  class CCALossLayer : public LossLayer<Dtype> {
  public:
    explicit CCALossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "CCALoss"; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    virtual inline int ExactNumBottomBlobs() const { return 3; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // Intermidiate blob to hold temporary results.
    Blob<Dtype> temp_buffer_;
  };
}

#endif
