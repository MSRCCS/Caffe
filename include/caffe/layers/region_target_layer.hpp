#ifndef CAFFE_REGION_TARGET_LAYER_HPP_
#define CAFFE_REGION_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/region_common.hpp"

namespace caffe {
template <typename Dtype>
class RegionTargetLayer : public Layer<Dtype> {
 public:
  explicit RegionTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionTarget"; }
  virtual inline int ExactNumTopBlobs() const {
      return 6;
  }

  virtual inline int ExactNumBottomBlobs() const {
      return -1;
  }

  virtual inline int MinBottomBlobs() const {
      return 4;
  }

  virtual inline int MaxBottomBlobs() const {
      return 5;
  }

protected:
  /// @copydoc RegionLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  uint64_t anchor_aligned_images_;
  bool rescore_;
  Dtype coord_scale_;
  Dtype positive_thresh_;
  Blob<Dtype> biases_;

  Blob<Dtype> ious_;
  Blob<Dtype> bbs_;
  Blob<int> gt_target_;

  Dtype* seen_images_;
};
}
#endif
