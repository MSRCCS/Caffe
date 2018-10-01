#ifndef CAFFE_SOFTMAXTREE_LAYER_HPP_
#define CAFFE_SOFTMAXTREE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/tree_common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Computes the softmax function for a taxonomy tree of classes.
 *
 * This is a generalization of softmax (softmax is a tree with only a single group, all roots)
 * IOW softmaxtree can be interpreted as a softmax function that can operate on a dense matrix of sparse groups of channels 
 * (i.e. softmax_axis_ is the flattened sparse matrix).
 * Forward and backward are computed similar to softmax, but per-group of siblings
 */
template <typename Dtype>
class SoftmaxTreeLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxTreeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxTree"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;

public:
  Tree softmax_tree_;

};

}  // namespace caffe

#endif  // CAFFE_SOFTMAXTREE_LAYER_HPP_
