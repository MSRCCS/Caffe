#ifndef CAFFE_SOFTMAXTREE_LAYER_HPP_
#define CAFFE_SOFTMAXTREE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

struct Tree {
    int *leaf;
    int n; // Total number of nodes in the tree
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int* group_size_cpu_ptr;
    int* group_offset_cpu_ptr;
    shared_ptr<SyncedMemory> group_size;
    shared_ptr<SyncedMemory> group_offset;

public:
    Tree() : leaf(NULL), parent(NULL), child(NULL),
        group(NULL), name(NULL), groups(0),
        group_size_cpu_ptr(NULL), group_offset_cpu_ptr(NULL),
        group_size(), group_offset() {
    }
};

/**
 * @brief Computes the softmax_tree function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
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
  Tree softmax_tree_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAXTREE_LAYER_HPP_
