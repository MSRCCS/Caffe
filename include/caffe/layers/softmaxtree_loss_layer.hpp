#ifndef CAFFE_SOFTMAXTREE_WITH_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAXTREE_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmaxtree_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss, 
 *        passing real-valued predictions through a softmaxtree to get 
 *        a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxTreeLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxTreeLayer.
 *
 */
template <typename Dtype>
class SoftmaxTreeWithLossLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    * @param param provides SoftmaxTreeLossParameter softmaxtree_loss_param, with options:
    *  - with_objectness (optional)
    *    Specify if the tree label targets should be computed with objectness (bottom[2]).
    */
  explicit SoftmaxTreeWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxTreeWithLoss"; }
  virtual inline int ExactNumBottomBlobs() const {
      if (this->layer_param_.softmaxtree_loss_param().with_objectness()) {
          return 3;
      }
      return 2; 
  }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { 
      if (this->layer_param_.softmaxtree_loss_param().with_objectness()) {
          return 3;
      }
      return 2;
  }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
      // This is a loss layer with potentially 3 bottoms, only the first one can ever have backward
      return bottom_index == 0;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the softmaxtree loss error gradient w.r.t. the predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]), 
   * nor objectness inputs (bottom[2]) so this method ignores bottom[1] and bottom[2] and 
   * requires !propagate_down[1] && !propagate_down[2], crashing if they are set.
   *
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SoftmaxTreeLayer used to map predictions to a distribution.
  shared_ptr<SoftmaxTreeLayer<Dtype>> softmaxtree_layer_;
  /// prob stores the output probability predictions from the SoftmaxTreeLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxTreeLayer::Forward
  vector<Blob<Dtype>*> softmaxtree_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxTreeLayer::Forward
  vector<Blob<Dtype>*> softmaxtree_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Whether objectness bottom is used for target label index.
  bool with_objectness_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_, objectness_label_stride_;
  /// Used to keep the hierarchical objectness probability
  Blob<double> label_prob_;
  Blob<Dtype> label_index_;
  Blob<Dtype> loss_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
