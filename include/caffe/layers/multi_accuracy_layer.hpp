#ifndef CAFFE_MULTI_ACCURACY_LAYER_HPP_
#define CAFFE_MULTI_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for multiple classes
 *        classification task, i.e., each sample can be 
 *		  labeled/predicted as multiple classes.
 */
template <typename Dtype>
class MultiAccuracyLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with MultiAccuracyLayer options:
   *		no params are supported for now
   */
  explicit MultiAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // only one top blob is allowed
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy: 
   *      @f$\frac{1}{N} \sum\limits_{n=1}^N |\{ \hat{l}_1,...,\hat{l}_{C_n}\}\cap \{l_1, ..., l_{C_n}\} | @f$, 
   *      where 
   *      @f$\{l_1,...,l_{C_n}\}@f$ is the set containing @f$C_n@f$  ground truth labels of the @f$n@f$-th sample
   *      @f$\{\hat{l}_1,...,\hat{l}_{C_n}\} @f$ is the set containing the top @f$C_n@f$  predicted labels of the @f$n@f$-th sample
   *      @f$|A|@f$ is the number of elements of set @f$A@f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- MultiAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_MULTI_ACCURACY_LAYER_HPP_
