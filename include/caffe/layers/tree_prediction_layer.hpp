#ifndef CAFFE_TREEPREDICTION_LAYER_HPP_
#define CAFFE_TREEPREDICTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/tree_common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Computes the prediction result for a taxonomy tree of classes.
*
* This layer accepts the output of the TreePredictionLayer and performs an tree search 
* for maximum hierarchical probability starting from the root.
*
* @param bottom input Blob vector
*   -# @f$ (N \times C \times H \times W) @f$
*      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
* @param top output Blob vector (length 2)
*   -# @f$ (N \times 1 \times H \times W) @f$
*      the label with highest hierarchical probability (argmax indices)
*      if a map, only labels in the map is considered
*   -# (optional) @f$ (N \times M \times H \times W) @f$
*      the hierarchical probability for each of the M classes
*      M is 1 if no label map is specified
*      M is the size of the map, if a label map is provided
*      if a map is provided, only the probability of the labels in the map is calculated
*      otherwise, only the top probability is calculated (for which the index is also found)
*   -# (optional) @f$ (N \times 1 \times H \times W) @f$
*      the highest hierarchical probability within the map (for which the label is also found)
*/
template <typename Dtype>
class TreePredictionLayer : public Layer<Dtype> {
public:
    explicit TreePredictionLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "TreePrediction";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 1;
    }
    virtual inline int ExactNumTopBlobs() const {
        return -1;
    }
    virtual inline int MinTopBlobs() const {
        return 1;
    }
    virtual inline int MaxTopBlobs() const {
        if (!this->layer_param_.treeprediction_param().has_map()) {
            return 2;
        }
        return 3;
    }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    /// @brief Not implemented (non-differentiable function)
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }
    }
    /// @brief Not implemented (non-differentiable function)
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }
    }

    int outer_num_;
    int inner_num_;
    int axis_;
    Tree tree_;
    float threshold_; // Hierarchical probability threshold
    bool has_map_;
    Blob<int> label_map_;
    Blob<Dtype> prob_;
};

}  // namespace caffe

#endif  // CAFFE_TreePrediction_LAYER_HPP_
