#ifndef CAFFE_SOFTMAXTREEPREDICTION_LAYER_HPP_
#define CAFFE_SOFTMAXTREEPREDICTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/tree_common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Computes the prediction result for a taxonomy tree of classes.
*
* This layer accepts the output of the SoftmaxTreeLayer and performs an tree search 
* for maximum hierarchical probability starting from the root.
*
* @param bottom input Blob vector
*   -# @f$ (N \times C \times H \times W) @f$
*      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
*   -# (optional) @f$ (N \times 1 \times H \times W) @f$
*      the prior probabilities for each spatial dimension and each batch
* @param top output Blob vector (length 2)
*   -# @f$ (N \times M \times H \times W) @f$
*      the probabilities for each of the M classes
*      if append_max property is set M == C + 1 otherwise M == C
*/
template <typename Dtype>
class SoftmaxTreePredictionLayer : public Layer<Dtype> {
public:
    explicit SoftmaxTreePredictionLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "SoftmaxTreePrediction";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return -1;
    }
    virtual inline int MinBottomBlobs() const {
        return 1;
    }
    virtual inline int MaxBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }
    int StackSize() const {
        return stack_size_;
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
    bool append_max_;
    bool with_objectness_;
    bool output_tree_path_;

    Blob<double> stack_parent_p_;
    Blob<int> stack_parent_argmax_;
    Blob<int> stack_g_;

private:
    int find_max_stack_size(int g);
    int stack_size_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAXTREEPREDICTION_LAYER_HPP_
