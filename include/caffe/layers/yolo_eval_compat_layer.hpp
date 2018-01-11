#ifndef CAFFE_YOLO_EVAL_COMPAT_LAYER_HPP_
#define CAFFE_YOLO_EVAL_COMPAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Convert probability shape to old Yolo format
*        Also apply a threshold if provided.
*
* @param bottom input Blob vector (length 3 or 5 if with objectness)
*   -# @f$ (N \times M \times \prod\limits_{d=1}^{D}S_d) @f$
*      the computed probabilities.
*      Either M must be equal to C (the number classes provided as a parameter)
*      Or the argmax indices (corresponding to each member of bottom[0]) must be provided by bottom[1]
*   -# (optional) @f$ (N \times \times 1 \times \prod\limits_{d=1}^{D}S_d) @f$
*      The argmax indices
* @param top output Blob vector (length 1 or 2 if with objectness)
*   -# @f$ (N \times \prod\limits_{d=1}^{D}S_d \times (C + 1)) @f$
*      The converted probabilities compatible with Yolo.
*      C is the number classes provided as a parameter.
*/
template <typename Dtype>
class YoloEvalCompatLayer : public Layer<Dtype> {
public:
    explicit YoloEvalCompatLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "YoloEvalCompat";
    }

    virtual inline int ExactNumTopBlobs() const { 
        return 1;
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

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }
    }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }

    }

private:
    int outer_num_;
    int inner_num_;
    int classes_;
    float threshold_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_EVAL_COMPAT_LAYER_HPP_
