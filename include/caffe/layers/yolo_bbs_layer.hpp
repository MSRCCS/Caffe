#ifndef CAFFE_YOLO_BBS_LAYER_HPP_
#define CAFFE_YOLO_BBS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Computes the Yolo bounding boxex (x,y,w,h) applying Yolo's reversed kernel
*        Optionally also computes the conditional probability with objectness
*
* @param bottom input Blob vector (length 3 or 5 if with objectness)
*   -# @f$ (N \times (2 \times A) \times H \times W) @f$
*      the xy block: 2 raw corrordinates (x,y) for each anchor (A)
*   -# @f$ (N \times (2 \times A) \times H \times W) @f$
*      the wh block: 2 raw corrordinates (w,h) for each anchor (A)
*   -# (optional) @f$ (N \times 2 \times 1 \times 1) @f$
*      the image_info block (height, width) for each batch ( 1 image per-batch)
* @param top output Blob vector (length 1 or 2 if with objectness)
*   -# @f$ (N \times A \times H \times W \times 4) @f$
*      the computed xywh bounding boxes
*/
template <typename Dtype>
class YoloBBsLayer : public Layer<Dtype> {
public:
    explicit YoloBBsLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "YoloBBs";
    }

    virtual inline int ExactNumTopBlobs() const { 
        return 1;
    }

    virtual inline int ExactNumBottomBlobs() const { 
        return -1;
    }

    virtual inline int MinBottomBlobs() const {
        return 2;
    }

    virtual inline int MaxBottomBlobs() const {
        return 3;
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
    int feat_stride_;
    Blob<Dtype> biases_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_BBS_LAYER_HPP_
