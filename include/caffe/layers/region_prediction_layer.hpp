#ifndef CAFFE_REGION_PREDICTION_LAYER_HPP_
#define CAFFE_REGION_PREDICTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class RegionPredictionLayer : public Layer<Dtype> {
public:
    explicit RegionPredictionLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "RegionPrediction"; }

    virtual inline int ExactNumBottomBlobs() const { return 5; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
    void GetRegionBoxes(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) { NOT_IMPLEMENTED; }
        }
    }
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    int net_w_;
    int net_h_;
    Blob<Dtype> biases_;
    bool class_specific_nms_;
    float thresh_;
    Dtype nms_;
    int feat_stride_;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
