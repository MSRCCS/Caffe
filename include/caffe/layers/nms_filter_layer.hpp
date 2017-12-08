#ifndef CAFFE_NMS_FILTER_LAYER_HPP_
#define CAFFE_NMS_FILTER_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies Non-Maximal Suppression filter to bounding box confidence values
 *        Each class (and each batch) will be filtered independently
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times \prod\limits_{d=1}^{D}S_d \times 4) @f$
 *      the (x,y,w,h) bounding boxes for each batch in each of the spatial dimension
 *      @f$\prod\limits_{d=1}^{D}S_d@f$ are 1 to D arbitrary spatial dimenstions
 *      e.g. @f$ (A \times H \times W) @f$ with @f$D=3@f$
 *   -# @f$ (N \times M \times \prod\limits_{d=1}^{D}S_d) @f$
 *      The unfiltered probabilities for each bounding box (for each class in M)
 *      M is the total number of classes, for each bounding box.
 *       if M is 1 then it could be omitted, i.e. @f$ (N \times \prod\limits_{d=1}^{D}S_d) @f$ will be valid.
 *       if M > 1 then the maximum probability among the classes will be used as bounding-box score
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times M \times \prod\limits_{d=1}^{D}S_d) @f$ (the shape will be same as bottom[1])
 *      the probabilities filtered by NMS (low IOU values are zeroed out according to NMS)
 */
template <typename Dtype>
class NMSFilterLayer : public Layer<Dtype> {
public:
    explicit NMSFilterLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "NMSFilter";
    }

    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
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
    float nms_;
    int classes_;
    int channels_;
    int outer_num_;
    int inner_num_;
    Blob<int> idx_;
    Blob<unsigned int> mask_;
};

}  // namespace caffe

#endif  // CAFFE_NMS_FILTER_LAYER_HPP_
