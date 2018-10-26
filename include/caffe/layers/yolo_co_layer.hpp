#ifndef CAFFE_YOLO_CO_LAYER_HPP_
#define CAFFE_YOLO_CO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Apply CoOccurrence objectness logic to negative objectness samples from RegionTarget
* based on "PFDet: 2nd Place Solution to Open Images Challenge 2018 Object Detection Track"
* Adapted to apply to Yolo as a weakly supervised training method: 
*  if class Y may co-occure inside class X, avoid penalizing Y (predictions) inside X (ground truths).
*  e.g. "Arm" may co-occure inside "Person" => 
*       avoid penalizing a detected "Arm" inside a "Person" ground truth bounding box
*
* @param bottom input Blob vector
*   -# @f$ (N \times A \times H \times W) @f$
*      the objectness probabilities for each spatial dimension and each batch
*   -# @f$ (N \times A \times H \times W) @f$
*      the no-objectness probabilities (for negative samples) for each spatial dimension and each batch
*      if objectness == no-objectness in a pixel, the loss should be zero
*   -# @f$ (N \times G \times 1 \times 1) @f$
*      The labels (each batch has 4 corners, and the 1 class index for each object ground truth)
*      G = (4 + 1) x max_gt
*   -# @f$ (N \times A \times H \times W \times 4) @f$ 
*      generic: @f$ (N \times \prod\limits_{d=1}^{D}S_d \times 4) @f$
*      the (x,y,w,h) bounding boxes for each batch in each of the spatial dimension
*      @f$\prod\limits_{d=1}^{D}S_d@f$ are 1 to D arbitrary spatial dimenstions
*       e.g. @f$ (A \times H \times W) @f$ with @f$D=3@f$
*   -# @f$ (N \times (C + 1) \times A \times H \times W) @f$
*      generic: @f$ (N \times (C + 1) \times \prod\limits_{d=1}^{D}S_d) @f$
*      the predicted probabilities for each of the C classes (and objectness) for each spatial dimension and each batch
* @param top output Blob vector
*   -# @f$ (N \times A \times H \times W) @f$
*      the modified no-objectness probabilities (for negative samples) for each spatial dimension and each batch
*/
template <typename Dtype>
class YoloCoOccurrenceLayer : public Layer<Dtype> {
public:
    explicit YoloCoOccurrenceLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "YoloCoOccurrence";
    }

    virtual inline int ExactNumTopBlobs() const { 
        return 1;
    }

    virtual inline int ExactNumBottomBlobs() const { 
        return 5;
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
    int channels_;
    int classes_;
    int max_gt_;
    map<string, int> labelmap_;
    int* comap_class_cpu_ptr_;
    Blob<int> comap_class_;
    int* comap_offset_cpu_ptr_;
    Blob<int> comap_offset_;
    int* comap_size_cpu_ptr_;
    Blob<int> comap_size_;
    int* comap_cpu_ptr_;
    Blob<int> comap_; // co-occurrence relation heap
    float* comap_thresh_cpu_ptr_;
    Blob<float> comap_thresh_; // class threshold
    float* comap_obj_thresh_cpu_ptr_;
    Blob<float> comap_obj_thresh_; // objectness threshold
    float* comap_ixr_cpu_ptr_;
    Blob<float> comap_ixr_; // intersection ratio

    void load_labelmap(const string &filename);
    void load_comap(const string &filename);
};

}  // namespace caffe

#endif  // CAFFE_YOLO_CO_LAYER_HPP_
