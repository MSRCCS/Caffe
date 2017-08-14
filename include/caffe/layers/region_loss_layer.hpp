#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
struct network {
    const float* input;
    const float* input_gpu;
    const float* truth;
};

struct tree {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
};

struct layer {
    float* output;
    float* output_gpu;
    float* delta;
    float* delta_gpu;
    float* biases;
    float* cost;
    float loss_weight;
    int batch;
    int inputs, outputs;
    int n;
    int w, h;
    int coords;
    int classes;
    int softmax;
    int truths;
    bool bias_match;
    bool rescore;

    float noobject_scale;
    float object_scale;
    float class_scale;
    float coord_scale;
    float thresh;
    float temperature;

    tree *softmax_tree;

    layer(): output(NULL), output_gpu(NULL), delta(NULL),
            delta_gpu(NULL), biases(NULL), cost(NULL), 
            loss_weight(0), batch(0), inputs(0), outputs(0),
            n(0), w(0), h(0), coords(0), classes(0), softmax(0),
            truths(0), bias_match(0), rescore(0), noobject_scale(0),
            object_scale(0), class_scale(0), coord_scale(0), thresh(0),
            temperature(0), softmax_tree(NULL){}
};

enum ACTIVATION {
    LOGISTIC
};

int entry_index(layer l, int batch, int location, int entry);

/**
 * @brief Computes the region loss as defined in Yolo 9000 @f$
 */
template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
 public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }

protected:
  /// @copydoc RegionLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  Blob<Dtype> output_;
  vector<float> biases_;
  network net_;
  layer l_;
  uint64_t anchor_aligned_images_;

  Dtype* seen_images_;
  
  void prepare_net_layer(network &net, layer &l, const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void forward_for_loss(network &net, layer &l);
};

/**
* @brief Computes the region output as defined in Yolo 9000 @f$
*  bottom[0]: the blob after region_loss_layer computation
*  bottom[1]: im_info, the input image size in the order of (Height, Width)
*  top[0]: bounding boxes in the format of {x,y,w,h} with absolute coord values in original image
*  top[1]: class prob and detection confidence for each bbox.
*          class prob: p(0),...,p(K-1) for K classes
*          det conf: objectiveness score * max(p). If being suppressed by nms, its value will be set to 0.
*/
template <typename Dtype>
class RegionOutputLayer : public Layer<Dtype> {
public:
    explicit RegionOutputLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "RegionOutput"; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }
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
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) { NOT_IMPLEMENTED; }
        }
    }

private:
    Blob<Dtype> output_;
    layer l_;
    int net_w_;
    int net_h_;
    int classes_;
    vector<float> biases_;
    float thresh_;
    float hier_thresh_;
    float nms_;
    int feat_stride_;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
