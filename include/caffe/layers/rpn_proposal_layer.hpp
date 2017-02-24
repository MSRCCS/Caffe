// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Python code Written by Ross Girshick
// C++ implementation by Lei Zhang
// ------------------------------------------------------------------

#ifndef CAFFE_RPN_PROPOSAL_LAYER_HPP_
#define CAFFE_RPN_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* RPNProposalLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class RPNProposalLayer : public Layer<Dtype> {
 class Point
 {
  public:
   Dtype x, y;
   Point(Dtype _x, Dtype _y) :
       x(_x), y(_y)
   {}
 };
 class Rect
 {
  public:
   Dtype x1, y1, x2, y2;
   Rect() : x1(0), y1(0), x2(0), y2(0)
   {}

   Rect(Dtype _x1, Dtype _y1, Dtype _x2, Dtype _y2) :
       x1(_x1), y1(_y1), x2(_x2), y2(_y2) 
   {}

   Rect Shift(Dtype x, Dtype y)
   {
       return Rect(x1 + x, y1 + y, x2 + x, y2 + y);
   }

   inline Dtype Area()
   {
       return (x2 - x1 + 1) * (y2 - y1 + 1);
   }

   inline Dtype Width()
   {
       return x2 - x1 + 1;
   }

   inline Dtype Height()
   {
       return y2 - y1 + 1;
   }
 };

 vector<Rect> bbox_transform_inv(vector<Rect>& boxes, vector<Rect>& deltas)
 {
     vector<Rect> pred_boxes;
     for (int i = 0; i < boxes.size(); i++)
     {
         Rect &box = boxes[i];
         Dtype width = box.x2 - box.x1 + 1.0;
         Dtype height = box.y2 - box.y1 + 1.0;
         Dtype ctr_x = box.x1 + 0.5 * width;
         Dtype ctr_y = box.y1 + 0.5 * height;

         Rect &delta = deltas[i];
         Dtype dx = delta.x1;
         Dtype dy = delta.y1;
         Dtype dw = delta.x2;
         Dtype dh = delta.y2;

         Dtype pred_ctr_x = dx * width + ctr_x;
         Dtype pred_ctr_y = dy * height + ctr_y;
         Dtype pred_w = exp(dw) * width;
         Dtype pred_h = exp(dh) * height;

         pred_boxes.push_back(Rect(pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
             pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h));
     }
     return pred_boxes;
 }

 public:
  explicit RPNProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RpnProposal"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int feat_stride_;
  vector<Rect> anchors_;
};

}  // namespace caffe

#endif  // CAFFE_RPN_PROPOSAL_LAYER_HPP_
