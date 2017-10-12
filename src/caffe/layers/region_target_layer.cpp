
#include <vector>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/region_target_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void RegionTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Layer<Dtype>::LayerSetUp(bottom, top);

    const RegionTargetParameter &region_param = this->layer_param().region_target_param();
    if (region_param.biases_size() > 0) {
        this->biases_.Reshape(region_param.biases_size(), 1, 1, 1);
        for (int i = 0; i < region_param.biases_size(); ++i) {
            *(this->biases_.mutable_cpu_data() + i) = region_param.biases(i);
        }
        CHECK(this->biases_.count() % 2 == 0) << "the number of biases must be even: " << this->biases_.count();
    }

    this->positive_thresh_ = region_param.thresh();
    CHECK_GE(this->positive_thresh_, 0);
    this->rescore_ = region_param.rescore();

    this->coord_scale_ = region_param.coord_scale();
    
    anchor_aligned_images_ = region_param.anchor_aligned_images();
    CHECK_GE(anchor_aligned_images_, 0);
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    seen_images_ = this->blobs_[0]->mutable_cpu_data();
    *seen_images_ = 0;

    CHECK_EQ(bottom.size(), 4);
    CHECK_EQ(top.size(), 6);
}

template <typename Dtype>
void RegionTargetLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];
    auto blob_obj = bottom[blob_idx++];
    auto blob_truth = bottom[blob_idx++];

    blob_idx = 0;
    auto target_xy = top[blob_idx++];
    auto target_wh = top[blob_idx++];
    auto target_xywh_weight = top[blob_idx++];
    auto target_obj_obj = top[blob_idx++];
    auto target_obj_noobj = top[blob_idx++];
    auto target_class = top[blob_idx++];

    target_xy->ReshapeLike(*blob_xy);
    target_wh->ReshapeLike(*blob_wh);
    target_xywh_weight->ReshapeLike(*blob_xy);
    target_obj_obj->ReshapeLike(*blob_obj);
    target_obj_noobj->ReshapeLike(*blob_obj);
    target_class->ReshapeLike(*blob_obj);

    int num = blob_xy->num();
    int num_anchor = blob_xy->channels() / 2;
    int height = blob_xy->height();
    int width = blob_xy->width();
    CHECK_EQ(blob_wh->num(), num);
    CHECK_EQ(blob_wh->channels(), 2 * num_anchor);
    CHECK_EQ(blob_wh->height(), height);
    CHECK_EQ(blob_wh->width(), width);
    CHECK_EQ(blob_obj->num(), num);
    CHECK_EQ(blob_obj->channels(), num_anchor);
    CHECK_EQ(blob_obj->height(), height);
    CHECK_EQ(blob_obj->width(), width);

    int num_gt = blob_truth->channels() / 5;
    CHECK_EQ(5 * num_gt, blob_truth->channels());
    CHECK_EQ(blob_truth->num(), num);

    ious_.Reshape({num, num_anchor, height, width, num_gt});
    bbs_.Reshape({num, num_anchor, height, width, 4});
    gt_target_.Reshape(num, num_gt, 3, 1);
}

template <typename Dtype>
void RegionTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    seen_images_ = this->blobs_[0]->mutable_cpu_data();
    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];
    auto blob_obj = bottom[blob_idx++];
    auto blob_truth = bottom[blob_idx++];

    blob_idx = 0;
    auto target_xy = top[blob_idx++];
    auto target_wh = top[blob_idx++];
    auto target_xywh_weight = top[blob_idx++];
    auto target_obj_obj = top[blob_idx++];
    auto target_obj_noobj = top[blob_idx++];
    auto target_class = top[blob_idx++];

    auto target_xy_data = target_xy->mutable_cpu_data();
    auto target_wh_data = target_wh->mutable_cpu_data();
    auto target_xywh_weight_data = target_xywh_weight->mutable_cpu_data();
    auto target_obj_noobj_data = target_obj_noobj->mutable_cpu_data();
    auto target_obj_obj_data = target_obj_obj->mutable_cpu_data();
    auto target_class_data = target_class->mutable_cpu_data();

    auto biases = this->biases_.cpu_data();
    auto iou_data = this->ious_.mutable_cpu_data();
    auto bbs_data = this->bbs_.mutable_cpu_data();
    auto gt_target_data = this->gt_target_.mutable_cpu_data();

    if ((*seen_images_) * Caffe::solver_count() < this->anchor_aligned_images_) {
        // if it is at the very begiining, let's align the output
        caffe_set(target_xy->count(), Dtype(0.5), target_xy_data);
        caffe_set(target_wh->count(), Dtype(0), target_wh_data);
        caffe_set(target_xywh_weight->count(), Dtype(0.01), target_xywh_weight_data);
    } else {
        // by default, we set the target of xywh as itself, that mean 0 penalty
        caffe_copy(target_xy->count(), blob_xy->cpu_data(), target_xy_data);
        caffe_copy(target_wh->count(), blob_wh->cpu_data(), target_wh_data);
        caffe_set(target_xywh_weight->count(), (Dtype)0., target_xywh_weight_data);
    }
    
    // for no-objectiveness, by default all of them be 0. we will zero-out the
    // position if it is 1) gt or 2) the predicted result is good enought
    caffe_set(target_obj_noobj->count(), (Dtype)0, target_obj_noobj_data);
    
    // For this one, we will only pernalize the position which should be
    // responsible for the gt
    caffe_copy(target_obj_obj->count(), blob_obj->cpu_data(), target_obj_obj_data);

    // by default, dont penalize the results
    caffe_set(target_class->count(), (Dtype)-1, target_class_data);

    caffe_set(this->ious_.count(), (Dtype)0, iou_data);

    int batches = blob_xy->num();
    int height = blob_xy->height();
    int width = blob_xy->width();
    int num_anchor = blob_xy->channels() / 2;

    const int max_gt = blob_truth->channels() / 5;
    CHECK_EQ(blob_truth->height(), 1);
    CHECK_EQ(blob_truth->width(), 1);

    for (int b = 0; b < batches; b++) {
        for (int n = 0; n < num_anchor; n++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    *(bbs_data + this->bbs_.offset({b, n, j, i, 0})) = 
                        (blob_xy->data_at(b, n, j, i) + i) / width;
                    *(bbs_data + this->bbs_.offset({b, n, j, i, 1})) = 
                        (blob_xy->data_at(b, n + num_anchor, j, i) + j) / height;
                    *(bbs_data + this->bbs_.offset({b, n, j, i, 2})) = 
                        exp(blob_wh->data_at(b, n, j, i)) * biases[2 * n] / width;
                    *(bbs_data + this->bbs_.offset({b, n, j, i, 3})) = 
                        exp(blob_wh->data_at(b, n + num_anchor, j, i)) * biases[2 * n + 1] / height;
                }
            }
        }
    }

    // calculate the IOU
    for (int b = 0; b < batches; ++b) {
        for (int n = 0; n < num_anchor; ++n) {
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    Dtype px = bbs_.data_at({b, n, j, i, 0});
                    Dtype py = bbs_.data_at({b, n, j, i, 1});
                    Dtype pw = bbs_.data_at({b, n, j, i, 2});
                    Dtype ph = bbs_.data_at({b, n, j, i, 3});
                    for (int t = 0; t < max_gt; ++t) {
                        Dtype tx = blob_truth->data_at(b, t * 5 + 0, 0, 0);
                        Dtype ty = blob_truth->data_at(b, t * 5 + 1, 0, 0);
                        Dtype tw = blob_truth->data_at(b, t * 5 + 2, 0, 0);
                        Dtype th = blob_truth->data_at(b, t * 5 + 3, 0, 0);
                        Dtype curr_iou = 0;
                        if (tx) {
                            curr_iou = TBoxIou<Dtype>(px, py, pw, ph, 
                                    tx, ty, tw, th);
                        }
                        vector<int> index = {b, n, j, i, t};
                        *(iou_data + ious_.offset(index)) = curr_iou; 
                    }
                }
            }
        }
    }
    
    // if the iou is large enough, let's not penalize the objectiveness
    for (int b =0; b < batches; b++) {
        for (int n = 0; n < num_anchor; n++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    for (int t = 0; t < max_gt; t++) {
                        vector<int> index = {b, n, j, i, t};
                        auto curr_iou = *(iou_data + ious_.offset(index));
                        if (curr_iou > this->positive_thresh_) {
                            *(target_obj_noobj_data + target_obj_noobj->offset(b, n, j, i)) = 
                                blob_obj->data_at(b, n, j, i);
                            break;
                        }
                    }
                }
            }
        }
    }
    

    caffe_set(this->gt_target_.count(), -1, gt_target_data);
    
    for (int b = 0; b < batches; b++) {
        for (int t = 0; t < max_gt; ++t) {
            Dtype tx = blob_truth->data_at(b, t * 5 + 0, 0, 0);
            Dtype ty = blob_truth->data_at(b, t * 5 + 1, 0, 0);

            int target_i = -1;
            int target_j = -1;
            int target_n = -1;
            if (tx) {
                target_i = tx * width;
                target_j = ty * height;
                Dtype tw = blob_truth->data_at(b, t * 5 + 2, 0, 0);
                Dtype th = blob_truth->data_at(b, t * 5 + 3, 0, 0);

                Dtype max_iou = -1;

                target_n = -1;
                for (int n = 0; n < num_anchor; n++) {
                    Dtype curr_iou = TBoxIou<Dtype>(0, 0, tw, th, 0, 0, biases[2 * n] / width, biases[2 * n + 1] / height);

                    if (curr_iou > max_iou) {
                        max_iou = curr_iou;
                        target_n = n;
                    }
                }
            }

            *(gt_target_data + b * max_gt * 3 + t * 3 + 0) = target_i;
            *(gt_target_data + b * max_gt * 3 + t * 3 + 1) = target_j;
            *(gt_target_data + b * max_gt * 3 + t * 3 + 2) = target_n;
        }
    }

    for (int b = 0; b < batches; b++) {
        for (int t = 0; t < max_gt; ++t) {
            int target_i = gt_target_.data_at(b, t, 0, 0);
            int target_j = gt_target_.data_at(b, t, 1, 0);
            int target_n = gt_target_.data_at(b, t, 2, 0);

            if (target_i < 0) {
                continue;
            }

            Dtype tx = blob_truth->data_at(b, t * 5 + 0, 0, 0);
            Dtype ty = blob_truth->data_at(b, t * 5 + 1, 0, 0);
            Dtype tw = blob_truth->data_at(b, t * 5 + 2, 0, 0);
            Dtype th = blob_truth->data_at(b, t * 5 + 3, 0, 0);

            *(target_xy_data + target_xy->offset(b, target_n, target_j, target_i)) = tx * width - target_i;
            *(target_xy_data + target_xy->offset(b, target_n+ num_anchor, target_j, target_i)) = ty * height - target_j;
            *(target_wh_data + target_wh->offset(b, target_n, target_j, target_i)) = log(tw * width / biases[2 * target_n]);
            *(target_wh_data + target_wh->offset(b, target_n+ num_anchor, target_j, target_i)) = log(th * height / biases[2 * target_n + 1]);
            *(target_xywh_weight_data + target_xywh_weight->offset(b, target_n, target_j, target_i)) = coord_scale_ * (2 - tw * th);
            *(target_xywh_weight_data + target_xywh_weight->offset(b, target_n + num_anchor, target_j, target_i)) = coord_scale_ * (2 - tw * th);
            
            if (!this->rescore_) {
                *(target_obj_obj_data + target_obj_obj->offset(b, target_n, target_j, target_i)) = 1;
            } else {
                vector<int> index = {b, target_n, target_j, target_i, t};
                *(target_obj_obj_data + target_obj_obj->offset(b, target_n, target_j, target_i)) = 
                    this->ious_.data_at(index);
            }
            *(target_obj_noobj_data + target_obj_noobj->offset(b, target_n, target_j, target_i)) = 
                blob_obj->data_at(b, target_n, target_j, target_i);

            int cls = blob_truth->data_at(b, 5 * t + 4, 0, 0);
            *(target_class_data + target_class->offset(b, target_n, target_j, target_i)) = cls;
        }
    }

    (*seen_images_) += batches;
}

template <typename Dtype>
void RegionTargetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(RegionTargetLayer);
#endif

INSTANTIATE_CLASS(RegionTargetLayer);
REGISTER_LAYER_CLASS(RegionTarget);

}
