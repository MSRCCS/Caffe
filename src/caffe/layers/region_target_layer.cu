
#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"

#include "caffe/layers/region_target_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ExtractBoundingBox(int total, int num_anchor, int height, int width, 
        Dtype* bbs_data, const Dtype* blob_xy_data, const Dtype* blob_wh_data, 
        const Dtype* biases) {
  CUDA_KERNEL_LOOP(index, total) {
      int b = index / (num_anchor * height * width);
      int left = index % (num_anchor * height * width);
      int n = left / (height * width);
      left = left % (height * width);
      int j = left / width;
      int i = left % width;
      Dtype* curr_bbs_data = bbs_data + index * 4;
      int offset_double_bnji = b * (2 * num_anchor) * height * width + n * height * width + j * width + i;
      int offset_double_bnji_next = offset_double_bnji + num_anchor * height * width;
      *(curr_bbs_data + 0) = (*(blob_xy_data + offset_double_bnji) + i) / width;
      *(curr_bbs_data + 1) = (*(blob_xy_data + offset_double_bnji_next) + j) / height;
      *(curr_bbs_data + 2) = exp(*(blob_wh_data + offset_double_bnji)) * biases[2 * n] / width;
      *(curr_bbs_data + 3) = exp(*(blob_wh_data + offset_double_bnji_next)) * biases[2 * n + 1] / height;
  }
}

template <typename Dtype>
__global__ void CalculateIOU(int total, Dtype* iou_data, const Dtype* bbs_data, const Dtype* truth_data, int num_anchor, int height, int width, int max_gt) {
  CUDA_KERNEL_LOOP(index, total) {
      int b = index / (num_anchor * height * width * max_gt);
      int left = index % (num_anchor * height * width * max_gt);
      int n = left / (height * width * max_gt);
      left = left % (height * width * max_gt);
      int j = left / (width * max_gt);
      left = left % (width * max_gt);
      int i = left / max_gt;
      int t = left % max_gt;
      Dtype tx = *(truth_data + b * 5 * max_gt + t * 5 + 0);
      Dtype ty = *(truth_data + b * 5 * max_gt + t * 5 + 1);
      Dtype tw = *(truth_data + b * 5 * max_gt + t * 5 + 2);
      Dtype th = *(truth_data + b * 5 * max_gt + t * 5 + 3);
      Dtype curr_iou = 0;
      if (tx) {
          int curr_index = (b * num_anchor * height * width + n * height * width + j * width + i) * 4;
          Dtype px = *(bbs_data + curr_index + 0);
          Dtype py = *(bbs_data + curr_index + 1);
          Dtype pw = *(bbs_data + curr_index + 2);
          Dtype ph = *(bbs_data + curr_index + 3);
          curr_iou = TBoxIou(px, py, pw, ph, 
                  tx, ty, tw, th);
      }
      *(iou_data + index) = curr_iou;
  }
}

template <typename Dtype>
__global__ void NoPenaltyIfIouLargeEnough(int total, Dtype positive_thresh, 
        const Dtype* iou_data, const Dtype* blob_obj_data, Dtype* target_obj_noobj_data, int max_gt) {
    CUDA_KERNEL_LOOP(index, total) {
        auto curr_iou = *(iou_data + index);
        if (curr_iou > positive_thresh) {
            // multiple threads might write this address at the same time, but
            // at least one will succeeds. It is safe to do this. 
            *(target_obj_noobj_data + index / max_gt) = 
                *(blob_obj_data + index / max_gt);
        }
    }
}


template <typename Dtype> 
__global__ void GroundTruthTarget(int total, int max_gt, 
        const Dtype* truth_data, int num_anchor, int height, int width, 
        const Dtype* biases, int* gt_target_data) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / max_gt;
        int t = index % max_gt;
        Dtype tx = *(truth_data + b * max_gt * 5 + 5 * t + 0);
        Dtype ty = *(truth_data + b * max_gt * 5 + 5 * t + 1);
        
        int target_i = -1;
        int target_j = -1;
        int target_n = -1;
        if (tx) {
            target_i = tx * width;
            target_j = ty * height;
            Dtype tw = *(truth_data + b * max_gt * 5 + 5 * t + 2);
            Dtype th = *(truth_data + b * max_gt * 5 + 5 * t + 3);

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

template <typename Dtype>
__global__ void RemoveDuplicateTarget(int total, int* gt_target_data, int max_gt) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / (max_gt * max_gt);
        int left_index = index % (max_gt * max_gt);
        int left_t = left_index / max_gt;
        int right_t = left_index % max_gt;
        if (left_t == right_t) {
            continue;
        }

        int left_target_i = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 0);
        int left_target_j = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 1);
        int left_target_n = *(gt_target_data + b * max_gt * 3 + left_t * 3 + 2);
        if (left_target_i < 0) {
            continue;
        }

        int right_target_i = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 0);
        int right_target_j = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 1);
        int right_target_n = *(gt_target_data + b * max_gt * 3 + right_t * 3 + 2);
        if (right_target_i < 0) {
            continue;
        }
        if (left_target_i == right_target_i && 
                left_target_j == right_target_j &&
                left_target_n == right_target_n) {
            if (left_t < right_t) {
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 0) = -1;
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 1) = -1;
                *(gt_target_data + b * max_gt * 3 + left_t * 3 + 2) = -1;
            } else {
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 0) = -1;
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 1) = -1;
                *(gt_target_data + b * max_gt * 3 + right_t * 3 + 2) = -1;
            }
        }
    }
}

template <typename Dtype> 
__global__ void AlignGroudTruth(int total, const int* gt_target_data, int max_gt,
        const Dtype* truth_data, Dtype* target_xy_data, Dtype* target_wh_data, 
        Dtype* target_xywh_weight_data, Dtype coord_scale,
        int num_anchor, int height, int width, bool rescore, Dtype* target_obj_obj_data, 
        const Dtype* iou_data, Dtype* target_obj_noobj_data, Dtype* target_class_data,
        const Dtype* biases, const Dtype* blob_obj_data) {
    CUDA_KERNEL_LOOP(index, total) {
        int b = index / max_gt;
        int t = index % max_gt;

        int target_i = *(gt_target_data + b * max_gt * 3 + t * 3 + 0);
        int target_j = *(gt_target_data + b * max_gt * 3 + t * 3 + 1);
        int target_n = *(gt_target_data + b * max_gt * 3 + t * 3 + 2);

        if (target_i < 0) {
            continue;
        }

        int offset_bt = b * max_gt * 5 + 5 * t;
        Dtype tx = *(truth_data + offset_bt + 0);
        Dtype ty = *(truth_data + offset_bt + 1);
        Dtype tw = *(truth_data + offset_bt + 2);
        Dtype th = *(truth_data + offset_bt + 3);

        int offset_bnji = b * num_anchor * height * width + target_n * height * width + 
            target_j * width + target_i;

        int offset_double_bnji = offset_bnji + b * num_anchor * height * width;
        int offset_double_bnji_next = offset_double_bnji + num_anchor * width * height;

        *(target_xy_data + offset_double_bnji) = tx * width - target_i;
        *(target_xy_data + offset_double_bnji_next) = ty * height - target_j;
        assert(tw > 0 && th > 0);
        *(target_wh_data + offset_double_bnji) = log(tw * width / biases[2 * target_n]);
        *(target_wh_data + offset_double_bnji_next) = log(th * height / biases[2 * target_n + 1]);
        *(target_xywh_weight_data + offset_double_bnji) = coord_scale * (2 - tw * th);
        *(target_xywh_weight_data + offset_double_bnji_next) = coord_scale * (2 - tw * th);

        if (!rescore) {
            *(target_obj_obj_data + offset_bnji) = 1;
        } else {
            *(target_obj_obj_data + offset_bnji) =  *(iou_data + offset_bnji * max_gt + t);
        }
        *(target_obj_noobj_data + offset_bnji) = *(blob_obj_data + offset_bnji);

        int cls = *(truth_data + offset_bt + 4);
        *(target_class_data + offset_bnji) = cls;
    }
}

template <typename Dtype>
void RegionTargetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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

    auto blob_xy_data = blob_xy->gpu_data();
    auto blob_wh_data = blob_wh->gpu_data();
    auto blob_obj_data = blob_obj->gpu_data();
    auto truth_data = blob_truth->gpu_data();

    auto target_xy_data = target_xy->mutable_gpu_data();
    auto target_wh_data = target_wh->mutable_gpu_data();
    auto target_xywh_weight_data = target_xywh_weight->mutable_gpu_data();
    auto target_obj_noobj_data = target_obj_noobj->mutable_gpu_data();
    auto target_obj_obj_data = target_obj_obj->mutable_gpu_data();
    auto target_class_data = target_class->mutable_gpu_data();

    auto biases = this->biases_.gpu_data();
    auto iou_data = this->ious_.mutable_gpu_data();
    auto bbs_data = this->bbs_.mutable_gpu_data();
    auto gt_target_data = this->gt_target_.mutable_gpu_data();

    // if the iou is large enough, let's not penalize the objectiveness
    if ((*seen_images_) * Caffe::solver_count() < this->anchor_aligned_images_) {
        // if it is at the very begiining, let's align the output
        caffe_gpu_set(target_xy->count(), Dtype(0.5), target_xy_data);
        caffe_gpu_set(target_wh->count(), Dtype(0), target_wh_data);
        caffe_gpu_set(target_xywh_weight->count(), Dtype(0.01), target_xywh_weight_data);
    } else {
        // by default, we set the target of xywh as itself, that mean 0 penalty
        caffe_copy(target_xy->count(), blob_xy->gpu_data(), target_xy_data);
        caffe_copy(target_wh->count(), blob_wh->gpu_data(), target_wh_data);
        caffe_gpu_set(target_xywh_weight->count(), (Dtype)0., target_xywh_weight_data);
    }
    
    // for no-objectiveness, by default all of them be 0. we will zero-out the
    // position if it is 1) gt or 2) the predicted result is good enought
    caffe_gpu_set(target_obj_noobj->count(), (Dtype)0, target_obj_noobj_data);
    
    // For this one, we will only pernalize the position which should be
    // responsible for the gt
    caffe_copy(target_obj_obj->count(), blob_obj->gpu_data(), target_obj_obj_data);

    // by default, dont penalize the results
    caffe_gpu_set(target_class->count(), (Dtype)-1, target_class_data);

    caffe_gpu_set(this->ious_.count(), (Dtype)0, iou_data);

    int batches = blob_xy->num();
    int height = blob_xy->height();
    int width = blob_xy->width();
    int num_anchor = blob_xy->channels() / 2;

    const int max_gt = blob_truth->channels() / 5;
    CHECK_EQ(blob_truth->height(), 1);
    CHECK_EQ(blob_truth->width(), 1);

    int total = batches * num_anchor * height * width;

    ExtractBoundingBox<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, num_anchor, height, width, 
        bbs_data, blob_xy_data, blob_wh_data, biases);
    CUDA_POST_KERNEL_CHECK;

    total = batches * num_anchor * height * width * max_gt;
    CalculateIOU<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, iou_data, 
            bbs_data, truth_data, num_anchor, height, width, max_gt);
    CUDA_POST_KERNEL_CHECK;

    NoPenaltyIfIouLargeEnough<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, this->positive_thresh_, 
        iou_data, blob_obj->gpu_data(), target_obj_noobj_data, max_gt);
    CUDA_POST_KERNEL_CHECK;

    total = batches * max_gt;
    GroundTruthTarget<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, max_gt, 
        blob_truth->gpu_data(), num_anchor, height, width, 
        biases, gt_target_data);
    CUDA_POST_KERNEL_CHECK;

    total = max_gt * max_gt * batches;
    RemoveDuplicateTarget<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, gt_target_data, max_gt);

    total = batches * max_gt;
    AlignGroudTruth<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, gt_target_data, max_gt,
        blob_truth->gpu_data(), target_xy_data, target_wh_data, 
        target_xywh_weight_data, coord_scale_,
        num_anchor, height, width, this->rescore_, target_obj_obj_data, 
        iou_data, target_obj_noobj_data, target_class_data, biases, blob_obj_data);
    CUDA_POST_KERNEL_CHECK;

    (*seen_images_) += batches;
}

template <typename Dtype>
void RegionTargetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionTargetLayer);

}  // namespace caffe
