#include <algorithm>
#include <vector>
#include <cfloat>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/yolo_co_layer.hpp"
#include "caffe/region_common.hpp"

namespace caffe {

template <typename Dtype>
__global__ void yolo_co_kernel(
    int outer_num, int inner_num, int co_classes, int classes, int max_gt,
    const int* comap_class_data, const int* comap_offset_data, const int* comap_size_data,
    const int* comap_data,
    const float* comap_thresh_data, const float* comap_obj_thresh_data, const float* comap_ixr_data,
    const Dtype* pred_data, const Dtype* bbs_data, const Dtype* truth_data,
    const Dtype* obj_data, Dtype* target_no_obj_data) {
    CUDA_KERNEL_LOOP(index, max_gt * outer_num * co_classes * inner_num) {
        const int s = index % inner_num;
        auto t = index / inner_num;
        const int cidx = t % co_classes;
        t /= co_classes;
        const int n = t % outer_num;
        t /= outer_num;

        auto obj_index = n * inner_num + s;
        // If this is a ground-truth already, nothing to do
        if (target_no_obj_data[obj_index] > 0)
            continue;

        auto offset_nt = n * 5 * max_gt + t * 5;
        Dtype tx = *(truth_data + offset_nt + 0);
        // If no ground-truth at this index
        if (!tx)
            continue;
        Dtype ty = *(truth_data + offset_nt + 1);
        Dtype tw = *(truth_data + offset_nt + 2);
        Dtype th = *(truth_data + offset_nt + 3);
        int cls = *(truth_data + offset_nt + 4); // Ground-truth class
        // we explicitly ignore this zero-length bounding boxes
        if (tw <= 0.00001 || th <= 0.00001)
            continue;

        int bbs_index = obj_index * 4;
        Dtype px = *(bbs_data + bbs_index + 0);
        Dtype py = *(bbs_data + bbs_index + 1);
        Dtype pw = *(bbs_data + bbs_index + 2);
        Dtype ph = *(bbs_data + bbs_index + 3);
        // Same as ground-truth logic:
        // we explicitly ignore this zero-length bounding boxes
        if (pw <= 0.00001 || ph <= 0.00001)
            continue;

        auto size = comap_size_data[cidx];
        auto offset = comap_offset_data[cidx];
        for (int i = 0; i < size; ++i) {
            auto co = comap_data[offset + i]; // class that c may co-occur with
            if (co != cls)
                continue;
            // c may co-occure with co only in one rule, so after this the loop will end

            auto obj_thresh = comap_obj_thresh_data[offset + i];
            auto offset_pred = n * (classes + 1) * inner_num + s;
            auto objectness = pred_data[offset_pred + classes * inner_num];
            if (objectness < obj_thresh)
                break;
            auto c = comap_class_data[cidx];
            auto conf = pred_data[offset_pred + c * inner_num];

            auto thresh = comap_thresh_data[offset + i];
            if (conf < thresh)
                break;
            // Check intersection with co-occured class
            auto ixr_thresh = comap_ixr_data[offset + i];
            auto ix = TBoxIntersection(px, py, pw, ph,
                                       tx, ty, tw, th);
            ix /= (pw * ph); // intersection ratio
            if (ix >= ixr_thresh)
                target_no_obj_data[obj_index] = obj_data[obj_index];

            break;
        }
    }
}

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_obj = bottom[blob_idx++];
    auto blob_no_obj = bottom[blob_idx++];
    auto blob_truth = bottom[blob_idx++];
    auto blob_bbs = bottom[blob_idx++];
    auto blob_pred = bottom[blob_idx++];

    auto target_no_obj = top[0];

    caffe_copy(blob_no_obj->count(), blob_no_obj->gpu_data(), target_no_obj->mutable_gpu_data());

    auto co_classes = comap_class_.count();
    if (!co_classes)
        return;
    auto classes = channels_ - 1;
    yolo_co_kernel << <CAFFE_GET_BLOCKS(max_gt_ * outer_num_ * co_classes * inner_num_),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, inner_num_, co_classes, classes, max_gt_,
                                     comap_class_.gpu_data(), comap_offset_.gpu_data(), comap_size_.gpu_data(),
                                     comap_.gpu_data(),
                                     comap_thresh_.gpu_data(), comap_obj_thresh_.gpu_data(), comap_ixr_.gpu_data(),
                                     blob_pred->gpu_data(), blob_bbs->gpu_data(), blob_truth->gpu_data(),
                                     blob_obj->gpu_data(), target_no_obj->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(YoloCoOccurrenceLayer);

}  // namespace caffe
