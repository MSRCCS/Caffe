#include <algorithm>
#include <vector>
#include <math.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cuda.h"

#include "caffe/layers/nms_filter_layer.hpp"
#include "caffe/region_common.hpp"

namespace caffe {

template <typename Dtype>
__device__ void bottom_up_argmerge(const Dtype* p, 
                                   int left, int right, int end,
                                   const int* src, int* dst) {
    int i = left;
    int j = right;
    // Merge 2 already sorted lists
    for (int k = left; k < end; ++k) {
        if (i < right && (j >= end || p[src[i]] > p[src[j]])) {
            dst[k] = src[i];
            i++;
        } else {
            dst[k] = src[j];
            j++;
        }
    }
}

template <typename Dtype>
__global__ void kernel_channel_argmergesort(
    int outer_num, int channels, int inner_num, int classes, int first_class,
    int width, int chunks,
    const Dtype* data,
    int* src, int* dst) {
    CUDA_KERNEL_LOOP(index, outer_num * classes * chunks) {
        const int i = index % chunks;
        const int c_idx = (index / chunks) % classes;
        const int c = c_idx + first_class;
        const int n = (index / chunks) / classes;
        const int dim = (n * channels + c) * inner_num;
        const int idx_dim = (n * classes + c_idx) * inner_num;
        int left = i * width;
        int right = min(left + width / 2, inner_num);
        int end = min(left + width, inner_num);
        int* src_idx = src + idx_dim;
        int* dst_idx = dst + idx_dim;
        if (width == 2) {
            // Initialize the index
            if (right < end)
                src_idx[right] = left + 1;
            src_idx[left] = left + 0;
        }
        bottom_up_argmerge(data + dim,
                           left, right, end,
                           src_idx, dst_idx);
    }
}

template <typename Dtype>
__global__ void kernel_pre_filter(
    int outer_num, int channels, int inner_num, int classes, int first_class,
    float thresh,
    Dtype* top_conf_data) {
    CUDA_KERNEL_LOOP(index, outer_num * classes * inner_num) {
        const int s = index % inner_num;
        const int c = (index / inner_num) % classes + first_class;
        const int n = (index / inner_num) / classes;
        int dim = (n * channels + c) * inner_num + s;
        if (top_conf_data[dim] <= thresh)
            top_conf_data[dim] = 0;
    }
}

template <typename Dtype>
__global__ void kernel_nms_filter(
    int outer_num, int channels, int inner_num, int classes, int first_class,
    const int* idx,
    const Dtype* bbs_data, float thresh,
    Dtype* top_conf_data) {
    CUDA_KERNEL_LOOP(index, outer_num * classes) {
        const int c_idx = index % classes;
        const int c = c_idx + first_class;
        const int n = index / classes;
        const int dim = (n * channels + c) * inner_num;
        const int idx_dim = (n * classes + c_idx) * inner_num;
        const int* src_idx = idx + idx_dim;
        for (int i_idx = 0; i_idx < inner_num; ++i_idx) {
            int i = src_idx[i_idx];
            if (top_conf_data[dim + i] == 0)
                continue;
            auto i_bb = bbs_data + (n * inner_num + i) * 4;
            for (int j_idx = i_idx + 1; j_idx < inner_num; ++j_idx) {
                int j = src_idx[j_idx];
                if (top_conf_data[dim + j] == 0)
                    continue;
                auto j_bb = bbs_data + (n * inner_num + j) * 4;
                Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3],
                                                j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
                if (curr_iou > thresh)
                    top_conf_data[dim + j] = 0;
            }
        }
    }
}

template <typename Dtype>
void NMSFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_bbs = bottom[blob_idx++];
    auto blob_conf = bottom[blob_idx++];
    auto conf_data = blob_conf->gpu_data();

    auto bbs_data = blob_bbs->gpu_data();

    auto top_conf = top[0];
    auto top_conf_data = top_conf->mutable_gpu_data();
    caffe_copy(blob_conf->count(), conf_data, top_conf_data);

    int actual_classes = classes_;
    if (actual_classes <= 0)
        actual_classes = channels_;
    if (thresh_ >= 0) {
        kernel_pre_filter << <CAFFE_GET_BLOCKS(outer_num_ * actual_classes * inner_num_),
            CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels_, inner_num_, actual_classes, first_class_,
                                         thresh_,
                                         top_conf_data
                                         );
    }

    if (nms_ <= 0 || inner_num_ == 1)
        return;

    int* idx_data = idx_.mutable_gpu_data();
    // This memory is safe to release afterwards but we keep it in GPU memory, 
    //  if there is a shortage of memory we can revisit this logic
    int* idx_tmp = idx_.mutable_gpu_diff();
    // Start swapped if loop runs for an odd number
    bool is_swapped = ((int)ceil(log2((double)inner_num_))) % 2 != 0;
    // TODO: Use dynamic parallelism for devices with 3.5 compute capability, and implement top_down
    for (int width = 2; width < inner_num_ * 2; width *= 2) {
        int chunks = (inner_num_ + width - 1) / width;
        int* src_idx = is_swapped ? idx_tmp : idx_data;
        int* dst_idx = is_swapped ? idx_data : idx_tmp;
        kernel_channel_argmergesort << <CAFFE_GET_BLOCKS(outer_num_ * actual_classes * chunks),
            CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels_, inner_num_, actual_classes, first_class_,
                                            width, chunks,
                                            conf_data,
                                            src_idx, dst_idx);
        CUDA_POST_KERNEL_CHECK;
        is_swapped = !is_swapped;
    }

    kernel_nms_filter << <CAFFE_GET_BLOCKS(outer_num_ * actual_classes),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels_, inner_num_, actual_classes, first_class_,
                                     idx_.gpu_data(),
                                     bbs_data, nms_,
                                     top_conf_data
                                     );
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NMSFilterLayer);

}  // namespace caffe

